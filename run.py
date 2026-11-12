import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from feeders.ntu import Feeder as NTUFeeder
from feeders.nwucla import Feeder as NWUCLAFeeder
from models.ntu import WGCN as NTUModel
from models.nwucla import WGCN as NWUCLAModel


ROOT = Path(__file__).resolve().parent
VARIANTS = {"ntu": (NTUModel, NTUFeeder), "nwucla": (NWUCLAModel, NWUCLAFeeder)}


def seed_worker(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_loader(feeder, path, options, batch_size, workers, training):
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers must be nonnegative")
    dataset = feeder(data_path=path, **options)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, worker_init_fn=seed_worker,
    )
    if not len(loader):
        raise ValueError(f"No batches available for {path}; check the data and batch size")
    return loader


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros(2, device=device, dtype=torch.long)
    total = 0
    for data, _, labels, _ in tqdm(loader, desc="Evaluate", leave=False):
        data = data.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        logits = model(data)
        prediction = logits.topk(5, dim=1).indices
        matches = prediction.eq(labels[:, None])
        correct += torch.stack((matches[:, 0].sum(), matches.any(dim=1).sum()))
        total += labels.numel()
    top1, top5 = (correct.cpu().numpy() * (100.0 / total)).tolist()
    return {"samples": total, "top1": top1, "top5": top5}


def train(model, train_loader, test_loader, device, cfg, epochs, run_dir, eval_every_epoch_from):
    from timm.scheduler.cosine_lr import CosineLRScheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    warmup_steps = min(cfg["warm_up_epoch"], max(epochs - 1, 0)) * steps_per_epoch
    schedule_steps = epochs * steps_per_epoch
    scheduler = CosineLRScheduler(
        optimizer, t_initial=schedule_steps - warmup_steps if cfg["warmup_prefix"] else schedule_steps,
        lr_min=cfg["min_lr"], warmup_lr_init=cfg["warmup_lr"], warmup_t=warmup_steps,
        cycle_limit=1, t_in_epochs=False, warmup_prefix=cfg["warmup_prefix"],
    )
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        batches = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for step, (data, _, labels, _) in enumerate(batches):
            scheduler.step_update((epoch - 1) * steps_per_epoch + step)
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)
            negative_log_likelihood = -log_probs.gather(1, labels[:, None]).squeeze(1)
            loss = (0.9 * negative_log_likelihood - 0.1 * log_probs.mean(dim=1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_max"], error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0:
                batches.set_postfix(loss=f"{loss.item():.4f}")
        if epoch == epochs or epoch % 5 == 0 or epoch >= epochs * eval_every_epoch_from:
            metrics = {"epoch": epoch, **evaluate(model, test_loader, device)}
            print(json.dumps(metrics), flush=True)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if metrics["top1"] > best:
                best = metrics["top1"]
                weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
                temporary = run_dir / "best.pt.tmp"
                torch.save(weights, temporary)
                temporary.replace(run_dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a joint-stream WST-SGCN (large) model.")
    parser.add_argument("mode", nargs="?", choices=("train", "eval"), help="Override config phase")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", type=Path, help="Override the prepared NTU NPZ file or NW-UCLA JSON directory")
    parser.add_argument("--weights", help="Override config weights; use none for training from scratch")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, for example cuda:0 or cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int, help="Override the training batch size")
    parser.add_argument("--test-batch-size", type=int)
    parser.add_argument("--epochs", type=int, help="Override the training epoch count")
    parser.add_argument("--run-dir", "--work-dir", dest="run_dir", type=Path, help="Override config work_dir with a new training output directory")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    phase = cfg.get("phase", "test")
    if phase not in ("train", "test"):
        parser.error("config phase must be train or test")
    mode = args.mode or ("eval" if phase == "test" else "train")
    weights_value = args.weights if args.weights is not None else cfg.get("weights")
    weights_path = None
    if weights_value is not None and str(weights_value).lower() not in ("none", "null"):
        weights_path = Path(weights_value)
        if args.weights is None:
            weights_path = ROOT / weights_path
    run_dir = args.run_dir
    if run_dir is None and cfg.get("work_dir"):
        run_dir = ROOT / cfg["work_dir"]

    if mode == "eval" and weights_path is None:
        parser.error("evaluation requires weights in the config or --weights")
    if mode == "train" and run_dir is None:
        parser.error("training requires work_dir in the config or --run-dir")
    if mode == "train" and run_dir.exists():
        parser.error(f"training output already exists: {run_dir}")
    if weights_path is not None and not weights_path.is_file():
        parser.error(f"checkpoint not found: {weights_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select an available device with --device.")

    settings = cfg["training"]
    data_path = args.data if args.data is not None else ROOT / cfg["data_path"]
    if not data_path.exists():
        parser.error(f"prepared dataset not found: {data_path}. Set config data_path or --data to its location.")
    seed_worker(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_cls, feeder = VARIANTS[cfg["model"]]
    model = model_cls(**cfg["model_args"])
    if weights_path is not None:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    model.to(device)
    workers = settings["num_worker"] if args.workers is None else args.workers
    test_batch = settings["test_batch_size"] if args.test_batch_size is None else args.test_batch_size
    test_loader = build_loader(feeder, data_path, cfg["test_data"], test_batch, workers, training=False)
    if mode == "eval":
        print(json.dumps(evaluate(model, test_loader, device), indent=2))
        return

    epochs = settings["num_epoch"] if args.epochs is None else args.epochs
    if epochs < 1:
        parser.error("--epochs must be positive")
    batch_size = settings["batch_size"] if args.batch_size is None else args.batch_size
    train_loader = build_loader(feeder, data_path, cfg["train_data"], batch_size, workers, training=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {**cfg, "phase": "train", "data_path": str(data_path),
                  "weights": str(weights_path) if weights_path is not None else None,
                  "work_dir": str(run_dir),
                  "training": {**settings, "num_epoch": epochs, "batch_size": batch_size,
                                     "test_batch_size": test_batch, "num_worker": workers}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    eval_every_epoch_from = 0.0 if cfg["model"] == "nwucla" else 0.8
    train(model, train_loader, test_loader, device, settings, epochs, run_dir, eval_every_epoch_from)


if __name__ == "__main__":
    main()
