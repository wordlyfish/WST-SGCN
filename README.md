# WST-SGCN

PyTorch implementation of **WST-SGCN: Wavelet-Based Spatial-Temporal Spectral Graph Convolutional Networks for Skeleton-Based Action Recognition**.

This repository provides the large, joint-stream models and pretrained weights for NTU RGB+D 60, NTU RGB+D 120, and NW-UCLA. The architecture consists of eight MSW-SGCN and DeepTCN blocks with four learnable Morlet wavelet scales.

## Pretrained models

| Dataset and protocol | Config | Weights | Top-1 (%) |
| --- | --- | --- | ---: |
| NTU RGB+D 60 X-Sub | [ntu60_xsub](configs/ntu60_xsub.yaml) | [Download](checkpoints/ntu60_xsub.pt) | 91.84 |
| NTU RGB+D 60 X-View | [ntu60_xview](configs/ntu60_xview.yaml) | [Download](checkpoints/ntu60_xview.pt) | 95.89 |
| NTU RGB+D 120 X-Sub | [ntu120_xsub](configs/ntu120_xsub.yaml) | [Download](checkpoints/ntu120_xsub.pt) | 87.47 |
| NTU RGB+D 120 X-Set | [ntu120_xset](configs/ntu120_xset.yaml) | [Download](checkpoints/ntu120_xset.pt) | 89.10 |
| NW-UCLA | [nwucla](configs/nwucla.yaml) | [Download](checkpoints/nwucla.pt) | 96.12 |

Checkpoint SHA-256 checksums and evaluation metrics are available in [checkpoints/manifest.json](checkpoints/manifest.json).

## Installation

Install the dependencies in a Python 3.9 environment.

```bash
python -m pip install -r requirements.txt
```

The dependency file specifies PyTorch 2.4.1 and torchvision 0.19.1. GPU execution requires a CUDA-enabled PyTorch installation.

## Datasets

Use prepared datasets with the following layout, or set `data_path` in the corresponding config to their location.

```text
datasets/
  NTU60_CS.npz
  NTU60_CV.npz
  NTU120_CSub.npz
  NTU120_CSet.npz
  NW-UCLA/
    all_sqe/
      *.json
```

### NTU RGB+D 60 and 120

Each NPZ archive contains the following arrays.

| Key | Shape | Content |
| --- | --- | --- |
| `x_train` | `[N_train, T, 150]` | Training skeleton sequences |
| `y_train` | `[N_train, C]` | One-hot training labels |
| `x_test` | `[N_test, T, 150]` | Test skeleton sequences |
| `y_test` | `[N_test, C]` | One-hot test labels |

`C` is 60 or 120. The last feature dimension stores coordinates in person, joint, and coordinate order, with two person slots, 25 joints, and three coordinates per joint. The loader samples sequences to 128 frames and loads each split into memory. Use the same dataset split, coordinate convention, and joint order as the pretrained model.

### NW-UCLA

Each JSON file contains a `skeletons` array with shape `[T, 20, 3]`. Store these files in `all_sqe` under the directory specified by `data_path`. Sample names, labels, and split order are defined in [feeders/nwucla_splits.json](feeders/nwucla_splits.json). The loader samples sequences to 64 frames.

## Evaluation

Run commands from the repository root. Each supplied config uses `phase: test` and selects its corresponding pretrained weights.

```bash
python run.py --config configs/ntu60_xsub.yaml
```

Replace `ntu60_xsub` with another config name from the pretrained-model table. To use datasets stored elsewhere, pass their path with `--data`.

```bash
python run.py --config configs/ntu60_xsub.yaml --data /path/to/NTU60_CS.npz
python run.py --config configs/nwucla.yaml --data /path/to/NW-UCLA
```

The command reports Top-1 and Top-5 accuracy on the configured test split. Select a device with `--device cuda:0` or `--device cpu`. The default device is `cuda:0`.

## Training

Copy a config for the dataset and protocol to train.

```bash
cp configs/ntu60_xsub.yaml configs/my_ntu60_xsub.yaml
```

For training from random initialization, set these fields in the copied config.

```yaml
phase: train
weights: null
work_dir: runs/my_ntu60_xsub
```

Start training with the copied config.

```bash
python run.py --config configs/my_ntu60_xsub.yaml
```

Set `data_path` in the config or pass `--data` if the dataset is stored elsewhere. Use a new `work_dir` for each run. Training saves `best.pt`, `metrics.jsonl`, and the effective `config.yaml` in that directory.

To fine-tune a pretrained model, set `weights` to its checkpoint path. Each training run creates a new optimizer and learning-rate schedule.

Command-line arguments override config values. For example, train from random initialization using an existing config with the following command.

```bash
python run.py train --config configs/ntu60_xsub.yaml --weights none --run-dir runs/my_ntu60_xsub
```

Training uses one device. NW-UCLA is evaluated every epoch. NTU is evaluated every five epochs before the final 20% of training and every epoch thereafter.

## Configuration

| Field | Purpose |
| --- | --- |
| `phase` | `train` or `test` |
| `weights` | Checkpoint path for evaluation or model initialization. Use `null` for training from random initialization. |
| `work_dir` | Output directory for a new training run |
| `data_path` | NTU NPZ file or NW-UCLA directory containing `all_sqe` |
| `model_args` | Class count, skeleton dimensions, channel configuration, and dropout |
| `train_data` / `test_data` | Sampling and augmentation settings |
| `training` | Epoch count, batch sizes, learning rates, and optimizer settings |

Relative paths in config files resolve from the repository root. The wavelet parameters are defined in `models/ntu.py` and `models/nwucla.py`. The pretrained models use the `morlet` wavelet and four learnable scales.
