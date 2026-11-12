import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, split, p_interval, window_size, thres,
                 aug_method="", intra_p=0.5, inter_p=0.0):
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.thres = thres
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.load_data()

    def load_data(self):
        if self.split not in ("train", "test"):
            raise ValueError(f"Unknown NTU split: {self.split}")
        with np.load(self.data_path, allow_pickle=False) as archive:
            self.data = archive[f"x_{self.split}"]
            labels = archive[f"y_{self.split}"]
        if self.data.ndim != 3 or self.data.shape[-1] != 150:
            raise ValueError("Expected NTU features with shape [N, T, 150]")
        if labels.ndim != 2 or len(labels) != len(self.data) or not np.all((labels > 0).sum(axis=1) == 1):
            raise ValueError("Expected one one-hot action label per NTU sample")
        self.label = labels.argmax(axis=1)
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if valid_frame_num == 0:
            raise ValueError(f"NTU sample {index} in {self.data_path} has no valid frames")

        data_numpy, index_t = tools.valid_crop_uniform(
            data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres,
        )

        if self.split == 'train':

            p = np.random.rand(1)
            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)


            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                if f_num == 0:
                    raise ValueError(f"NTU augmentation reference {adain_idx} has no valid frames")
                data_adain = data_adain[:, np.clip(t_idx, 0, data_adain.shape[1] - 1)]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

        return data_numpy, index_t, label, index
