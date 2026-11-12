from pathlib import Path
import numpy as np
import json
import random
import math

from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, p=0.5, partition=True):

        if label_path not in ("train", "val"):
            raise ValueError(f"Unknown NW-UCLA split: {label_path}")
        self.train_val = label_path
        manifest = Path(__file__).with_name("nwucla_splits.json")
        self.data_dict = json.loads(manifest.read_text(encoding="utf-8"))[label_path]
        self.nw_ucla_root = Path(data_path) / "all_sqe"
        self.time_steps = 64
        self.label = [int(info['label']) - 1 for info in self.data_dict]
        self.partition = partition
        self.repeat = repeat
        self.p = p
        self.load_data()

        if self.partition:
            self.right_arm = np.array([5, 6, 7, 8]) - 1
            self.left_arm = np.array([9, 10, 11, 12]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.torso), axis=-1)

    def load_data(self):

        self.data = []
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.nw_ucla_root / f"{file_name}.json", 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            if value.ndim != 3 or value.shape[1:] != (20, 3) or not len(value):
                raise ValueError(f"Expected NW-UCLA skeleton [T, 20, 3] in {file_name}")
            self.data.append(value)

    def __len__(self):
        return len(self.data_dict) * self.repeat

    @staticmethod
    def normalize(coordinates):
        minimum = coordinates.min(axis=0)
        span = coordinates.max(axis=0) - minimum
        return (coordinates - minimum) / np.where(span > 0, span, 1)

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = self.normalize(scalerValue)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.time_steps, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]
            index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1


            if random.random() < self.p:
                axis_next = random.randint(0, 2)
                temp = data.copy()
                T, V, C = data.shape
                x_new = np.zeros((T, V))
                temp[:, :, axis_next] = x_new
                data = temp


            if random.random() < self.p:
                temp = data.copy()
                T, V, C = data.shape
                random_int_v = random.randint(4, 12)
                all_joints = [i for i in range(V)]
                joint_list_ = random.sample(all_joints, random_int_v)
                joint_list_ = sorted(joint_list_)
                random_int_t = random.randint(16, 32)
                all_frames = [i for i in range(T)]
                time_range_ = random.sample(all_frames, random_int_t)
                time_range_ = sorted(time_range_)
                x_new = np.zeros((len(time_range_), len(joint_list_), C))
                temp2 = temp[time_range_, :, :].copy()
                temp2[:, joint_list_, :] = x_new
                temp[time_range_, :, :] = temp2
                data = temp

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = self.normalize(scalerValue)
            scalerValue = scalerValue * 2 - 1

            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.time_steps, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = value[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        if self.partition:
            data = data[:, :, self.new_idx]

        return data, index_t, label, index
