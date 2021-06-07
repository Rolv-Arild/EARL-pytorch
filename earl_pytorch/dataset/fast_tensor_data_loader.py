import h5py
import torch
import numpy as np
import os

# https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
from earl_pytorch.dataset.create_dataset import X_DATASET_NAMES, Y_DATASET_NAMES


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, h5_group, batch_size=128, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """

        self.h5_group = h5_group
        self.x_data = h5_group["x_data"]
        self.y_data = h5_group["y_data"]

        # self.x_data = []
        # for name in ("ball", "boost", "players"):
        #     np_arr = np.load(os.path.join(folder, f"x_{name}-{index}.npy"))
        #     self.x_data.append(torch.from_numpy(np_arr))
        #
        # self.y_data = []
        # for name in ("score", "next_touch", "collect", "demo",
        #              "throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"):
        #     np_arr = np.load(os.path.join(folder, f"y_{name}-{index}.npy"))
        #     np_arr[np_arr < -100] = -100
        #     np_arr[np_arr > 34] = -100
        #     self.y_data.append(torch.from_numpy(np_arr))

        # assert all(t.shape[0] == self.x_data[0].shape[0] for t in self.x_data)
        # assert all(t.shape[0] == self.y_data[0].shape[0] for t in self.y_data)

        self.dataset_len = self.x_data["balls"].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = np.random.permutation(self.dataset_len)
            self.x_data = [self.x_data[name][r] for name in X_DATASET_NAMES]
            self.y_data = [self.y_data[name][r] for name in Y_DATASET_NAMES]
        else:
            self.x_data = [self.x_data[name] for name in X_DATASET_NAMES]
            self.y_data = [self.y_data[name] for name in Y_DATASET_NAMES]

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch_x = tuple(torch.from_numpy(t[self.i:self.i + self.batch_size]).cuda() for t in self.x_data)
        batch_y = tuple(torch.from_numpy(t[self.i:self.i + self.batch_size]).cuda() for t in self.y_data)

        self.i += self.batch_size
        return batch_x, batch_y

    def __len__(self):
        return self.n_batches
