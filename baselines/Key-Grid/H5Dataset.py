import os
import random
import numpy
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch


class H5Dataset(Dataset):
    def __init__(self, h5_paths, normalize=False, include_label=False,
                 subclasses=tuple(range(40)), transform=None):
        self.files = h5_paths
        self.normalize = normalize
        self.include_label = include_label
        self.subclasses = subclasses
        self.transform = transform
        # self.sample = sample

        self.data = []
        self.labels = []

        # Load only metadata, not full arrays
        for path in self.files:
            with h5py.File(path, 'r') as f:
                x = f['data'][:]
                y = f['label'][:]

                for i in range(len(x)):
                    label = y[i][0] if y is not None else None
                    if label in subclasses:
                        self.data.append(x[i])
                        if include_label:
                            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc = self.data[idx]

        # Optional subsampling
        # if self.sample is not None:
        #     idxs = np.random.choice(len(pc), self.sample, replace=True)
        #     pc = pc[idxs]

        if self.normalize:
            dmin = pc.min(axis=0, keepdims=True)
            dmax = pc.max(axis=0, keepdims=True)
            pc = (pc - dmin) / (dmax - dmin + 1e-8)  # Avoid division by zero
            pc = 2.0 * (pc - 0.5)

        pc = torch.tensor(pc, dtype=torch.float32)

        return pc
