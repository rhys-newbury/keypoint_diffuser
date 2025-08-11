import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

KEYS = {'table': 18, 'car': 13, 'airplane': 0, 'cabinet': 9, 'birdhouse': 53, 'sofa': 48, 'bus': 8, 'chair': 14, 'rifle': 45, 'pot': 42, 'vessel': 50, 'bench': 5, 'monitor': 17, 'bathtub': 3, 'knife': 30, 'mailbox': 34, 'faucet': 25, 'telephone': 19, 'bottle': 6, 'lamp': 31, 'tower': 21, 'clock': 15, 'speaker': 33, 'microwave': 36, 'bowl': 7, 'remote_control': 44, 'skateboard': 47, 'tin_can': 20, 'laptop': 32, 'piano': 39, 'cellphone': 52, 'bed': 4, 'printer': 43, 'helmet': 28, 'dishwasher': 16, 'guitar': 27, 'can': 10, 'bookshelf': 54, 'file': 26, 'train': 22, 'jar': 29, 'mug': 38, 'washer': 51, 'motorcycle': 37, 'pistol': 41, 'stove': 49, 'camera': 11, 'pillow': 40, 'earphone': 24, 'bag': 1, 'basket': 2, 'keyboard': 23, 'cap': 12, 'rocket': 46, 'microphone': 35}

class H5Dataset(Dataset):
    def __init__(self, h5_paths, normalize=False, include_label=False,
                 object_name="chair", transform=None):
        self.files = h5_paths
        self.normalize = normalize
        self.include_label = include_label
        self.object_name = object_name
        self.transform = transform

        self.data = []
        self.labels = []
        
        assert self.object_name in KEYS.keys()

        # Load only metadata, not full arrays
        for path in self.files:
            with h5py.File(path, 'r') as f:
                x = f['data'][:]
                y = f['label'][:]
              

                for i in range(len(x)):
                    label = y[i][0]
                    if KEYS[self.object_name] == label:
                        self.data.append(x[i])
                        if include_label:
                            self.labels.append(label)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc = self.data[idx]

        if self.normalize:
            dmin = pc.min(axis=0, keepdims=True)
            dmax = pc.max(axis=0, keepdims=True)
            pc = (pc - dmin) / (dmax - dmin + 1e-8)  # Avoid division by zero
            pc = 2.0 * (pc - 0.5)

        pc = torch.tensor(pc, dtype=torch.float32)

        if self.transform:
            transformed, deformed = self.transform(
                {"coord": pc.cpu().numpy()}
            )  # Apply transform
            sample = {
                **{"target_shape": pc},
                **{
                    f"orig_{key}": value.cuda()
                    for key, value in transformed.items()
                },
                **{
                    f"deformed_{key}": value.cuda()
                    for key, value in deformed.items()
                },
            }

        return sample
