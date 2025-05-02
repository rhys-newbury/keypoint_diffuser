import os
import random
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


class Smplx(Dataset):
    DO_NOT_BATCH = ["smplx_path", "point_clouds"]

    def __init__(self, opt, split="train", normalize=True, transform=None):
        """
        Args:
            split_file (str): Path to CSV or TXT file listing samples and their split.
            split (str): One of 'train' or 'test'.
            normalize (bool): Whether to normalize point clouds to unit box.
            transform (callable): Optional transform to apply to data.
        """
        self.opt = opt
        split_file = "data/shapenet_split/smplx.csv"
        self.split = split
        self.normalize = normalize
        self.transform = transform

        self.samples = self._load_split_file(split_file)
        print(f"Loaded {len(self.samples)} {split} samples.")

    def _load_split_file(self, split_file):
        samples = []
        with open(split_file) as f:
            lines = f.read().splitlines()
        for line in lines:
            path, tag = line.strip().split(",")
            if tag.strip() == self.opt.split:
                samples.append(path.strip())
        return samples

    def _get_point_cloud_paths(self, base_dir):
        return [os.path.join(base_dir, f"new_samples_{i}.npy") for i in range(5)]

    def _load_point_cloud(self, base_dir):
        paths = self._get_point_cloud_paths(base_dir)
        available_paths = [p for p in paths if os.path.isfile(p)]
        if not available_paths:
            raise FileNotFoundError(f"No new_samples_.npy found in {base_dir}")
        path = random.choice(available_paths)
        points = np.load(path)
        if self.normalize:
            points[:, :3] = self._normalize_to_unit_box(points[:, :3])
        return torch.from_numpy(points).float()

    def _normalize_to_unit_box(self, points):
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (max_xyz + min_xyz) / 2
        scale = (max_xyz - min_xyz).max()
        return (points - center) / scale

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                smplx_path = self.samples[idx]
                frame_dir = f"/app/data/smplx/{smplx_path}"
                point_cloud = self._load_point_cloud(frame_dir)

                sample = {
                    "smplx_path": smplx_path,
                    "target_shape": point_cloud,
                }

                if self.transform:
                    transformed, deformed = self.transform(
                        {"coord": sample["target_shape"].cpu().numpy()}
                    )  # Apply transform
                    sample = {
                        **sample,
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

            except Exception as e:
                warnings.warn(
                    f"Error loading sample {idx} ({self.samples[idx]}):\n"
                    + "".join(traceback.format_exception(type(e), e, e.__traceback__))
                )
                idx = (idx + 1) % len(self)

    @staticmethod
    def collate(batch):
        batched = {}
        for key in batch[0]:
            if key in Smplx.DO_NOT_BATCH:
                batched[key] = [item[key] for item in batch]
            else:
                try:
                    batched[key] = default_collate([item[key] for item in batch])
                except Exception as e:
                    print(f"Collate error for key '{key}': {e}")
        return batched
