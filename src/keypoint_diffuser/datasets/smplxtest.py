import collections.abc as container_abcs
import json
import os
import random
import traceback
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


class Smplxtest(Dataset):
    DO_NOT_BATCH = ["smplx_path", "point_clouds"]

    def __init__(self, opt, split="train", normalize=True, transform=None):
        """
        Args:
            opt: options including opt.split, etc.
            split (str): 'train' or 'test'.
            normalize (bool): Normalize point clouds to unit box.
            transform (callable): Optional transform to apply.
        """
        self.opt = opt
        split_file = "data/shapenet_split/smplx.json"  # Now a JSON file
        self.split = split
        self.normalize = normalize
        self.transform = transform

        self.samples = self._load_split_file(split_file)
        print(f"Loaded {len(self.samples)} {split} samples.")

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def _load_split_file(self, split_file):
        with open(split_file) as f:
            data = json.load(f)
        # JSON expected format: {"train": [...], "test": [...]}
        return data

    def _get_point_cloud_paths(self, base_dir):
        return [os.path.join(base_dir, f"new_samples_{i}.npy") for i in range(10)]

    def _load_point_cloud_sequence(self, base_dir):
        paths = self._get_point_cloud_paths(base_dir)
        point_clouds = []
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing point cloud: {path}")
            points = np.load(path)
            if self.normalize:
                points[:, :3] = self._normalize_to_unit_box(points[:, :3])
            point_clouds.append(torch.from_numpy(points).float())
        return point_clouds  # List of 10 tensors

    def _normalize_to_unit_box(self, points):
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (max_xyz + min_xyz) / 2
        scale = (max_xyz - min_xyz).max()
        return (points - center) / scale

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        for _ in range(10):  # Retry logic
            try:
                trajectory_key = f"trajectory_{idx}"
                smplx_paths = self.samples["trajectories"][trajectory_key]

                point_clouds = []
                for smplx_pkl_path in smplx_paths:
                    frame_dir = os.path.dirname(
                        os.path.join("/app/data/smplx", smplx_pkl_path)
                    )

                    # List all candidate npy files
                    npy_candidates = [
                        os.path.join(frame_dir, f"new_samples_{i}.npy")
                        for i in range(6)
                    ]
                    npy_available = [p for p in npy_candidates if os.path.isfile(p)]

                    if not npy_available:
                        raise FileNotFoundError(
                            f"No new_samples_.npy found in {frame_dir}"
                        )

                    # Randomly select one
                    npy_path = random.choice(npy_available)

                    points = np.load(npy_path)
                    if self.normalize:
                        points[:, :3] = self._normalize_to_unit_box(points[:, :3])
                    point_clouds.append(torch.from_numpy(points).float())

                sample = {
                    "smplx_path": smplx_paths,  # List of frame pkl paths
                    "point_clouds": point_clouds,  # List[Tensor], length = sequence length
                }

                if self.transform:
                    transformed_acc = defaultdict(list)
                    deformed_acc = defaultdict(list)

                    for pc in point_clouds:
                        transformed, deformed = self.transform({"coord": pc.numpy()})
                        for k, v in transformed.items():
                            transformed_acc[k].append(torch.tensor(v))
                        for k, v in deformed.items():
                            deformed_acc[k].append(torch.tensor(v))

                    sample.update(
                        {
                            **{f"orig_{k}": (v) for k, v in transformed_acc.items()},
                            **{f"deformed_{k}": (v) for k, v in deformed_acc.items()},
                        }
                    )

                return sample

            except Exception as e:
                warnings.warn(
                    f"Error loading sample {idx} ({self.samples['trajectories'][f'trajectory_{idx}']}):\n"
                    + "".join(traceback.format_exception(type(e), e, e.__traceback__))
                )
                idx = (idx + 1) % len(self)

    @staticmethod
    def collate(batch):
        batched = {}
        for key in batch[0]:
            if key in Smplxtest.DO_NOT_BATCH:
                batched[key] = [item[key] for item in batch]  # List of samples
            else:
                try:
                    batched[key] = default_collate([item[key] for item in batch])
                except Exception as e:
                    print(f"Collate error for key '{key}': {e}")
        return batched

    @staticmethod
    def uncollate(batched):
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.cuda()
            elif isinstance(v, container_abcs.Sequence) and isinstance(
                v[0], torch.Tensor
            ):
                batched[k] = [e.cuda() for e in v]
        return batched
