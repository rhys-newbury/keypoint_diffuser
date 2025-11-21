import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from .utils import random_y_rotation_matrix, transform


class PeopleDataset(Dataset):
    """
    Loads converted SMPL-X samples (.ply + .kp.npz).
    Ignores semantic labels, only uses keypoints and midpoints.

    Each item can return:
        (1) {'source_shape': ..., 'target_shape': ...}    if get_two=True
        (2) (pc1, R1, pc2, R2)                            if random_rotate=True
        (3) single dict with optional transform outputs    otherwise
    """

    def __init__(
        self,
        root_dir,
        normalize=True,
        transform=None,
        get_two=False,
        random_rotate=False,
        get_keypoints=False,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.normalize = normalize
        self.transform = transform
        self.get_two = get_two
        self.random_rotate = random_rotate
        self.get_keypoints = get_keypoints

        self.samples = sorted(self.root_dir.glob("*.kp.npz"))
        if not self.samples:
            raise RuntimeError(f"No .kp.npz files found in {root_dir}")

        # preload rotations for all samples if random_rotate
        self.rotations = []
        if self.random_rotate:
            for _ in self.samples:
                self.rotations.append(self._random_poses())
            self.rotations = np.array(self.rotations).reshape(-1, 3, 3)

    def _random_poses(self, n=24):
        """Generate n random Y-axis rotations."""
        return np.stack([random_y_rotation_matrix() for _ in range(n)], axis=0)

    def __len__(self):
        return len(self.samples)

    def _normalize_pc(self, pc):
        pc = pc - pc.mean(0)
        pc /= np.max(np.linalg.norm(pc, axis=-1))
        return pc

    def _load_sample(self, idx):
        kp_path = self.samples[idx]
        ply_path = kp_path.with_name(kp_path.name.replace(".kp.npz", ".ply"))

        seg_path = kp_path.with_name(kp_path.name.replace(".kp.npz", ".semseg.npz"))
        # load all points
        all_points = np.asarray(
            trimesh.load(ply_path, process=False).vertices, dtype=np.float32
        )
        seg_data = np.load(seg_path, allow_pickle=True)

        labels = seg_data.get("labels")

        # load keypoints
        data = np.load(kp_path, allow_pickle=True)
        joints = data.get("joints", np.zeros((0, 3), np.float32))
        midpoints = data.get("midpoints", np.zeros((0, 3), np.float32))
        keypoints = (
            np.concatenate([joints, midpoints], axis=0) if midpoints.size else joints
        )

        return all_points, keypoints, labels

    def __getitem__(self, idx):
        pc, keypoints, labels = self._load_sample(idx)

        if self.random_rotate:
            # SC3K wants 'random' rotations.
            R1 = self.rotations[idx]
            R2 = self.rotations[-(idx + 1)]

            pc1 = transform(pc, R1)
            pc2 = transform(pc, R2)

            pc1 = self._normalize_pc(pc1)
            pc2 = self._normalize_pc(pc2)

            return (
                torch.tensor(pc1, dtype=torch.float32),
                torch.tensor(R1, dtype=torch.float32),
                torch.tensor(pc2, dtype=torch.float32),
                torch.tensor(R2, dtype=torch.float32),
            )

        # === PAIR MODE (for shape deformation training) ===
        if self.get_two:
            rand_idx = random.randrange(len(self.samples))
            pc_tgt, _ = self._load_sample(rand_idx)

            if self.normalize:
                pc = self._normalize_pc(pc)
                pc_tgt = self._normalize_pc(pc_tgt)

            return {
                "source_shape": torch.tensor(pc, dtype=torch.float32),
                "target_shape": torch.tensor(pc_tgt, dtype=torch.float32),
            }

        # === TRANSFORM MODE (augmentation or deformation pipeline) ===
        if self.transform:
            transformed = self.transform({"coord": pc})
            if isinstance(transformed, tuple) and len(transformed) == 2:
                orig, deformed = transformed
                return {
                    "target_shape": torch.tensor(pc, dtype=torch.float32),
                    **{f"orig_{k}": v.cuda() for k, v in orig.items()},
                    **{f"deformed_{k}": v.cuda() for k, v in deformed.items()},
                }
            else:
                return {
                    "target_shape": torch.tensor(pc, dtype=torch.float32),
                    **{f"orig_{k}": v for k, v in transformed.items()},
                }

        # === DEFAULT ===
        if self.normalize:
            pc = self._normalize_pc(pc)
            keypoints = self._normalize_pc(keypoints)

        return (
            pc
            if not self.get_keypoints
            else (np.hstack((pc, labels.reshape(-1, 1))), keypoints)
        )
