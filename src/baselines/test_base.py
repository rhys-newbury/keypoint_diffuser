import argparse
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


EPOCH_RE = re.compile(r"(?P<kp>\d+)kp_(?P<epoch>\d+)\.pth$", re.IGNORECASE)


@dataclass
class TestBase:
    model: torch.nn.Module = field(init=False, default=None)

    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return p

    @abstractmethod
    def load_model(self, model_path: Path):
        pass

    @abstractmethod
    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def filter_and_sort_ckpts(ckpt_paths):
        """Return only ckpts matching Nkp_epoch.pth, sorted by epoch integer."""
        valid = []
        for p in map(Path, ckpt_paths):
            for model in p.glob("*.pth"):
                m = EPOCH_RE.search(model.name)
                if m:
                    epoch = int(m.group("epoch"))
                    valid.append((epoch, m, model))
        if not valid:
            raise RuntimeError("No valid `Nkp_epoch.pth` checkpoints found.")
        # sort by epoch
        valid.sort(key=lambda x: x[0])
        # return paths only in sorted order
        return valid

    @staticmethod
    def _ema(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        if x.size == 0:
            return x
        y = np.empty_like(x, dtype=float)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    def get_reconstruction(self, pcd: np.ndarray) -> np.ndarray:
        raise NotImplementedError
