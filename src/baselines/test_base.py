import argparse
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


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

    def get_reconstruction(self, pcd: np.ndarray) -> np.ndarray:
        raise NotImplementedError
