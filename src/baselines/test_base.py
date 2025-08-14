import argparse
from abc import abstractmethod
from pathlib import Path

import numpy as np


class TestBase:
    @staticmethod
    @abstractmethod
    def get_parser() -> argparse.ArgumentParser:
        pass

    @abstractmethod
    def load_model(self, model_path: Path):
        pass

    @abstractmethod
    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        pass

    def get_reconstruction(self, pcd: np.ndarray) -> np.ndarray:
        raise NotImplementedError
