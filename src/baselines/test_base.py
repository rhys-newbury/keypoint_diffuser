import argparse
from abc import abstractmethod
from pathlib import Path

import numpy as np


class TestBase:
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
