import argparse
from pathlib import Path

import numpy as np
import torch
from baselines.key_grid import merger_net
from baselines.test_base import TestBase


class KeyGrid(TestBase):
    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--max-points",
            type=int,
            default=2048,
            help="Indicates maximum points in each input point cloud.",
        )
        return p

    def load_model(self, model_path: Path, cfg):
        self.model = merger_net.Net(
            cfg.max_points, cfg.key_points
        ).cuda()  # unsupervised network
        self.model.load_state_dict(
            torch.load(model_path)["model_state_dict"]
        )  # load weights
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        data = (torch.Tensor(pcd["orig"])).cuda()
        return self.model.get_keypoints(data).cpu().numpy()

    def get_reconstruction(self, pcd: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        _, reconstruct = self.model(pcd.cuda(), False)
        kps = self.model.get_keypoints(data).cpu().numpy()

        return reconstruct, pcd.cuda(), kps
