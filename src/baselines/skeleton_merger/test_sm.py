import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.skeleton_merger import merger_net
from baselines.test_base import TestBase


class SM(TestBase):
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
        # item 2 is keypoints
        return self.model.get_keypoints(data).cpu().numpy()

    @torch.no_grad()
    def get_reconstruction(
        self,
        pcd: np.ndarray,
        thresholds=(0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_points: int = 2048,
    ):
        device = next(self.model.parameters()).device
        data = torch.as_tensor(pcd, dtype=torch.float32, device=device)

        RPCD, _, _, _, MA = self.model(
            data
        )  # RPCD: list[P] of [B x m x 3], MA: [B x P]

        reconstructions = []
        for b in range(data.shape[0]):
            recos_b = []
            for th in thresholds:
                keep_ids = (MA[b] > th).nonzero(as_tuple=True)[0]
                if keep_ids.numel() == 0:
                    continue  # nothing passes this threshold

                parts = [RPCD[i][b] for i in keep_ids.tolist()]
                merged = torch.cat(parts, dim=0)  # [M x 3]

                M = merged.shape[0]
                if min_points > M:
                    continue  # skip candidates too small

                if min_points < M:
                    sel = torch.randperm(M, device=merged.device)[:min_points]
                    merged = merged[sel]

                recos_b.append(merged)  # each candidate is exactly [2048 x 3]
            reconstructions.append(recos_b)

        return reconstructions, data
