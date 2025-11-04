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
        min_points: int = 2048,
    ):
        """
        Greedy reconstruction by activation score:
        - For each item b, sort part indices by MA[b] descending
        - Accumulate corresponding RPCD parts until total points >= min_points
        - If overshoot: downsample to exactly min_points (no replacement)
        - If not enough even after all parts: upsample with replacement to min_points

        Returns:
            reconstructions: list length B; each item is [ merged_(min_points x 3) ]
            data           : torchified input (B, ..., 3)
            kps            : keypoints returned by the model
        """
        device = next(self.model.parameters()).device
        data = torch.as_tensor(pcd, dtype=torch.float32, device=device)

        # RPCD: list[P] of tensors shaped [B, m_i, 3]
        # MA   : [B, P] activation scores per part
        RPCD, kps, _, _, MA = self.model(data)

        B = data.shape[0]
        MA.shape[1]  # number of parts
        reconstructions = []

        for b in range(B):
            # sort parts by activation score (desc)
            scores = MA[b]  # (P,)
            order = torch.argsort(scores, descending=True)
            parts_points = []
            total = 0

            for idx in order.tolist():
                pts = RPCD[idx][b]  # [m_i, 3]
                if pts.numel() == 0:
                    continue
                parts_points.append(pts)
                total += pts.shape[0]
                if total >= min_points:
                    break

            if not parts_points:
                # no parts available for this item; return empty candidate
                reconstructions.append([])
                continue

            merged = torch.cat(parts_points, dim=0)  # [M, 3]
            M = merged.shape[0]

            if min_points <= M:
                # downsample without replacement
                sel = torch.randperm(M, device=merged.device)[:min_points]
                merged = merged[sel]

                reconstructions.append([merged])

            elif min_points > M:
                reconstructions.append([])

        return reconstructions, data, kps
