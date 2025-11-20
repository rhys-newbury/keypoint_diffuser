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
            cfg.max_points, cfg.key_point
        ).cuda()  # unsupervised network
        self.model.load_state_dict(
            torch.load(model_path)["model_state_dict"]
        )  # load weights
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        data = (torch.Tensor(pcd["orig"])).cuda()
        return self.model.get_keypoints(data).cpu().numpy()

    def get_reconstruction(
        self, pcd: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kps, reconstruct = self.model(pcd.cuda(), False)

        return reconstruct, pcd.cuda(), kps

    @staticmethod
    def choose_model(metrics_info, ckpt_paths, tail_frac=0.2):
        if "chamfer" not in metrics_info:
            raise KeyError("metrics_info must include 'chamfer'.")

        steps = np.asarray(metrics_info["chamfer"]["steps"], dtype=float)
        losses = np.asarray(metrics_info["chamfer"]["values"], dtype=float)
        assert steps.size == losses.size > 0, "Bad chamfer input"

        # Sort by step
        order = np.argsort(steps)
        steps, losses = steps[order], losses[order]

        # ---- Find global minimum ----
        j = int(np.nanargmin(losses))
        best_step = steps[j]
        best_val = losses[j]

        # ---- Map best training step → best checkpoint ----
        max_step = float(steps[-1])
        prog = best_step / max_step if max_step > 0 else 1.0

        ckpt_paths = TestBase.filter_and_sort_ckpts(ckpt_paths)
        N = len(ckpt_paths)

        idx = int(round(prog * (N - 1)))
        idx = min(max(idx, 0), N - 1)

        best_ckpt = ckpt_paths[idx][2]

        return {
            "ckpt": best_ckpt,
            "ckpt_index": idx,
            "total_ckpts": N,
            "best_step": float(best_step),
            "best_loss": float(best_val),
            "training_progress_%": round(prog * 100, 2),
            "picked_from": "global_min_chamfer_loss (step→epoch mapping)",
        }
