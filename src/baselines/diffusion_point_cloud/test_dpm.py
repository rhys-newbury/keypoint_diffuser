import argparse
from pathlib import Path

import numpy as np
import torch
from baselines.diffusion_point_cloud.autoencoder import AutoEncoder
from baselines.test_base import TestBase


class DPM(TestBase):  # inherit if you need the same interface
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_args = None
        self.model = None
        self.flexibility = None

    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--max-points",
            type=int,
            default=2048,
            help="Indicates maximum points in each input point cloud.",
        )
        p.add_argument("--num_steps", type=int, default=200)
        p.add_argument("--beta_1", type=float, default=1e-4)
        p.add_argument("--beta_T", type=float, default=0.05)
        p.add_argument("--residual", type=eval, default=True, choices=[True, False])
        p.add_argument("--sched_mode", type=str, default="linear")
        p.add_argument("--flexibility", type=float, default=0.0)

        return p

    def load_model(self, ckpt_path: Path, cfg):
        """Load AE checkpoint and build model on device."""
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        self.flexibility = 0.0  # getattr(self.ckpt_args, 'flexibility', 0.0)
        cfg.latent_dim = cfg.key_points + 3 * 5

        self.model = AutoEncoder(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    def get_reconstruction(self, pcd) -> tuple[torch.Tensor, torch.Tensor, None]:
        with torch.no_grad():
            code = self.model.encode(pcd.cuda())  # [B, Z]
            recons = self.model.decode(
                code, pcd.shape[1], flexibility=self.flexibility
            )  # [B, N, 3]
        return recons.unsqueeze(1).cpu(), pcd, code

    def get_keypoints(self, pcd):
        raise NotImplementedError()

    @staticmethod
    def choose_model(metrics_info, ckpt_paths, tail_frac=0.2):
        if "loss" not in metrics_info:
            raise KeyError("metrics_info must include 'loss'.")

        steps = np.asarray(metrics_info["loss"]["steps"], dtype=float)
        losses = np.asarray(metrics_info["loss"]["values"], dtype=float)
        assert steps.size == losses.size > 0, "Bad loss input"

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
