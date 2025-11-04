import argparse
from pathlib import Path

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

    def get_reconstruction(self, pcd) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            code = self.model.encode(pcd.cuda())  # [B, Z]
            recons = self.model.decode(
                code, pcd.shape[1], flexibility=self.flexibility
            )  # [B, N, 3]
        return recons.unsqueeze(1).cpu(), pcd

    def get_keypoints(self, pcd):
        raise NotImplementedError()
