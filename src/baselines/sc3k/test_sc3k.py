import argparse
from pathlib import Path

import numpy as np
import torch
from baselines.sc3k import network
from baselines.test_base import TestBase


class SC3K(TestBase):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Train SC3K (argparse version)")

        # Core training args
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--max-epoch", type=int, default=200)

        p.add_argument("--lr", type=float, default=1e-3)

        # Domain/task specifics you reference
        p.add_argument("--key-points", type=int, default=10)
        p.add_argument("--log-path", type=str, default=None)
        p.add_argument(
            "--category", type=str, help="Category of objects", default="chair"
        )

        p.add_argument("--overlap-threshold", type=float, default=0.05)

        # Loss term weights
        p.add_argument("--separation", type=float, default=0.5)
        p.add_argument("--shape", type=float, default=6.0)
        p.add_argument("--volume", type=float, default=1.0)
        p.add_argument("--overlap", type=float, default=0.07)

        p.add_argument("--consist", type=float, default=1.0)
        p.add_argument("--pose", type=float, default=0.05)

        p.add_argument("--lamda", type=float, default=0.0)
        p.add_argument("--lamda2", type=float, default=0.0)
        p.add_argument("--sample-points", type=int, default=2048)

        return p

    def load_model(self, model_path: Path, cfg):
        cfg.task = "canonical"
        cfg.split = "test"

        self.model = network.sc3k(cfg).cuda()  # unsupervised network
        self.model.load_state_dict(torch.load(model_path))  # load weights
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        return self.model(((torch.Tensor(pcd["orig"])).cuda(),)).cpu().numpy()
