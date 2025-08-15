from pathlib import Path

import numpy as np
import torch
from baselines.sc3k import network
from baselines.test_base import TestBase


class SC3K(TestBase):
    def load_model(self, model_path: Path, cfg):
        cfg.task = "canonical"
        cfg.split = "test"

        self.model = network.sc3k(cfg).cuda()  # unsupervised network
        self.model.load_state_dict(torch.load(model_path))  # load weights
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        return self.model(((torch.Tensor(pcd["orig"])).cuda(),)).cpu().numpy()
