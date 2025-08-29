import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from baselines.test_base import TestBase

from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.utils.nn import load_network
from keypoint_diffuser.utils.utils import reparameterize


class Ours(TestBase):
    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return AEOptions().initialize(p)

    def load_model(self, model_path: Path, cfg):
        self.model = AutoEncoder(cfg).cuda()  # unsupervised network
        load_network(self.model, str(model_path))
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        return self.model.encode(pcd)[0].cpu().numpy()

    def get_network_data(self, data: dict[str, Any], key="orig"):
        opp = "deformed" if key == "orig" else "orig"

        d = {}
        for k, v in data.items():
            if k.startswith(opp):
                continue
            elif k.startswith(key):
                d[k[len(key) + 1 :]] = v.cuda()
            else:
                d[k] = v.cuda()
        return d

    def get_reconstruction(self, pcd: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        z0, mu, logvar = self.model.encode(self.get_network_data(pcd))
        z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)
        z0 = z0.reshape(z0.shape[0], -1)
        z_full = torch.cat([z0, z_aux], dim=1)

        recons = self.model.decode(z_full, 2048).detach()
        return (
            recons.unsqueeze(1),
            pcd["target_shape"].reshape(z0.shape[0], -1, 3).cuda(),
        )
