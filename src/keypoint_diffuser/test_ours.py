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

    def get_reconstruction(
        self, pcd: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z0, mu, logvar = self.model.encode(self.get_network_data(pcd))
        z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)
        z0 = z0.reshape(z0.shape[0], -1)
        z_full = torch.cat([z0, z_aux], dim=1)

        recons = self.model.decode(z_full, 2048).detach()
        return (
            recons.unsqueeze(1),
            pcd["target_shape"].reshape(z0.shape[0], -1, 3).cuda(),
            z0,
        )

    @torch.no_grad()
    def interpolate_latent(
        self, pcd_a: dict, pcd_b: dict, n_steps: int = 10, batch_size: int = 12
    ):
        """
        Interpolate between two input point clouds in latent space.
        Returns a list of decoded reconstructions.
        """
        # Encode both point clouds
        z0_a, mu_a, logvar_a = self.model.encode(self.get_network_data(pcd_a))
        z_aux_a = reparameterize(mu_a, logvar_a)
        z_a = torch.cat([z0_a.reshape(z0_a.shape[0], -1), z_aux_a], dim=1)

        z0_b, mu_b, logvar_b = self.model.encode(self.get_network_data(pcd_b))
        z_aux_b = reparameterize(mu_b, logvar_b)
        z_b = torch.cat([z0_b.reshape(z0_b.shape[0], -1), z_aux_b], dim=1)

        # Precompute interpolation coefficients
        t_values = torch.linspace(0, 1, n_steps, device=z_a.device)
        z_interps = [(1 - t) * z_a + t * z_b for t in t_values]

        k_a = z0_a[0].reshape(-1, 3)
        k_b = z0_b[0].reshape(-1, 3)
        k_interps = [(1 - t) * k_a + t * k_b for t in t_values]

        # Decode in batches
        recons = []
        for i in range(0, len(z_interps), batch_size):
            z_batch = torch.cat(z_interps[i : i + batch_size], dim=0)
            recon_batch = self.model.decode(z_batch, 2048).detach().cpu()
            recons.extend(recon_batch.split(1))  # keep consistent shape list

        k_interps = [k.cpu() for k in k_interps]

        return recons, k_interps
