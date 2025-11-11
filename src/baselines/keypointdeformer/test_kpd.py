import argparse
from pathlib import Path

import numpy as np
import torch
from baselines.keypointdeformer.models import cage_skinning
from baselines.keypointdeformer.options.base_options import BaseOptions
from baselines.keypointdeformer.utils.nn import load_network
from baselines.test_base import TestBase


class KPD(TestBase):
    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        BaseOptions().initialize(p)
        return cage_skinning.CageSkinning.modify_commandline_options(p)

    def load_model(self, model_path: Path, cfg):
        self.model = cage_skinning.CageSkinning(cfg).cuda()  # unsupervised network
        load_network(self.model, str(model_path))
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        if "orig" in pcd:
            data = (torch.Tensor(pcd["orig"])).cuda().permute((0, 2, 1))
        else:
            data, _ = self.get_data(pcd)
        return self.model.keypoint_predictor(data).cpu().numpy().transpose(0, 2, 1)

    def get_latent(self, pcd):
        return torch.Tensor(self.get_keypoints(pcd)), self.get_data(pcd)[1]

    def generate(self, z0):
        latent, shape = z0
        shape = shape.cuda()
        latent = (
            latent.reshape(latent.shape[0], 10, -1).permute((0, 2, 1)).float().cuda()
        )

        outputs = []
        n = latent.shape[0]

        for i in range(0, n, 16):
            latent_batch = latent[i : i + 16]
            shape_batch = shape[i : i + 16]

            out = self.model.deform_from_keypoints(shape_batch, latent_batch)
            outputs.append(out.cpu())

        # concatenate along batch dimension
        return torch.cat(outputs, dim=0)

    def get_data(self, data):
        source_shape, target_shape = data["source_shape"], data["target_shape"]

        source_shape_t = source_shape.transpose(1, 2)
        target_shape_t = target_shape.transpose(1, 2)

        return source_shape_t.cuda(), target_shape_t.cuda()

    def get_reconstruction(
        self, pcd: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.get_data(pcd)
        out_dict = self.model(*data)
        return (
            (out_dict["deformed"].unsqueeze(1)),
            data[0].transpose(2, 1),
            out_dict["target_keypoints"],
        )

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
