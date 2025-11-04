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
        data = (torch.Tensor(pcd["orig"])).cuda()
        return (
            self.model.keypoint_predictor(data.permute((0, 2, 1)))
            .cpu()
            .numpy()
            .transpose(0, 2, 1)
        )
    def get_latent(self, pcd):
        return self.get_keypoints(pcd)

    def get_data(self, data):
        source_shape, target_shape = data["source_shape"], data["target_shape"]

        source_shape_t = source_shape.transpose(1, 2)
        target_shape_t = target_shape.transpose(1, 2)

        return source_shape_t.cuda(), target_shape_t.cuda()

    def get_reconstruction(self, pcd: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.get_data(pcd)
        out_dict = self.model(*data)
        return (out_dict["deformed"].unsqueeze(1)), data[0].transpose(2, 1)
