# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from pytorch3d.loss import chamfer_distance


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


class EDMLossCurriculum:
    def __init__(
        self,
        P_mean_init=-2.0,
        P_std_init=0.1,
        P_mean_final=-1.2,
        P_std_final=1.2,
        sigma_data=0.3,
        max_steps=100_000,
    ):
        self.P_mean_init = P_mean_init
        self.P_std_init = P_std_init
        self.P_mean_final = P_mean_final
        self.P_std_final = P_std_final
        self.sigma_data = sigma_data
        self.max_steps = max_steps

    def interpolate(self, start, end, pct):
        return start + pct * (end - start)

    def __call__(self, net, data, code, step, augment_pipe=None):
        pct = min(step / self.max_steps, 1.0)

        # Interpolate P_mean and P_std
        P_mean = self.interpolate(self.P_mean_init, self.P_mean_final, pct)
        P_std = self.interpolate(self.P_std_init, self.P_std_final, pct)
        rnd_normal = torch.randn([data.shape[0], 1, 1], device=data.device)
        P = rnd_normal * P_std + P_mean
        sigma = P.exp()

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y = data

        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, context=code)
        loss, _ = chamfer_distance(D_yn, y)
        loss = weight * loss
        return loss.mean()


# ----------------------------------------------------------------------------
