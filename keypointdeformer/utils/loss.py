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


# from torch_utils import persistence

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, data, code):
        rnd_uniform = torch.rand([data.shape[0], 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y = data
        # y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, context=code)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss, _ = chamfer_distance(D_yn, y)
        loss = weight * loss
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.3):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, data, code, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([data.shape[0], 1, 1], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y = data
        # y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, context=code)
        loss = weight * ((D_yn - y) ** 2)
        return loss


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
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


# ----------------------------------------------------------------------------
