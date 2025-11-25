# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


def local_repulsion(pred, k=8, margin=0.01):
    """
    pred: (B, N, 3)
    returns: (B,) repulsion penalty per batch element
    """
    B, N, _ = pred.shape
    dists = torch.cdist(pred, pred)  # (B,N,N)

    knn_dists, _ = torch.topk(dists, k=k + 1, dim=-1, largest=False)  # (B,N,k+1)
    knn_dists = knn_dists[:, :, 1:]  # (B,N,k) drop self

    penalty = F.relu(margin - knn_dists)  # (B,N,k)
    # average over neighbors and points → (B,)
    penalty_per_batch = penalty.mean(dim=(1, 2))
    return penalty_per_batch


class EDMLossCurriculum:
    def __init__(
        self,
        p_mean_init=-2.0,
        p_std_init=0.1,
        p_mean_final=-1.2,
        p_std_final=1.2,
        sigma_data=0.3,
        max_steps=100_000,
    ):
        self.p_mean_init = p_mean_init
        self.p_std_init = p_std_init
        self.p_mean_final = p_mean_final
        self.p_std_final = p_std_final
        self.sigma_data = sigma_data
        self.max_steps = max_steps

    def interpolate(self, start, end, pct):
        return start + pct * (end - start)

    def __call__(self, net, data, code, step, augment_pipe=None):
        pct = min(step / self.max_steps, 1.0)

        # Interpolate P_mean and P_std
        P_mean = self.interpolate(self.p_mean_init, self.p_mean_final, pct)
        P_std = self.interpolate(self.p_std_init, self.p_std_final, pct)
        rnd_normal = torch.randn([data.shape[0], 1, 1], device=data.device)
        P = rnd_normal * P_std + P_mean
        sigma = P.exp()

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y = data

        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, context=code)

        # One-way Chamfer (pred → gt)
        loss_pred_to_gt, _ = chamfer_distance(
            D_yn, y, single_directional=True, batch_reduction=None
        )

        # One-way Chamfer (gt → pred)
        # this is how close from each ground truth point to a point in the prediction.
        # we care a lot more about this.
        loss_gt_to_pred, _ = chamfer_distance(
            y, D_yn, single_directional=True, batch_reduction=None
        )

        # Weighted asymmetric combination
        recon_loss = 0.5 * loss_pred_to_gt + 1.5 * loss_gt_to_pred

        repel_per_sample = local_repulsion(D_yn, k=8, margin=0.01)  # (B,)
        pct_clean = (
            (self.sigma_data / (sigma + self.sigma_data)).clamp(0, 1).flatten()
        )  # (B,)

        spread_strength = 0.02  # base lambda
        spread_term = spread_strength * pct_clean * repel_per_sample  # (B,)
        loss = (weight.flatten() * recon_loss).mean() + spread_term.mean()

        return loss


# ----------------------------------------------------------------------------