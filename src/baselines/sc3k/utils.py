"""
This file contains additional functions for the unsupervised keypoints estimation method

engr.mz@hotmail.com
May 31, 2022
"""

import math

import torch
import torch.nn.functional as F


def compute_loss(kp1, kp2, data, cfg, split="split??"):
    device = kp1.device
    l_sep1 = cfg.separation * separation_loss(kp1)
    l_sep2 = cfg.separation * separation_loss(kp2)
    l_overlap1 = cfg.overlap * overlap_loss(kp1, cfg.overlap_threshold)
    l_overlap2 = cfg.overlap * overlap_loss(kp2, cfg.overlap_threshold)
    l_shape1 = cfg.shape * shape_loss(data[0].float().to(device), kp1)
    l_shape2 = cfg.shape * shape_loss(data[2].float().to(device), kp2)
    l_consist = cfg.consist * consistancy_loss(
        kp1, kp2, data[1].float().to(device), data[3].float().to(device)
    )
    l_volume1 = cfg.volume * volume_loss(kp1, data[0].float().to(device))
    l_volume2 = cfg.volume * volume_loss(kp2, data[2].float().to(device))
    l_pose = cfg.pose * pose_loss(
        kp1, kp2, data[1].float().to(device), data[3].float().to(device)
    )
    d = {
        f"{split}_loss/consist": l_consist,
        f"{split}_loss/relative_pose": l_pose,
        f"{split}_loss/sep1": l_sep1,
        f"{split}_loss/sep2": l_sep2,
        f"{split}_loss/overlap1": l_overlap1,
        f"{split}_loss/overlap2": l_overlap2,
        f"{split}_loss/shape1": l_shape1,
        f"{split}_loss/shape2": l_shape2,
        f"{split}_loss/volume1": l_volume1,
        f"{split}_loss/volume2": l_volume2,
    }

    return (
        l_sep1
        + l_sep2
        + l_overlap1
        + l_overlap2
        + l_shape1
        + l_shape2
        + l_consist
        + l_volume1
        + l_volume2
        + l_pose
    ), d  # + l_reconstruction


def consistancy_loss(kp1, kp2, rot1, rot2):
    """

    Parameters
    ----------
    kp1     Estimated key-points 1
    kpT     Transformed version of the estimated key-points 2

    Returns     Loss => the corresponding key-points should be estimated in the same 3D positions
    -------

    """

    kp2_to_kp1 = torch.transpose(
        torch.bmm(
            torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2)),
            torch.transpose(kp2.double(), 1, 2),
        ),
        1,
        2,
    )
    return F.mse_loss(kp1, kp2_to_kp1.float())


def shape_loss(pc, kp):
    """
    Parameters
    ----------
    pc      Input point cloud
    kp      Estimated key-points

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    """

    D = torch.cdist(kp, pc, p=2)  # [B, K, P]
    nn = D.min(dim=2).values  # nearest pc point per keypoint: [B, K]
    return nn.mean()  # equals (1/K) sum_i (...) then mean over batch


def overlap_loss(kp, threshold=0.05):
    """
    Parameters
    ----------
    kp:         Key-points
    threshold   allowable overlap between the key-points
    Method:     Find distance of every point from all the points
                select the minimum distances that are greater than 0 (distance from itself)
                return count of the separated distances => final loss

    Returns     separation loss -> avoid estimation of multiple key-points on the same 3D location
    -------
    """

    _, K, _ = kp.shape
    D = torch.cdist(kp, kp, p=2)  # [B, K, K]

    # mask out i == j, but keep K^2 in the denominator (as in the paper)
    offdiag = ~torch.eye(K, dtype=torch.bool, device=kp.device).unsqueeze(0)

    hits = ((threshold > D) & offdiag).float()  # 1 if pair overlaps, else 0
    per_batch = hits.sum(dim=(1, 2)) / (K * K)  # divide by K^2
    return per_batch.mean()


def separation_loss(kp, floor=0.01):
    """
    Parameters
    ----------
    kp:         Key-points
    Method:     compute distances of each point from all the points in "kp"
                consider minimum two distances
                take mean of the distances from the closest point (distance>0)

    Returns     separation loss ->  average distance of every point from closest points
    -------
    """
    _, K, _ = kp.shape
    D = torch.cdist(kp, kp, p=2)  # [B,K,K], Euclidean

    # exclude self-distances
    eye = torch.eye(K, dtype=torch.bool, device=kp.device).unsqueeze(0)
    D = D.masked_fill(eye, float("inf"))  # or: D + eye.float()*1e6

    # nearest neighbor distance for each keypoint
    nn = D.min(dim=-1).values  # [B,K]

    # per-sample mean, then floor by 0.01, then invert; finally average over batch
    mean_nn = nn.mean(dim=1)  # [B]
    denom = torch.clamp(mean_nn, min=floor)
    return (1.0 / denom).mean()


def volume_loss(kp, pc):
    """

    Parameters: 3D IoU loss
                => same as coverage loss of clara's Paper
                => see:
                https://github.com/cfernandezlab/Category-Specific-Keypoints/
                blob/master/models/losses.py
        Smooth L1 loss
        https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    """
    val_max_pc, _ = torch.max(pc, 1)  # Bx3
    val_min_pc, _ = torch.min(pc, 1)  # Bx3
    dim_pc = val_max_pc - val_min_pc  # Bx3
    val_max_kp, _ = torch.max(kp, 1)  # Bx3
    val_min_kp, _ = torch.min(kp, 1)  # Bx3
    dim_kp = val_max_kp - val_min_kp  # Bx3

    return F.smooth_l1_loss(dim_kp, dim_pc)


def pose_loss(kp1, kp2, rot1, rot2):
    """

    Parameters
    ----------
    kp1     Estimated key-points 1
    kp2     Transformed version of the estimated key-points 2
    rot1    pose of KP1
    rot2    pose of KP2

    rot     GT relative pose b/w kp1 and kp2

    Returns     Loss => Error in relative pose b/w kp1 and kp2 [Forbunius Norm]
    -------

    """

    device = kp1.device
    gt_rot = torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2))
    mat = batch_compute_similarity_transform_torch(kp1, kp2)
    frob = torch.norm(gt_rot - mat, dim=(1, 2))  # Forbunius Norm

    frob_min = torch.min(torch.tensor(1.0).to(device), 1 / (2 * math.sqrt(2)) * frob)
    frob_clamped = torch.clamp(frob_min, -1, 1)
    angle = 2 * torch.arcsin(frob_clamped)

    return torch.mean(angle)


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    help: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427

    """
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))  # position
    R = torch.linalg.inv(R)  # rotation

    return R


class AverageMeter:
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
