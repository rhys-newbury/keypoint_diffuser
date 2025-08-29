"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""

import torch
from tqdm.auto import tqdm

from .emd_loss.emd_module import EMDModule


def downsample_batched_point_cloud(point_clouds, num_samples=4096):
    """
    Vectorized downsampling of batched point clouds using random sampling.

    Args:
        point_clouds (torch.Tensor): Batched point clouds of shape (B, N, 3)
        num_samples (int): Number of points to sample per batch

    Returns:
        torch.Tensor: Downsampled batched point clouds of shape (B, num_samples, 3)
    """
    B, N, C = point_clouds.shape
    if num_samples > N:
        raise ValueError("Cannot sample more points than exist in the point cloud.")

    # Generate random indices for each batch element
    rand_vals = torch.rand(B, N, device=point_clouds.device)
    _, indices = rand_vals.topk(num_samples, dim=1, largest=False, sorted=False)

    # Expand batch indices to match indices shape
    batch_indices = (
        torch.arange(B, device=point_clouds.device).view(-1, 1).expand(-1, num_samples)
    )

    # Gather points using advanced indexing
    downsampled = point_clouds[batch_indices, indices]  # Shape: (B, num_samples, 3)

    return downsampled


def emd_approx(sample, ref):
    N, N_ref = sample.size(1), ref.size(1)
    assert N_ref == N, "Not sure what would EMD do in this case"
    emd = EMDModule()
    sample_ = (downsample_batched_point_cloud(sample) + 1) / 2
    ref_ = (downsample_batched_point_cloud(ref) + 1) / 2
    dis, _ = emd(sample_, ref_, 0.002, 10000)  # 0.005, 50 for training
    emd_norm = torch.sqrt(dis).mean(dim=1) / N
    return emd_norm


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in tqdm(iterator, desc="EMD-CD"):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        "MMD-CD": cd,
        "MMD-EMD": emd,
    }
    return results, cd_lst, emd_lst


if __name__ == "__main__":
    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
    print(EMD_CD(a, b, batch_size=8))
