"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""

import pytorch3d.loss
import torch
from tqdm.auto import tqdm

from .emd_loss.emd_module import EMDModule


def downsample_batched_point_cloud(point_clouds, num_samples=2048):
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


def EMD_CD_recon(sample_pcs, ref_pcs, batch_size=8, reduced=True):
    """
    Computes Chamfer and EMD distances for reconstruction, in batches.

    Args:
        sample_pcs (B, N, 3): Predicted point clouds
        ref_pcs (B, N, 3): Ground-truth point clouds
        batch_size (int): Batch size for evaluation
        reduced (bool): If True, returns mean CD and EMD. If False, returns per-sample tensors.

    Returns:
        dict with 'CD' and 'EMD' (averaged if reduced=True)
    """
    assert (
        sample_pcs.shape == ref_pcs.shape
    ), "Shape mismatch between prediction and reference"

    cd_lst = []
    emd_lst = []

    B = sample_pcs.shape[0]
    for b_start in range(0, B, batch_size):
        b_end = min(B, b_start + batch_size)
        samp_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]
        cd, _ = pytorch3d.loss.chamfer_distance(samp_batch, ref_batch)
        emd = emd_approx(samp_batch, ref_batch)  # per-sample EMD

        cd_lst.append(cd.reshape(1))
        emd_lst.append(emd)

    cd_all = torch.cat(cd_lst)
    emd_all = torch.cat(emd_lst)

    if reduced:
        return {
            "CD": cd_all.mean(),
            "EMD": emd_all.mean(),
        }
    else:
        return {
            "CD": cd_all,
            "EMD": emd_all,
        }


def MMD_CD_EMD(
    gen_pcs: torch.Tensor,  # (Ns, P, 3) e.g. generated
    ref_pcs: torch.Tensor,     # (Nr, P, 3) e.g. real
    batch_size: int = 16,
    reduced: bool = True,
    ref_chunk_size: int = 32,
    symmetric: bool = False,
):
    """
    True MMD but in two levels of batching:
      - outer over query (gen or ref), size <= batch_size
      - inner over reference chunks, size <= ref_chunk_size

    For each query point cloud q:
       best_cd[q] = min_j CD(q, ref_j)
       best_emd[q] = min_j EMD(q, ref_j)
    """
    device = gen_pcs.device
    ref_pcs = ref_pcs.to(device)

    def one_direction(query: torch.Tensor, target: torch.Tensor):
        Nq = query.shape[0]
        Nt = target.shape[0]
        all_best_cd = []
        # all_best_emd = []

        for q_start in tqdm(range(0, Nq, batch_size), desc="MMD query batches"):
            q_batch = query[q_start : q_start + batch_size]  # (B, P, 3)
            B = q_batch.shape[0]
            # start with +inf best
            best_cd = torch.full((B,), float("inf"), device=device)
            # best_emd = torch.full((B,), float("inf"), device=device)

            for t_start in range(0, Nt, ref_chunk_size):
                t_batch = target[t_start : t_start + ref_chunk_size]  # (C, P, 3)
                C = t_batch.shape[0]

                # expand to (B*C, P, 3) but C is small
                q_exp = q_batch[:, None, :, :].expand(B, C, -1, -1).reshape(B * C, -1, 3)
                t_exp = t_batch[None, :, :, :].expand(B, C, -1, -1).reshape(B * C, -1, 3)

                # CD for this block
                # print("calc CD")
                cd_block, _ = pytorch3d.loss.chamfer_distance(q_exp, t_exp, batch_reduction=None)  # (B*C,)
                cd_block = cd_block.view(B, C)       # (B, C)
                # print("calc cd done.")
                # EMD for this block (loop over pairs in the block)
                # print("calc emd")
                # emd_block = emd_approx(q_exp, t_exp)
                # emd_block = emd_block.view(B, C)     # (B, C)
                # print("calc emd done")

                # update best
                cd_min_block, _ = cd_block.min(dim=1)   # (B,)
                # emd_min_block, _ = emd_block.min(dim=1) # (B,)

                best_cd = torch.minimum(best_cd, cd_min_block)
                # best_emd = torch.minimum(best_emd, emd_min_block)

            all_best_cd.append(best_cd)
            # all_best_emd.append(best_emd)

        all_best_cd = torch.cat(all_best_cd, dim=0)
        # all_best_emd = torch.cat(all_best_emd, dim=0)
        return all_best_cd #, all_best_emd

    cd_s2r = one_direction(gen_pcs, ref_pcs)

    if symmetric:
        cd_r2s = one_direction(ref_pcs, gen_pcs)
        cd_all = torch.cat([cd_s2r, cd_r2s], dim=0)
        # emd_all = torch.cat([emd_s2r, emd_r2s], dim=0)
    else:
        cd_all = cd_s2r
        # emd_all = emd_s2r

    return {
        "MMD-CD": cd_all.mean(),
        # "MMD-EMD": emd_all.mean(),
    }



if __name__ == "__main__":
    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
