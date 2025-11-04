#!/usr/bin/env python3
"""
Latent interpolation evaluation script (diverse pairs).

- Loads *.h5 files for a category
- Builds a pool of candidate shapes (size = --pool-size)
- Computes pairwise symmetric Chamfer distances between all candidates
- Chooses the top --num-pairs *most dissimilar* pairs (highest CD)
- For each chosen pair, performs latent interpolation using model.interpolate_latent
- Saves reconstructions + keypoints for visualization

Run:
  python interpolate_latent_pairs.py Ours \
         --ckpt checkpoints/foo.pth \
         --dataset_path /path/to/ShapeNetH5/val \
         --category chair \
         --batch-size 1 \
         --pool-size 200 \
         --num-pairs 50 \
         --interpolation-steps 10
"""

import argparse
import contextlib
from glob import glob
from pathlib import Path

import numpy as np
import torch
import tqdm
from classes import MODEL_CLASSES
from pytorch3d.loss import chamfer_distance
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.H5Datset import H5Dataset
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import Collect, GridSample, ToTensor


TESTSET = "/app/shapenetcorev2_hdf5_2048/val"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        p.add_argument(*names, **kwargs)


def get_pointcloud(sample) -> torch.Tensor:
    """
    Extracts an (N,3) point cloud in torch.float32 on CUDA from a dataset item.
    Adjust this if your dataset dict uses a different key.

    We expect something like sample["target_shape"] or sample["coord"].
    We'll try a few common keys.
    """
    if isinstance(sample, dict):
        if "target_shape" in sample:
            pts = sample["target_shape"]
        elif "coord" in sample:
            pts = sample["coord"]
        else:
            raise KeyError(
                "Can't find point cloud in sample; expected 'target_shape' or 'coord'."
            )

    else:
        raise TypeError("dataset[i] did not return a dict-like sample.")

    pts = torch.as_tensor(pts, dtype=torch.float32)
    if pts.ndim == 3 and pts.shape[0] == 1:
        pts = pts[0]
    # pts should now be (N,3)
    return pts.cuda(non_blocking=True)


def batched_chamfer(pool_pcs, batch_size=8):
    """
    Compute symmetric Chamfer distance between all unique pairs in pool_pcs.
    Uses batched GPU computation for speed.

    pool_pcs: list[torch.Tensor] of shape (N,3)
    Returns: list[(cd_val, i, j)] sorted by cd_val descending
    """
    pair_scores = []
    n = len(pool_pcs)

    for i in tqdm.trange(n, desc="Computing Chamfer (batched)"):
        pc_i = pool_pcs[i].unsqueeze(0)  # (1, N, 3)
        batch_list = []
        idx_list = []

        # gather a batch of j's
        for j in range(i + 1, n):
            batch_list.append(pool_pcs[j])
            idx_list.append(j)

            # process once batch full or last pair
            if len(batch_list) == batch_size or j == n - 1:
                pc_j = torch.stack(batch_list, dim=0)  # (B, N, 3)
                # repeat i shape B times to match
                pc_i_rep = pc_i.repeat(pc_j.shape[0], 1, 1)  # (B, N, 3)

                # compute Chamfer distance (symmetric)
                _cd_batch, _ = chamfer_distance(pc_i_rep, pc_j)
                # cd_batch is a scalar (mean over batch)
                # We need per-sample distance, so compute manually:
                # we can compute separately to get per-batch losses
                dists = torch.cdist(pc_i_rep, pc_j)  # (B, N, N)
                # one-directional means
                a_to_b = dists.min(dim=2)[0].mean(dim=1)
                b_to_a = dists.min(dim=1)[0].mean(dim=1)
                cd_vals = (a_to_b + b_to_a).cpu().numpy()

                for k, j_idx in enumerate(idx_list):
                    pair_scores.append((float(cd_vals[k]), i, j_idx))

                # clear batch
                batch_list = []
                idx_list = []

    pair_scores.sort(key=lambda x: x[0], reverse=True)
    return pair_scores


def save_interpolations(pair_idx, interps, kps, opt, out_dir=Path("interps_out")):
    """
    Save interpolated reconstructions + keypoints for a chosen pair.
    """
    out_dir = out_dir / opt.model / opt.category / f"pair_{pair_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (rec, kp) in enumerate(zip(interps, kps)):
        np.save(out_dir / f"interp_{i:02d}.npy", rec.cpu().numpy())
        np.save(out_dir / f"kps{i:02d}.npy", kp.cpu().numpy())



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpolate between maximally different latent shapes."
    )
    subparsers = p.add_subparsers(dest="model", required=True)

    for model_name, model_cls in MODEL_CLASSES.items():
        sp = subparsers.add_parser(model_name)
        model_cls.get_parser(sp)

        try_add_arg(sp, "--ckpt", type=Path)
        try_add_arg(sp, "--category", type=str, default="chair")
        try_add_arg(sp, "--dataset_path", type=str, default=TESTSET)
        try_add_arg(sp, "--batch-size", type=int, default=1)
        try_add_arg(
            sp,
            "--pool-size",
            type=int,
            default=200,
            help="How many shapes to consider when mining diverse pairs.",
        )
        try_add_arg(
            sp,
            "--num-pairs",
            type=int,
            default=50,
            help="How many top-dissimilar pairs to interpolate.",
        )
        try_add_arg(sp, "--interpolation-steps", type=int, default=10)
        try_add_arg(sp, "--num-workers", type=int, default=4)

    return p


# ------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------
def make_loader(opt: argparse.Namespace):
    h5_files = glob(f"{opt.dataset_path}/**/*.h5", recursive=True)

    t = (
        transforms.Compose(
            [
                GridSample(
                    keys=("coord",),
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                ToTensor(),
                Collect(
                    keys=("coord", "grid_coord"),
                    feat_keys=("coord",),
                ),
            ]
        )
        if opt.model == "Ours"
        else None
    )

    dataset = H5Dataset(
        h5_files,
        normalize=True,
        get_two=False,
        include_label=False,
        object_name=opt.category,
        transform=t,
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # do not shuffle; we index into this later
        num_workers=opt.num_workers,
        collate_fn=collate_fn if opt.model == "Ours" else None,
        drop_last=False,
    )
    return loader, dataset


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = build_argparser()
    opt = p.parse_args()

    # load model
    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)
    model.model.eval().cuda()

    # load dataset
    _loader, dataset = make_loader(opt)
    num_total = len(dataset)
    print(f"[INFO] dataset size = {num_total}")

    # pick a pool of candidate indices to mine pairs from
    # strategy: either take the first pool_size, or sample if dataset is huge.
    candidate_indices = list(range(num_total))

    print(f"[INFO] mining pairs from pool of {len(candidate_indices)} shapes")

    # pre-load point clouds for that pool (on GPU once)
    pool_pcs = {}
    for idx in tqdm.tqdm(candidate_indices, desc="Loading point clouds"):
        sample = dataset[idx]
        pool_pcs[idx] = get_pointcloud(sample)  # (N,3) cuda tensor

    # compute pairwise distances
    print(f"[INFO] Computing pairwise Chamfer for {len(candidate_indices)} shapes ...")
    pool_pcs = [get_pointcloud(dataset[idx]) for idx in candidate_indices]

    pair_scores = batched_chamfer(pool_pcs, batch_size=8)
    top_pairs = pair_scores[: opt.num_pairs]
    print(f"[INFO] Selected {len(top_pairs)} most dissimilar pairs.")

    print(f"[INFO] selected {len(top_pairs)} diverse pairs")

    # now run interpolation for just those
    for pair_idx, (_cd_val, ia, ib) in enumerate(
        tqdm.tqdm(top_pairs, desc="Interpolating top pairs")
    ):
        pcd_a = dataset[ia]
        pcd_b = dataset[ib]

        interps, kps = model.interpolate_latent(
            pcd_a,
            pcd_b,
            n_steps=opt.interpolation_steps,
        )

        save_interpolations(pair_idx, interps, kps, opt, out_dir=Path("interps_out"))

    print("[✓] Done.")


if __name__ == "__main__":
    main()
