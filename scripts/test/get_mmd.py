#!/usr/bin/env python3
"""
Generate shapes by PCA+KDE over training keypoints,
evaluate MMD (CD + EMD) against a reference set,
and log results to SQLite.

Usage:
  python gen_mmd_log.py Ours \
      --ckpt checkpoints/foo.pth \
      --category chair \
      --num-gen 256 \
      --pca-dim 16 \
      --db-path results.db
"""

import argparse
import contextlib
import sqlite3
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
import tqdm

# your project imports
from classes import MODEL_CLASSES
from datasets.H5Datset import H5Dataset
from keypoint_diffuser.utils.eval_metrics import MMD_CD_EMD
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import Collect, GridSample, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms


DEFAULT_TRAIN = "/app/shapenetcorev2_hdf5_2048/train"
DEFAULT_REF = "/app/shapenetcorev2_hdf5_2048/val"


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def make_loader(root, model_name, category, batch_size, num_workers, is_train):
    h5_files = glob(f"{root}**/*.h5", recursive=True)
    if model_name == "Ours":
        t = transforms.Compose(
            [
                GridSample(
                    keys=("coord",),
                    hash_type="fnv",
                    mode="train" if True else "test",
                    return_grid_coord=True,
                ),
                ToTensor(),
                Collect(keys=("coord", "grid_coord"), feat_keys=("coord",)),
            ]
        )
    else:
        t = None

    ds = H5Dataset(
        h5_files,
        normalize=True,
        get_two=model_name == "KPD",
        include_label=False,
        object_name=category,
        transform=t,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn if model_name == "Ours" else None,
        drop_last=False,
    )


# ------------------------------------------------------------
# Keypoint extraction
# ------------------------------------------------------------
def extract_latent_from_model(model, batch):
    kp, z_aux = model.get_latent(batch)
    return kp, z_aux


def collect_train_keypoints(model, loader, opt, device="cuda"):
    model.model.eval().to(device)
    all_kps, all_aux = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Collect train keypoints"):
            kps, z_aux = extract_latent_from_model(model, batch)
            if opt.model == "DPM":
                kps = kps.to(device).float()
                all_kps.append(kps.cpu())
            else:
                kps = kps.to(device).float().reshape(kps.shape[0], 10, -1)
                B, K, _ = kps.shape
                all_kps.append(kps.reshape(B, K * 3).cpu())

            if z_aux is not None:
                all_aux.append(z_aux.cpu())
    all_kps = torch.cat(all_kps, dim=0)
    if opt.model == "Ours":
        z_aux_mean = (
            torch.cat(all_aux, dim=0).mean(dim=0, keepdim=True)
            if len(all_aux)
            else None
        )
    else:
        z_aux_mean = torch.cat(all_aux, dim=0) if len(all_aux) else None

    return all_kps, z_aux_mean


# ------------------------------------------------------------
# PCA + KDE
# ------------------------------------------------------------
def fit_pca(x, pca_dim):
    x_np = x.numpy()
    mu = x_np.mean(axis=0, keepdims=True)
    x_centered = x_np - mu
    _U, _S, Vt = np.linalg.svd(x_centered, full_matrices=False)
    comps = Vt[:pca_dim]
    z_pca = x_centered @ comps.T
    return mu, comps, z_pca


def fit_diag_kde(z_pca, bw_scale=1.0):
    mu = z_pca.mean(axis=0, keepdims=True)
    std = z_pca.std(axis=0, keepdims=True) + 1e-6
    return mu, std * bw_scale


def sample_from_pca_kde(num, pca_mu, pca_comps, kde_mu, kde_std):
    z = kde_mu + kde_std * np.random.randn(num, kde_mu.shape[1])
    x = pca_mu + z @ pca_comps
    return x


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------
def generate_from_keypoints(model, kp_samples, z_aux_mean, opt):
    device = kp_samples.device

    if z_aux_mean is not None:
        if opt.model == "Ours":
            z_aux_rep = z_aux_mean.to(device).expand(kp_samples.shape[0], -1)
            z0 = torch.cat([kp_samples, z_aux_rep], dim=1)
        else:
            z0 = kp_samples, z_aux_mean
    else:
        z0 = kp_samples

    model.model.eval().to(device)
    with torch.no_grad():
        return model.generate(z0)


# ------------------------------------------------------------
# SQLite logging
# ------------------------------------------------------------
def init_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mmd_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            ckpt TEXT,
            category TEXT,
            pca_dim INTEGER,
            mmd_cd REAL,
            mmd_emd REAL,
            timestamp TEXT
        )
        """
    )
    con.commit()
    return con


def save_mmd_run(db_path: Path, opt, mmd_cd):
    con = init_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO mmd_results
        (model, ckpt, category, pca_dim, mmd_cd, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            opt.model,
            str(opt.ckpt) if opt.ckpt else None,
            opt.category,
            int(opt.pca_dim),
            float(mmd_cd),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()
    run_id = cur.lastrowid
    con.close()
    return run_id


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser("Generate by PCA+KDE and compute MMD metrics.")
    sub = p.add_subparsers(dest="model", required=True)
    for name, cls in MODEL_CLASSES.items():
        sp = sub.add_parser(name)
        cls.get_parser(sp)
        with contextlib.suppress(Exception):
            sp.add_argument("--ckpt", type=Path)
            sp.add_argument("--category", type=str)

        sp.add_argument("--key_point", type=int, default=10)

        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--num-workers", type=int, default=4)
        sp.add_argument("--trainset", type=str, default=DEFAULT_TRAIN)
        sp.add_argument("--refset", type=str, default=DEFAULT_REF)
        sp.add_argument("--pca-dim", type=int, default=16)
        sp.add_argument("--symmetric-mmd", action="store_true")
        sp.add_argument("--db-path", type=Path, default=Path("results.db"))
    return p


def covariance_scale_to_keypoints(points, keypoints, eps=1e-8):
    """
    points:    (B, N, 3)
    keypoints: (B, M, 3)
    returns:   (B, N, 3), (B, 1)
    """
    # 1) center
    keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
    p_mean = points.mean(dim=1, keepdim=True)
    k_mean = keypoints.mean(dim=1, keepdim=True)
    p_c = points - p_mean
    k_c = keypoints - k_mean

    # 2) covariance trace (sum of variances)
    # var = E[||x||^2]/d  up to constants; we'll just use mean squared norm
    p_var = (p_c**2).sum(dim=-1).mean(dim=1)  # (B,)
    k_var = (k_c**2).sum(dim=-1).mean(dim=1)  # (B,)

    s = torch.sqrt((k_var + eps) / (p_var + eps)).view(-1, 1, 1)  # (B,1,1)
    # 3) apply
    p_scaled = p_mean + s * p_c
    return p_scaled, s


def main():
    p = build_argparser()
    opt = p.parse_args()

    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)

    # 1) Collect training keypoints
    train_loader = make_loader(
        opt.trainset, opt.model, opt.category, opt.batch_size, opt.num_workers, True
    )
    all_kps, z_aux_mean = collect_train_keypoints(model, train_loader, opt)

    # 2) Fit PCA + KDE
    pca_mu, pca_comps, z_pca = fit_pca(all_kps, opt.pca_dim)
    kde_mu, kde_std = fit_diag_kde(z_pca)

    # 3) Sample keypoints & generate

    ref_loader = make_loader(
        opt.refset, opt.model, opt.category, opt.batch_size, opt.num_workers, False
    )
    num_ref = len(ref_loader.dataset)

    kp_samples_flat = sample_from_pca_kde(num_ref, pca_mu, pca_comps, kde_mu, kde_std)
    kp_samples = torch.from_numpy(kp_samples_flat).float().cuda()

    if z_aux_mean:
        idx = torch.randperm(z_aux_mean.shape[0])[: kp_samples.shape[0]]
        gen_pcs = (
            generate_from_keypoints(model, kp_samples, z_aux_mean[idx], opt)
            .float()
            .cuda()
        )
    else:
        gen_pcs = generate_from_keypoints(model, kp_samples, None, opt).float().cuda()

    if opt.model == "Ours":
        gen_pcs = covariance_scale_to_keypoints(gen_pcs, kp_samples)[0]

    # 4) Load reference pcs
    ref_pcs_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(ref_loader, desc="Load ref pcs"):
            if opt.model == "Ours":
                ref_pcs_list.append(
                    batch["target_shape"]
                    .float()
                    .reshape(batch["orig_offset"].shape[0], -1, 3)
                )
            elif opt.model == "KPD":
                ref_pcs_list.append(batch["source_shape"])
            else:
                ref_pcs_list.append(batch)
    ref_pcs = torch.cat(ref_pcs_list, dim=0).cuda()

    # 5) Compute MMD
    out = MMD_CD_EMD(gen_pcs, ref_pcs, batch_size=16, symmetric=opt.symmetric_mmd)
    print(f"MMD-CD  : {float(out['MMD-CD']):.6f}")

    # 6) Log results
    run_id = save_mmd_run(opt.db_path, opt, float(out["MMD-CD"]))
    print(f"[✓] Logged to {opt.db_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
