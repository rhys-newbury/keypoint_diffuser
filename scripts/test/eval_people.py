#!/usr/bin/env python3
"""
Evaluate keypoint-label correlation and DAS-style alignment on a PeopleDataset.
"""

import argparse
import contextlib
import sqlite3
from pathlib import Path

import numpy as np
import torch
import tqdm
from baselines.test_base import TestBase
from classes import MODEL_CLASSES
from datasets.people_dataset import PeopleDataset
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)
from torchvision import transforms


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


# ----------------------------
# Argparse helpers
# ----------------------------


def try_add_arg(p: argparse.ArgumentParser, arg, type=None, default=None, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        if type:
            p.add_argument(arg, type=type, default=default, **kwargs)
        else:
            p.add_argument(arg, default=default, **kwargs)


p = argparse.ArgumentParser()
subparsers = p.add_subparsers(dest="model", required=True)

# Create a subparser for each model
for model_name, model_cls in MODEL_CLASSES.items():
    subparser = subparsers.add_parser(model_name)
    model_cls.get_parser(subparser)

    try_add_arg(subparser, "--ckpt", type=Path)
    try_add_arg(
        subparser,
        "--people-dir",
        type=Path,
        default=Path("/app/ego"),
        help="Root dir for PeopleDataset (SMPL-X converted people).",
    )
    try_add_arg(subparser, "--batch-size", type=int, default=32)
    try_add_arg(subparser, "--key_point", type=int, default=10)
    try_add_arg(
        subparser,
        "--db-path",
        type=Path,
        default=Path("results_people.db"),
        help="SQLite DB file to store correlation + DAS runs.",
    )
    try_add_arg(
        subparser,
        "--category",
        type=str,
        help="Optional category tag for runs (for filtering later).",
        default="people",
    )
    try_add_arg(
        subparser,
        "--normalize",
        action="store_true",
        help="(Kept for compatibility; PeopleDataset already normalizes if desired.)",
    )
    try_add_arg(
        subparser,
        "--random-rotate",
        action="store_true",
        help="Ask PeopleDataset to apply random Y rotations.",
    )


# ----------------------------
# Utilities
# ----------------------------


def split_xyz_labels(arr: np.ndarray):
    """
    Expect arr shape (N, 4): xyz + label.
    Returns:
        xyz: (N, 3) float64
        labels: (N,) original dtype copy
    """
    xyz = arr[:, :3].astype(np.float64)
    labels = arr[:, 3].copy()
    return xyz, labels


# ----------------------------
# Prediction over PeopleDataset
# ----------------------------


def run_prediction_people(
    model: TestBase,
    opt: argparse.Namespace,
    dataset: PeopleDataset,
):
    """
    - pulls samples directly from PeopleDataset
    - uses pts[:, 3] as segmentation labels
    - also collects GT keypoints from the dataset for DAS
    """
    model.model.eval()
    model.model.cuda()

    t = transforms.Compose(
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

    out_kpcd = []  # predicted keypoints
    labels = []  # per-point xyz+merged label
    gt_kpts = []  # GT keypoints from dataset

    num_samples = len(dataset)

    for i in tqdm.tqdm(
        range(0, num_samples, opt.batch_size), unit_scale=opt.batch_size
    ):
        Q = []
        batch_gt_kpts = []

        for j in range(opt.batch_size):
            idx = i + j
            if idx >= num_samples:
                continue

            pts, kpts = dataset[idx]  # pts: (N,4), kpts: (K,3)

            xyz, pt_labels = split_xyz_labels(pts)

            # merge arm and leg classes: 2+3, 4+5 → 3,4 groups
            # lut index = old label, value = merged label
            lut = np.array([0, 1, 2, 2, 3, 3])
            pt_labels = lut[pt_labels.astype(int)]

            pc_aligned = np.concatenate([xyz, pt_labels[:, None]], axis=1)
            labels.append(pc_aligned)

            Q.append(xyz.astype(np.float32))
            batch_gt_kpts.append(np.asarray(kpts, dtype=np.float32))

        if len(Q) == 0:
            continue

        # keep behaviour: ensure batch size >= 2
        if len(Q) == 1:
            Q.append(Q[-1])
            batch_gt_kpts.append(batch_gt_kpts[-1])

        with torch.no_grad():
            Q_np = np.array(Q, dtype=np.float32)

            T_nb = []
            for b in range(Q_np.shape[0]):
                data = {"coord": Q_np[b]}
                T_nb.append(t(data))

            batch = collate_fn(T_nb)
            batch = {k: v.cuda() for k, v in batch.items()}
            batch["orig"] = Q_np

            key_points = model.get_keypoints(batch)  # iterable of (K, 3)

            # collect preds + GTs (note: we added the duplicated sample above)
            for kp, gk in zip(key_points, batch_gt_kpts):
                out_kpcd.append(kp)
                gt_kpts.append(gk)

    return out_kpcd, labels, gt_kpts


# ----------------------------
# ----------------------------


def project_to_surface(pred, surface_points, eps=0.05):
    """
    Hard project keypoints so they lie within `eps` distance of surface.
    If already closer than eps, no change.

    pred: (K, 3)
    surface_points: (N, 3)
    eps: allowed max distance
    """
    dists = torch.cdist(pred[None, ...], surface_points[None, ...])  # (1, K, N)
    idx = dists.argmin(dim=-1)  # (1, K)
    nearest = surface_points[idx]  # (K, 3)

    vec = nearest - pred
    dist = vec.norm(dim=-1, keepdim=True) + 1e-8

    overshoot = dist > eps
    scale = torch.where(overshoot, (dist - eps) / dist, torch.zeros_like(dist))
    pred_clamped = pred + vec * scale

    return pred_clamped.squeeze(0)


@torch.no_grad()
def keypoint_label_correlation(
    out_kpcd,  # List[ArrayLike] or (B, K, 3)
    labels,  # List[np.ndarray] each (Ni, 4): xyz + label
    threshold=0.05,
    device=None,
    model=None,
):
    """
    Returns:
        avg_corr: scalar tensor — mean over keypoints
        closest_labels_tensor: Bool tensor of shape (B, K, L)
        per_kp_toplabel_freq: Tensor (K,)
        per_kp_toplabel_id:   Long Tensor (K,)
    """
    B = len(out_kpcd)
    if B == 0:
        raise RuntimeError("No keypoints predicted; check your dataset / batching.")

    K = int(np.asarray(out_kpcd[0]).shape[0])

    # determine global label count L
    all_labels = np.concatenate([arr[:, 3] for arr in labels], axis=0)
    L = int(all_labels.max()) + 1

    d = (
        torch.device(device)
        if device is not None
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )

    closest_labels_list = []

    for b in range(B):
        kp = torch.as_tensor(out_kpcd[b], dtype=torch.float64, device=d)  # (K, 3)
        seg_points = torch.as_tensor(
            labels[b][:, :3], dtype=torch.float64, device=d
        )  # (N, 3)
        seg_labels = torch.as_tensor(
            labels[b][:, 3], dtype=torch.long, device=d
        )  # (N,)

        if model == "Ours":
            kp = project_to_surface(kp, seg_points, eps=threshold)

        distances = torch.cdist(kp, seg_points)  # (K, N)
        within = distances <= float(threshold)

        kp_idx, seg_idx = torch.nonzero(within, as_tuple=True)
        valid_labels = seg_labels[seg_idx]

        presence = torch.zeros((K, L), dtype=torch.bool, device=d)
        presence[kp_idx, valid_labels] = True

        closest_labels_list.append(presence)

    closest_labels_tensor = torch.stack(closest_labels_list, dim=0)  # (B, K, L)
    counts_per_label = closest_labels_tensor.sum(dim=0)  # (K, L)
    per_kp_toplabel_freq, per_kp_toplabel_id = counts_per_label.max(dim=1)

    per_kp_toplabel_freq = per_kp_toplabel_freq.to(torch.float64) / float(B)
    avg_corr = per_kp_toplabel_freq.mean()

    return (
        avg_corr,
        closest_labels_tensor,
        per_kp_toplabel_freq,
        per_kp_toplabel_id,
    )


# ----------------------------
# DAS-style alignment (using People GT keypoints)
# ----------------------------


def fwd_alignment_scores_people_threshold_multi(
    gt_kpts,
    predicted_kpcd,
    threshold=0.05,
):
    """
    Forward alignment with MULTI matches per predicted kp:
      - For each predicted kp p, collect all GT joints j with dist(p, gt_j) <= threshold
      - Build a presence tensor B x K_pred x K_gt
      - For each kp index k (0..K_pred-1), compute which GT joint is most
        frequently present across the batch (mode label)
      - The score per kp is: freq(top_label_k) / B
      - Final FWD score = mean over keypoints

    Returns:
        fwd_score: scalar float
        presence:  np.ndarray of shape (B, K_pred, K_gt), bool
                   presence[b, k, j] = True if GT j is within threshold of pred kp k in sample b
        per_kp_toplabel_freq: (K_pred,) frequency of most common GT per kp
        per_kp_toplabel_id:   (K_pred,) GT index of that most common label
    """
    B = len(predicted_kpcd)
    if B == 0:
        raise RuntimeError("No keypoints predicted; empty batch.")

    presence_list = []

    for pred, gt in zip(predicted_kpcd, gt_kpts):
        pred = np.asarray(pred, float)  # (K_pred, 3)
        gt = np.asarray(gt, float)  # (K_gt, 3)

        dist = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(axis=-1))

        # Find closest GT distance per predicted kp
        d_min = dist.min(axis=1, keepdims=True)  # (K_pred, 1)

        # dynamic threshold: closest * 1.10
        dynamic_thresh = d_min * 1.20  # (K_pred, 1)

        # broadcast compare: (K_pred, K_gt)
        within = dist <= dynamic_thresh
        presence_list.append(within)

    presence = np.stack(presence_list, axis=0)

    # For each kp and GT id, count in how many samples it`s present
    # counts_per_label[k, j] = number of samples where kp k was close to GT j
    counts_per_label = presence.sum(axis=0)  # (K_pred, K_gt)

    # For each kp, pick the GT joint with highest count (the "most common")
    per_kp_toplabel_freq = counts_per_label.max(axis=1)  # (K_pred,)
    per_kp_toplabel_id = counts_per_label.argmax(axis=1)  # (K_pred,)

    # normalize by B to get fraction of samples
    per_kp_toplabel_freq = per_kp_toplabel_freq.astype(np.float64) / float(B)

    # final scalar: mean over kps
    fwd_score = float(per_kp_toplabel_freq.mean())

    return fwd_score, presence, per_kp_toplabel_freq, per_kp_toplabel_id


def bwd_alignment_scores_people(gt_kpts, predicted_kpcd):
    """
    Backward alignment:
      for each GT keypoint i, see which predicted index it maps to across instances
      and measure consistency.
    """
    preds = {}  # semantic_id -> list of predicted indices

    for kpcd, gt in zip(predicted_kpcd, gt_kpts):
        kpcd = np.asarray(kpcd)
        gt = np.asarray(gt)

        dist = np.sum(
            (kpcd[:, None, :] - gt[None, :, :]) ** 2, axis=-1
        )  # (K_pred, K_gt)

        argminbwd = np.argmin(dist, axis=0)  # (K_gt,) predicted index for each GT

        K_gt = gt.shape[0]
        for i in range(K_gt):
            sid = i  # semantic id = joint index
            preds.setdefault(sid, []).append(argminbwd[i])

    q = []
    for arr in preds.values():
        arr = np.asarray(arr)
        q.append(np.mean(arr[:, None] == arr[None, :]))
    return float(np.mean(q))


# ----------------------------
# Database helpers
# ----------------------------


def init_corr_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs_correlation_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model TEXT NOT NULL,
            ckpt TEXT,
            people_dir TEXT,
            batch_size INTEGER,
            key_points INTEGER,
            category TEXT,
            correlation REAL,
            fwd REAL,
            bwd REAL,
            das REAL
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_corr_people_model ON runs_correlation_people(model);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_corr_people_time "
        "ON runs_correlation_people(created_at);"
    )

    con.commit()
    return con


def save_corr_run(
    db_path: Path, opt, correlation: float, fwd: float, bwd: float, das: float
) -> int:
    con = init_corr_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO runs_correlation_people (
            model, ckpt, people_dir, batch_size, key_points, category,
            correlation, fwd, bwd, das
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            opt.model,
            str(opt.ckpt) if getattr(opt, "ckpt", None) is not None else None,
            str(opt.people_dir),
            int(opt.batch_size),
            int(opt.key_point),
            str(opt.category),
            float(correlation),
            float(fwd),
            float(bwd),
            float(das),
        ),
    )
    con.commit()
    run_id = cur.lastrowid
    con.close()
    return run_id


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    opt = p.parse_args()

    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)

    dataset = PeopleDataset(
        root_dir=opt.people_dir,
        normalize=True,  # PeopleDataset handles normalization
        random_rotate=opt.random_rotate,
        get_keypoints=True,
    )

    out_kpcd, labels, gt_kpts = run_prediction_people(model, opt, dataset)

    # segment-label correlation
    avg_corr, pres_tensor, kp_freq, kp_label = keypoint_label_correlation(
        out_kpcd, labels, threshold=0.05, model=opt.model
    )

    # DAS-style metrics using GT keypoints
    fwd, assignments, _, _ = fwd_alignment_scores_people_threshold_multi(
        gt_kpts, out_kpcd
    )
    bwd = bwd_alignment_scores_people(gt_kpts, out_kpcd)
    das = 0.5 * (fwd + bwd)

    print("People correlation @ radius 0.05:", float(avg_corr))
    print("Forward Alignment Score (people):", float(fwd))
    print("Backward Alignment Score (people):", float(bwd))
    print("DAS (people):", float(das))

    run_id = save_corr_run(
        db_path=opt.db_path,
        opt=opt,
        correlation=float(avg_corr),
        fwd=float(fwd),
        bwd=float(bwd),
        das=float(das),
    )
    print(f"[✓] saved to {opt.db_path} (run_id={run_id})")
