import argparse
import contextlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import tqdm
from classes import MODEL_CLASSES
from icp import icp_align_identity
from torchvision import transforms

from baselines.test_base import TestBase
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


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
    model_cls.get_parser(
        subparser
    )  # pass the subparser in instead of creating inside get_parser
    try_add_arg(subparser, "--ckpt", type=Path)
    try_add_arg(
        subparser,
        "--annotation-json",
        type=Path,
        default="/app/annotations/table.json",
    )
    try_add_arg(subparser, "--pcd-path", type=Path, default="/app/pcds")
    try_add_arg(subparser, "--batch-size", type=int, default=32)
    try_add_arg(subparser, "--key_point", type=int, default=10)

    try_add_arg(
        subparser, "--category", type=str, help="Category of objects", default="chair"
    )
    try_add_arg(subparser, "--db-path", type=Path, default=Path("results.db"))
    try_add_arg(subparser, "--save", action="store_true")
    try_add_arg(
        subparser, "--label-path", type=Path, default=Path("../../copied_points")
    )

# ----------------------------
# Utilities
# ----------------------------


def naive_read_pcd(path):
    with open(path) as f:
        lines = f.readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith("DATA ascii"):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(" ") for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float32)
    return pc


# ----------------------------
# Prediction Function
# ----------------------------
def split_xyz_labels(arr):
    xyz = arr[:, :3].astype(np.float64)
    labels = arr[:, 3].copy()
    return xyz, labels


def run_prediction(model: TestBase, opt: argparse.Namespace):
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

    kpn_ds = json.load(open(opt.annotation_json))
    out_kpcd = []
    labels = []

    for i in tqdm.tqdm(
        range(0, len(kpn_ds), opt.batch_size), unit_scale=opt.batch_size
    ):
        Q = []
        for j in range(opt.batch_size):
            if i + j >= len(kpn_ds):
                continue
            entry = kpn_ds[i + j]
            cid = entry["class_id"]
            mid = entry["model_id"]

            pc_path = (
                opt.label_path / cid / mid / "models" / "point_resampled_labeled.npy"
            )
            if not pc_path.exists():
                continue

            pc = np.load(pc_path)
            pc, pc_labels = split_xyz_labels(pc)

            orig_pc_path = opt.pcd_path / cid / f"{mid}.pcd"
            orig_pc = naive_read_pcd(orig_pc_path)

            orig_pcmax = orig_pc.max()
            orig_pcmin = orig_pc.min()
            orig_pcn = (orig_pc - orig_pcmin) / (orig_pcmax - orig_pcmin)
            orig_pcn = 2.0 * (orig_pcn - 0.5)

            pcmax = pc.max()
            pcmin = pc.min()
            pcn = (pc - pcmin) / (pcmax - pcmin)
            pcn = 2.0 * (pcn - 0.5)

            _T, _, _, pc_aligned = icp_align_identity(pcn, orig_pcn)
            pc_aligned = np.concatenate([pc_aligned, pc_labels[:, None]], axis=1)

            Q.append(orig_pcn)
            labels.append(pc_aligned)

        if len(Q) == 1:
            Q.append(Q[-1])

        with torch.no_grad():
            Q = np.array(Q)
            T_nb = []
            for i in range(Q.shape[0]):
                data = {"coord": Q[i]}
                T_nb.append(t(data))

            batch = collate_fn(
                T_nb
            )  # assumes collate_fn knows how to stack dictionaries correctly
            batch = {k: v.cuda() for k, v in batch.items()}
            batch["orig"] = Q

            with torch.no_grad():
                key_points = model.get_keypoints(batch)

            for kp in key_points:
                out_kpcd.append(kp)

    return out_kpcd, labels


# ----------------------------
# Evaluation Functions
# ----------------------------


def project_to_surface(pred, surface_points, eps=0.05):
    """
    Hard project keypoints so they lie within `eps` distance of surface.
    If already closer than eps, no change.

    pred: (K, 3)
    surface_points: (N, 3)
    eps: allowed max distance
    """
    # Compute nearest surface point for each keypoint
    dists = torch.cdist(pred[None, ...], surface_points[None, ...])  # (1, K, N)
    idx = dists.argmin(dim=-1)  # (1, K)
    nearest = surface_points[idx]  # (K, 3)

    # Vector from keypoint → nearest surface
    vec = nearest - pred
    dist = vec.norm(dim=-1, keepdim=True) + 1e-8

    # If farther than eps, pull it just enough to reach eps
    overshoot = dist > eps
    scale = torch.where(overshoot, (dist - eps) / dist, torch.zeros_like(dist))
    pred_clamped = pred + vec * scale

    return pred_clamped.squeeze(0)


@torch.no_grad()
def keypoint_label_correlation(
    out_kpcd,  # List[ArrayLike] or (B, K*3) / (B, K, 3)
    labels,  # List[np.ndarray] with shape (Ni, 4): xyz + label
    threshold=0.05,  # radius for "nearby" match (same units as coords)
    device=None,
):
    """
    Returns:
        avg_corr: scalar tensor — mean over keypoints of (most-frequent nearby label frequency across batch)
        closest_labels_tensor: Bool tensor of shape (B, K, L) — per sample/keypoint/label presence
        per_kp_toplabel_freq: Tensor (K,) — frequency of the top label across batch per keypoint
        per_kp_toplabel_id:   Long Tensor (K,) — which label is top per keypoint
    """
    B = len(out_kpcd)

    # determine K and global label count L
    K = int(np.asarray(out_kpcd[0]).shape[0])

    # collect max label across batch (assumes non-negative integer labels)
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
        kp = torch.as_tensor(out_kpcd[b], dtype=torch.float64, device=d)  # (K,3)
        seg_points = torch.as_tensor(
            labels[b][:, :3], dtype=torch.float64, device=d
        )  # (N,3)
        seg_labels = torch.as_tensor(
            labels[b][:, 3], dtype=torch.long, device=d
        )  # (N,)

        kp = project_to_surface(kp, seg_points)
        # pairwise distances KP x SEG
        distances = torch.cdist(kp, seg_points)  # (K,N), float64
        within = distances <= float(threshold)  # boolean (K,N)

        # Find all (kp_idx, seg_idx) within threshold
        kp_idx, seg_idx = torch.nonzero(within, as_tuple=True)
        valid_labels = seg_labels[seg_idx]  # (M,)

        # presence matrix (K,L) — True if any seg point of label l near kp k
        presence = torch.zeros((K, L), dtype=torch.bool, device=d)
        presence[kp_idx, valid_labels] = True

        closest_labels_list.append(presence)

    closest_labels_tensor = torch.stack(closest_labels_list, dim=0)

    # For each keypoint (across batch), count how many samples had label l present
    counts_per_label = closest_labels_tensor.sum(
        dim=0
    )  # sum over batch -> frequency per label

    # Top label per keypoint and its batch frequency
    per_kp_toplabel_freq, per_kp_toplabel_id = counts_per_label.max(dim=1)  # (K,)
    # Normalize by B to get fraction
    per_kp_toplabel_freq = per_kp_toplabel_freq.to(torch.float64) / float(B)

    # Final scalar: average over keypoints
    avg_corr = per_kp_toplabel_freq.mean()

    return (
        avg_corr,
        closest_labels_tensor,
        per_kp_toplabel_freq,
        per_kp_toplabel_id,
    )  # ----------------------------


# Database
# ----------------------------


def init_corr_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs_correlation2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model TEXT NOT NULL,
            ckpt TEXT,
            annotation_json TEXT,
            pcd_path TEXT,
            batch_size INTEGER,
            key_points INTEGER,
            category TEXT,
            correlation REAL
        );
        """
    )
    # optional: helpful indexes
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_corr_model ON runs_correlation2(model);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_corr_time ON runs_correlation2(created_at);"
    )
    con.commit()
    return con


def save_corr_run(db_path: Path, opt, correlation: float) -> int:
    con = init_corr_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO runs_correlation2 (
            model, ckpt, annotation_json, pcd_path, batch_size, key_points, category, correlation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            opt.model,
            str(opt.ckpt) if getattr(opt, "ckpt", None) is not None else None,
            str(opt.annotation_json),
            str(opt.pcd_path),
            int(opt.batch_size),
            int(opt.key_point),
            str(opt.category),
            float(correlation),
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
    model = model_cls()  # or model_cls(opt) if your ctor expects args
    model.load_model(opt.ckpt, opt)

    out_kpcd, labels = run_prediction(model, opt)
    avg_corr, pres_tensor, kp_freq, kp_label = keypoint_label_correlation(
        out_kpcd, labels, threshold=0.05
    )

    print("correlation 0.05: ", avg_corr)

    run_id = save_corr_run(db_path=opt.db_path, opt=opt, correlation=avg_corr)
    print(f"[✓] saved to {opt.db_path} (run_id={run_id})")
