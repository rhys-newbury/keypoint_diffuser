#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from classes import MODEL_CLASSES
from torchvision import transforms

from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import Collect, GridSample, ToTensor


# ------------------------------------------------------------
# basic io
# ------------------------------------------------------------
def naive_read_pcd(path: Path):
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


def split_xyz_labels(arr: np.ndarray):
    xyz = arr[:, :3].astype(np.float32)
    labels = arr[:, 3].copy()
    return xyz, labels


def build_transform():
    return transforms.Compose(
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


# ------------------------------------------------------------
# normalization (old style, per-sample)
# ------------------------------------------------------------
def old_normalize(pc: np.ndarray):
    pcmax = pc.max()
    pcmin = pc.min()
    pcn = (pc - pcmin) / (pcmax - pcmin)
    pcn = 2.0 * (pcn - 0.5)
    return pcn.astype(np.float32), pcmax, pcmin


def old_apply_to_xyz(xyz: np.ndarray, pcmax: float, pcmin: float):
    """Apply the same min–max → [-1,1] normalization used on the PC."""
    x = (xyz - pcmin) / (pcmax - pcmin)
    x = 2.0 * (x - 0.5)
    return x.astype(np.float32)


# ------------------------------------------------------------
# mesh utils
# ------------------------------------------------------------
def load_obj_as_single_mesh(obj_path: Path):
    tm = trimesh.load(obj_path, process=False)
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) == 0:
            raise ValueError("Scene has no geometries")
        meshes = list(tm.geometry.values())
        tm = trimesh.util.concatenate(meshes)
    verts = np.asarray(tm.vertices, dtype=np.float32)
    faces = np.asarray(tm.faces, dtype=np.int64)
    return verts, faces


# ------------------------------------------------------------
# batching
# ------------------------------------------------------------
def prepare_batch_entries(kpn_ds_slice, opt, tform):
    Q = []
    label_clouds = []
    meta = []
    norm_facts = []  # (pcmax, pcmin) per sample

    for entry in kpn_ds_slice:
        cid = entry["class_id"]
        mid = entry["model_id"]

        # labeled cloud
        pc_labeled_path = opt.label_path / cid / mid / "models" / "point_resampled_labeled.npy"
        if not pc_labeled_path.exists():
            continue

        pc_labeled = np.load(pc_labeled_path)
        pc_xyz, pc_lbl = split_xyz_labels(pc_labeled)

        # original raw pc
        orig_pc_path = opt.pcd_path / cid / f"{mid}.pcd"
        orig_pc = naive_read_pcd(orig_pc_path)

        # use OLD normalization (per-sample)
        orig_pcn, pcmax, pcmin = old_normalize(orig_pc)
        labeled_pcn = old_apply_to_xyz(pc_xyz, pcmax, pcmin)

        # align labeled to orig
        from icp import icp_align_identity

        _, _, _, pc_aligned = icp_align_identity(labeled_pcn, orig_pcn)
        pc_aligned_labeled = np.concatenate(
            [pc_aligned, pc_lbl[:, None]], axis=1
        ).astype(np.float32)

        Q.append(orig_pcn.astype(np.float32))
        label_clouds.append(pc_aligned_labeled)
        meta.append((cid, mid))
        norm_facts.append((pcmax, pcmin))

    # make at least 2 for collate
    if len(Q) == 1:
        Q.append(Q[-1])
        label_clouds.append(label_clouds[-1])
        meta.append(meta[-1])
        norm_facts.append(norm_facts[-1])

    T_list = []
    for q in Q:
        data = {"coord": q}
        T_list.append(tform(data))

    batch = collate_fn(T_list)
    batch["orig"] = np.array(Q)

    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()

    return batch, label_clouds, meta, norm_facts


# ------------------------------------------------------------
# keypoint + mesh processing
# ------------------------------------------------------------
def project_to_surface(pred: torch.Tensor, surface_points: torch.Tensor, eps=0.05):
    dists = torch.cdist(pred[None, ...], surface_points[None, ...])  # (1,K,N)
    idx = dists.argmin(dim=-1)
    nearest = surface_points[idx]
    vec = nearest - pred[None, ...]
    dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-8
    overshoot = dist > eps
    scale = torch.where(overshoot, (dist - eps) / dist, torch.zeros_like(dist))
    pred_clamped = pred[None, ...] + vec * scale
    return pred_clamped[0]


def make_color_palette(K):
    base = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.8, 0.1],
            [0.8, 0.1, 0.8],
            [0.1, 0.8, 0.8],
            [0.9, 0.5, 0.1],
            [0.5, 0.1, 0.9],
            [0.3, 0.7, 0.3],
            [0.7, 0.3, 0.7],
            [0.6, 0.6, 0.6],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    reps = int(np.ceil(K / base.shape[0]))
    return np.tile(base, (reps, 1))[:K]


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="model", required=True)

    for model_name, model_cls in MODEL_CLASSES.items():
        subp = subparsers.add_parser(model_name)
        model_cls.get_parser(subp)
        subp.add_argument("--annotation-json", type=Path, required=True)
        subp.add_argument("--pcd-path", type=Path, required=True)
        subp.add_argument("--label-path", type=Path, required=True)
        subp.add_argument("--out-path", type=Path, default=Path("qual_keypoints.npz"))

    opt = parser.parse_args()

    # 1) load model
    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)
    model.model.eval().cuda()

    # 2) load full annotation file
    kpn_ds = json.load(open(opt.annotation_json))

    # 3) build transform + batch
    tform = build_transform()
    batch, label_clouds, meta, norm_facts = prepare_batch_entries(kpn_ds, opt, tform)

    # 4) run model
    with torch.no_grad():
        key_points = model.get_keypoints(batch)
        if isinstance(key_points, torch.Tensor):
            key_points = key_points.detach().cpu().numpy()

    # 5) per-sample: project kps to surface, load OBJ, map OBJ to PC space
    verts_list = []
    faces_list = []
    kps_list = []
    pc_list = []

    for (kp_np, lbl_pc, (cid, mid), (_pcmax, _pcmin)) in zip(
        key_points, label_clouds, meta, norm_facts
    ):
        surf_np = lbl_pc[:, :3]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kp_t = torch.tensor(kp_np, dtype=torch.float32, device=device)
        surf_t = torch.tensor(surf_np, dtype=torch.float32, device=device)
        kp_proj = project_to_surface(kp_t, surf_t).cpu().numpy()

        obj_path = opt.label_path / cid / mid / "models" / "model_normalized.obj"
        if obj_path.exists():
            try:
                obj_verts, obj_faces = load_obj_as_single_mesh(obj_path)
                obj_verts, _, _ = old_normalize(obj_verts)
            except Exception:
                hull = trimesh.Trimesh(vertices=surf_np, process=False).convex_hull
                obj_verts = np.asarray(hull.vertices, dtype=np.float32)
                obj_faces = np.asarray(hull.faces, dtype=np.int64)
        else:
            hull = trimesh.Trimesh(vertices=surf_np, process=False).convex_hull
            obj_verts = np.asarray(hull.vertices, dtype=np.float32)
            obj_faces = np.asarray(hull.faces, dtype=np.int64)

        verts_list.append(obj_verts.astype(np.float32))
        faces_list.append(obj_faces)
        kps_list.append(kp_proj.astype(np.float32))
        pc_list.append(surf_np.astype(np.float32))

    # 6) global bounds
    all_pts = np.concatenate(verts_list + kps_list + pc_list, axis=0)
    bounds_lo = all_pts.min(axis=0)
    bounds_hi = all_pts.max(axis=0)

    # 7) colors
    K = kps_list[0].shape[0]
    colors = np.asarray(make_color_palette(K), dtype=np.float32)

    # 8) save
    out_npz = opt.out_path.with_suffix(".npz")
    np.savez_compressed(
        out_npz,
        verts=np.array(verts_list, dtype=object),
        faces=np.array(faces_list, dtype=object),
        kps=np.array(kps_list, dtype=object),
        pcs=np.array(pc_list, dtype=object),
        colors=colors,
        bounds_lo=bounds_lo.astype(np.float32),
        bounds_hi=bounds_hi.astype(np.float32),
    )


if __name__ == "__main__":
    main()
