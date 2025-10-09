#!/usr/bin/env python3
"""
SMPL-X PKL -> point cloud (.ply) + labeled **major keypoints** (.keypoints.npz)
+ **long-part midpoints**, and **6-way semantic labels** for the sampled
point cloud (.semseg.npz) with robust head/arm/torso boundaries.

Buckets: ["torso","head","left_arm","right_arm","left_leg","right_leg"]
"""

import argparse
import pickle
import sys
from pathlib import Path
from random import shuffle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pymeshfix
import smplx
import torch
import trimesh
from tqdm import tqdm


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def load_pkl(pkl_path: str) -> dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = {k.decode() if isinstance(k, bytes) else k: v for k, v in data.items()}
    return data


def to_tensor(x, dtype=torch.float32, device="cpu"):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


# -----------------------------------------------------------------------------
# SMPL-X utilities
# -----------------------------------------------------------------------------


def extract_smplx_params(pkl_dict: dict[str, Any]) -> dict[str, Any]:
    container = None
    for key in [
        "results",
        "smplx",
        "smplx_params",
        "body_params",
        "output",
        "pred_params",
    ]:
        if key in pkl_dict and isinstance(pkl_dict[key], dict):
            container = pkl_dict[key]
            break
    params_src = container if container is not None else pkl_dict

    def first_present(*names):
        for n in names:
            if n in params_src:
                return params_src[n]
        return None

    params = {
        "betas": first_present("betas", "shape", "shapes"),
        "global_orient": first_present(
            "global_orient", "root_orient", "global_orient_rotmat"
        ),
        "body_pose": first_present("body_pose", "pose_body", "body_pose_rotmat"),
        "transl": first_present("transl", "translation"),
        "left_hand_pose": first_present("left_hand_pose", "hand_pose_left"),
        "right_hand_pose": first_present("right_hand_pose", "hand_pose_right"),
        "jaw_pose": first_present("jaw_pose"),
        "expression": first_present("expression", "expr"),
    }

    for k, v in params.items():
        if v is None:
            continue
        t = to_tensor(v)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim > 2 and k.endswith("_pose"):
            t = t.reshape(t.shape[0], -1) if t.shape[0] == 1 else t[:1].reshape(1, -1)
        else:
            if t.shape[0] > 1:
                t = t[:1]
        params[k] = t
    return params


def smplx_model(model_dir: str, gender: str):
    return smplx.create(
        model_dir,
        model_type="smplx",
        gender=gender,
        ext="npz",
        num_pca_comps=12,
        create_global_orient=True,
        create_transl=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
    )


def smplx_mesh_from_params(
    model, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    kwargs = {}
    for k in [
        "betas",
        "global_orient",
        "body_pose",
        "transl",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "expression",
    ]:
        if params.get(k) is not None:
            kwargs[k] = params[k]
    output = model(**kwargs)
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int64)

    joints = None
    if hasattr(output, "joints") and output.joints is not None:
        joints = output.joints[0].detach().cpu().numpy()

    joint_names = None
    if hasattr(model, "joint_names"):
        try:
            joint_names = list(model.joint_names)
        except Exception:
            joint_names = None
    parents = None
    if hasattr(model, "parents"):
        parents = np.array(model.parents, dtype=np.int64)

    return (
        vertices,
        faces,
        {"joints": joints, "joint_names": joint_names, "parents": parents},
    )


# -----------------------------------------------------------------------------
# Mesh sampling (watertight via PyMeshFix)
# -----------------------------------------------------------------------------


def sample_point_cloud_with_meshfix(vertices, faces, n_points, *, interior_ratio=0.0):
    mf = pymeshfix.MeshFix(vertices, faces)
    mf.repair(verbose=False)
    mesh = trimesh.Trimesh(vertices=mf.v, faces=mf.f, process=False)
    if not mesh.is_watertight:
        print("[warn] mesh not watertight after repair")

    r = float(np.clip(interior_ratio, 0.0, 1.0))
    n_interior = int(round(r * n_points))
    n_surface = n_points - n_interior

    # robust resampling
    def safe_sample_surface(mesh, n):
        out = []
        while len(out) < n:
            remain = n - len(out)
            pts, _ = trimesh.sample.sample_surface(mesh, remain)
            out.append(pts)
            if len(pts) == 0:  # bail if it`s stuck
                break
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3), np.float32)

    points = []
    if n_surface > 0:
        surf_pts = safe_sample_surface(mesh, n_surface)
        points.append(surf_pts.astype(np.float32))
    if n_interior > 0:
        int_pts = trimesh.sample.volume_mesh(mesh, n_interior).astype(np.float32)
        points.append(int_pts)

    all_points = np.vstack(points) if points else np.zeros((0, 3), np.float32)

    # pad or crop to exactly n_points
    if all_points.shape[0] < n_points:
        extra = np.random.choice(all_points.shape[0], n_points - all_points.shape[0])
        all_points = np.vstack([all_points, all_points[extra]])
    elif all_points.shape[0] > n_points:
        all_points = all_points[
            np.random.choice(all_points.shape[0], n_points, replace=False)
        ]

    assert all_points.shape[0] == 2048

    return all_points, np.zeros((n_points,), np.int64)


def save_ply(points: np.ndarray, out_path: Path):
    trimesh.points.PointCloud(points).export(out_path)


# -----------------------------------------------------------------------------
# Keypoints + midpoints (compact, same as before)
# -----------------------------------------------------------------------------


def _norm_name(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "_")


def _find_index(
    joint_names: list[str] | None, candidates: list[str], J: int
) -> int | None:
    if not joint_names:
        return None
    names = [_norm_name(n) for n in joint_names]
    Jn = min(len(names), J)
    cands = [_norm_name(c) for c in candidates]
    for i in range(Jn):
        for c in cands:
            if c in names[i]:
                return i
    return None


def _sanitize_parents(parents, J: int) -> np.ndarray:
    """Return a length-J parent array with invalid entries set to -1."""
    if parents is None:
        return np.full((J,), -1, dtype=np.int64)
    p = np.asarray(parents).reshape(-1).astype(np.int64, copy=True)

    # truncate/pad to length J
    if p.shape[0] >= J:
        p = p[:J]
    else:
        tmp = np.full((J,), -1, dtype=np.int64)
        tmp[: p.shape[0]] = p
        p = tmp

    # clamp invalid parent indices
    p[(p < -1) | (p >= J)] = -1
    return p


def select_major_joints(
    joints: np.ndarray, joint_names: list[str] | None, parents: np.ndarray | None
):
    J = joints.shape[0]
    wanted = [
        ("pelvis", ["pelvis", "hips", "root"]),
        ("chest", ["chest", "upper_chest", "spine3", "spine2", "neck"]),
        ("head", ["head", "head_top"]),
        ("l_shoulder", ["left_shoulder", "l_shoulder", "left_collar", "l_collar"]),
        ("r_shoulder", ["right_shoulder", "r_shoulder", "right_collar", "r_collar"]),
        ("l_elbow", ["left_elbow", "l_elbow"]),
        ("r_elbow", ["right_elbow", "r_elbow"]),
        ("l_wrist", ["left_wrist", "l_wrist", "left_hand"]),
        ("r_wrist", ["right_wrist", "r_wrist", "right_hand"]),
        ("l_hip", ["left_hip", "l_hip", "leftupleg", "left_upleg", "left_upLeg"]),
        ("r_hip", ["right_hip", "r_hip", "rightupleg", "right_upleg", "right_upLeg"]),
        ("l_knee", ["left_knee", "l_knee"]),
        ("r_knee", ["right_knee", "r_knee"]),
        ("l_ankle", ["left_ankle", "l_ankle", "left_foot"]),
        ("r_ankle", ["right_ankle", "r_ankle", "right_foot"]),
    ]

    if joint_names:
        idxs, labels, seen = [], [], set()
        for label, cands in wanted:
            i = _find_index(joint_names, cands, J)
            if i is not None and 0 <= i < J and i not in seen:
                seen.add(i)
                idxs.append(i)
                labels.append(label)
        if idxs:
            return joints[np.array(idxs, dtype=int)], labels

    # fallback: graphy selection
    p = _sanitize_parents(parents, J)

    root = int(np.where(p == -1)[0][0]) if np.any(p == -1) else 0
    depth = np.full(J, -1, int)
    depth[root] = 0
    changed = True
    while changed:
        changed = False
        for c in range(J):
            if p[c] >= 0 and depth[p[c]] >= 0 and depth[c] < depth[p[c]] + 1:
                depth[c] = depth[p[c]] + 1
                changed = True
    head_like = int(np.argmax(depth))
    edges = []
    for c in range(J):
        par = int(p[c])
        if par >= 0:
            edges.append((np.linalg.norm(joints[c] - joints[par]), par, c))
    edges.sort(reverse=True, key=lambda t: t[0])
    majors_idx = {root, head_like}
    for _, par, c in edges:
        majors_idx.add(par)
        majors_idx.add(c)
        if len(majors_idx) >= 15:
            break
    majors_idx = sorted(majors_idx)
    majors = joints[np.array(majors_idx, dtype=int)]
    labels = [
        f"joint_{i}"
        if i not in (root, head_like)
        else ("pelvis" if i == root else "head")
        for i in majors_idx
    ]
    return majors, labels


def compute_midpoints_longparts(
    joints: np.ndarray, joint_names: list[str] | None, parents: np.ndarray | None
):
    J = joints.shape[0]
    mids, labels = [], []
    if joint_names:
        PAIRS = [
            (
                ["left_shoulder", "l_shoulder", "left_collar", "l_collar"],
                ["left_elbow", "l_elbow"],
                "mid_left_upperarm",
            ),
            (
                ["left_elbow", "l_elbow"],
                ["left_wrist", "l_wrist", "left_hand"],
                "mid_left_forearm",
            ),
            (
                ["right_shoulder", "r_shoulder", "right_collar", "r_collar"],
                ["right_elbow", "r_elbow"],
                "mid_right_upperarm",
            ),
            (
                ["right_elbow", "r_elbow"],
                ["right_wrist", "r_wrist", "right_hand"],
                "mid_right_forearm",
            ),
            (
                ["left_hip", "l_hip", "leftupleg", "left_upleg", "left_upLeg"],
                ["left_knee", "l_knee"],
                "mid_left_thigh",
            ),
            (
                (["left_knee", "l_knee"]),
                (["left_ankle", "l_ankle", "left_foot"]),
                "mid_left_shin",
            ),
            (
                ["right_hip", "r_hip", "rightupleg", "right_upleg", "right_upLeg"],
                ["right_knee", "r_knee"],
                "mid_right_thigh",
            ),
            (
                ["right_knee", "r_knee"],
                ["right_ankle", "r_ankle", "right_foot"],
                "mid_right_shin",
            ),
            (
                ["left_shoulder", "l_shoulder", "left_collar", "l_collar"],
                ["right_shoulder", "r_shoulder", "right_collar", "r_collar"],
                "mid_chest",
            ),
            (
                ["pelvis", "hips", "root"],
                ["chest", "upper_chest", "spine3", "spine2", "neck"],
                "mid_torso",
            ),
        ]
        for a_names, b_names, lbl in PAIRS:
            ia = _find_index(joint_names, a_names, J)
            ib = _find_index(joint_names, b_names, J)
            if ia is None or ib is None:
                continue
            if 0 <= ia < J and 0 <= ib < J:
                mids.append(0.5 * (joints[ia] + joints[ib]))
                labels.append(lbl)
        if mids:
            return np.stack(mids, axis=0), labels

    # fallback simple
    return np.zeros((0, 3), dtype=joints.dtype), []


# -----------------------------------------------------------------------------
# 6-way labeling from LBS with gates/priors/denoising
# -----------------------------------------------------------------------------


def _resolve_joint(joint_names, cands):
    if not joint_names:
        return None
    nn = [s.lower().replace(" ", "_").replace("-", "_") for s in joint_names]
    for cand in cands:
        c = cand.lower().replace(" ", "_").replace("-", "_")
        for i, name in enumerate(nn):
            if c in name:
                return i
    return None


def _segdist(P, A, B):
    V = B - A
    VV = float(np.dot(V, V)) + 1e-12
    t = np.clip(((P - A) @ V) / VV, 0.0, 1.0)
    proj = A + t[:, None] * V[None, :]
    return np.linalg.norm(P - proj, axis=1), t, proj


def _knn_mode(labels, points, k=12, iters=1):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        _, nn = tree.query(points, k=k + 1)
        nn = nn[:, 1:]
    except Exception:
        D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        nn = np.argsort(D, axis=1)[:, 1 : k + 1]
    lab = labels.copy()
    for _ in range(iters):
        new = lab.copy()
        for i in range(points.shape[0]):
            b = np.bincount(lab[nn[i]], minlength=lab.max() + 1)
            m = np.argmax(b)
            if b[m] > b[lab[i]]:
                new[i] = m
        lab = new
    return lab


def get_joint_names_for_model(model, n_joints: int):
    if hasattr(model, "joint_names"):
        try:
            names = list(model.joint_names)
            if len(names) == n_joints:
                return names
        except Exception:
            pass
    if n_joints == 55:
        return [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "jaw",
            "left_eye",
            "right_eye",
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
        ]
    return [f"joint_{i}" for i in range(n_joints)]


def label_points_from_lbs_interp_6(
    points: np.ndarray,
    model_mesh: trimesh.Trimesh,
    model,
    joints: np.ndarray | None = None,
    joint_names: list[str] | None = None,
    normalize: str = "max",  # sharper boundaries than "mean"
    use_spatial_prior: bool = True,
    sigma_arm: float = 0.05,
    sigma_torso: float = 0.08,
    torso_weight: float = 0.9,
):
    # LBS weights
    lbs = getattr(model, "lbs_weights", None)
    if lbs is None:
        raise RuntimeError("SMPL-X model doesn't expose lbs_weights.")
    W = lbs.detach().cpu().numpy()
    V, J = W.shape

    # joint names / buckets
    jn = joint_names if joint_names is not None else get_joint_names_for_model(model, J)

    def _bucket(n: str):
        n = n.lower().replace(" ", "_").replace("-", "_")
        if any(
            k in n
            for k in [
                "hand",
                "wrist",
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "finger",
                "shoulder",
                "collar",
                "clav",
                "upperarm",
                "forearm",
                "arm",
            ]
        ):
            if n.startswith(("left_", "l_")) or "left" in n or "_l" in n:
                return "left_arm"
            if n.startswith(("right_", "r_")) or "right" in n or "_r" in n:
                return "right_arm"
        if any(
            k in n
            for k in [
                "hip",
                "upleg",
                "thigh",
                "shin",
                "knee",
                "ankle",
                "foot",
                "toe",
                "leg",
            ]
        ):
            if n.startswith(("left_", "l_")) or "left" in n or "_l" in n:
                return "left_leg"
            if n.startswith(("right_", "r_")) or "right" in n or "_r" in n:
                return "right_leg"
        if any(k in n for k in ["head", "jaw", "eye"]):
            return "head"
        if any(
            k in n
            for k in ["torso", "pelvis", "spine", "chest", "root", "hips", "neck"]
        ):
            return "torso"
        return "torso"

    buckets = ["torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"]
    bucket_to_idx = {b: i for i, b in enumerate(buckets)}
    joint_to_bucket = np.array([bucket_to_idx[_bucket(n)] for n in jn], dtype=np.int32)

    # project points → faces → barycentrics
    closest_pts, _, face_ids = trimesh.proximity.closest_point(model_mesh, points)
    tris = model_mesh.triangles[face_ids]
    bc = trimesh.triangles.points_to_barycentric(tris, closest_pts)
    face_vidx = model_mesh.faces[face_ids]
    W_face = W[face_vidx]
    Wp = (W_face * bc[:, :, None]).sum(axis=1)  # (N,J)

    # bucket aggregation
    scores = np.zeros((points.shape[0], 6), dtype=np.float32)
    if normalize == "max":
        for b in range(6):
            idx = joint_to_bucket == b
            if np.any(idx):
                scores[:, b] = Wp[:, idx].max(axis=1)
    else:  # "mean"
        for b in range(6):
            idx = joint_to_bucket == b
            if np.any(idx):
                scores[:, b] = Wp[:, idx].mean(axis=1)

    # optional spatial prior (helps wrists/shoulders)
    if use_spatial_prior and joints is not None:

        def _idx(names):
            return _resolve_joint(jn, names)

        ls, le, lw = (
            _idx(["left_shoulder", "l_shoulder", "left_collar", "l_collar"]),
            _idx(["left_elbow", "l_elbow"]),
            _idx(["left_wrist", "l_wrist", "left_hand"]),
        )
        rs, re, rw = (
            _idx(["right_shoulder", "r_shoulder", "right_collar", "r_collar"]),
            _idx(["right_elbow", "r_elbow"]),
            _idx(["right_wrist", "r_wrist", "right_hand"]),
        )
        pel, chest = _idx(["pelvis", "hips", "root"]), _idx(
            ["chest", "upper_chest", "spine3", "spine2", "neck"]
        )

        def segdist(P, A, B):
            V = B - A
            VV = float(np.dot(V, V)) + 1e-12
            t = np.clip(((P - A) @ V) / VV, 0.0, 1.0)
            proj = A + t[:, None] * V[None, :]
            return np.linalg.norm(P - proj, axis=1)

        prior = np.ones_like(scores)
        if ls is not None and le is not None and lw is not None:
            dL = np.minimum(
                segdist(points, joints[ls], joints[le]),
                segdist(points, joints[le], joints[lw]),
            )
            prior[:, bucket_to_idx["left_arm"]] *= np.exp(
                -(dL**2) / (2 * sigma_arm**2)
            )
        if rs is not None and re is not None and rw is not None:
            dR = np.minimum(
                segdist(points, joints[rs], joints[re]),
                segdist(points, joints[re], joints[rw]),
            )
            prior[:, bucket_to_idx["right_arm"]] *= np.exp(
                -(dR**2) / (2 * sigma_arm**2)
            )
        if pel is not None and chest is not None:
            dT = segdist(points, joints[pel], joints[chest])
            prior[:, bucket_to_idx["torso"]] *= torso_weight * np.exp(
                -(dT**2) / (2 * sigma_torso**2)
            )
        scores *= prior

    # ------------------------------------------------------------------
    # HEAD vs TORSO: robust bisector gate + cylinder around head axis
    # ------------------------------------------------------------------
    torso_id, head_id = bucket_to_idx["torso"], bucket_to_idx["head"]
    ch = _resolve_joint(jn, ["chest", "upper_chest", "spine3", "spine2"])
    hd = _resolve_joint(jn, ["head"])
    pel = _resolve_joint(jn, ["pelvis", "hips", "root"])

    if joints is not None and ch is not None and hd is not None:
        head_axis = joints[hd] - joints[ch]
        L = float(np.linalg.norm(head_axis))
        if L > 1e-8:
            uh = head_axis / L  # chest -> head

            # torso axis (chest -> pelvis). If missing, use the opposite of head.
            if pel is not None:
                ut = joints[pel] - joints[ch]
                ut /= np.linalg.norm(ut) + 1e-8
            else:
                ut = -uh

            # ensure opposite orientation
            if np.dot(uh, ut) > 0.0:
                ut = -ut

            # bisector plane normal pointing toward the head side
            gate_n = uh - ut

            # half-space test: head side if (p - chest)·gate_n > 0
            s = (points - joints[ch][None, :]) @ gate_n
            head_halfspace = s > 0.0

            # cylinder around head axis (keep a generous radius)
            t = (points - joints[ch][None, :]) @ uh
            proj = joints[ch][None, :] + t[:, None] * uh[None, :]
            r = np.linalg.norm(points - proj, axis=1)

            # geometric plausibility region for the head
            in_cyl = r <= 0.12  # ~12cm radius; tweak 0.10-0.14
            along = (t > -0.02) & (t < 1.6 * L)
            head_zone = head_halfspace & in_cyl & along

            # soften torso and lift head inside the head zone
            # (prevents back-of-head from flipping to torso)
            scores[head_zone, torso_id] *= 0.02
            scores[head_zone, torso_id] -= 1.0
            scores[head_zone, head_id] = np.maximum(scores[head_zone, head_id], 0.25)

    # TORSO suppression around limbs (half-space + distance)
    def _suppress_torso_along(A, B, radius):
        if A is None or B is None:
            return
        d, t, _ = _segdist(points, joints[A], joints[B])
        limb_dir = joints[B] - joints[A]
        side = ((points - joints[A]) @ limb_dir) > 0.0
        near = d <= radius
        mask = side & near
        scores[mask, torso_id] *= 0.05
        scores[mask, torso_id] -= 1.0

    ls = _resolve_joint(jn, ["left_shoulder", "l_shoulder", "left_collar", "l_collar"])
    le = _resolve_joint(jn, ["left_elbow", "l_elbow"])
    lw = _resolve_joint(jn, ["left_wrist", "l_wrist", "left_hand"])
    rs = _resolve_joint(
        jn, ["right_shoulder", "r_shoulder", "right_collar", "r_collar"]
    )
    re = _resolve_joint(jn, ["right_elbow", "r_elbow"])
    rw = _resolve_joint(jn, ["right_wrist", "r_wrist", "right_hand"])
    lh = _resolve_joint(jn, ["left_hip", "l_hip"])
    lk = _resolve_joint(jn, ["left_knee", "l_knee"])
    rh = _resolve_joint(jn, ["right_hip", "r_hip"])
    rk = _resolve_joint(jn, ["right_knee", "r_knee"])

    _suppress_torso_along(ls, le, radius=0.07)
    _suppress_torso_along(le, lw, radius=0.06)
    _suppress_torso_along(rs, re, radius=0.07)
    _suppress_torso_along(re, rw, radius=0.06)
    _suppress_torso_along(lh, lk, radius=0.09)
    _suppress_torso_along(rh, rk, radius=0.09)

    labels6 = np.argmax(scores, axis=1).astype(np.int32)

    # lock tiny sphere around head joint to head
    if joints is not None and hd is not None:
        head_core = np.linalg.norm(points - joints[hd][None, :], axis=1) <= 0.055
        labels6[head_core] = head_id

    # light denoise
    labels6 = _knn_mode(labels6, points, k=12, iters=1)
    return labels6, buckets


# -----------------------------------------------------------------------------
# Semseg + keypoint save/preview
# -----------------------------------------------------------------------------


def make_flat_outpath(base_pkl: Path, extension: str, parts: int = 5) -> Path:
    """
    Convert a nested .pkl path into a flat file name with given extension.
    Keeps the last `parts` components of the path joined by '_'.
    Example:
        recording_20210910_S06_S05_03/body_idx_0/results/frame_02368.pkl
    →  recording_20210910_S06_S05_03_body_idx_0_results_frame_02368.semseg.npz
    """
    suffixless = base_pkl.with_suffix("")  # remove .pkl
    tail_parts = suffixless.parts[-parts:]  # take last N parts
    flat_name = Path("_".join(tail_parts) + extension)
    return flat_name


def save_semseg_npz(
    base_pkl: Path, output_dir: Path, labels: np.ndarray, label_names: list[str]
) -> Path:
    out_path = output_dir / make_flat_outpath(base_pkl, ".semseg.npz")

    np.savez_compressed(
        out_path,
        labels=labels.astype(np.int32),
        label_names=np.array(label_names, dtype=object),
    )
    return out_path


def save_keypoints_npz(
    base_pkl: Path,
    output_dir: Path,
    joints: np.ndarray | None,
    joint_names: list[str] | None,
    parents: np.ndarray | None,
    midpoints: np.ndarray | None,
    mid_names: list[str] | None,
) -> Path:
    out_path = output_dir / make_flat_outpath(base_pkl, ".kp.npz")

    np.savez_compressed(
        out_path,
        joints=joints if joints is not None else np.zeros((0, 3), dtype=np.float32),
        joint_names=np.array(
            joint_names if joint_names is not None else [], dtype=object
        ),
        parents=parents if parents is not None else np.array([], dtype=np.int64),
        midpoints=midpoints
        if midpoints is not None
        else np.zeros((0, 3), dtype=np.float32),
        mid_names=np.array(mid_names if mid_names is not None else [], dtype=object),
    )
    return out_path


def save_keypoints_ply(
    base_pkl: Path,
    output_dir: Path,
    joints: np.ndarray | None,
    midpoints: np.ndarray | None,
) -> Path | None:
    if (joints is None or not joints.size) and (
        midpoints is None or not midpoints.size
    ):
        return None
    parts = []
    if joints is not None and joints.size:
        parts.append(joints)
    if midpoints is not None and midpoints.size:
        parts.append(midpoints)
    pts = np.concatenate(parts, axis=0)
    out_path = output_dir / make_flat_outpath(base_pkl, ".kp.ply")
    save_ply(pts, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Conversion + visualization
# -----------------------------------------------------------------------------


def convert_pkl_to_artifacts(
    pkl_path: Path,
    smplx_dir: Path,
    output_dir: Path,
    gender: str,
    n_points: int,
    with_normals: bool,
    inside: bool,
    interior_ratio: float,
    write_kpts_ply: bool,
) -> tuple[Path | None, Path | None, Path | None]:
    data = load_pkl(pkl_path)
    params = extract_smplx_params(data)
    model = smplx_model(smplx_dir, gender)

    vertices, faces, meta = smplx_mesh_from_params(model, params)
    joints = meta.get("joints", None)
    joint_names = meta.get("joint_names", None)
    parents = meta.get("parents", None)

    if joints is not None and joints.shape[0] > 55:
        joints = joints[:55]
        if parents is not None:
            parents = parents[:55]
        if joint_names is None or len(joint_names) != 55:
            joint_names = get_joint_names_for_model(model, 55)

    pts, _ = sample_point_cloud_with_meshfix(
        vertices, faces, n_points, interior_ratio=interior_ratio
    )

    # keypoints/mids
    if joints is not None:
        major_joints, major_names = select_major_joints(joints, joint_names, parents)
        if major_joints.size == 0:
            keep = min(15, joints.shape[0])
            major_joints = joints[:keep]
            major_names = [f"joint_{i}" for i in range(keep)]
        midpoints, mid_names = compute_midpoints_longparts(joints, joint_names, parents)
        major_parents = np.full((major_joints.shape[0],), -1, dtype=np.int64)
    else:
        major_joints = np.zeros((0, 3), np.float32)
        major_names = []
        major_parents = np.array([], dtype=np.int64)
        midpoints, mid_names = np.zeros((0, 3), np.float32), []

    # 6-way labels directly from LBS + gates
    model_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    seg_labels, bucket_names = label_points_from_lbs_interp_6(
        pts,
        model_mesh,
        model,
        joints=joints,
        joint_names=joint_names,
        normalize="max",
        use_spatial_prior=True,
    )
    save_semseg_npz(pkl_path, output_dir, seg_labels, bucket_names)

    # save cloud + kpts
    pcd_path = output_dir / make_flat_outpath(pkl_path, ".ply")
    save_ply(pts, pcd_path)
    kpt_npz_path = save_keypoints_npz(
        pkl_path,
        output_dir,
        major_joints,
        major_names,
        major_parents,
        midpoints,
        mid_names,
    )
    kpt_ply_path = (
        save_keypoints_ply(pkl_path, output_dir, major_joints, midpoints)
        if write_kpts_ply
        else None
    )
    return pcd_path, kpt_npz_path, kpt_ply_path


#     print(f"[WARN] Failed on {pkl_path}: {e}", file=sys.stderr)


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    r = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - r, x_middle + r])
    ax.set_ylim3d([y_middle - r, y_middle + r])
    ax.set_zlim3d([z_middle - r, z_middle + r])


def visualize_pointcloud_and_keypoints(pcd_path: Path, kpt_npz_path: Path):
    pts = np.asarray(trimesh.load(pcd_path, process=False).vertices)
    data = np.load(kpt_npz_path, allow_pickle=True)
    joints = data.get("joints", np.zeros((0, 3)))
    mids = data.get("midpoints", np.zeros((0, 3)))
    joint_names = (
        [str(x) for x in data.get("joint_names", [])] if "joint_names" in data else []
    )
    mid_names = (
        [str(x) for x in data.get("mid_names", [])] if "mid_names" in data else []
    )
    semseg_path = pcd_path.with_suffix(".semseg.npz")
    colors = None
    if semseg_path.exists():
        seg = np.load(semseg_path, allow_pickle=True)
        labels = seg["labels"].astype(int)
        cmap = plt.get_cmap("tab10")
        uniq = np.unique(labels)
        lut = {lab: np.array(cmap(i % 10)[:3]) for i, lab in enumerate(uniq)}
        colors = np.vstack([lut[int(l)] for l in labels])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, alpha=0.35, depthshade=False)
    else:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=0.6,
            alpha=0.6,
            depthshade=False,
            c=colors,
        )

    if joints.size:
        ax.scatter(
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
            s=18,
            marker="o",
            depthshade=False,
            label="joints",
        )
        for i in range(min(joints.shape[0], 8)):
            name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            ax.text(joints[i, 0], joints[i, 1], joints[i, 2], name, fontsize=6)

    if mids.size:
        ax.scatter(
            mids[:, 0],
            mids[:, 1],
            mids[:, 2],
            s=28,
            marker="^",
            edgecolors="k",
            linewidths=0.5,
            depthshade=False,
            label="midpoints",
        )
        for i in range(min(mids.shape[0], 4)):
            name = mid_names[i] if i < len(mid_names) else f"mid_{i}"
            ax.text(mids[i, 0], mids[i, 1], mids[i, 2], name, fontsize=6)

    _set_axes_equal(ax)
    ax.legend(loc="upper left")
    ax.set_title(pcd_path.name)
    plt.show()


# -----------------------------------------------------------------------------
# Walk & CLI
# -----------------------------------------------------------------------------


def walk_and_convert(
    root: Path,
    smplx_dir: Path,
    output_dir: Path,
    gender: str,
    n_points: int,
    with_normals: bool,
    write_kpts_ply: bool,
    inside: bool,
    interior_ratio: float,
) -> None:
    converted, failed, limit = 0, 0, 5000
    l = [p.strip() for p in open(root / "pkl_files.txt").readlines() if p.strip()]
    shuffle(l)
    preview_count = 0
    with tqdm(total=min(limit, len(l)), desc="Converted", unit="pcd") as pbar:
        for pkl_rel in l:
            pkl_path = root / pkl_rel
            pcd_path, kpt_npz_path, _ = convert_pkl_to_artifacts(
                pkl_path,
                smplx_dir,
                output_dir,
                gender,
                n_points,
                with_normals,
                inside,
                interior_ratio,
                write_kpts_ply,
            )
            if pcd_path is None or kpt_npz_path is None:
                failed += 1
                pbar.set_postfix(failed=failed)
                continue
            if preview_count < 3:
                try:
                    visualize_pointcloud_and_keypoints(pcd_path, kpt_npz_path)
                    preview_count += 1
                except Exception as e:
                    print(f"[WARN] preview failed for {pcd_path}: {e}", file=sys.stderr)
            converted += 1
            pbar.update(1)
            pbar.set_postfix(failed=failed)
            if converted >= limit:
                break
    print(f"\nDone. Converted: {converted}, Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SMPL-X PKLs to point clouds with robust 6-way semantic labels."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder. Must contain pkl_files.txt",
    )
    parser.add_argument(
        "--smplx-dir",
        type=Path,
        required=True,
        help="Directory containing SMPL-X model files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))

    parser.add_argument(
        "--gender", type=str, default="neutral", choices=["neutral", "male", "female"]
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--normals", action="store_true")
    parser.add_argument("--kpts-ply", action="store_true")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--interior-ratio", "-i", type=float, default=0.0)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(
            f"ERROR: root path {args.root} does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.smplx_dir.is_dir():
        print(
            f"ERROR: smplx-dir {args.smplx_dir} does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    walk_and_convert(
        args.root,
        args.smplx_dir,
        args.output_dir,
        args.gender,
        args.points,
        args.normals,
        args.kpts_ply,
        args.inside,
        args.interior_ratio,
    )
