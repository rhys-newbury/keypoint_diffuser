#!/usr/bin/env python3

import numpy as np
import open3d as o3d


def to_o3d_pcd(pts):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return p


def icp_align_identity(pc, orig_pc, max_iters=50):
    """
    Align `pc` to `orig_pc` using ICP with identity as the initial guess.
    Returns (T, R, t, pc_aligned)
    """
    src = to_o3d_pcd(pc)
    tgt = to_o3d_pcd(orig_pc)

    # Optional downsampling
    diag = np.linalg.norm(orig_pc.max(0) - orig_pc.min(0))
    voxel = max(diag * 0.01, 1e-3)
    src_ds = src.voxel_down_sample(voxel)
    tgt_ds = tgt.voxel_down_sample(voxel)

    # Normals for point-to-plane ICP
    radius = voxel * 2.5
    src_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    tgt_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    # ICP thresholds
    thresh = diag * 0.02

    # Run ICP from identity
    reg = o3d.pipelines.registration.registration_icp(
        src_ds,
        tgt_ds,
        thresh,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters),
    )

    T = reg.transformation
    R = T[:3, :3]
    t = T[:3, 3]
    pc_aligned = (pc @ R.T) + t

    return T, R, t, pc_aligned
