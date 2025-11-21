#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tqdm


def knn_outlier_filter(points, k=16, std_ratio=2.5, min_points=50):
    """
    Remove obvious outliers from a point cloud using k-NN distance statistics.

    points: (N,3) ndarray
    k: how many neighbors to consider (including self)
    std_ratio: reject points whose mean neighbor distance
               > mean + std_ratio * std
    min_points: if N < min_points, skip filtering (return input)

    Returns:
        filtered_points: (M,3) ndarray
    """
    N = points.shape[0]
    if min_points > N or k >= N:
        return points

    # pairwise squared distances
    # (N,N) = ||p_i - p_j||^2
    diffs = points[:, None, :] - points[None, :, :]  # (N,N,3)
    dists2 = np.sum(diffs * diffs, axis=-1)  # (N,N)

    # take k nearest neighbors per point (including itself)
    # argsort dists2 along axis=1
    nn_idx = np.argpartition(dists2, kth=k - 1, axis=1)[:, :k]  # (N,k) unsorted subset
    # gather those distances
    nn_d2 = np.take_along_axis(dists2, nn_idx, axis=1)  # (N,k)
    nn_d = np.sqrt(nn_d2 + 1e-9)  # (N,k)

    # mean distance to k-NN for each point
    mean_d = nn_d.mean(axis=1)  # (N,)

    mu = mean_d.mean()
    sigma = mean_d.std() + 1e-9

    # keep points that are not "too far" from neighborhood
    keep_mask = mean_d <= (mu + std_ratio * sigma)

    filtered = points[keep_mask]
    # fallback: if we dropped basically everything, keep original
    if filtered.shape[0] < min_points:
        return points
    return filtered


def load_pair_folder(pair_dir: Path, do_filter=True):
    """
    Load all interp_XX.npy and kpsXX.npy for a pair.
    Returns (steps, clouds, kps) where:
      steps:  [s0, s1, ...] sorted ints
      clouds: list of (N,3) float arrays
      kps:    list of (K,3) float arrays
    Applies outlier filtering to each reconstruction point cloud if do_filter=True.
    """
    interp_files = sorted(pair_dir.glob("interp_*.npy"))
    step_pattern_interp = re.compile(r"interp_(\d+)\.npy")

    step_to_cloud = {}
    for f in interp_files:
        m = step_pattern_interp.match(f.name)
        if m:
            step_idx = int(m.group(1))
            arr = np.load(f)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # optional denoising
            if do_filter:
                arr = knn_outlier_filter(arr, k=16, std_ratio=2.5)
            step_to_cloud[step_idx] = arr

    kp_files = sorted(pair_dir.glob("kps*.npy"))
    step_pattern_kp = re.compile(r"kps(\d+)\.npy")

    step_to_kp = {}
    for f in kp_files:
        m = step_pattern_kp.match(f.name)
        if m:
            step_idx = int(m.group(1))
            arr = np.load(f)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # keypoints are few / stable, so we typically keep them as-is
            step_to_kp[step_idx] = arr

    steps_common = sorted(set(step_to_cloud.keys()) & set(step_to_kp.keys()))
    if not steps_common:
        return [], [], []

    clouds = [step_to_cloud[s] for s in steps_common]
    kps = [step_to_kp[s] for s in steps_common]
    return steps_common, clouds, kps


def compute_global_bounds(all_clouds, all_kps):
    mins = []
    maxs = []

    for c in all_clouds:
        if c is not None and len(c) > 0:
            mins.append(c.min(axis=0))
            maxs.append(c.max(axis=0))
    for k in all_kps:
        if k is not None and len(k) > 0:
            mins.append(k.min(axis=0))
            maxs.append(k.max(axis=0))

    if not mins:
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([+1.0, +1.0, +1.0])
        return lo, hi

    xyz_min = np.min(np.stack(mins, axis=0), axis=0)
    xyz_max = np.max(np.stack(maxs, axis=0), axis=0)

    center = 0.5 * (xyz_min + xyz_max)
    half_range = 0.5 * (xyz_max - xyz_min).max()
    lo = center - half_range
    hi = center + half_range
    return lo, hi


def build_plot(
    model,
    category,
    root_dir,
    cloud_size=2.5,
    cloud_color="red",
    kp_size=6,
    kp_color="blue",
    std_ratio=2.5,
    k_nn=16,
):
    root_dir = Path(root_dir)
    pair_dirs = sorted([d for d in root_dir.glob("pair_*") if d.is_dir()])
    if not pair_dirs:
        raise RuntimeError(f"No pair_* folders in {root_dir}")

    # Load + filter all data
    pairs_data = []
    all_clouds = []
    all_kps = []
    for p in tqdm.tqdm(pair_dirs, total=len(pair_dirs)):
        steps, clouds, kps = load_pair_folder(
            p,
            do_filter=True,  # outlier removal ON
        )
        if not steps:
            continue
        pairs_data.append(
            {
                "name": p.name,
                "steps": steps,  # [0..T-1]
                "clouds": clouds,  # list len T of (N,3) filtered
                "kps": kps,  # list len T of (K,3)
            }
        )
        all_clouds.extend(clouds)
        all_kps.extend(kps)

    if not pairs_data:
        raise RuntimeError("No valid interpolation data found after filtering.")

    lo, hi = compute_global_bounds(all_clouds, all_kps)

    fig = go.Figure()

    # Build traces: two per (pair, step)
    trace_idx_cloud = []  # trace_idx_cloud[pair_i][step_j]
    trace_idx_kps = []  # trace_idx_kps[pair_i][step_j]

    for pair_i, pdata in enumerate(pairs_data):
        steps = pdata["steps"]
        clouds_list = pdata["clouds"]
        kps_list = pdata["kps"]

        cloud_idx_list = []
        kps_idx_list = []

        for step_j, (xyz_cloud, xyz_kp) in enumerate(zip(clouds_list, kps_list)):
            cx, cy, cz = xyz_cloud[:, 0], xyz_cloud[:, 1], xyz_cloud[:, 2]
            kx, ky, kz = xyz_kp[:, 0], xyz_kp[:, 1], xyz_kp[:, 2]

            visible_init = pair_i == 0 and step_j == 0

            # reconstruction cloud trace
            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="markers",
                    marker={
                        "size": cloud_size,
                        "color": cloud_color,
                        "opacity": 1.0,
                    },
                    name=f"{pdata['name']} shape t={steps[step_j]}",
                    visible=visible_init,
                    hovertemplate="shape<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
            cloud_trace_id = len(fig.data) - 1

            # keypoint trace
            fig.add_trace(
                go.Scatter3d(
                    x=kx,
                    y=ky,
                    z=kz,
                    mode="markers",
                    marker={
                        "size": kp_size,
                        "color": kp_color,
                        "symbol": "diamond",
                        "opacity": 1.0,
                        "line": {"width": 1, "color": "black"},
                    },
                    name=f"{pdata['name']} kps t={steps[step_j]}",
                    visible=visible_init,
                    hovertemplate="kp<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
            kps_trace_id = len(fig.data) - 1

            cloud_idx_list.append(cloud_trace_id)
            kps_idx_list.append(kps_trace_id)

        trace_idx_cloud.append(cloud_idx_list)
        trace_idx_kps.append(kps_idx_list)

    # Build per-(pair,step) visibility masks
    num_traces_total = len(fig.data)
    visible_mask = []
    for pair_i, pdata in enumerate(pairs_data):
        pair_masks = []
        num_steps_i = len(pdata["steps"])
        for step_j in range(num_steps_i):
            mask = [False] * num_traces_total
            mask[trace_idx_cloud[pair_i][step_j]] = True
            mask[trace_idx_kps[pair_i][step_j]] = True
            pair_masks.append(mask)
        visible_mask.append(pair_masks)

    # Slider steps for first pair
    first_pair_steps = []
    for step_j, _ in enumerate(pairs_data[0]["steps"]):
        first_pair_steps.append(
            {
                "method": "update",
                "args": [
                    {"visible": visible_mask[0][step_j]},
                    {
                        "title": f"{model}/{category} :: {pairs_data[0]['name']} :: step {step_j}"
                    },
                ],
                "label": str(step_j),
            }
        )

    # Dropdown buttons, each rewires slider to that pair
    dropdown_buttons = []
    for pair_i, pdata in enumerate(pairs_data):
        pair_slider_steps = []
        for step_j, _ in enumerate(pdata["steps"]):
            pair_slider_steps.append(
                {
                    "method": "update",
                    "args": [
                        {"visible": visible_mask[pair_i][step_j]},
                        {
                            "title": f"{model}/{category} :: {pdata['name']} :: step {step_j}"
                        },
                    ],
                    "label": str(step_j),
                }
            )

        dropdown_buttons.append(
            {
                "label": pdata["name"],
                "method": "update",
                "args": [
                    {"visible": visible_mask[pair_i][0]},
                    {
                        "title": f"{model}/{category} :: {pdata['name']} :: step 0",
                        "sliders": [
                            {
                                "active": 0,
                                "currentvalue": {"prefix": "step: "},
                                "pad": {"t": 50},
                                "steps": pair_slider_steps,
                            }
                        ],
                    },
                ],
            }
        )

    # Layout
    fig.update_layout(
        title=f"{model}/{category} :: {pairs_data[0]['name']} :: step 0",
        scene={
            "xaxis": {
                "title": "x",
                "showticklabels": False,
                "range": [lo[0], hi[0]],
            },
            "yaxis": {
                "title": "y",
                "showticklabels": False,
                "range": [lo[1], hi[1]],
            },
            "zaxis": {
                "title": "z",
                "showticklabels": False,
                "range": [lo[2], hi[2]],
            },
            "aspectmode": "cube",
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "step: "},
                "pad": {"t": 50},
                "steps": first_pair_steps,
            }
        ],
    )

    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Interactive Plotly viewer for latent interpolations"
    )
    ap.add_argument("--model", required=True, help="e.g. Ours")
    ap.add_argument("--category", required=True, help="e.g. airplane")
    ap.add_argument(
        "--root",
        type=str,
        default="interps_out",
        help="Root: interps_out/<model>/<category>/pair_xxxxx/",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="interps_viewer.html",
        help="Output HTML file",
    )
    args = ap.parse_args()

    base_dir = Path(args.root) / args.model / args.category
    fig = build_plot(
        model=args.model,
        category=args.category,
        root_dir=base_dir,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


if __name__ == "__main__":
    main()
