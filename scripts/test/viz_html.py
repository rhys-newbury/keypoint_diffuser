#!/usr/bin/env python3
"""
Interactive Plotly visualization for reconstruction results (GT, recon, keypoints).

- Loads recon.npy, gt.npy, input.npy, pred_kp.npy, and GT keypoints (from annotations) for each sample
- Provides a dropdown menu to choose which sample to view
- Displays reconstruction (red), ground truth (gray), model input (green), keypoints (blue diamonds), and GT keypoints (orange x)
- Saves an interactive HTML file (self-contained)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_sample_dirs(root_pattern: Path):
    return [
        d
        for d in root_pattern.glob("*")
        if d.is_dir()
        and (d / "recon.npy").exists()
        and (d / "gt.npy").exists()
        and (d / "pred_kp.npy").exists()
        # and (d / "input.npy").exists()
    ]


def _load_gt_keypoints_normalized(root: Path, category_fallback: str, gt_cloud: np.ndarray) -> np.ndarray | None:
    """
    Load ground-truth keypoints from annotation JSON using the model_id in meta.json.
    Normalize to [-1, 1] using scalar pcmin/pcmax computed from the GT point cloud.
    Returns (10, 3) float32 if found, else None.
    """
    meta_file = root / "meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    model_id = meta.get("model_id")
    category = meta.get("category", category_fallback)
    if not (model_id and category):
        return None

    annotation_json = Path(f"/mnt/slow/shapenetcorev2-h5/annotations/{category}.json")
    if not annotation_json.exists():
        return None

    try:
        ann = json.loads(annotation_json.read_text())
    except Exception:
        return None

    kps_list = None
    for entry in ann:
        if entry.get("model_id") == model_id:
            kps_list = [np.asarray(kp["xyz"], dtype=np.float32) for kp in entry.get("keypoints", [])]
            break
    if not kps_list:
        return None

    gt_kp_raw = np.stack(kps_list, axis=0)  # (N,3)

    pcmin = float(gt_cloud.min())
    pcmax = float(gt_cloud.max())
    if pcmax == pcmin:
        return None
    gt_kp = 2.0 * ((gt_kp_raw - pcmin) / (pcmax - pcmin) - 0.5)
    gt_kp = gt_kp.reshape(-1, 3)
    if gt_kp.shape[0] >= 10:
        gt_kp = gt_kp[:10]
    else:
        pad = np.zeros((10 - gt_kp.shape[0], 3), dtype=gt_kp.dtype)
        gt_kp = np.concatenate([gt_kp, pad], axis=0)
    
    # return gt_kp.astype(np.float32)
    return gt_kp_raw.astype(np.float32)


def _pairwise_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (Na,3), b: (Nb,3) -> (Na,Nb)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def compute_dual_alignment_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Proxy for DAS on a single sample: fraction of mutual NN matches,
    averaged over forward (pred->gt) and backward (gt->pred).
    """
    if pred.size == 0 or gt.size == 0:
        return float("nan")
    D = _pairwise_dists(pred, gt)
    p2g = np.argmin(D, axis=1)          # each pred -> nearest gt index
    g2p = np.argmin(D, axis=0)          # each gt -> nearest pred index
    # mutual matches
    mutual_pred = (np.arange(len(pred)) == g2p[p2g])     # which preds are mutual
    mutual_gt = (np.arange(len(gt)) == p2g[g2p])         # which gts are mutual
    fwd = mutual_pred.mean() if len(mutual_pred) else 0.0
    bwd = mutual_gt.mean() if len(mutual_gt) else 0.0
    return float((fwd + bwd) * 0.5)


def compute_miou_at_threshold(pred: np.ndarray, gt: np.ndarray, thr: float = 0.1) -> float:
    """
    mIoU@thr for a single sample via bipartite matching:
      - Solve min-distance assignment (Hungarian) between pred and gt
      - Count matches with distance < thr as TP
      - IoU = TP / (|pred| + |gt| - TP)
    """
    if pred.size == 0 and gt.size == 0:
        return 1.0
    if pred.size == 0 or gt.size == 0:
        return 0.0

    # Hungarian matching
    D = _pairwise_dists(pred, gt)
    # Convert to square cost matrix by padding with large values
    n, m = D.shape
    size = max(n, m)
    pad = np.full((size, size), fill_value=D.max() + 1.0, dtype=D.dtype)
    pad[:n, :m] = D
    # simple Hungarian implementation via scipy if available; else greedy fallback
    try:
        from scipy.optimize import linear_sum_assignment  # optional
        ri, cj = linear_sum_assignment(pad)
        matched_d = pad[ri, cj]
        # keep only real indices
        mask = (ri < n) & (cj < m)
        matched_d = matched_d[mask]
    except Exception:
        # Greedy fallback: repeatedly take global min
        used_r = set()
        used_c = set()
        pairs = []
        flat = [(D[i, j], i, j) for i in range(n) for j in range(m)]
        for _, i, j in sorted(flat):
            if i not in used_r and j not in used_c:
                pairs.append((i, j))
                used_r.add(i)
                used_c.add(j)
        matched_d = np.array([D[i, j] for i, j in pairs], dtype=D.dtype)

    tp = int((matched_d < thr).sum())
    iou = tp / float(n + m - tp)
    return float(iou)


def plot_recon_collage_interactive(
    root_pattern: Path,
    out_path: Path,
    background_alpha=0.8,
    recon_color="red",
    gt_color="gray",
    input_color="green",
    kp_color="blue",
    gt_kp_color="orange",
):
    dirs = load_sample_dirs(root_pattern)
    if not dirs:
        print(f"[!] No samples with recon.npy + gt.npy + input.npy + pred_kp.npy under {root_pattern}")
        return

    dirs = sorted(dirs, key=lambda d: d.name)

    fig = go.Figure()
    buttons = []

    for idx, root in enumerate(dirs):
        recon = np.load(root / "recon.npy")
        gt = np.load(root / "gt.npy")
        try:
            inp = np.load(root / "input.npy")
        except Exception:
            inp = None
        kp = np.load(root / "pred_kp.npy")
        if 3 not in kp.shape:
            kp = kp.reshape(-1, 3)

        # Load normalized GT keypoints (if available)
        category_fallback = root_pattern.name
        gt_kp = _load_gt_keypoints_normalized(root, category_fallback, gt)

        # ---------------- hacky filter for the recon ----------------
        # dist_thresh = 3
        # mean_pt = np.mean(recon, 0)
        # dist_from_mean = np.linalg.norm(recon - mean_pt, axis=1)
        # in_mask = dist_from_mean < dist_thresh
        # while not np.all(in_mask):
        #     recon = recon[in_mask]
        #     mean_pt = np.mean(recon, 0)
        #     dist_from_mean = np.linalg.norm(recon - mean_pt, axis=1)
        #     in_mask = dist_from_mean < dist_thresh
        # -----------------------------------------------------------

        # Build title: prefer meta['das'] and meta['miou_at_0_1']; if missing, compute here
        title = root.name
        meta_file = root / "meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                meta = {}

        parts = [root.name]

        if "cd" in meta:
            parts.append(f"CD={meta['cd']:.4e}")
        if "emd" in meta:
            parts.append(f"EMD={meta['emd']:.4e}")

        # Try read DAS/mIoU from meta; if missing, compute on the fly (per-sample)
        # ** Since DAS should be calculated with pairs of point clouds it doesn't really make sense to compute it here individually **
        # das_val = meta.get("das", None)
        # miou_val = meta.get("miou_at_0_1", None)

        # if das_val is None and gt_kp is not None:
        #     try:
        #         das_val = compute_dual_alignment_score(kp, gt_kp)
        #     except Exception:
        #         das_val = None
        # if miou_val is None and gt_kp is not None:
        #     try:
        #         miou_val = compute_miou_at_threshold(kp, gt_kp, thr=0.1)
        #     except Exception:
        #         miou_val = None

        # if das_val is not None:
        #     parts.append(f"DAS~={float(das_val):.3f}")
        # if miou_val is not None:
        #     parts.append(f"mIoU@0.1~={float(miou_val):.3f}")

        title = "  |  ".join(parts)

        # 5 traces per sample: GT cloud, Recon cloud, Input cloud, Pred KPs, GT KPs
        if inp is not None:
            traces_per_sample = 5
        else:
            traces_per_sample = 4
        visible = [False] * (traces_per_sample * len(dirs))

        # Ground-Truth cloud
        fig.add_trace(
            go.Scatter3d(
                x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
                mode="markers",
                marker={"size": 2, "color": gt_color, "opacity": 1.0},
                name="Ground Truth",
                visible=(idx == 0),
            )
        )

        # Reconstruction
        fig.add_trace(
            go.Scatter3d(
                x=recon[:, 0], y=recon[:, 1], z=recon[:, 2],
                mode="markers",
                marker={"size": 2.5, "color": recon_color, "opacity": 1.0},
                name="Reconstruction",
                visible=(idx == 0),
            )
        )

        # Input point cloud
        if inp is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=inp[:, 0], y=inp[:, 1], z=inp[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": input_color, "opacity": 1.0},
                    name="Input",
                    visible=(idx == 0),
                )
            )

        # Predicted keypoints
        fig.add_trace(
            go.Scatter3d(
                x=kp[:, 0], y=kp[:, 1], z=kp[:, 2],
                mode="markers",
                marker={
                    "size": 6,
                    "color": kp_color,
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "line": {"width": 1, "color": "black"},
                },
                name="Keypoints",
                visible=(idx == 0),
            )
        )

        # Ground-truth keypoints (if found)
        # TODO REMEMBER TO REMOVE THIS AFTER FIXING GT KEYPOINTS NORMALIZATION
        gt_kp = None
        if gt_kp is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=gt_kp[:, 0], y=gt_kp[:, 1], z=gt_kp[:, 2],
                    mode="markers",
                    marker={
                        "size": 7,
                        "color": gt_kp_color,
                        "symbol": "x",
                        "opacity": 1.0,
                        "line": {"width": 2, "color": "black"},
                    },
                    name="GT Keypoints",
                    visible=(idx == 0),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode="markers",
                    marker={"size": 7, "color": gt_kp_color, "symbol": "x"},
                    name="GT Keypoints",
                    visible=False,
                )
            )

        # Update button visibility for this sample
        for i in range(traces_per_sample):
            visible[idx * traces_per_sample + i] = True

        buttons.append(
            {
                "label": title,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": title},
                ],
            }
        )

    fig.update_layout(
        title="Interactive Reconstruction Viewer",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"[✓] Saved interactive HTML to {out_path}")


def simple_plot(points_list: list[np.ndarray], name_list, out_html: Path, write=False):
    import plotly.graph_objects as go
    fig = go.Figure()
    
    traces = len(points_list)
    visible = [False] * (traces)
    
    for points, name in zip(points_list, name_list):
        shape = points.shape
        # assume it is object point cloud if a lot of points
        if np.max(shape) > 1000:
            # randomise color for each point cloud
            marker = {"size": 2, "opacity": 1.0}
        # else assume it is keypoints
        else:
            marker = {"size": 3, "opacity": 1.0, "color": np.random.randint(0, 255, size=3), "symbol": "diamond", "line": {"width": 2, "color": "black"}}
            
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker=marker,
                name=name,
            )
        )
    
    if write:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs=True, full_html=True)
        print(f"[✓] Saved interactive HTML to {out_html}")
    else:
        return fig  # return for customization of layout or other features
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot interactive reconstructions with keypoints.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", required=True, help="Category name (used as a fallback if meta.json lacks 'category')")
    ap.add_argument("--root-dir", type=Path, default=Path("recons_out"))
    args = ap.parse_args()

    folder = args.root_dir / args.model / args.category
    out_html = folder.parent / f"{args.category}_recons_interactive.html"

    plot_recon_collage_interactive(folder, out_html)
