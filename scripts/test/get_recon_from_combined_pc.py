#!/usr/bin/env python3
"""
Two-mode reconstruction from all_splits.csv (no DataLoader).

- Walk all_splits.csv and filter rows by --class-id (synset) and optional --split.
- For each (cid, mid):
    * Load partial A and partial B from the SAME model (random sample ids).
    * Load GT (full) point cloud from --pcd-path.
    * Normalise all to [-1, 1] using GT's min/max.
    * Compute min inter-set distance between A and B.
    * If threshold passes, pick/merge and downsample, then reconstruct.
    * Record CD/EMD; try to read coverage and plot CD/EMD vs coverage later.

Coverage is optional. The script tries:
  models/coverage_{mode}_{n}.npy (scalar)  or  models/coverage_{mode}_{n}.json {"coverage": float}
If not found, stores NaN (these are masked out in plots).
"""

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
import torch
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classes import MODEL_CLASSES
from keypoint_diffuser.utils.transforms import GridSample, ToTensor, Collect
from keypoint_diffuser.utils.pc_utils import collate_fn
from torchvision import transforms
from keypoint_diffuser.utils.eval_metrics import EMD_CD_recon

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:
    go = None
    pio = None


# ----------------------------
# CLI
# ----------------------------

def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        p.add_argument(*names, **kwargs)

def build_argparser():
    p = argparse.ArgumentParser(
        description="Two-mode selection + reconstruction over all_splits.csv (same model_id for A, B, GT)."
    )
    sub = p.add_subparsers(dest="model", required=True)
    for _, model_cls in MODEL_CLASSES.items():
        sp = sub.add_parser(model_cls.__name__, conflict_handler="resolve")
        model_cls.get_parser(sp)
        # IO / dataset
        try_add_arg(sp, "--all-splits-csv", type=Path, default=Path("data/shapenet_split/splits_out.csv"))
        try_add_arg(sp, "--class-id", type=str, default="02691156", help="ShapeNet synset id")
        try_add_arg(sp, "--split", type=str, default="test", help="Optional split to filter (e.g. test/train/val)")
        try_add_arg(sp, "--source-points-dir", type=Path, default=Path("/mnt/slow/shapenetcorev2-source"))
        try_add_arg(sp, "--pcd-path", type=Path, default=Path("/mnt/slow/shapenetcorev2-h5/pcds"))
        try_add_arg(sp, "--n-partial-samples", type=int, default=2)

        # modes / selection
        try_add_arg(sp, "--mode-a", type=str, default="tac", choices=["default", "myopia", "patch", "tac"])
        try_add_arg(sp, "--mode-b", type=str, default="tac",   choices=["default", "myopia", "patch", "tac"])
        try_add_arg(sp, "--min-dist-thresh", type=float, default=0.1)
        try_add_arg(sp, "--accept-if", type=str, default="gt", choices=["lt", "gt"])
        try_add_arg(sp, "--use-which", type=str, default="merge", choices=["a", "b", "smaller", "merge"])

        # downsampling
        try_add_arg(sp, "--downsample-method", type=str, default="random", choices=["voxel", "random"])
        try_add_arg(sp, "--voxel-size", type=float, default=0.01)
        try_add_arg(sp, "--num-points", type=int, default=2048)

        # viz
        try_add_arg(sp, "--viz-pairs", action="store_true")
        try_add_arg(sp, "--viz-dir", type=Path, default=Path("out_pair/viz"))
        try_add_arg(sp, "--viz-format", type=str, default="both",
                    choices=["mpl", "plotly", "both"],
                    help="How to save pair visualisations when --viz-pairs is set.")


        # coverage plotting
        try_add_arg(sp, "--plot-vs-coverage", action="store_true")
        try_add_arg(sp, "--plots-dir", type=Path, default=Path("out_pair"))

        # run / save
        try_add_arg(sp, "--ckpt", type=Path)
        try_add_arg(sp, "--output-dir", type=Path, default=Path("out_pair"))
        try_add_arg(sp, "--db-path", type=Path, default=Path("pair_recon_results.db"))
        try_add_arg(sp, "--batch-size", type=int, default=1)  # kept for model.get_parser compatibility
    return p

# ----------------------------
# File readers (aligned with get_das_new.py)  :contentReference[oaicite:1]{index=1}
# ----------------------------

def naive_read_pcd(path: Path) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith("DATA ascii"):
            idx = i + 1
            break
    pts = [row.strip().split() for row in lines[idx:]]
    arr = np.asarray(pts)
    return np.array(arr[:, :3], dtype=np.float32)

def read_full_npy(points_dir: Path, cid: str, mid: str) -> int:
    """
    Load {points_dir}/{cid}/{mid}/models/new_samples_{n}.npy
    with a random n in [0, n_partial_samples).
    Returns (pc[:, :3], n)
    """
    n = int(np.random.randint(0, 5))
    p = points_dir / cid / mid / "models" / f"new_samples_{n}.npy"
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected partial npy shape at {p}: {arr.shape}")
    
    return arr[:, :3].astype(np.float32)

def read_partial_npy(points_dir: Path, cid: str, mid: str, mode: str, n: int) -> np.ndarray:
    """
    Load {points_dir}/{cid}/{mid}/models/partial_samples_{mode}_{n}.npy
    with a random n in [0, n_partial_samples).
    Returns (pc[:, :3], n)
    """
    p = points_dir / cid / mid / "models" / f"partial_samples_{mode}_{n}.npy"
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected partial npy shape at {p}: {arr.shape}")
    return arr[:, :3].astype(np.float32)

def try_read_coverage(points_dir: Path, cid: str, mid: str, mode: str, n: int) -> float:
    """
    Best-effort coverage reader (optional).
    Tries:
      coverage_{mode}_{n}.npy  (scalar or 1-element array)
      coverage_{mode}_{n}.json {"coverage": float}
    Returns float('nan') if not found.
    """
    # get partial point cloud coverage fraction
    csv_path = points_dir / cid / mid / "models" / f"coverage_{mode}.csv"
    
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        coverage = None
        radius = None
        for row in reader:
            # hack for crappy initial tac csv files
            try:
                p, c, r = row
            except ValueError:
                p, c = row
                r = -1.0
                
            m = re.search(r"_(\d+)\.npy$", p)
            if m and int(m.group(1)) == n:
                coverage = float(c)
                radius = float(r)
                break
        # didn't find matching case by the end
        if coverage is None or radius is None:
            raise RuntimeError(f"failed to find matching coverage values for {partial_path} in {csv_path}")
    
    return coverage

# ----------------------------
# Geometry utils
# ----------------------------

def normalize_to_unit_range(pc: np.ndarray, dmin: np.ndarray, dmax: np.ndarray) -> np.ndarray:
    """Use GT's min/max to map to [-1, 1] like your DAS code path.  :contentReference[oaicite:2]{index=2}"""
    pc = (pc - dmin) / (dmax - dmin + 1e-12)
    pc = 2.0 * (pc - 0.5)
    return pc.astype(np.float32)

def normalize_to_box(inp, centroid=None, furthest_distance=None):
    """
    normalize point cloud to unit bounding box
    keep statefulness to apply the same normalization to a second point cloud
    center = (max - min)/2
    scale = max(abs(x))
    inp: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(inp.shape) == 2:
        axis = 0
        P = inp.shape[0]
        D = inp.shape[1]
    elif len(inp.shape) == 3:
        axis = 1
        P = inp.shape[1]
        D = inp.shape[2]
    else:
        raise ValueError()

    if centroid is None or furthest_distance is None:
        if isinstance(inp, np.ndarray):
            maxP = np.amax(inp, axis=axis, keepdims=True)
            minP = np.amin(inp, axis=axis, keepdims=True)
            centroid = (maxP + minP) / 2
            inp = inp - centroid
            furthest_distance = np.amax(np.abs(inp), axis=(axis, -1), keepdims=True)
            inp = inp / furthest_distance
        elif isinstance(inp, torch.Tensor):
            maxP = np.amax(inp, axis=axis, keepdims=True)
            minP = np.amin(inp, axis=axis, keepdims=True)
            centroid = (maxP + minP) / 2
            inp = inp - centroid
            in_shape = [*list(inp.shape[:axis]), P * D]
            furthest_distance = torch.max(
                torch.abs(inp).reshape(in_shape), dim=axis, keepdim=True
            )[0]
            furthest_distance = furthest_distance.unsqueeze(-1)
            inp = inp / furthest_distance
        else:
            raise ValueError()
    else:
        inp = inp - centroid
        inp = inp / furthest_distance

    return inp, centroid, furthest_distance

def min_interset_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_t = torch.from_numpy(a).float().unsqueeze(0)  # [1, Na, 3]
    b_t = torch.from_numpy(b).float().unsqueeze(0)  # [1, Nb, 3]
    with torch.no_grad():
        d = torch.cdist(a_t, b_t, p=2).min().item()
    return float(d)

def downsample(pc: np.ndarray, method: str, voxel_size: float, num_points: int) -> np.ndarray:
    if method == "voxel":
        if voxel_size <= 0:
            return pc
        vox = np.floor(pc / float(voxel_size))
        _, idx = np.unique(vox, axis=0, return_index=True)
        return pc[np.sort(idx)]
    if method == "random":
        if num_points <= 0 or pc.shape[0] <= num_points:
            return pc
        sel = np.random.choice(pc.shape[0], size=num_points, replace=False)
        return pc[sel]
    return pc

def pick_candidate(a: np.ndarray, b: np.ndarray, which: str) -> np.ndarray:
    if which == "a":
        return a
    if which == "b":
        return b
    if which == "smaller":
        return a if a.shape[0] <= b.shape[0] else b
    if which == "merge":
        return np.vstack([a, b])
    return a

def save_pair_viz(a: np.ndarray, b: np.ndarray, gt: np.ndarray, out_png: Path):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=1)
    ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=1)
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=1, alpha=0.5, color="gray")
    mins = np.min(np.vstack([a, b, gt]), axis=0)
    maxs = np.max(np.vstack([a, b, gt]), axis=0)
    rng = (maxs - mins).max()
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0]-rng/2, mid[0]+rng/2)
    ax.set_ylim(mid[1]-rng/2, mid[1]+rng/2)
    ax.set_zlim(mid[2]-rng/2, mid[2]+rng/2)
    ax.set_title("Mode A vs Mode B")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def save_pair_viz_plotly(a: np.ndarray, b: np.ndarray, gt, out_html: Path, title: str = "Mode A vs Mode B", recon=None, kp=None):
    """
    Save an interactive 3D scatter (two point clouds) to an HTML file using Plotly.
    - a, b: (N,3) float32 arrays
    - out_html: output .html path
    """
    if go is None or pio is None:
        # graceful fallback if plotly isn't installed
        out_png = out_html.with_suffix(".png")
        save_pair_viz(a, b, gt, out_png)
        return

    # Axis limits: equal cube
    if recon is None:
        mins = np.min(np.vstack([a, b, gt]), axis=0)
        maxs = np.max(np.vstack([a, b, gt]), axis=0)
    else:
        mins = np.min(np.vstack([a, b, gt, recon]), axis=0)
        maxs = np.max(np.vstack([a, b, gt, recon]), axis=0)
    rng = float((maxs - mins).max())
    mid = (maxs + mins) / 2.0
    x_range = [mid[0] - rng / 2, mid[0] + rng / 2]
    y_range = [mid[1] - rng / 2, mid[1] + rng / 2]
    z_range = [mid[2] - rng / 2, mid[2] + rng / 2]



    trace_a = go.Scatter3d(
        x=a[:, 0], y=a[:, 1], z=a[:, 2],
        mode="markers",
        name="Mode A",
        marker=dict(size=2, opacity=0.9)
    )
    trace_b = go.Scatter3d(
        x=b[:, 0], y=b[:, 1], z=b[:, 2],
        mode="markers",
        name="Mode B",
        marker=dict(size=2, opacity=0.9)
    )
    trace_gt = go.Scatter3d(
        x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
        mode="markers",
        name="Ground truth",
        marker=dict(size=2, opacity=0.5, color="gray")
    )
    if recon is not None:
        trace_recon = go.Scatter3d(
            x=recon[:, 0], y=recon[:, 1], z=recon[:, 2],
            mode="markers",
            name="Reconstruction",
            marker=dict(size=2, opacity=0.9, color="green")
        )
    if kp is not None:
        trace_kp = go.Scatter3d(
            x=kp[:, 0], y=kp[:, 1], z=kp[:, 2],
            mode="markers",
            name="Keypoints",
            marker=dict(size=5, opacity=1.0, color="blue", symbol="diamond")
        )

    dt = [trace_a, trace_b, trace_gt]
    if recon is not None:
        dt.append(trace_recon)
    if kp is not None:
        dt.append(trace_kp)

    fig = go.Figure(data=dt)
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=x_range, title="x"),
            yaxis=dict(range=y_range, title="y"),
            zaxis=dict(range=z_range, title="z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", auto_open=False)


# ----------------------------
# Save geoms (same spirit as your prior recon script)
# ----------------------------

def save_recon_geoms(recons, gts, inputs, names, cds, emds, opt, out_root: Path):
    out_dir = out_root / opt.model / opt.class_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for recon, gt, inpc, name, cd, emd in zip(recons, gts, inputs, names, cds, emds):
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "recon.npy", recon.astype(np.float32))
        np.save(dst / "gt.npy", gt.astype(np.float32))
        np.save(dst / "input.npy", inpc.astype(np.float32))
        with open(dst / "meta.json", "w") as f:
            json.dump({
                "model": opt.model,
                "class_id": opt.class_id,
                "model_id": name,
                "cd": float(cd),
                "emd": float(emd),
                "mode_a": opt.mode_a,
                "mode_b": opt.mode_b
            }, f, indent=2)

# ----------------------------
# Debug utils
# ----------------------------
def simple_plot(points_list: list[np.ndarray], out_html: Path):
    import plotly.graph_objects as go
    fig = go.Figure()
    
    traces = len(points_list)
    visible = [False] * (traces)
    
    for points in points_list:
        shape = points.shape
        if np.max(shape) > 1000:
            # randomise color for each point cloud
            marker = {"size": 2, "opacity": 1.0, "color": np.random.randint(0, 255, size=3)}
        else:
            marker = {"size": 6, "opacity": 1.0, "color": np.random.randint(0, 255, size=3), "symbol": "diamond", "line": {"width": 1, "color": "black"}}
            
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 1.0},
                name=f"Trace {len(fig.data)+1}",
            )
        )
    
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    # print(f"[✓] Saved interactive HTML to {out_html}")


# ----------------------------
# Main loop (no DataLoader)
# ----------------------------

def main():
    p = build_argparser()
    opt = p.parse_args()

    # model
    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)
    model.model.eval()
    model.model.cuda()

    # light transforms + collate exactly like DAS path  :contentReference[oaicite:3]{index=3}
    t = transforms.Compose([
        GridSample(keys=("coord",), hash_type="fnv", mode="train", return_grid_coord=True),
        ToTensor(),
        Collect(keys=("coord", "grid_coord"), feat_keys=("coord",)),
    ])

    # gather candidate rows
    rows = []
    with open(opt.all_splits_csv, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("synsetId") and r.get("modelId"):
                if r["synsetId"] != opt.class_id:
                    continue
                if opt.split and r.get("split") and r["split"] != opt.split:
                    continue
                rows.append((r["synsetId"], r["modelId"]))

    if not rows:
        print("[WARN] No rows matched --class-id and --split. Nothing to do.")
        return

    cds, emds, coverages = [], [], []
    recons, gts, inputs, names = [], [], [], []

    for cid, mid in tqdm.tqdm(rows, desc="Reconstruct (two modes; same model_id)"):
        # load GT (full) and min/max
        gt_path = opt.pcd_path / cid / f"{mid}.pcd"
        # gt = naive_read_pcd(gt_path)  # full cloud
        gt_full = read_full_npy(opt.source_points_dir, cid, mid)
        # dmin = gt_full.min(axis=0)
        # dmax = gt_full.max(axis=0)
        dmin = np.amax(gt_full, axis=0, keepdims=True)
        dmax = np.amin(gt_full, axis=0, keepdims=True)
        # unit_gt_n_full = normalize_to_unit_range(gt_full, dmin, dmax)
        gt_n_full, centroid, furthest_distance = normalize_to_box(gt_full)
        
        gt_n = downsample(gt_n_full, opt.downsample_method, opt.voxel_size, opt.num_points)

        # load partial A / B from same model
        nA = int(np.random.randint(0, opt.n_partial_samples))
        if opt.mode_a == opt.mode_b:
            # ensure different samples if same mode
            while True:
                nB = int(np.random.randint(0, opt.n_partial_samples))
                if nB != nA:
                    break
        else:
            nB = int(np.random.randint(0, opt.n_partial_samples))
            
        
        a = read_partial_npy(opt.source_points_dir, cid, mid, opt.mode_a, nA)  # :contentReference[oaicite:5]{index=5}
        b = read_partial_npy(opt.source_points_dir, cid, mid, opt.mode_b, nB)  # :contentReference[oaicite:6]{index=6}
        a_n, _, _ = normalize_to_box(a, centroid, furthest_distance)
        b_n, _, _ = normalize_to_box(b, centroid, furthest_distance)
        # simple_plot([gt_full, gt_n_full, a, a_n, b, b_n], Path("debug_gt_normalizations.html"))
        simple_plot([gt_n_full, a_n, b_n], Path("debug_gt_normalizations.html"))
        # input("paused")


        # threshold on min inter-set distance
        dmin_ab = min_interset_distance(a_n, b_n)
        # print(dmin_ab)
        passes = (dmin_ab < opt.min_dist_thresh) if (opt.accept_if == "lt") else (dmin_ab > opt.min_dist_thresh)
        chosen = pick_candidate(a_n, b_n, opt.use_which)# if passes else a_n
        chosen = downsample(chosen, opt.downsample_method, opt.voxel_size, opt.num_points)

        # choose coverage that corresponds to the chosen set
        if passes:
            # coverage (best effort)
            covA = try_read_coverage(opt.source_points_dir, cid, mid, opt.mode_a, nA)
            covB = try_read_coverage(opt.source_points_dir, cid, mid, opt.mode_b, nB)
            cov = covA + covB
            
            if opt.use_which == "a":
                coverages.append(covA)
            elif opt.use_which == "b":
                coverages.append(covB)
            elif opt.use_which == "smaller":
                coverages.append(covA if a.shape[0] <= b.shape[0] else covB)
            elif opt.use_which == "merge":
                # simple average when merged (adjust if you prefer a different rule)
                coverages.append(cov)
            else:
                coverages.append(covA)
        else:
            # skip if the pair does not pass
            continue

        # assemble a single-sample batch for reconstruction
        data = {"coord": chosen}
        Tb = [t(data)]
        batch = collate_fn(Tb)
        batch = {k: v.cuda() for k, v in batch.items()}
        batch["orig"] = gt_n[None, ...]              # provide GT to the model path if it uses it
        batch["target_partial_shape"] = torch.from_numpy(chosen)[None, ...].float().cuda()
        batch["target_shape"] = torch.from_numpy(gt_n)[None, ...].float().cuda()

        # forward + metrics
        with torch.no_grad():
            recon, input_pc, full_pc, _kp = model.get_reconstruction(batch, key="partial_orig")
            res = EMD_CD_recon(recon.squeeze(0), full_pc, reduced=False)
            cd = float(res["CD"].mean().detach().cpu().item())
            emd = float(res["EMD"].mean().detach().cpu().item())

        cds.append(cd); emds.append(emd)
        recons.append(recon.squeeze().detach().cpu().numpy())
        gts.append(gt_n)
        inputs.append(input_pc.squeeze().detach().cpu().numpy())
        names.append(mid)

        
        # optional viz
        if opt.viz_pairs:
            base = Path(opt.viz_dir) / f"{cid}_{mid}__{opt.mode_a}-{nA}_vs_{opt.mode_b}-{nB}"
            if opt.viz_format in ("mpl", "both"):
                save_pair_viz(a, b, gt_full, base.with_suffix(".png"))
            if opt.viz_format in ("plotly", "both"):
                # save_pair_viz_plotly(a, b, gt_full, base.with_suffix(".html"),
                #                      title=f"{opt.mode_a}-{nA} vs {opt.mode_b}-{nB}", recon=recon.squeeze().detach().cpu().numpy())
                save_pair_viz_plotly(a_n, b_n, gt_n_full, base.with_suffix(".html"),
                                     title=f"{opt.mode_a}-{nA} vs {opt.mode_b}-{nB}", 
                                     recon=recon.squeeze().detach().cpu().numpy(), 
                                     kp=_kp.reshape(-1, 3).detach().cpu().numpy())
            print(f"\n[INFO] Saved pair viz for {cid}/{mid}.")
            print(f"[INFO] Passed threshold: {passes}; with min distance {dmin_ab}.")
            print(f"[INFO] Combined covariance: {covA} + {covB} = {cov}.")
            if not passes:
                input(f"Press Enter to continue...")
        
        
        
        # for testing: ########
        # if len(cds) == 300:
            # break
        #######################

    # raw CSV
    plots_dir = Path(opt.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    with open(plots_dir / f"coverage_cd_emd_{opt.model}_{opt.class_id}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_id", "coverage", "cd", "emd"])
        for n, c, cd_, emd_ in zip(names, coverages, cds, emds):
            w.writerow([n, c, cd_, emd_])

    # save npys per model
    save_recon_geoms(recons, gts, inputs, names, cds, emds, opt, out_root=opt.output_dir)

    # plots vs coverage
    if opt.plot_vs_coverage and len(coverages):
        cov = np.asarray(coverages, float)
        cdv = np.asarray(cds, float)
        emdv = np.asarray(emds, float)
        m = ~np.isnan(cov)

        if np.any(m):
            plt.figure(figsize=(6,4))
            plt.scatter(cov[m], cdv[m], s=10)
            plt.xlabel("Coverage"); plt.ylabel("Chamfer Distance (CD)")
            plt.title(f"CD vs Coverage ({opt.class_id})"); plt.tight_layout()
            plt.savefig(plots_dir / f"cd_vs_coverage_{opt.model}_{opt.class_id}.png", dpi=200); plt.close()

            plt.figure(figsize=(6,4))
            plt.scatter(cov[m], emdv[m], s=10)
            plt.xlabel("Coverage"); plt.ylabel("Earth Mover's Distance (EMD)")
            plt.title(f"EMD vs Coverage ({opt.class_id})"); plt.tight_layout()
            plt.savefig(plots_dir / f"emd_vs_coverage_{opt.model}_{opt.class_id}.png", dpi=200); plt.close()

        

    # quick summary
    cd_mean = float(np.mean(cds)) if cds else math.nan
    emd_mean = float(np.mean(emds)) if emds else math.nan
    print(f"[✓] {len(names)} models processed | mean CD={cd_mean:.6f} | mean EMD={emd_mean:.6f} | mean coverage={np.mean(coverages)}")

if __name__ == "__main__":
    main()
