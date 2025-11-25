#!/usr/bin/env python3
"""
Compare two CSV roots (A vs B) with binned CD/EMD vs coverage and add
coverage distribution EDFs (PDF via normalized histogram) for TRAIN (via dataloader; split='train')
vs TEST (from CSVs).

Figure layout:
  --show-diff (default: off):
    1 x 3: [ CD mean±SEM ] [ EMD mean±SEM ] [ EDF: coverage (Train vs Test) ]
  --show-diff (on):
    2 x 3: [ CD mean±SEM ] [ EMD mean±SEM ] [ EDF ]
            [ CD diff A−B ] [ EMD diff A−B ] [ (empty) ]

Also writes:
  - train_coverages.csv  (all training coverage values; one value per row)

Usage (example):
  python plot_recon_v_cov_from_csvs.py \
    --csv-root-a recons_out/Ours/chair \
    --csv-root-b recons_out/Partial/chair \
    --modes default,myopia,patch,tac \
    --out compare_out_with_edf \
    --bins 20 --cov-min 0.0 --cov-max 1.0 \
    --cd-yscale log --emd-yscale log \
    --cd-min 1e-4 --cd-max 1e-1 --emd-min 1e-4 --emd-max 1e-1 \
    --min-count 5 \
    --label-a "Full PC Paradigm" --label-b "Partial PC Paradigm" \
    --category chair --train-batch-size 64 --train-num-workers 8 \
    Ours \
    --show-diff
"""

import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from classes import MODEL_CLASSES  # not used, kept for parity
from keypoint_diffuser.options.ae_options import AEConfig
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth, Collect, Deform, GridSample, ToTensor
)

ALL_INPUT_TYPES = ["default", "myopia", "patch", "tac"]


# -------------------- CSV helpers (TEST) --------------------
def kde1d(samples: np.ndarray, xs: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """
    Simple Gaussian KDE (no scipy). Returns density on xs.
    Silverman's rule used if bandwidth is None.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros_like(xs, dtype=float)

    if bandwidth is None:
        # Silverman's rule of thumb
        std = np.std(x, ddof=1) if n > 1 else 0.0
        iqr = np.subtract(*np.percentile(x, [75, 25])) if n > 1 else 0.0
        sigma = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else (std if std > 0 else (1e-6))
        bandwidth = 0.9 * sigma * n ** (-1 / 5)

    h = float(bandwidth) if bandwidth > 0 else 1e-6
    # Gaussian kernel: (1/(n*h*sqrt(2π))) * Σ exp(-(x - xi)^2 / (2h^2))
    diffs = (xs[:, None] - x[None, :]) / h
    norm = (n * h * np.sqrt(2 * np.pi))
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / norm
    return dens

def edf_from_samples(samples: np.ndarray, edges: np.ndarray):
    """(kept for backward-compat step-EDF) normalized histogram over given edges."""
    hist, _ = np.histogram(samples, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def plot_edf(ax, train_cov, test_cov, edges, label_train="Train coverage",
             label_test="Test coverage", smooth=False, bandwidth=None,
             x_min=None, x_max=None, grid_n=400):
    """
    If smooth=False: step histogram EDF (density).
    If smooth=True : Gaussian KDE EDF on a regular grid [x_min, x_max].
    """
    if not smooth:
        c_train, d_train = edf_from_samples(train_cov, edges)
        c_test,  d_test  = edf_from_samples(test_cov,  edges)
        ax.step(c_train, d_train, where="mid", label=label_train)
        ax.step(c_test,  d_test,  where="mid", label=label_test)
    else:
        # grid over requested coverage range
        if x_min is None: x_min = float(np.nanmin([np.min(train_cov), np.min(test_cov)]))
        if x_max is None: x_max = float(np.nanmax([np.max(train_cov), np.max(test_cov)]))
        xs = np.linspace(x_min, x_max, int(grid_n))
        d_train = kde1d(train_cov, xs, bandwidth)
        d_test  = kde1d(test_cov,  xs, bandwidth)
        ax.plot(xs, d_train, label=label_train)
        ax.plot(xs, d_test,  label=label_test)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_title("Coverage distribution (EDF)")
    ax.legend()


def read_combined_csv(path: Path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "coverage": float(row["coverage"]),
                "cd": float(row["cd"]),
                "emd": float(row["emd"]),
            })
    return rows

def read_per_mode_csv(root: Path, modes):
    rows = []
    for m in modes:
        p = root / f"{m}_cov" / "metrics_vs_coverage.csv"
        if not p.exists():
            print(f"[WARN] Missing CSV for mode '{m}' under {root}. Skipping.")
            continue
        with open(p, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append({
                    "coverage": float(row["coverage"]),
                    "cd": float(row["cd"]),
                    "emd": float(row["emd"]),
                })
    return rows

def load_rows_from_root(root: Path, modes):
    combined = root / "ALL_cov" / "metrics_vs_coverage_all.csv"
    if combined.exists():
        print(f"[INFO] Using combined CSV: {combined}")
        return read_combined_csv(combined)
    print(f"[INFO] Reading per-mode CSVs under: {root}")
    return read_per_mode_csv(root, modes)

# -------------------- Dataloader (TRAIN coverage) --------------------
def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        p.add_argument(*names, **kwargs)

def _config_from_namespace_for_split(ns: argparse.Namespace, partial_view_mode: str, split: str) -> AEConfig:
    from dataclasses import fields, MISSING
    cfg_kwargs = {}
    for f in fields(AEConfig):
        name = f.name
        val = getattr(ns, name, MISSING)
        if val is MISSING:
            if f.default is not MISSING:
                val = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:
                val = f.default_factory()  # type: ignore[misc]
            else:
                continue
        if f.type is Path and isinstance(val, str):
            val = Path(val)
        cfg_kwargs[name] = val

    if cfg_kwargs.get("normalization") == "none":
        cfg_kwargs["normalization"] = None

    cfg_kwargs.update(
        dict(
            name="recon",
            split=split,
            mesh_dir="/mnt/slow/shapenetcorev2-source/",
            points_dir="/mnt/slow/shapenetcorev2-source/",
            split_file="./data/shapenet_split/splits_out.csv",
            n_partial_samples=2,
            category=ns.category,
            partial_view_mode=partial_view_mode,
        )
    )
    return AEConfig(**cfg_kwargs)

def make_loader_for_split(opt: argparse.Namespace, mode: str, split: str) -> DataLoader:
    t = (
        transforms.Compose([
            Deform(),
            ApplyToBoth(
                transforms.Compose([
                    GridSample(keys=("coord",), hash_type="fnv", mode="train", return_grid_coord=True),
                    ToTensor(),
                    Collect(keys=("coord", "grid_coord", "transformation", "shape"), feat_keys=("coord",)),
                ])
            ),
        ])
        if (opt.model == "Ours" or opt.model == "Partial")
        else None
    )
    cfg = _config_from_namespace_for_split(opt, mode, split=split)
    dataset = ShapesPartial(cfg, transform=t)
    return DataLoader(
        dataset,
        batch_size=opt.train_batch_size if split == "train" else opt.batch_size,
        shuffle=False,
        num_workers=opt.train_num_workers if split == "train" else opt.num_workers,
        collate_fn=collate_fn if (opt.model == "Ours" or opt.model == "Partial") else None,
        drop_last=False,
    )

def collect_train_coverages(opt: argparse.Namespace, modes):
    coverages = []
    for mode in modes:
        loader = make_loader_for_split(opt, partial_view_mode_for_train(mode), split="train")
        with torch.no_grad():
            for batch in loader:
                cov = batch.get("target_coverage", None)
                if cov is None:
                    raise KeyError("Batch is missing 'target_coverage'")
                if torch.is_tensor(cov):
                    cov = cov.detach().cpu().numpy()
                coverages.extend(np.asarray(cov).reshape(-1).tolist())
    return np.asarray(coverages, dtype=float)

def partial_view_mode_for_train(mode: str) -> str:
    return "default" if mode == "full" else mode

# -------------------- Binning + EDF --------------------
def bin_stats(x, y, edges, min_count=5):
    x = np.asarray(x); y = np.asarray(y)
    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(edges.size - 1, np.nan, dtype=float)
    sem = np.full(edges.size - 1, np.nan, dtype=float)
    count = np.zeros(edges.size - 1, dtype=int)
    for i in range(edges.size - 1):
        sel = (idx == i)
        n = int(np.sum(sel))
        count[i] = n
        if n >= min_count:
            vals = y[sel]
            m = np.mean(vals)
            s = np.std(vals, ddof=1) if n > 1 else 0.0
            mean[i] = m
            sem[i] = s / np.sqrt(n) if n > 1 else 0.0
    return centers, mean, sem, count

def binned_compare(cov_A, y_A, cov_B, y_B, edges, min_count=5):
    cA, mA, sA, nA = bin_stats(cov_A, y_A, edges, min_count=min_count)
    cB, mB, sB, nB = bin_stats(cov_B, y_B, edges, min_count=min_count)
    assert np.allclose(cA, cB)
    centers = cA
    valid = (~np.isnan(mA)) & (~np.isnan(mB))
    diff = np.full_like(mA, np.nan)
    sem_diff = np.full_like(sA, np.nan)
    diff[valid] = mA[valid] - mB[valid]
    sem_diff[valid] = np.sqrt(np.square(sA[valid]) + np.square(sB[valid]))
    return {
        "centers": centers,
        "A_mean": mA, "A_sem": sA, "A_count": nA,
        "B_mean": mB, "B_sem": sB, "B_count": nB,
        "diff": diff, "diff_sem": sem_diff,
        "valid": valid
    }

def edf_from_samples(samples: np.ndarray, edges: np.ndarray):
    """EDF (density): normalized histogram over given edges."""
    hist, _ = np.histogram(samples, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

# -------------------- Plotting --------------------
def plot_binned(ax, centers, meanA, semA, meanB, semB, labelA="A", labelB="B",
                colorA="#1f77b4", colorB="#d62728", markerA="o", markerB="s"):
    ax.plot(centers, meanA, label=labelA, color=colorA, marker=markerA, linewidth=1.8, markersize=4)
    ax.fill_between(centers, meanA - semA, meanA + semA, alpha=0.20, color=colorA)
    ax.plot(centers, meanB, label=labelB, color=colorB, marker=markerB, linewidth=1.8, markersize=4)
    ax.fill_between(centers, meanB - semB, meanB + semB, alpha=0.20, color=colorB)

def plot_diff(ax, centers, diff, sem_diff, color="#444"):
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.plot(centers, diff, color=color, linewidth=1.8)
    ax.fill_between(centers, diff - sem_diff, diff + sem_diff, alpha=0.20, color=color)

# def plot_edf(ax, train_cov, test_cov, edges, label_train="Train coverage", label_test="Test coverage"):
#     c_train, d_train = edf_from_samples(train_cov, edges)
#     c_test, d_test = edf_from_samples(test_cov, edges)
#     ax.step(c_train, d_train, where="mid", label=label_train)
#     ax.step(c_test, d_test, where="mid", label=label_test)
#     ax.set_xlabel("Coverage")
#     ax.set_ylabel("Density")
#     ax.set_title("Coverage distribution (EDF)")
#     ax.legend()

# -------------------- CLI / main --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Binned CD/EMD vs coverage (A vs B) plus EDFs of Train vs Test coverage.")
    # CSV roots (TEST)
    p.add_argument("--csv-root-a", type=Path, required=True)
    p.add_argument("--csv-root-b", type=Path, required=True)
    p.add_argument("--modes", type=str, default=",".join(ALL_INPUT_TYPES),
                   help="Comma-separated modes if per-mode CSVs are used (ignored if combined CSV exists).")
    # Output
    p.add_argument("--out", type=Path, default=Path("compare_out_binned"))
    # Binning + scales
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--cov-min", type=float, default=None)
    p.add_argument("--cov-max", type=float, default=None)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--cd-yscale", type=str, default="log", choices=["linear", "log", "symlog", "logit"])
    p.add_argument("--emd-yscale", type=str, default="linear", choices=["linear", "log", "symlog", "logit"])
    p.add_argument("--cd-min", type=float, default=None)
    p.add_argument("--cd-max", type=float, default=None)
    p.add_argument("--emd-min", type=float, default=None)
    p.add_argument("--emd-max", type=float, default=None)
    # Labels
    p.add_argument("--label-a", type=str, default="Full PC Paradigm")
    p.add_argument("--label-b", type=str, default="Partial PC Paradigm")
    # Dataloader args to fetch TRAIN coverage
    p.add_argument("model", type=str, help="Model name for dataset pipeline (e.g., Ours, Partial)")
    p.add_argument("--category", type=str, default="airplane")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--train-num-workers", type=int, default=8)
    # Optional difference plots
    p.add_argument("--show-diff", action="store_true", help="Add (A−B) difference plots in a second row.")
    
    p.add_argument("--edf-smooth", action="store_true",
               help="Plot coverage EDF as a smooth KDE instead of a step histogram.")
    p.add_argument("--edf-bandwidth", type=float, default=None,
               help="KDE bandwidth for --edf-smooth (default: Silverman's rule).")
    p.add_argument("--edf-grid", type=int, default=400,
               help="Number of x points for the smooth EDF curve.")

    
    return p.parse_args()

def _nan(x):
    try:
        return float(x) if np.isfinite(x) else ""
    except Exception:
        return ""

def main():
    opt = parse_args()
    opt.out.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in (ALL_INPUT_TYPES if opt.modes == "all" else opt.modes.split(",")) if m.strip()]

    # Load TEST data (CSV)
    rowsA = load_rows_from_root(opt.csv_root_a, modes)
    rowsB = load_rows_from_root(opt.csv_root_b, modes)

    covA = np.array([r["coverage"] for r in rowsA], dtype=float)
    cdA  = np.array([r["cd"]       for r in rowsA], dtype=float)
    emdA = np.array([r["emd"]      for r in rowsA], dtype=float)
    covB = np.array([r["coverage"] for r in rowsB], dtype=float)
    cdB  = np.array([r["cd"]       for r in rowsB], dtype=float)
    emdB = np.array([r["emd"]      for r in rowsB], dtype=float)

    # TRAIN coverage (aggregate across modes; ignore 'full' for train)
    train_modes = [m for m in modes if m != "full"]
    if not train_modes:
        train_modes = ALL_INPUT_TYPES
    print(f"[INFO] Collecting TRAIN coverage over modes: {train_modes}")
    train_cov = collect_train_coverages(opt, train_modes)

    # Save training coverages for reproducibility
    train_cov_csv = opt.out / "train_coverages.csv"
    with open(train_cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage"])
        for v in train_cov:
            w.writerow([float(v)])
    print(f"[✓] Saved training coverages: {train_cov_csv}")

    # TEST coverage to compare in EDF (use A's CSV; typically same across A/B)
    test_cov = covA.copy()

    # Coverage bin edges
    cov_min = opt.cov_min if opt.cov_min is not None else float(np.nanmin([covA.min(), covB.min(), train_cov.min()]))
    cov_max = opt.cov_max if opt.cov_max is not None else float(np.nanmax([covA.max(), covB.max(), train_cov.max()]))
    edges = np.linspace(cov_min, cov_max, opt.bins + 1)

    # Binned A vs B
    cd_stats  = binned_compare(covA, cdA,  covB, cdB,  edges, min_count=opt.min_count)
    emd_stats = binned_compare(covA, emdA, covB, emdB, edges, min_count=opt.min_count)

    # Figure
    if opt.show_diff:
        fig = plt.figure(figsize=(16, 8))
        ax_cd    = fig.add_subplot(2, 3, 1)
        ax_emd   = fig.add_subplot(2, 3, 2)
        ax_edf   = fig.add_subplot(2, 3, 3)
        ax_cd_d  = fig.add_subplot(2, 3, 4)
        ax_emd_d = fig.add_subplot(2, 3, 5)
        ax_empty = fig.add_subplot(2, 3, 6); ax_empty.axis("off")
    else:
        fig = plt.figure(figsize=(16, 4.5))
        ax_cd    = fig.add_subplot(1, 3, 1)
        ax_emd   = fig.add_subplot(1, 3, 2)
        ax_edf   = fig.add_subplot(1, 3, 3)

    # Colors/markers
    colorA, colorB = "#1f77b4", "#d62728"
    markerA, markerB = "o", "s"

    # Top row: mean ± SEM
    plot_binned(ax_cd,  cd_stats["centers"],  cd_stats["A_mean"],  cd_stats["A_sem"],
                               cd_stats["B_mean"],  cd_stats["B_sem"],
                               labelA=opt.label_a, labelB=opt.label_b,
                               colorA=colorA, colorB=colorB, markerA=markerA, markerB=markerB)
    plot_binned(ax_emd, emd_stats["centers"], emd_stats["A_mean"], emd_stats["A_sem"],
                               emd_stats["B_mean"], emd_stats["B_sem"],
                               labelA=opt.label_a, labelB=opt.label_b,
                               colorA=colorA, colorB=colorB, markerA=markerA, markerB=markerB)

    ax_cd.set_xlabel("Coverage");  ax_cd.set_ylabel("Chamfer Distance (CD)"); ax_cd.set_title("CD vs Coverage (mean ± SEM)")
    ax_emd.set_xlabel("Coverage"); ax_emd.set_ylabel("EMD");       ax_emd.set_title("EMD vs Coverage (mean ± SEM)")
    ax_cd.legend(); ax_emd.legend()

    # Y scales and limits
    ax_cd.set_yscale(opt.cd_yscale)
    if opt.cd_min is not None or opt.cd_max is not None:
        ax_cd.set_ylim(opt.cd_min, opt.cd_max)
    ax_emd.set_yscale(opt.emd_yscale)
    if opt.emd_min is not None or opt.emd_max is not None:
        ax_emd.set_ylim(opt.emd_min, opt.emd_max)

    # EDFs: Train vs Test coverage
    # plot_edf(ax_edf, train_cov, test_cov, edges, label_train="Train coverage", label_test="Test coverage")
    plot_edf(
        ax_edf,
        train_cov,
        test_cov,
        edges,
        label_train="Train coverage",
        label_test="Test coverage",
        smooth=opt.edf_smooth,
        bandwidth=opt.edf_bandwidth,
        x_min=float(cov_min),
        x_max=float(cov_max),
        grid_n=opt.edf_grid,
    )



    # Optional bottom row: differences (A - B) ± SEM
    if opt.show_diff:
        plot_diff(ax_cd_d,  cd_stats["centers"],  cd_stats["diff"],  cd_stats["diff_sem"], color="#444")
        plot_diff(ax_emd_d, emd_stats["centers"], emd_stats["diff"], emd_stats["diff_sem"], color="#444")
        ax_cd_d.set_xlabel("Coverage");  ax_cd_d.set_ylabel(f"CD: {opt.label_a} − {opt.label_b}")
        ax_cd_d.set_title("CD difference (A − B) ± SEM")
        ax_emd_d.set_xlabel("Coverage"); ax_emd_d.set_ylabel(f"EMD: {opt.label_a} − {opt.label_b}")
        ax_emd_d.set_title("EMD difference (A − B) ± SEM")

    plt.tight_layout()
    fig_name = "binned_cd_emd_vs_coverage_ABonly_with" + ("_diffs_and_edf.png" if opt.show_diff else "_edf.png")
    fig_path = opt.out / fig_name
    plt.savefig(fig_path, dpi=220)
    print(f"[✓] Saved figure: {fig_path}")

    # Save binned numbers
    csv_out = opt.out / ("binned_stats_ABonly" + ("_with_diffs.csv" if opt.show_diff else ".csv"))
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bin_center",
            "cd_mean_A","cd_sem_A","cd_count_A",
            "cd_mean_B","cd_sem_B","cd_count_B",
            "cd_diff","cd_diff_sem",
            "emd_mean_A","emd_sem_A","emd_count_A",
            "emd_mean_B","emd_sem_B","emd_count_B",
            "emd_diff","emd_diff_sem"
        ])
        for i in range(cd_stats["centers"].size):
            w.writerow([
                float(cd_stats["centers"][i]),
                _nan(cd_stats["A_mean"][i]), _nan(cd_stats["A_sem"][i]), int(cd_stats["A_count"][i]),
                _nan(cd_stats["B_mean"][i]), _nan(cd_stats["B_sem"][i]), int(cd_stats["B_count"][i]),
                _nan(cd_stats["diff"][i]),   _nan(cd_stats["diff_sem"][i]),
                _nan(emd_stats["A_mean"][i]), _nan(emd_stats["A_sem"][i]), int(emd_stats["A_count"][i]),
                _nan(emd_stats["B_mean"][i]), _nan(emd_stats["B_sem"][i]), int(emd_stats["B_count"][i]),
                _nan(emd_stats["diff"][i]),   _nan(emd_stats["diff_sem"][i]),
            ])
    print(f"[✓] Saved binned stats: {csv_out}")

if __name__ == "__main__":
    main()
