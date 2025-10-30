#!/usr/bin/env python3
"""
plot_coverage_distribution.py

Scan coverage CSVs and visualize distributions with flexible grouping.
Now supports:
  - Grouping by MODE and by OBJECT CLASS (taxonomy via synset_utils.py)
  - Plot styles:
      * 'hist'          : single histogram for ALL filtered samples
      * 'kde-modes'     : KDE overlay (one curve per mode)
      * 'kde-classes'   : KDE overlay (one curve per class)

CSV layout per instance:
  <root>/<synsetId>/<instance>/models/coverage_<mode>.csv
Columns required:
  partial_pc_path, coverage [, radius]

Examples
--------
# All chairs, all discovered modes, single histogram
python plot_coverage_distribution.py \
  --root /mnt/slow/shapenetcorev2-source \
  --taxonomy filtered_taxonomy.json \
  --classes chair \
  --mode all \
  --style hist

# Compare modes (KDE overlay) for mugs + chairs, keep partial index n<5
python plot_coverage_distribution.py \
  --root /data/shapenet \
  --taxonomy filtered_taxonomy.json \
  --classes "mug, chair" \
  --mode "default,tac" \
  --n 5 \
  --style kde-modes

# Compare classes (KDE overlay) for a single mode
python plot_coverage_distribution.py \
  --root /data/shapenet \
  --taxonomy filtered_taxonomy.json \
  --classes "chair, car, mug" \
  --mode default \
  --style kde-classes
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Shared taxonomy helpers
from synset_utils import (
    names_to_synsets,
    synsets_to_names,
)


# ---------------------- helpers ----------------------

def parse_classes_arg(classes_arg: str) -> List[str]:
    """Accept comma and/or whitespace separated list of class names."""
    if not classes_arg:
        return []
    parts = [p for chunk in classes_arg.split(",") for p in chunk.split()]
    return [p for p in parts if p]


def discover_modes(root: Path) -> List[str]:
    """Find unique mode names by globbing for coverage_*.csv under root."""
    modes = set()
    for p in root.rglob("coverage_*.csv"):
        m = re.match(r"coverage_(.+)\.csv$", p.name)
        if m:
            modes.add(m.group(1))
    return sorted(modes)


def rows_with_n_less_than(df: pd.DataFrame, n_max: int) -> pd.DataFrame:
    """
    Keep only rows whose partial_pc_path ends with _<n>.npy where n < n_max.
    Rows not matching the pattern are dropped.
    """
    n_values = df["partial_pc_path"].str.extract(r"_(\d+)\.npy$", expand=False)
    valid_mask = n_values.notna()
    df = df.loc[valid_mask].copy()
    n_values = n_values.loc[valid_mask].astype(int)
    return df.loc[n_values < n_max]


def _synset_from_path(csv_path: Path) -> Optional[str]:
    """Infer 8-digit ShapeNet synset ID from a path's components."""
    for comp in csv_path.parts:
        if re.fullmatch(r"\d{8}", comp):
            return comp
    return None


def collect_grouped_coverages(
    root: Path,
    modes: List[str],
    allowed_synsets: Optional[List[str]],
    n_max: int,
    taxonomy_path: Path,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]], Dict[Tuple[str, str], List[float]]]:
    """
    Collect coverage values grouped by mode and by class.

    Returns:
        all_values: list of all coverage floats after filters
        by_mode:    dict[mode] -> list[coverage]
        by_class:   dict[class_name] -> list[coverage]
        by_mode_cls:dict[(mode, class_name)] -> list[coverage]
    """
    all_values: List[float] = []
    by_mode: Dict[str, List[float]] = {m: [] for m in modes}
    by_class: Dict[str, List[float]] = {}
    by_mode_cls: Dict[Tuple[str, str], List[float]] = {}

    # Cache synset->name to avoid repeated file reads
    synset_name_cache: Dict[str, str] = {}

    def _class_name(sid: str) -> str:
        if sid not in synset_name_cache:
            try:
                synset_name_cache[sid] = synsets_to_names([sid], str(taxonomy_path))[0]
            except Exception:
                synset_name_cache[sid] = sid  # fallback to id
        return synset_name_cache[sid]

    for mode in modes:
        for csv_path in root.rglob(f"coverage_{mode}.csv"):
            try:
                sid = _synset_from_path(csv_path)
                if allowed_synsets and sid and sid not in allowed_synsets:
                    continue

                df = pd.read_csv(csv_path)
                if not {"coverage", "partial_pc_path"}.issubset(df.columns):
                    continue

                if n_max is not None:
                    df = rows_with_n_less_than(df, n_max)

                vals = df["coverage"].dropna().astype(float).tolist()
                if not vals:
                    continue

                all_values.extend(vals)
                by_mode.setdefault(mode, []).extend(vals)

                # Class bucket
                cls_name = _class_name(sid) if sid else "unknown"
                by_class.setdefault(cls_name, []).extend(vals)
                by_mode_cls.setdefault((mode, cls_name), []).extend(vals)

            except Exception as e:
                print(f"[WARN] Skipping {csv_path} due to error: {e}")

    return all_values, by_mode, by_class, by_mode_cls


# ---------------------- plotting ----------------------

def plot_hist_all(values: List[float], title: str, out_path: Path, bins: int = 40, logy: bool = False) -> None:
    """Single histogram for all values."""
    if not values:
        print(f"[INFO] No values to plot for {title}. Skipping.")
        return
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    a = a[(a >= 0.0) & (a <= 1.0)]
    if a.size == 0:
        print(f"[INFO] No valid values in [0,1] for {title}.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=bins, range=(0, 1), edgecolor="black", alpha=0.8)
    plt.xlim(0, 1)
    plt.xlabel("Coverage"); plt.ylabel("Frequency")
    plt.title(f"{title}\nMean {a.mean():.4f} ± Std {a.std():.4f} (n={a.size})")
    plt.grid(alpha=0.3)
    if logy:
        plt.yscale("log")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_kde_overlay(
    groups: Dict[str, List[float]],
    title: str,
    out_path: Path,
    logy: bool = False,
    grid_pts: int = 400,
    min_n: int = 2,
    bw_adjust: float = 1.2,  # increase for smoother, decrease for sharper
) -> None:
    """
    Overlay KDE curves using seaborn, with the area under each curve filled.

    Parameters
    ----------
    groups    : mapping label -> list of coverage values
    title     : plot title
    out_path  : where to save the figure (PNG)
    logy      : log-scale y-axis if True
    grid_pts  : KDE evaluation resolution (seaborn 'gridsize')
    min_n     : minimum group size to plot
    bw_adjust : seaborn bandwidth adjustment factor (1.0=default)
    """

    plt.figure(figsize=(9, 5))
    any_plotted = False

    # set colour palette for seaborn
    n_curves = len(groups.keys())+1     # add one to prevent looping around to use the same colour
    palette = sns.husl_palette(n_colors=n_curves, s=0.6, l=0.85)  # tweak s,l to taste
    sns.set_palette(palette)

    for label, vals in groups.items():
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        a = a[(a >= 0.0) & (a <= 1.0)]  # coverage in [0,1]
        if a.size < min_n:
            continue

        sns.kdeplot(
            a,
            fill=True,                # fill area under curve
            bw_adjust=bw_adjust,      # smoothing knob
            clip=(0, 1),              # keep support in [0,1]
            common_norm=False,        # normalize each curve independently
            gridsize=max(50, int(grid_pts)),
            linewidth=2.0,
            alpha=0.1,
            label=f"{label} (n={a.size}, μ={a.mean():.3f})",
        )
        any_plotted = True

    if not any_plotted:
        print(f"[INFO] Nothing to plot for {title}.")
        plt.close()
        return

    plt.xlabel("Coverage")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.title(title)
    plt.grid(alpha=0.3)
    if logy:
        plt.yscale("log")
    plt.legend(frameon=False, fontsize=9, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Plot coverage distributions grouped by mode/class.")
    ap.add_argument("--root", type=str, default="/mnt/slow/shapenetcorev2-source",
                    help="Dataset root containing <synset>/<instance>/models/coverage_<mode>.csv files.")
    ap.add_argument("--taxonomy", type=str, default="filtered_taxonomy.json",
                    help="Path to taxonomy JSON under --root (or absolute path).")
    ap.add_argument("--classes", type=str, default="",
                    help="Class names to include (comma/space-separated). Blank=ALL.")
    ap.add_argument("--mode", type=str, default="default",
                    help="Mode name, comma-separated modes, or 'all' to discover from disk.")
    ap.add_argument("--n", type=int, default=5,
                    help="Keep rows whose trailing '_<n>.npy' index is less than this value.")
    ap.add_argument("--style", type=str, choices=["hist", "kde-modes", "kde-classes"], default="hist",
                    help="Plot style: single histogram of all values, or KDE overlay by modes/classes.")
    ap.add_argument("--bins", type=int, default=40,
                    help="Histogram bins (hist) or x-grid points (KDE).")
    ap.add_argument("--logy", action="store_true", help="Use log-scale on Y axis.")
    ap.add_argument("--outdir", type=str, default="coverage_plots",
                    help="Directory to save output plots (created under --root unless absolute).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.is_absolute():
        taxonomy_path = (root / taxonomy_path).resolve()

    # Resolve classes -> synset IDs
    classes_raw = parse_classes_arg(args.classes)
    if classes_raw:
        try:
            allowed_synsets = names_to_synsets(classes_raw, str(taxonomy_path))
        except Exception as e:
            print(f"[ERROR] Failed to resolve classes via taxonomy: {e}")
            return
    else:
        allowed_synsets = None

    # Modes
    if args.mode.strip().lower() == "all":
        modes = discover_modes(root)
        if not modes:
            print("[INFO] No coverage_*.csv files found under root. Exiting.")
            return
    else:
        modes = [m.strip() for m in args.mode.split(",") if m.strip()]

    # Collect
    all_values, by_mode, by_class, by_mode_cls = collect_grouped_coverages(
        root=root,
        modes=modes,
        allowed_synsets=allowed_synsets,
        n_max=args.n,
        taxonomy_path=taxonomy_path,
    )

    # Output dir
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Labels for titles/files
    if allowed_synsets is None:
        cls_label = "ALL"
    else:
        try:
            cls_names = synsets_to_names(allowed_synsets, str(taxonomy_path))
        except Exception:
            cls_names = allowed_synsets
        cls_label = "-".join(sorted({re.sub(r'[^A-Za-z0-9\-]+', '_', n) for n in cls_names}))

    modes_label = "-".join(modes) if len(modes) <= 4 else f"{len(modes)}modes"

    # Plot according to style
    if args.style == "hist":
        title = f"Coverage Histogram | classes={cls_label} | modes={modes_label} | n<{args.n}"
        out_path = outdir / f"hist_all_classes-{cls_label}_modes-{modes_label}_nlt{args.n}.png"
        plot_hist_all(all_values, title, out_path, bins=args.bins, logy=args.logy)

    elif args.style == "kde-modes":
        # Filter empty groups
        groups = {k: v for k, v in by_mode.items() if len(v) > 1}
        title = f"KDE Overlay by Mode | classes={cls_label} | n<{args.n}"
        out_path = outdir / f"kde_by-mode_classes-{cls_label}_nlt{args.n}.png"
        plot_kde_overlay(groups, title, out_path, logy=args.logy, grid_pts=args.bins)

    elif args.style == "kde-classes":
        groups = {k: v for k, v in by_class.items() if len(v) > 1}
        title = f"KDE Overlay by Class | modes={modes_label} | n<{args.n}"
        out_path = outdir / f"kde_by-class_modes-{modes_label}_nlt{args.n}.png"
        plot_kde_overlay(groups, title, out_path, logy=args.logy, grid_pts=args.bins)

    # Console summary
    print("\n[Summary]")
    print(f"  Total values: {len(all_values)}")
    for m in modes:
        n = len(by_mode.get(m, []))
        print(f"  Mode {m:>12}: n={n}")
    top_classes = sorted(by_class.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
    if top_classes:
        print("  Top classes by count:")
        for name, vals in top_classes:
            print(f"    - {name:>16}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()