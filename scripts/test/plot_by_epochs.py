#!/usr/bin/env python3
"""
Plot metrics vs epoch from SQLite DBs produced by:
- get_das.py              -> table 'runs' with: category, ckpt, das, miou_at_0_1
- get_reconstruction.py   -> table 'reconstruction' with: category, ckpt, cd_recon, emd_recon

X axis: epoch parsed from ckpt path (last integer).
Y axis: one of: das | miou | cd_recon | emd_recon

Directory support:
  --db can be files and/or directories.
  If a directory is given, we include all *.db inside (non-recursive by default).
  Use --recursive to include **/*.db.
"""

import argparse
import collections
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot DAS/mIoU/CD/EMD vs epoch from experiment result DBs."
    )
    p.add_argument(
        "--db", type=Path, nargs="+", required=True,
        help="SQLite DB path(s) or dir(s) containing *.db files."
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="When a --db path is a directory, also search subdirectories for *.db."
    )
    p.add_argument(
        "--metric", choices=["das", "miou", "cd_recon", "emd_recon"], default="das",
        help="Metric to plot on Y axis."
    )
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="Categories to include (e.g. chair mug table)."
    )
    p.add_argument(
        "--all", action="store_true",
        help="If set, include all categories found."
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Optional plot title."
    )
    p.add_argument(
        "--output", type=Path, default=Path("metric_vs_epoch.png"),
        help="Output figure path."
    )
    p.add_argument(
        "--show", action="store_true",
        help="Show the figure interactively."
    )
    p.add_argument(
        "--average-duplicates", action="store_true",
        help="Average values when multiple rows share the same (category, epoch)."
    )
    p.add_argument(
        "--max-epoch", type=int, default=100,
        help="the max epoch corresponding to the model net_final"
    )
    return p.parse_args()


# ----------------------------
# DB helpers
# ----------------------------

def expand_db_paths(paths: List[Path], recursive: bool) -> List[Path]:
    """Expand any directory to its *.db files (optionally recursive)."""
    dbs: List[Path] = []
    for p in paths:
        if p.is_dir():
            pattern = "**/*.db" if recursive else "*.db"
            found = sorted(p.glob(pattern))
            dbs.extend([f for f in found if f.is_file()])
        else:
            if p.suffix.lower() == ".db" and p.is_file():
                dbs.append(p)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for d in dbs:
        if d.resolve() not in seen:
            seen.add(d.resolve())
            uniq.append(d)
    return uniq


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def extract_epoch(ckpt: str) -> int | None:
    if not ckpt:
        return None
    ints = re.findall(r"(\d+)", str(ckpt))
    return int(ints[-1]) if ints else None


def read_runs_rows(db_paths: List[Path]) -> List[Tuple[str, str, float, float]]:
    """Return (category, ckpt, das, miou_at_0_1)."""
    rows: List[Tuple[str, str, float, float]] = []
    for db in db_paths:
        con = sqlite3.connect(str(db))
        if table_exists(con, "runs"):
            cur = con.cursor()
            cur.execute("SELECT category, ckpt, das, miou_at_0_1 FROM runs WHERE ckpt IS NOT NULL;")
            rows.extend(cur.fetchall())
        con.close()
    return rows


def read_recon_rows(db_paths: List[Path]) -> List[Tuple[str, str, float, float]]:
    """Return (category, ckpt, cd_recon, emd_recon)."""
    rows: List[Tuple[str, str, float, float]] = []
    for db in db_paths:
        con = sqlite3.connect(str(db))
        if table_exists(con, "reconstruction"):
            cur = con.cursor()
            cur.execute("SELECT category, ckpt, cd_recon, emd_recon FROM reconstruction WHERE ckpt IS NOT NULL;")
            rows.extend(cur.fetchall())
        con.close()
    return rows


# ----------------------------
# Aggregation
# ----------------------------

def gather_metric_by_category(
    metric: str,
    runs_rows: List[Tuple[str, str, float, float]],
    recon_rows: List[Tuple[str, str, float, float]],
    max_epoch: int,
) -> Dict[str, List[Tuple[int, float]]]:
    """Build: category -> list of (epoch, metric_value)."""
    cat2vals: Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    if metric in ("das", "miou"):
        for category, ckpt, das, miou in runs_rows:
            epoch = extract_epoch(ckpt)
            if "net_final" in str(ckpt):
                epoch = max_epoch
            if epoch is None:
                continue
            val = float(das) if metric == "das" else float(miou)
            cat2vals[category].append((epoch, val))
    else:  # cd_recon, emd_recon
        for category, ckpt, cd, emd in recon_rows:
            epoch = extract_epoch(ckpt)
            if "net_final" in str(ckpt):
                epoch = max_epoch
            if epoch is None:
                continue
            val = float(cd) if metric == "cd_recon" else float(emd)
            cat2vals[category].append((epoch, val))
    return cat2vals


def average_duplicate_epochs(pairs: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Given [(epoch, value), ...], average duplicate epochs and return sorted arrays."""
    if not pairs:
        return np.array([]), np.array([])
    per_epoch = collections.defaultdict(list)
    for e, v in pairs:
        per_epoch[e].append(v)
    epochs = np.array(sorted(per_epoch.keys()), dtype=int)
    vals   = np.array([np.mean(per_epoch[e]) for e in epochs], dtype=float)
    return epochs, vals


# ----------------------------
# Plot
# ----------------------------

def main():
    args = parse_args()

    db_paths = expand_db_paths(args.db, args.recursive)
    if not db_paths:
        raise SystemExit("No .db files found from the provided --db paths.")
    print(f"[INFO] Using {len(db_paths)} DB file(s).")

    runs_rows  = read_runs_rows(db_paths)
    recon_rows = read_recon_rows(db_paths)

    if args.metric in ("das", "miou") and not runs_rows:
        raise SystemExit("No 'runs' rows found for metrics das/miou.")
    if args.metric in ("cd_recon", "emd_recon") and not recon_rows:
        raise SystemExit("No 'reconstruction' rows found for metrics cd_recon/emd_recon.")

    by_cat = gather_metric_by_category(args.metric, runs_rows, recon_rows, args.max_epoch)
    if not by_cat:
        raise SystemExit("No (category, epoch, value) pairs found after parsing ckpt epochs.")

    # choose categories
    all_cats = sorted(by_cat.keys())
    if args.all or (not args.classes):
        cats = all_cats if args.all or not args.classes else args.classes
    else:
        cats = [c for c in args.classes if c in by_cat]
        missing = set(args.classes) - set(cats)
        for m in sorted(missing):
            print(f"[WARN] No data for requested category: {m}")

    if not cats:
        raise SystemExit("No categories to plot.")

    # plot
    plt.figure(figsize=(9, 5))
    for cat in cats:
        pairs = by_cat.get(cat, [])
        if not pairs:
            print(f"[WARN] No data for '{cat}', skipping.")
            continue
        if args.average_duplicates:
            epochs, values = average_duplicate_epochs(pairs)
        else:
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            epochs = np.array([e for e, _ in pairs_sorted], dtype=int)
            values = np.array([v for _, v in pairs_sorted], dtype=float)
        if epochs.size == 0:
            continue
        plt.plot(epochs, values, marker="o", linewidth=1.5, label=cat)

    plt.xlabel("Epoch")
    ylabels = {
        "das": "DAS (Dual Alignment Score)",
        "miou": "mIoU @ 0.1",
        "cd_recon": "Chamfer Distance (recon)",
        "emd_recon": "EMD (recon)",
    }
    plt.ylabel(ylabels[args.metric])
    ttl = args.title or f"{ylabels[args.metric]} vs Epoch"
    plt.title(ttl)
    plt.axvline(x=50, color='k')
    plt.grid(True, alpha=0.35)
    plt.legend(title="Category", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"[✓] Saved plot to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
