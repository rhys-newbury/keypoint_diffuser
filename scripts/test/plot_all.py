#!/usr/bin/env python3
"""
Plot best result per (category, algorithm), then group bars by algorithm for each category.
- X-axis: categories
- For each category: one bar per algorithm (the best checkpoint for that algo in that category on the chosen metric)

Usage examples:
  python plot_grouped_by_algo.py --db results.db --metric das --out grouped_algo.png
  python plot_grouped_by_algo.py --db results.db --metric fwd --model PointNet --out -
"""
import argparse
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from classes import MODEL_CLASSES


VALID_METRICS = {"das", "fwd", "bwd", "miou_at_0_1"}


def fetch_rows(db_path: Path, model_filter: str | None):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    if model_filter:
        cur.execute(
            "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, key_points, category, fwd, bwd, das, miou_at_0_1 "
            "FROM runs WHERE model = ?",
            (model_filter,),
        )
    else:
        cur.execute(
            "SELECT id, model, ckpt, annotation_json, pcd_path, batch_size, key_points, category, fwd, bwd, das, miou_at_0_1 "
            "FROM runs"
        )
    rows = cur.fetchall()
    con.close()
    return rows


def pick_best_per_algo_category(rows, metric: str):
    """Return nested dict: {category: {algo: {metric, ckpt, id, fwd, bwd, das, miou_at_0_1}}}
    Chooses the best checkpoint *within each (category, algo)* by the selected metric.
    """
    idx = {"fwd": 8, "bwd": 9, "das": 10, "miou_at_0_1": 11}[metric]
    best: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        # 0:id,1:model,2:ckpt,3:annotation_json,4:pcd_path,5:batch_size,6:key_points,7:category,8:fwd,9:bwd,10:das,11:miou_at_0_1
        cat = r[7]
        algo = r[1]
        val = r[idx]
        if val is None:
            continue
        if cat not in best:
            best[cat] = {}
        if algo not in best[cat] or val > best[cat][algo]["metric"]:
            best[cat][algo] = {
                "metric": float(val),
                "ckpt": r[2],
                "id": r[0],
                "fwd": r[8],
                "bwd": r[9],
                "das": r[10],
                "miou_at_0_1": r[11],
            }
    return best


def print_table_grouped(best_map, metric: str):
    cats = sorted(best_map.keys())
    algos = sorted({algo for cat in best_map.values() for algo in cat})

    print(f"\nBest rows by {metric} (grouped by category → algo):")
    header = f"{'category':<12} {'algo':<20} {'metric':>8} {'fwd':>8} {'bwd':>8} {'das':>8} {'miou@0.1':>10}  ckpt"
    print(header)
    for cat in cats:
        for algo in algos:
            b = best_map[cat].get(algo)
            if not b:
                continue
            print(
                f"{cat:<12} {algo:<20} {b['metric']:>8.4f} "
                f"{(b['fwd'] if b['fwd'] is not None else float('nan')):>8.4f} "
                f"{(b['bwd'] if b['bwd'] is not None else float('nan')):>8.4f} "
                f"{(b['das'] if b['das'] is not None else float('nan')):>8.4f} "
                f"{(b['miou_at_0_1'] if b['miou_at_0_1'] is not None else float('nan')):>10.4f}  "
                f"{b['ckpt']}"
            )


def plot_grouped_by_algo(best_map, metric: str, out_path: Path | None):
    cats = sorted(best_map.keys())
    # ensure stable set of algorithms across categories
    algos = sorted({algo for cat in best_map.values() for algo in cat})

    x = np.arange(len(cats))
    n_algos = len(algos)
    if n_algos == 0 or len(cats) == 0:
        print("Nothing to plot.")
        return

    # bar width so that groups fit nicely
    width = min(0.8 / max(n_algos, 1), 0.25)

    plt.figure(figsize=(max(8, 1.2 * len(cats)), 5))

    for i, algo in enumerate(algos):
        vals = [best_map[c].get(algo, {}).get("metric", float("nan")) for c in cats]
        offset = (i - (n_algos - 1) / 2) * width
        bars = plt.bar(x + offset, vals, width=width, label=algo)

        # annotate bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.title(f"Best {metric} per category, grouped by algorithm")
    plt.ylabel(metric)
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.legend(title="Algorithm", fontsize=8)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"[✓] saved grouped-by-algo plot -> {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot best results per (category, algorithm) for a chosen metric, "
            "grouped by algorithm within each category."
        )
    )
    ap.add_argument("--db", type=Path, required=True, help="Path to results.db")
    ap.add_argument(
        "--metric",
        choices=sorted(VALID_METRICS),
        default="das",
        help="Metric to rank by within each (category, algo). Default: das",
    )
    ap.add_argument(
        "--model",
        choices=MODEL_CLASSES.keys(),
        default=None,
        help="Optional: filter rows by a specific algorithm (model).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("grouped_by_algo.png"),
        help="Output image path (use '-' to show instead of saving)",
    )
    args = ap.parse_args()

    rows = fetch_rows(args.db, args.model)
    if not rows:
        print("No rows found in runs table.")
        return

    best = pick_best_per_algo_category(rows, args.metric)
    if not best:
        print("No valid rows with the chosen metric.")
        return

    print_table_grouped(best, args.metric)

    out_path = None if str(args.out) == "-" else args.out
    plot_grouped_by_algo(best, args.metric, out_path)


if __name__ == "__main__":
    main()
