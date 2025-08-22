#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from classes import MODEL_CLASSES


VALID_METRICS = {"das", "fwd", "bwd", "miou_at_0_1"}
DB = "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=bryce-rhys/results.db"


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


def pick_best_per_category(rows, metric: str):
    # rows columns:
    # 0:id,1:model,2:ckpt,3:annotation_json,4:pcd_path,5:batch_size,6:key_points,7:category,8:fwd,9:bwd,10:das,11:miou_at_0_1
    best = {}
    idx = {"fwd": 8, "bwd": 9, "das": 10, "miou_at_0_1": 11}[metric]
    for r in rows:
        cat = r[7]
        val = r[idx]
        if val is None:
            continue
        if cat not in best or val > best[cat]["metric"]:
            best[cat] = {
                "metric": float(val),
                "ckpt": r[2],
                "model": r[1],
                "id": r[0],
                "fwd": r[8],
                "bwd": r[9],
                "das": r[10],
                "miou_at_0_1": r[11],
            }
    return best


def plot_best(best_map, metric: str, out_path: Path | None):
    cats = sorted(best_map.keys())
    vals = [best_map[c]["metric"] for c in cats]

    plt.figure(figsize=(max(8, 0.8 * len(cats)), 5))
    bars = plt.bar(cats, vals)
    plt.title(f"Best {metric} per category")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")

    # annotate bars with short ckpt name and value
    for cat, bar in zip(cats, bars):
        v = best_map[cat]["metric"]
        ckpt_short = (
            Path(best_map[cat]["ckpt"]).name if best_map[cat]["ckpt"] else "N/A"
        )
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.3f}\n{ckpt_short}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"[✓] saved plot -> {out_path}")
    else:
        plt.show()


def print_table(best_map, metric: str):
    print(f"\nBest rows by {metric}:")
    print(
        f"{'category':<12} {'metric':>8} {'fwd':>8} {'bwd':>8} {'das':>8} {'miou@0.1':>10}  ckpt"
    )
    for cat in sorted(best_map.keys()):
        b = best_map[cat]
        print(
            f"{cat:<12} {b['metric']:>8.4f} {b['fwd']:>8.4f} {b['bwd']:>8.4f} {b['das']:>8.4f} {b['miou_at_0_1']:>10.4f}  {b['ckpt']}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Plot best results per category from SQLite runs table."
    )
    ap.add_argument("--db", type=Path, required=True, help="Path to results.db")
    ap.add_argument(
        "--metric",
        choices=sorted(VALID_METRICS),
        default="das",
        help="Metric to rank by (default: das)",
    )
    ap.add_argument(
        "--model",
        choices=MODEL_CLASSES.keys(),
        default=None,
        help="Optional: filter rows by model",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("best_per_category.png"),
        help="Output image path (use '-' to show instead of saving)",
    )
    args = ap.parse_args()

    rows = fetch_rows(args.db, args.model)
    if not rows:
        print("No rows found in runs table.")
        return

    best = pick_best_per_category(rows, args.metric)
    if not best:
        print("No valid rows with the chosen metric.")
        return

    print_table(best, args.metric)
    out_path = None if str(args.out) == "-" else args.out
    plot_best(best, args.metric, out_path)


if __name__ == "__main__":
    main()
