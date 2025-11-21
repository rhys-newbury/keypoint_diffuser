#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classes import MODEL_CLASSES
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_collage(
    root_pattern: Path,
    out_path: Path,
    grid=(6, 6),
    point_size=1,
    kp_size=40,
    sort_by_score=True,
    ascending=True,
    show_assign=True,
):
    """
    Makes a collage of PC + pred/gt keypoints in a grid.
    Args:
        root_pattern : glob pattern for sample dirs (category folder containing model subdirs)
        out_path     : where to save the collage PNG
        grid         : (rows, cols)
        sort_by_score: if True, load meta.json and sort by 'score'
        ascending    : sort order for scores
        show_assign  : if True, draw assignment lines from pred→GT
    """
    # gather subdirs
    dirs = sorted(x for x in root_pattern.glob("*") if x.is_dir())

    scored_dirs = []
    for d in dirs:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        score = meta.get("score", None)
        if score is not None:
            scored_dirs.append((score, d))
    scored_dirs.sort(key=lambda x: x[0], reverse=not ascending)
    dirs = [d for _, d in scored_dirs]
    scores = [s for s, _ in scored_dirs]

    nplots = grid[0] * grid[1]
    dirs = dirs[:nplots]

    fig = plt.figure(figsize=(grid[1] * 5, grid[0] * 5), dpi=150)

    colors = None
    for idx, root in enumerate(dirs):
        pc = np.load(root / "pc.npy")
        pred = np.load(root / "pred_kp.npy")
        gt = np.load(root / "gt_kp.npy")

        assign_path = root / "assign.npy"
        assign = np.load(assign_path) if assign_path.exists() else None

        if colors is None:
            num_kp = pred.shape[0]
            cmap = plt.cm.get_cmap("rainbow", num_kp)
            colors = cmap(np.arange(num_kp))[:, :3]

        ax = fig.add_subplot(grid[0], grid[1], idx + 1, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=point_size, c="gray", alpha=0.3)
        ax.scatter(pred[:, 0], pred[:, 2], pred[:, 1], s=kp_size, c=colors, marker="o")
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=kp_size, c="blue", marker="^")

        # Draw assignment lines
        if show_assign and assign is not None:
            for i, j in enumerate(assign):
                if j < 0 or j >= gt.shape[0]:
                    continue
                x = [pred[i, 0], gt[j, 0]]
                y = [pred[i, 2], gt[j, 2]]
                z = [pred[i, 1], gt[j, 1]]
                ax.plot(x, y, z, c=colors[i], alpha=0.6, linewidth=1.0)

        if score is not None:
            ax.set_title(f"score={scores[idx]:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)

        # equal scaling
        all_pts = np.vstack([pc, pred, gt])
        max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2
        mid = all_pts.mean(0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Saved collage with {len(dirs)} samples to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot collage of saved numpy geoms")
    ap.add_argument("--model", choices=MODEL_CLASSES.keys(), required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--descending", action="store_true", help="Sort scores high→low")
    ap.add_argument(
        "--no-assign", action="store_true", help="Do not draw assignment lines"
    )
    args = ap.parse_args()

    folder = Path("geom_np") / args.category
    out_path = Path("output") / args.model / f"{args.category}_collage.png"

    plot_collage(
        folder,
        out_path=out_path,
        grid=(6, 6),
        sort_by_score=True,
        ascending=not args.descending,
        show_assign=not args.no_assign,
    )
