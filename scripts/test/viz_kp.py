import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classes import MODEL_CLASSES
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_collage(
    root_pattern: Path, out_path: Path, grid=(6, 6), point_size=1, kp_size=40
):
    """
    Makes a collage of PC + pred/gt keypoints in a grid.
    Args:
        root_pattern : glob pattern for sample dirs
        out_path     : where to save the collage PNG
        grid         : (rows, cols)
    """
    dirs = sorted(x for x in root_pattern.glob("*") if x.is_dir())
    nplots = grid[0] * grid[1]
    dirs = dirs[:nplots]

    fig = plt.figure(figsize=(grid[1] * 5, grid[0] * 5), dpi=150)

    for idx, root in enumerate(dirs):
        pc = np.load(os.path.join(root, "pc.npy"))
        pred = np.load(os.path.join(root, "pred_kp.npy"))
        gt = np.load(os.path.join(root, "gt_kp.npy"))

        ax = fig.add_subplot(grid[0], grid[1], idx + 1, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=point_size, c="gray", alpha=0.3)
        ax.scatter(pred[:, 0], pred[:, 2], pred[:, 1], s=kp_size, c="red", marker="o")
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=kp_size, c="blue", marker="^")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)

        # keep same scale
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
    ap = argparse.ArgumentParser(
        description="Plot best results per category from SQLite runs table."
    )
    ap.add_argument(
        "--model",
        choices=MODEL_CLASSES.keys(),
        default=None,
        help="Optional: filter rows by model",
    )
    ap.add_argument("--category", type=str, help="Category of objects")
    args = ap.parse_args()
    folder = Path("output") / args.model
    plot_collage(
        folder / args.category,
        out_path=folder / f"{args.category}_collage.png",
        grid=(6, 6),
    )
