#!/usr/bin/env python3
"""
Visualize an NxN collage of random SMPL-X converted samples using PeopleDataset.
Usage:
    python viz_collage.py --dir path/to/output --grid 3
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.people_dataset import PeopleDataset  # <-- import your dataset class


# -----------------------------------------------------------------------------
# grid visualization
# -----------------------------------------------------------------------------


def visualize_collage(dataset: PeopleDataset, grid: int = 3):
    n = min(grid * grid, len(dataset))
    fig, axes = plt.subplots(
        grid, grid, figsize=(4 * grid, 4 * grid), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten()

    # pick random samples
    idxs = random.sample(range(len(dataset)), n)

    for ax, idx in zip(axes[:n], idxs):
        sample = dataset[idx]
        pts = (
            sample["target_shape"].cpu().numpy()
            if isinstance(sample["target_shape"], torch.Tensor)
            else sample["target_shape"]
        )

        # color by z for aesthetics (optional)
        z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-8)
        colors = plt.get_cmap("viridis")(z_norm)

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=0.5,
            alpha=0.6,
            c=colors,
            depthshade=False,
        )

        # plot keypoints if present
        if "keypoints" in sample and sample["keypoints"].numel() > 0:
            kpts = sample["keypoints"].cpu().numpy()
            ax.scatter(
                kpts[:, 0],
                kpts[:, 1],
                kpts[:, 2],
                s=20,
                c="red",
                depthshade=False,
                marker="^",
                edgecolors="k",
                linewidths=0.4,
            )

        ax.set_title(sample.get("name", f"sample_{idx}"), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # equal axes
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        center = (mins + maxs) / 2
        r = (maxs - mins).max() / 2
        for dim, c in zip("xyz", center):
            getattr(ax, f"set_{dim}lim")([c - r, c + r])

    # Hide unused axes if fewer samples than grid²
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig("output.png")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NxN collage from PeopleDataset."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/app/ego"),
        help="Output directory containing .ply/.kp.npz files.",
    )
    parser.add_argument("--grid", type=int, default=3, help="Grid size (NxN).")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize point clouds to unit sphere.",
    )
    parser.add_argument(
        "--random-rotate",
        action="store_true",
        help="Apply random Y rotations for visualization.",
    )
    args = parser.parse_args()

    dataset = PeopleDataset(
        root_dir=args.dir,
        normalize=args.normalize,
        random_rotate=args.random_rotate,
    )

    visualize_collage(dataset, grid=args.grid)


if __name__ == "__main__":
    main()
