import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classes import MODEL_CLASSES
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_recon_collage(
    root_pattern: Path,
    out_path: Path,
    grid=(6, 6),
    gt_point_size=1,
    recon_point_size=2,
    gt_color="gray",
    recon_color="red",
    alpha_gt=0.25,
    worst: bool = False,  # <<< show worst (highest CD) first
):
    dirs = [
        d
        for d in root_pattern.glob("*")
        if d.is_dir() and (d / "recon.npy").exists() and (d / "gt.npy").exists()
    ]
    if not dirs:
        print(f"[!] No samples with recon.npy + gt.npy under {root_pattern}")
        return

    def _cd(d: Path):
        m = d / "meta.json"
        if m.exists():
            try:
                return float(json.loads(m.read_text()).get("cd", float("inf")))
            except Exception:
                pass
        return float("inf")

    # sort by CD; worst=True => descending
    dirs = sorted(dirs, key=_cd, reverse=worst)

    nplots = grid[0] * grid[1]
    dirs = dirs[:nplots]

    fig = plt.figure(figsize=(grid[1] * 5, grid[0] * 5), dpi=150)
    for idx, root in enumerate(dirs):
        recon = np.load(root / "recon.npy")
        gt = np.load(root / "gt.npy")
        title = root.name
        m = root / "meta.json"
        if m.exists():
            try:
                meta = json.loads(m.read_text())
                cd = meta.get("cd")
                emd = meta.get("emd")
                parts = [title]
                if cd is not None:
                    parts.append(f"CD={cd:.4e}")
                if emd is not None:
                    parts.append(f"EMD={emd:.4e}")
                title = "  |  ".join(parts)
            except Exception:
                pass

        ax = fig.add_subplot(grid[0], grid[1], idx + 1, projection="3d")
        ax.scatter(
            gt[:, 0], gt[:, 2], gt[:, 1], s=gt_point_size, c=gt_color, alpha=alpha_gt
        )
        ax.scatter(
            recon[:, 0],
            recon[:, 2],
            recon[:, 1],
            s=recon_point_size,
            c=recon_color,
            alpha=1.0,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)

        all_pts = np.vstack([gt, recon])
        max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2.0
        mid = all_pts.mean(0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(
        f"[✓] Saved collage with {len(dirs)} worst (highest-CD) samples to {out_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot collage of reconstructions vs GT.")
    ap.add_argument(
        "--model",
        choices=MODEL_CLASSES.keys(),
        default=None,
        help="Optional: filter rows by model",
    )
    ap.add_argument(
        "--category", required=True, help="Category subfolder (under output/<model>/)"
    )
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    folder = Path("recons_out") / args.model / args.category
    out_png = folder.parent / f"{args.category}_recon_collage.png"

    plot_recon_collage(
        folder,
        out_path=out_png,
        grid=(args.rows, args.cols),
        gt_point_size=1,
        recon_point_size=2,
    )
