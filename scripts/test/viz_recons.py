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
    alpha_gt=0.5,
    worst: bool = False,  # <<< show worst (highest CD) first
    plot_kps: bool = False,
    kps_root: Path = None, 
):
    dirs = [
        d
        for d in root_pattern.glob("*")
        if d.is_dir() and (d / "recon.npy").exists() and (d / "gt.npy").exists()
    ]
    if not dirs:
        print(f"[!] No samples with recon.npy + gt.npy under {root_pattern}")
        return

    if plot_kps:
        assert kps_root is not None
        
    # kps_dirs = [
    #     d
    #     for d in kps_root.glob("*")
    #     if d.is_dir() and (d / "gt_kp.npy").exists() and (d / "pred_kp.npy").exists()
    # ]
    # if not kps_dirs:
    #     print(f"[!] No samples with gt_kp.npy + pred_kp.npy under {kps_root}")
    #     return

    def _cd(d: Path):
        m = d / "meta.json"
        if m.exists():
            try:
                return float(json.loads(m.read_text()).get("cd", float("inf")))
            except Exception:
                pass
        return float("inf")
    
    def _emd(d: Path):
        m = d / "meta.json"
        if m.exists():
            try:
                return float(json.loads(m.read_text()).get("emd", float("inf")))
            except Exception:
                pass
        return float("inf")

    # sort by CD; worst=True => descending
    dirs = sorted(dirs, key=_cd, reverse=worst)

    nplots = grid[0] * grid[1]
    dirs = dirs[:nplots]

    fig = plt.figure(figsize=(grid[1] * 5, grid[0] * 5), dpi=150)
    for idx, root in enumerate(dirs):
        model_id = root.name
        kps_dir = kps_root / model_id
        
        recon = np.load(root / "recon.npy")
        gt = np.load(root / "gt.npy")
        inpc = np.load(root / "input.npy")
        title = root.name
        m = root / "meta.json"
        try:
            pred_kp = np.load(kps_dir / "pred_kp.npy")
            gt_kp = np.load(kps_dir / "gt_kp.npy")
        except FileNotFoundError:
            continue
        
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

        # hacky filter for the recon ################################################################
        num_removed = 0
        dist_thresh = 3
        # take the average of all points in the reconstructed point cloud
        mean_pt = np.mean(recon, 0)
        dist_from_mean = np.linalg.norm(recon - mean_pt, axis=1)
        # print(f"before filtering recon shape: {recon.shape}")
        # print(dist_from_mean.shape)
        # print(f"max dist from mean: {max(dist_from_mean)}")
        # print(f"mean dist from mean: {np.mean(dist_from_mean)}")
        # print(f"median dist from mean: {np.median(dist_from_mean)}")
        # remove points that are further to the mean than the threshold
        in_mask = dist_from_mean < dist_thresh
        while not all(in_mask):
            # print(np.size(in_mask))
            # print(np.count_nonzero(in_mask))
            # print(np.size(in_mask) - np.count_nonzero(in_mask))
            # print(f"removing {len(in_mask) - sum(in_mask)} outlier points")
            num_removed += len(in_mask) - sum(in_mask)
            recon = recon[in_mask]
            mean_pt = np.mean(recon, 0)
            dist_from_mean = np.linalg.norm(recon - mean_pt, axis=1)
            in_mask = dist_from_mean < dist_thresh
            
        # print(f"after filtering recon shape: {recon.shape}")
        print(f"total number of points removed: {num_removed}")
        # hacky filter for the recon ################################################################

        ax = fig.add_subplot(grid[0], grid[1], idx + 1, projection="3d")
        ax.scatter(
            gt[:, 0], gt[:, 2], gt[:, 1], s=gt_point_size, c=gt_color, alpha=alpha_gt
        )
        ax.scatter(
            inpc[:, 0], inpc[:, 2], inpc[:, 1], s=0.5, c="green", alpha=alpha_gt
        )
        ax.scatter(
            recon[:, 0],
            recon[:, 2],
            recon[:, 1],
            s=recon_point_size,
            c=recon_color,
            alpha=1.0,
        )
        ax.scatter(pred_kp[:, 0], pred_kp[:, 2], pred_kp[:, 1], s=50, c="blue", marker="o")
        # ax.scatter(gt_kp[:, 0], gt_kp[:, 2], gt_kp[:, 1], s=kp_size, c="blue", marker="^")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=0)

        all_pts = np.vstack([gt, recon])
        max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2.0
        mid = all_pts.mean(0)
        # ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        # ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        # ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    # print(
        # f"[✓] Saved collage with {len(dirs)} worst (highest-CD) samples to {out_path}"
    # )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot collage of reconstructions vs GT.")
    ap.add_argument(
        "--model",
        choices=MODEL_CLASSES.keys(),
        default="Ours",
        help="Optional: filter rows by model",
    )
    ap.add_argument(
        "--category", required=True, help="Category subfolder (under output/<model>/)"
    )
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--worst", action="store_true")
    ap.add_argument("--recon-out-dir", type=Path, default=Path("recons_out"))
    ap.add_argument("--plot-kps", action="store_true")
    ap.add_argument("--kps-out-dir", type=Path, default=Path("kps_out"))
    args = ap.parse_args()

    folder = args.recon_out_dir / args.model / args.category
    kpfolder = args.kps_out_dir / args.model / args.category
    if not args.worst:
        out_png = folder.parent / f"{args.category}_recon_collage.png"
    else:
        out_png = folder.parent / f"{args.category}_worst_recon_collage.png"

    plot_recon_collage(
        folder,
        out_path=out_png,
        grid=(args.rows, args.cols),
        gt_point_size=1,
        recon_point_size=2,
        worst=args.worst,
        plot_kps=args.plot_kps,
        kps_root=kpfolder, 
    )
