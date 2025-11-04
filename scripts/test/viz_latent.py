#!/usr/bin/env python3
import argparse

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------- same k-NN denoise as before ----------
def knn_outlier_filter(points, k=16, std_ratio=2.5, min_points=50):
    N = points.shape[0]
    if min_points > N or k >= N:
        return points

    diffs = points[:, None, :] - points[None, :, :]
    dists2 = np.sum(diffs * diffs, axis=-1)

    nn_idx = np.argpartition(dists2, kth=k - 1, axis=1)[:, :k]
    nn_d2 = np.take_along_axis(dists2, nn_idx, axis=1)
    nn_d = np.sqrt(nn_d2 + 1e-9)

    mean_d = nn_d.mean(axis=1)
    mu = mean_d.mean()
    sigma = mean_d.std() + 1e-9

    keep = mean_d <= (mu + std_ratio * sigma)
    filtered = points[keep]
    if filtered.shape[0] < min_points:
        return points
    return filtered


# ---------- axis / bounds helpers ----------
def auto_pick_up_axis(clouds, kps):
    mins = []
    maxs = []
    for c in clouds:
        if c is not None and len(c) > 0:
            mins.append(c.min(axis=0))
            maxs.append(c.max(axis=0))
    for k in kps:
        if k is not None and len(k) > 0:
            mins.append(k.min(axis=0))
            maxs.append(k.max(axis=0))
    if not mins:
        return "z"
    mins = np.stack(mins, axis=0)
    maxs = np.stack(maxs, axis=0)
    span = (maxs.max(axis=0) - mins.min(axis=0))
    up_idx = int(np.argmin(span))
    for name, idx in AXIS_INDEX.items():
        if idx == up_idx:
            return name
    return "z"


def compute_bounds_2d(clouds, kps, ax1, ax2):
    vs1, vs2 = [], []
    for c in clouds:
        if len(c):
            vs1.append(c[:, ax1])
            vs2.append(c[:, ax2])
    for k in kps:
        if len(k):
            vs1.append(k[:, ax1])
            vs2.append(k[:, ax2])
    if not vs1:
        return (-1, 1), (-1, 1)
    v1 = np.concatenate(vs1)
    v2 = np.concatenate(vs2)
    v1_min, v1_max = v1.min(), v1.max()
    v2_min, v2_max = v2.min(), v2.max()
    c1 = 0.5 * (v1_min + v1_max)
    c2 = 0.5 * (v2_min + v2_max)
    half = max(v1_max - v1_min, v2_max - v2_min) * 0.5
    return (c1 - half, c1 + half), (c2 - half, c2 + half)


# ---------- pastel palette for keypoints ----------
def make_pastel_palette(n):
    # soft, distinct-ish; repeat if n > len(base)
    base10 = [
        "#ffb3ba",  # soft pink
        "#bae1ff",  # sky blue
        "#baffc9",  # mint
        "#ffffba",  # pale yellow
        "#ffdfba",  # peach
        "#e2baff",  # lavender
        "#ffd1dc",  # blush
        "#c9fff5",  # aqua
        "#d1b3ff",  # lilac
        "#ffe4b5",  # light apricot
    ]

    if n <= len(base10):
        return base10[:n]
    # repeat
    return [base10[i % len(base10)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(
        description="Top-down pastel GIF, per-keypoint color, centered per frame"
    )
    ap.add_argument("--npz", default="pair_4.npz")
    ap.add_argument("--out", default="pair_4_topdown.gif")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument(
        "--look-from",
        choices=["x", "y", "z"],
        default=None,
        help="axis to look from (treat as up); if none, auto-detect",
    )
    ap.add_argument(
        "--skip-denoise",
        action="store_true",
        help="don't run the extra k-NN filter",
    )
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    steps = data["steps"]
    clouds = list(data["clouds"])
    kps = list(data["kps"])
    T = len(steps)

    # 1) optional extra denoise
    if not args.skip_denoise:
        for i in range(T):
            clouds[i] = knn_outlier_filter(clouds[i], k=16, std_ratio=2.5, min_points=30)

    # 2) center per frame (translate cloud + kps so cloud center is at origin)
    for i in range(T):
        c = clouds[i]
        if len(c) == 0:
            continue
        center = c.mean(axis=0)  # you can switch to bbox center if you prefer
        clouds[i] = c - center
        kps[i] = kps[i] - center

    # 3) decide view plane
    if args.look_from is None:
        up_axis = auto_pick_up_axis(clouds, kps)
        print(f"[i] auto-picked up axis: {up_axis}")
    else:
        up_axis = args.look_from
        print(f"[i] using up axis: {up_axis}")
    up_idx = AXIS_INDEX[up_axis]
    axes_2d = [0, 1, 2]
    axes_2d.remove(up_idx)
    ax1, ax2 = axes_2d  # e.g. up=y -> plot x,z

    # 4) global bounds (after centering!)
    (min1, max1), (min2, max2) = compute_bounds_2d(clouds, kps, ax1, ax2)

    # 5) make per-keypoint colors — we want SAME order across steps
    max_k = max(k.shape[0] for k in kps)
    kp_palette = make_pastel_palette(max_k)

    cloud_color = "#f7c9c9"  # whole cloud stays faint pink
    bg = "#ffffff"

    frames = []
    for i in range(T):
        c = clouds[i]
        k = kps[i]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # draw cloud
        ax.scatter(
            c[:, ax1],
            c[:, ax2],
            s=3,
            c=cloud_color,
            edgecolors="none",
            alpha=1.0,
        )

        # draw each keypoint with its own color
        for ki in range(k.shape[0]):
            col = kp_palette[ki]
            ax.scatter(
                [k[ki, ax1]],
                [k[ki, ax2]],
                s=30,
                c=col,
                edgecolors="black",
                linewidths=0.3,
                alpha=1.0,
            )

        ax.set_xlim(min1, max1)
        ax.set_ylim(min2, max2)
        ax.set_aspect("equal", "box")
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_title(f"step {i} (file step {steps[i]})", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # turn off spines (black square)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # no title
        ax.set_title("")

        ax.invert_yaxis()  # keep it as "looking down"

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((*fig.canvas.get_width_height()[::-1], 3))
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(args.out, frames, loop=0, duration=1.0 / args.fps)


if __name__ == "__main__":
    main()
