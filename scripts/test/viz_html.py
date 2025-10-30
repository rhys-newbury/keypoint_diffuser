#!/usr/bin/env python3
"""
Interactive Plotly visualization for reconstruction results (GT, recon, keypoints).

- Loads recon.npy, gt.npy, and kp.npy for each sample under a results folder
- Provides a dropdown menu to choose which sample to view
- Displays reconstruction (red), ground truth (gray), and keypoints (blue diamonds)
- Saves an interactive HTML file (self-contained)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_sample_dirs(root_pattern: Path):
    return [
        d
        for d in root_pattern.glob("*")
        if d.is_dir()
        and (d / "recon.npy").exists()
        and (d / "gt.npy").exists()
        and (d / "kp.npy").exists()
    ]


def plot_recon_collage_interactive(
    root_pattern: Path,
    out_path: Path,
    background_alpha=0.2,
    recon_color="red",
    gt_color="gray",
    kp_color="blue",
):
    dirs = load_sample_dirs(root_pattern)
    if not dirs:
        print(f"[!] No samples with recon.npy + gt.npy + kp.npy under {root_pattern}")
        return

    dirs = sorted(dirs, key=lambda d: d.name)

    fig = go.Figure()
    buttons = []

    for idx, root in enumerate(dirs):
        recon = np.load(root / "recon.npy")
        gt = np.load(root / "gt.npy")
        kp = np.load(root / "kp.npy")

        # Metadata if available
        title = root.name
        meta_file = root / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                parts = [root.name]
                if "cd" in meta:
                    parts.append(f"CD={meta['cd']:.4e}")
                if "emd" in meta:
                    parts.append(f"EMD={meta['emd']:.4e}")
                title = "  |  ".join(parts)
            except Exception:
                pass

        visible = [False] * (3 * len(dirs))  # 3 traces per sample

        # GT points
        fig.add_trace(
            go.Scatter3d(
                x=gt[:, 0],
                y=gt[:, 1],
                z=gt[:, 2],
                mode="markers",
                marker={"size": 2, "color": gt_color, "opacity": background_alpha},
                name="Ground Truth",
                visible=(idx == 0),
            )
        )

        # Reconstruction
        fig.add_trace(
            go.Scatter3d(
                x=recon[:, 0],
                y=recon[:, 1],
                z=recon[:, 2],
                mode="markers",
                marker={"size": 2.5, "color": recon_color, "opacity": 1.0},
                name="Reconstruction",
                visible=(idx == 0),
            )
        )

        # Keypoints
        fig.add_trace(
            go.Scatter3d(
                x=kp[:, 0],
                y=kp[:, 1],
                z=kp[:, 2],
                mode="markers",
                marker={
                    "size": 6,
                    "color": kp_color,
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "line": {"width": 1, "color": "black"},
                },
                name="Keypoints",
                visible=(idx == 0),
            )
        )

        # Update button visibility
        for i in range(3):
            visible[idx * 3 + i] = True

        buttons.append(
            {
                "label": title,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": title},
                ],
            }
        )

    fig.update_layout(
        title="Interactive Reconstruction Viewer",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"[✓] Saved interactive HTML to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot interactive reconstructions with keypoints."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", required=True)
    args = ap.parse_args()

    folder = Path("recons_out") / args.model / args.category
    out_html = folder.parent / f"{args.category}_recons_interactive.html"

    plot_recon_collage_interactive(folder, out_html)