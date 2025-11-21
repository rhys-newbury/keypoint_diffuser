#!/usr/bin/env python3
"""
Interactive Plotly visualization for attention → keypoints.

- Scans a folder for samples saved like the eval script does:
    attn.npy
    encoder_coord.npy
    pred_kp.npy
    meta.json   (optional, just for title bits)

- Builds ONE HTML with a dropdown to switch between samples
- For each sample:
    * background: all encoder coords (gray, faint)
    * for every keypoint:
        - top-N attended points (colored)
        - the keypoint itself (diamond, same color)

- Additionally: saves a PNG of the FIRST sample's point cloud/attn view.
"""

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors  # add this import at the top
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.graph_objs import Layout


def load_attn_sample_dirs(root_pattern: Path):
    return [
        d
        for d in root_pattern.glob("**/*")
        if d.is_dir()
        and (d / "attn.npy").exists()
        and (d / "encoder_coord.npy").exists()
        and (d / "pred_kp.npy").exists()
    ]


def build_sample_traces(
    attn: np.ndarray,
    coords: np.ndarray,
    keypoints: np.ndarray,
    sample_title: str,
    top_n: int = 100,
    background_alpha: float = 0.5,
):
    """
    Returns a list of plotly traces for ONE sample.
    """
    K, N = attn.shape
    top_n = int(min(top_n, N))

    palette = qualitative.Plotly
    if len(palette) < K:
        times = (K + len(palette) - 1) // len(palette)
        palette = (palette * times)[:K]
    else:
        palette = palette[:K]

    traces = []

    # background cloud
    traces.append(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker={"size": 4, "opacity": background_alpha, "color": "gray"},
            name="all points",
            showlegend=True,
            legendgroup="__background__",
        )
    )

    for k in range(K):
        w = attn[k]

        # thresholded selection
        top_mask = w > 0.005
        pts = coords[top_mask]
        color = palette[k]

        base_rgb = np.array(mcolors.to_rgb(color))

        # normalize for opacity
        w_ = (w - w.min()) / (w.max() - w.min() + 1e-8)
        opacities = 0.5 + 0.5 * w_[top_mask]

        r, g, b = base_rgb
        rgba_colors = [
            f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.3f})"
            for alpha in opacities
        ]

        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 4, "color": rgba_colors, "opacity": 1.0},
                name=f"k{k} top-{top_n}",
                legendgroup=f"k{k}",
                hovertemplate=(
                    f"<b>{sample_title}</b><br>"
                    f"k{k} top-{top_n}<br>"
                    "x:%{x:.3f}<br>y:%{y:.3f}<br>z:%{z:.3f}<br>"
                    "w:%{customdata:.5f}<extra></extra>"
                ),
                customdata=w[top_mask],
            )
        )

        # keypoint marker
        kp = keypoints[k]
        traces.append(
            go.Scatter3d(
                x=[kp[0]],
                y=[kp[1]],
                z=[kp[2]],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "color": color,
                    "line": {"width": 1, "color": "black"},
                },
                name=f"k{k} (keypoint)",
                legendgroup=f"k{k}",
            )
        )

    return traces


def plot_attention_interactive_collage(
    root_pattern: Path,
    out_path: Path,
    top_n: int = 100,
    background_alpha: float = 0.24,
):
    dirs = load_attn_sample_dirs(root_pattern)
    if not dirs:
        return

    dirs = sorted(dirs, key=lambda d: d.name)[:50]

    layout = Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene={
            "bgcolor": "white",
            "xaxis": {"showbackground": False, "visible": False},
            "yaxis": {"showbackground": False, "visible": False},
            "zaxis": {"showbackground": False, "visible": False},
        },
    )

    fig = go.Figure(layout=layout)
    buttons = []

    for sample_idx, sample_dir in enumerate([dirs[0]]):
        attn = np.load(sample_dir / "attn.npy")
        coords = np.load(sample_dir / "encoder_coord.npy")
        keypoints = np.load(sample_dir / "pred_kp.npy")

        title = sample_dir.name
        meta_file = sample_dir / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                parts = [title]
                if "score" in meta:
                    parts.append(f"score={meta['score']:.3f}")
                if "model" in meta:
                    parts.append(f"model={meta['model']}")
                title = "  |  ".join(parts)
            except Exception:
                pass

        sample_traces = build_sample_traces(
            attn,
            coords,
            keypoints,
            sample_title=title,
            top_n=top_n,
            background_alpha=background_alpha,
        )

        # add traces to main fig
        for tr in sample_traces:
            tr.visible = sample_idx == 0
            fig.add_trace(tr)

        # remember how many traces the first sample contributed
        if sample_idx == 0:
            len(sample_traces)

        total_traces_so_far = len(fig.data)
        traces_per_sample = len(sample_traces)

        visible_mask = [False] * total_traces_so_far
        start = sample_idx * traces_per_sample
        end = start + traces_per_sample
        for i in range(start, end):
            visible_mask[i] = True

        buttons.append(
            {
                "label": title,
                "method": "update",
                "args": [
                    {"visible": visible_mask},
                    {"title": title},
                ],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)

    # ---------- NEW: save the FIRST sample as PNG ----------
    # if first_sample_trace_count is not None and first_sample_trace_count > 0:
    #     # copy first-sample traces
    #     for i in range(first_sample_trace_count):

    #     # top-down camera

    # apply layout to the figure we're about to export
    # first_fig.update_layout(
    #         ),
    #         ),
    #         ),
    #     ),

    #     first_fig.write_image(


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Interactive attention → keypoints viewer."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument(
        "--top-n", type=int, default=100, help="Top-N attended points per keypoint."
    )
    ap.add_argument(
        "--background-alpha",
        type=float,
        default=0.24,
        help="Opacity for full background cloud.",
    )
    args = ap.parse_args()

    base = Path("geom_np") / args.category
    out_html = base.parent / f"{args.category}_attn_interactive.html"

    plot_attention_interactive_collage(
        base,
        out_html,
        top_n=args.top_n,
        background_alpha=args.background_alpha,
    )
