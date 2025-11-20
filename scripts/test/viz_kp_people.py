#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from classes import MODEL_CLASSES
from baselines.test_base import TestBase
from datasets.people_dataset import PeopleDataset
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)
import plotly.graph_objects as go


def build_transforms():
    return transforms.Compose(
        [
            GridSample(
                keys=("coord",),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            ToTensor(),
            Collect(
                keys=("coord", "grid_coord"),
                feat_keys=("coord",),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--people-dir",
        type=Path,
        default=Path("/app/ego"),
        help="Root dir for PeopleDataset.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=10,
        help="Number of dataset samples to visualize (from index 0).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="(Kept for compatibility; PeopleDataset already normalizes if desired.)",
    )
    parser.add_argument(
        "--random-rotate",
        action="store_true",
        help="Use PeopleDataset random Y rotation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("people_viz"),
        help="Where to save the HTML visualization.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2048,
        help="Max number of points to subsample per person for visualization.",
    )
    parser.add_argument(
        "--key-point",
        type=int,
        default=10,
        help="Number of keypoints the model predicts (for info only).",
    )
    args = parser.parse_args()

    # 1) model
    model_cls = MODEL_CLASSES[args.model]
    model: TestBase = model_cls()
    model.load_model(args.ckpt, args)
    model.model.eval().cuda()

    # 2) dataset
    dataset = PeopleDataset(
        root_dir=args.people_dir,
        normalize=True,
        random_rotate=args.random_rotate,
        get_keypoints=True,
    )

    num_samples = min(args.idx, len(dataset))
    if num_samples <= 0:
        raise ValueError("idx must be > 0 to visualize at least one sample.")
    print(f"[info] Visualizing first {num_samples} samples.")

    t = build_transforms()

    fig = go.Figure()
    sample_traces = []  # list of list of trace indices per sample
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    trace_idx = 0

    for i in tqdm.tqdm(range(num_samples), total=num_samples):
        pts, kpts = dataset[i]  # pts: (N, 4) [x,y,z,label]

        pcn = pts[:, :3].astype(np.float32)
        labels = pts[:, 3].astype(int)

        # merge arm/leg classes: [0,1,2,2,3,3]
        lut = np.array([0, 1, 2, 2, 3, 3])
        labels = lut[labels]

        # subsample for speed / clarity
        if pcn.shape[0] > args.max_points:
            idx_choice = np.random.choice(pcn.shape[0], args.max_points, replace=False)
            pcn = pcn[idx_choice]
            labels = labels[idx_choice]

        # prepare model batch for keypoints
        data = {"coord": pcn}
        data_t = t(data)
        batch = collate_fn([data_t])
        batch = {k: v.cuda() for k, v in batch.items()}
        batch["orig"] = np.expand_dims(pcn, axis=0)

        with torch.no_grad():
            key_points_list = model.get_keypoints(batch)
            keypoints = key_points_list[0]

        keypoints_np = (
            keypoints.detach().cpu().numpy()
            if isinstance(keypoints, torch.Tensor)
            else np.asarray(keypoints)
        )

        sample_trace_indices = []

        # Add one trace per merged class so legend is discrete
        unique_labels = np.unique(labels)
        for l in sorted(unique_labels):
            mask = labels == l
            clr = color_palette[l % len(color_palette)]
            visible = (i == 0)  # only first sample visible initially

            fig.add_trace(
                go.Scatter3d(
                    x=pcn[mask, 0],
                    y=pcn[mask, 1],
                    z=pcn[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=clr, opacity=0.6),
                    name=f"Class {l}",
                    showlegend=(i == 0),  # legend only once
                    visible=visible,
                )
            )
            sample_trace_indices.append(trace_idx)
            trace_idx += 1

        # keypoint trace
        visible = (i == 0)
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=keypoints_np[:, 0],
        #         y=keypoints_np[:, 1],
        #         z=keypoints_np[:, 2],
        #         mode="markers",
        #         marker=dict(
        #             size=6,
        #             color="red",
        #             symbol="cross",
        #             line=dict(width=1, color="black"),
        #         ),
        #         name="Keypoints",
        #         showlegend=(i == 0),
        #         visible=visible,
        #     )
        # )
        fig.add_trace(
            go.Scatter3d(
                x=keypoints_np[:, 0],
                y=keypoints_np[:, 1],
                z=keypoints_np[:, 2],
                mode="markers",
                marker=dict(
                    size=7,
                    color="red",
                    symbol="cross",
                    line=dict(width=1, color="black"),
                ),
                name="Pred Keypoints",
                showlegend=(i == 0),
                visible=(i == 0),
            )
        )
        sample_trace_indices.append(trace_idx)
        trace_idx += 1

        # -------------------------
        # GT KEYS from PeopleDataset (blue)
        # -------------------------
        # kpts_np = kpts.astype(float)   # (K, 3)
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=kpts_np[:, 0],
        #         y=kpts_np[:, 1],
        #         z=kpts_np[:, 2],
        #         mode="markers",
        #         marker=dict(
        #             size=8,
        #             color="blue",
        #             symbol="diamond",
        #             line=dict(width=1, color="black"),
        #         ),
        #         name="GT Keypoints",
        #         showlegend=(i == 0),
        #         visible=(i == 0),
        #     )
        # )
        # sample_trace_indices.append(trace_idx)
        # trace_idx += 1        
        # sample_trace_indices.append(trace_idx)
        # trace_idx += 1

        kpts_np = kpts.astype(float)
        K_gt = kpts_np.shape[0]

        # any large palette; tab20 has 20 distinct colors
        gt_cmap = plt.get_cmap("tab20")

        for j in range(K_gt):
            clr = f"rgba({int(gt_cmap(j % 20)[0] * 255)},"
            clr += f"{int(gt_cmap(j % 20)[1] * 255)},"
            clr += f"{int(gt_cmap(j % 20)[2] * 255)}, 1.0)"

            fig.add_trace(
                go.Scatter3d(
                    x=[kpts_np[j, 0]],
                    y=[kpts_np[j, 1]],
                    z=[kpts_np[j, 2]],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=clr,
                        symbol="diamond",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"GT {j}",
                    showlegend=(i == 0),
                    visible=(i == 0),
                )
            )

            sample_trace_indices.append(trace_idx)
            trace_idx += 1

        sample_traces.append(sample_trace_indices)

    # Dropdown to select which sample to show
    buttons = []
    total_traces = trace_idx

    for s, indices in enumerate(sample_traces):
        visible = [False] * total_traces
        for idx in indices:
            visible[idx] = True

        buttons.append(
            dict(
                label=f"Sample {s}",
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{args.model} | idx={s}"},
                ],
            )
        )

    fig.update_layout(
        title=f"{args.model} | idx=0",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
        margin=dict(l=0, r=0, t=50, b=0),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / "people_viz.html"
    fig.write_html(str(html_path))
    print(f"[viz] saved interactive HTML to {html_path}")


if __name__ == "__main__":
    main()
