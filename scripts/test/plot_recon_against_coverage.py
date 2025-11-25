#!/usr/bin/env python3
"""
Plot CD/EMD vs coverage (no geometry saving).

- Single mode: makes one plot for that mode.
- --input-type all: runs full, default, myopia, patch, tac and overlays them on ONE plot.
- --reuse-csv: if per-mode CSV already exists, read it and skip recomputation.

Saved outputs:
- Per-mode CSVs: metrics_vs_coverage.csv (under .../<mode>_cov/)
- Combined CSV (when --input-type all): metrics_vs_coverage_all.csv
- Figure(s):
    * Single mode: cd_emd_vs_coverage.png (under that mode's folder)
    * All modes:  cd_emd_vs_coverage_ALL.png (under .../ALL_cov/)
    
Example usage:
For the original paradigm trained:
python scripts/test/plot_recon_against_coverage.py Ours --ckpt logs/original-300/faithful-dust-6/net_final.pth --category airplane --input-type all --reuse-csv
For the partial trained:
python scripts/test/plot_recon_against_coverage.py Ours --ckpt logs/partial/deft-paper-30/net_final.pth --category airplane --input-type all --reuse-csv --output-dir recon_vs_coverage_partial-train
"""

import argparse
import contextlib
from pathlib import Path
import csv
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from classes import MODEL_CLASSES
from keypoint_diffuser.options.ae_options import AEConfig
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.utils.eval_metrics import EMD_CD_recon
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth, Collect, Deform, GridSample, ToTensor
)
from torchvision import transforms
from torch.utils.data import DataLoader

ALL_INPUT_TYPES = ["default", "myopia", "patch", "tac"]


def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        p.add_argument(*names, **kwargs)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute CD/EMD vs coverage (no geometry saving)."
    )
    subparsers = p.add_subparsers(dest="model", required=True)

    for model_name, model_cls in MODEL_CLASSES.items():
        sp = subparsers.add_parser(model_name)
        model_cls.get_parser(sp)

        try_add_arg(sp, "--ckpt", type=Path, required=True)
        try_add_arg(sp, "--category", type=str, default="chair")
        try_add_arg(sp, "--batch-size", type=int, default=16)
        try_add_arg(sp, "--num-workers", type=int, default=8)
        try_add_arg(sp, "--key-points", type=int, default=10)
        try_add_arg(sp, "--output-dir", type=Path, default=Path("recon_vs_coverage"))
        try_add_arg(
            sp,
            "--input-type",
            type=str,
            default="full",
            choices=["full", "default", "myopia", "patch", "tac", "all"],
            help='Choose one mode or "all" to run all modes.',
        )
        try_add_arg(
            sp,
            "--reuse-csv",
            action="store_true",
            help="If per-mode CSV exists, read it and skip recomputation."
        )
    return p


def _config_from_namespace(ns: argparse.Namespace, partial_view_mode: str) -> AEConfig:
    """Lift argparse namespace to AEConfig and inject dataset fields."""
    from dataclasses import fields, MISSING
    cfg_kwargs = {}
    for f in fields(AEConfig):
        name = f.name
        val = getattr(ns, name, MISSING)
        if val is MISSING:
            if f.default is not MISSING:
                val = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:
                val = f.default_factory()  # type: ignore[misc]
            else:
                continue
        if f.type is Path and isinstance(val, str):
            val = Path(val)
        cfg_kwargs[name] = val

    if cfg_kwargs.get("normalization") == "none":
        cfg_kwargs["normalization"] = None

    cfg_kwargs.update(
        dict(
            name="recon",
            split="test",
            mesh_dir="/mnt/slow/shapenetcorev2-source/",
            points_dir="/mnt/slow/shapenetcorev2-source/",
            split_file="./data/shapenet_split/splits_out.csv",
            n_partial_samples=2,
            category=ns.category,
            partial_view_mode=partial_view_mode if partial_view_mode != "full" else "default",
        )
    )
    return AEConfig(**cfg_kwargs)


def make_loader(opt: argparse.Namespace, mode: str) -> DataLoader:
    t = (
        transforms.Compose([
            Deform(),
            ApplyToBoth(
                transforms.Compose([
                    GridSample(keys=("coord",), hash_type="fnv", mode="train", return_grid_coord=True),
                    ToTensor(),
                    Collect(keys=("coord", "grid_coord", "transformation", "shape"), feat_keys=("coord",)),
                ])
            ),
        ])
        if (opt.model == "Ours" or opt.model == "Partial")
        else None
    )
    cfg = _config_from_namespace(opt, mode)
    dataset = ShapesPartial(cfg, transform=t)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_fn if (opt.model == "Ours" or opt.model == "Partial") else None,
        drop_last=False,
    )


def mode_out_dir(opt, mode) -> Path:
    return (opt.output_dir / opt.model / opt.category / f"{mode}_cov").resolve()


def mode_csv_path(opt, mode) -> Path:
    return mode_out_dir(opt, mode) / "metrics_vs_coverage.csv"


def read_mode_csv(csv_path: Path):
    coverage, cd, emd = [], [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            coverage.append(float(row["coverage"]))
            cd.append(float(row["cd"]))
            emd.append(float(row["emd"]))
    return {
        "coverage": np.asarray(coverage, dtype=float),
        "cd": np.asarray(cd, dtype=float),
        "emd": np.asarray(emd, dtype=float),
    }


def run_one_mode(model, opt: argparse.Namespace, mode: str):
    """Compute one mode, write per-mode CSV, and return arrays."""
    key = "orig" if mode == "full" else "partial_orig"
    loader = make_loader(opt, mode)

    coverages, cds, emds, names, patsams = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"[{mode}] reconstruct & score vs coverage"):
            recon_batch, input_pc, full_pc, _kp = model.get_reconstruction(batch, key=key)

            if recon_batch.dim() == 4 and recon_batch.shape[1] == 1:
                recon_batch = recon_batch.squeeze(1)

            res = EMD_CD_recon(recon_batch.float(), full_pc.float(), reduced=False)
            cd_all = res["CD"].detach().cpu().numpy()
            emd_all = res["EMD"].detach().cpu().numpy()

            cov = batch.get("target_coverage", None)
            if cov is None:
                raise KeyError("Batch is missing 'target_coverage'")
            if torch.is_tensor(cov):
                cov = cov.detach().cpu().numpy()
            cov = np.asarray(cov).reshape(-1)

            name_batch = batch.get("target_file", ["<unknown>"] * len(cd_all))
            patsam_batch = batch.get("target_partial_sample_name", ["<unknown>"] * len(cd_all))

            coverages.extend(list(cov))
            cds.extend(list(cd_all))
            emds.extend(list(emd_all))
            names.extend(list(name_batch))
            patsams.extend(list(patsam_batch))

    out_dir = mode_out_dir(opt, mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_vs_coverage.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "partial_sample_name", "coverage", "cd", "emd"])
        for n, p, c, cdv, emdv in zip(names, patsams, coverages, cds, emds):
            w.writerow([n, p, float(c), float(cdv), float(emdv)])

    print(f"[{mode}] ✓ CSV: {csv_path}")

    return {
        "mode": mode,
        "coverage": np.asarray(coverages, dtype=float),
        "cd": np.asarray(cds, dtype=float),
        "emd": np.asarray(emds, dtype=float),
    }


def plot_single_mode(coverage, cd, emd, title_suffix, out_dir: Path):
    fig = plt.figure(figsize=(9, 4.8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(coverage, cd, s=8, alpha=0.7)
    ax1.yscale("log")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Chamfer Distance (CD)")
    ax1.set_title(f"CD vs Coverage ({title_suffix})")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(coverage, emd, s=8, alpha=0.7)
    ax2.set_xlabel("Coverage")
    ax2.set_ylabel("EMD (Sinkhorn)")
    ax2.set_title(f"EMD vs Coverage ({title_suffix})")

    plt.tight_layout()
    fig_path = out_dir / "cd_emd_vs_coverage.png"
    plt.savefig(fig_path, dpi=180)
    print(f"[{title_suffix}] ✓ FIG: {fig_path}")


def plot_overlay(modes_data, out_dir: Path):
    """Overlay all modes on one figure (two panels)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # # separate modes display ############################################################
    # for d in modes_data:
    #     label = d["mode"]
    #     ax1.scatter(d["coverage"], d["cd"], s=8, alpha=0.7, label=label)
    #     ax2.scatter(d["coverage"], d["emd"], s=8, alpha=0.7, label=label)
    # # separate modes display ############################################################
    # no modes display ##################################################################
    for d in modes_data:
        ax1.scatter(d["coverage"], d["cd"], s=8, alpha=0.7, color="steelblue")
        ax2.scatter(d["coverage"], d["emd"], s=8, alpha=0.7, color="steelblue")
    # no modes display ##################################################################


    ax1.set_yscale("log")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Chamfer Distance (CD)")
    ax1.set_title("CD vs Coverage (ALL)")
    # ax1.legend(markerscale=2, frameon=True)

    ax2.set_xlabel("Coverage")
    ax2.set_ylabel("EMD (Sinkhorn)")
    ax2.set_title("EMD vs Coverage (ALL)")
    # ax2.legend(markerscale=2, frameon=True)

    plt.tight_layout()
    fig_path = out_dir / "cd_emd_vs_coverage_ALL.png"
    plt.savefig(fig_path, dpi=200)
    print(f"[ALL] ✓ FIG: {fig_path}")


def write_combined_csv(modes_data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "coverage", "cd", "emd"])
        for d in modes_data:
            mode = d["mode"]
            for c, cdv, emdv in zip(d["coverage"], d["cd"], d["emd"]):
                w.writerow([mode, float(c), float(cdv), float(emdv)])
    print(f"[ALL] ✓ CSV: {out_path}")


def main():
    p = build_argparser()
    opt = p.parse_args()

    # Determine which modes to run
    if opt.input_type == "all":
        modes = ALL_INPUT_TYPES
    else:
        modes = [opt.input_type]

    modes_data = []
    modes_to_compute = []

    # First pass: try to read CSVs if --reuse-csv
    if opt.reuse_csv:
        for mode in modes:
            csv_path = mode_csv_path(opt, mode)
            if csv_path.exists():
                d = read_mode_csv(csv_path)
                d["mode"] = mode
                modes_data.append(d)
                print(f"[{mode}] ✓ Reused CSV: {csv_path}")
            else:
                modes_to_compute.append(mode)
                print(f"[{mode}] CSV not found → will compute.")
    else:
        modes_to_compute = modes

    # Compute missing modes (if any)
    if len(modes_to_compute) > 0:
        # Load model only if needed
        model_cls = MODEL_CLASSES[opt.model]
        model = model_cls()
        model.load_model(opt.ckpt, opt)
        model.model.eval()
        model.model.cuda()

        for mode in modes_to_compute:
            d = run_one_mode(model, opt, mode)
            modes_data.append(d)

    # Plot & write combined when needed
    if opt.input_type == "all":
        # Ensure mode labels present on reused-only paths
        modes_data = [{**d, "mode": d.get("mode", m)} for d, m in zip(modes_data, [d["mode"] for d in modes_data])]
        all_dir = (opt.output_dir / opt.model / opt.category / "ALL_cov").resolve()
        write_combined_csv(modes_data, all_dir / "metrics_vs_coverage_all.csv")
        plot_overlay(modes_data, all_dir)
    else:
        # Single mode plot
        d = modes_data[0]
        out_dir = mode_out_dir(opt, modes[0])
        plot_single_mode(d["coverage"], d["cd"], d["emd"], modes[0], out_dir)


if __name__ == "__main__":
    main()
