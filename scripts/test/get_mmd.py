#!/usr/bin/env python3
"""
Generative-distribution evaluation (MMD / COV / 1-NNA) on the *source sampled* data.

Same idea as get_mmd_h5.py - fit a distribution over the encoder's keypoint latents,
sample new latents from it, decode them into shapes, and score that generated set
against a reference set of real shapes - but built on the ShapesPartial loader used
by get_reconstruction.py instead of the H5 test set. That buys three things the H5
version cannot do:

  * input modes: the latent distribution can be fit from the full clouds
    (--input-type full) or from any partial view profile (default / myopia / patch /
    tac / uniform / all), so you can ask whether a model that only ever sees a
    partial view still induces the same shape distribution.
  * real object identities (target_file), partial view names and coverage, so every
    generated / reference shape is traceable back to a model id on disk.
  * visualisation: an interactive HTML viewer of each generated shape next to its
    nearest reference, plus the distance histograms behind MMD and COV.

Pipeline
  1. encode the fit split under the chosen input mode      -> keypoint latents + z_aux
  2. fit PCA over vec(K), then a Gaussian / KDE in PCA space
  3. sample latents, pair them with z_aux, decode           -> generated point clouds
  4. pairwise CD (and optionally EMD) against the reference split
  5. MMD, COV and optionally 1-NNA; save npys, HTML, plots and a SQLite row

Metric conventions follow PointFlow (as does keypoint_diffuser.utils.eval_metrics):
  MMD  - mean over references of the distance to the nearest generated shape (lower better)
  COV  - fraction of references that are some generated shape's nearest neighbour (higher better)
  1-NNA- leave-one-out 1-NN classifier accuracy over gen+ref; 0.5 is perfect (needs --nna)

Run:
  python scripts/test/get_mmd.py Ours -t configs/test.yaml \\
      --ckpt logs/foo/checkpoints/net_final.pth \\
      --category airplane --input-type full \\
      --fit-split train --ref-split val \\
      --num-gen 256 --pca-dim 16 \\
      --viz --db-path db/test_db/mmd.db
"""

import argparse
import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.loss
import torch
import tqdm

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from classes import MODEL_CLASSES
from keypoint_diffuser.utils.eval_metrics import emd_approx

# shared with the reconstruction script: the loader wiring (ShapesPartial + the
# GridSample/Collect transform stack) and the viewer's colour conventions, so a
# keypoint slot has the same colour in both viewers
from get_reconstruction import (
    PARTIAL_VIEW_MODES,
    build_partial_transform,
    config_from_namespace,
    try_add_arg,
)
from viz_utils import slot_color, subsample
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.utils.pc_utils import collate_fn
from torch.utils.data import DataLoader


# ----------------------------
# CLI
# ----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit a latent distribution, generate shapes and score MMD / COV / 1-NNA "
                    "against the sampled source data."
    )

    subparsers = p.add_subparsers(dest="model", required=True)

    for model_name, model_cls in MODEL_CLASSES.items():
        # baselines whose import failed are unusable here
        if model_cls is None:
            continue
        sp = subparsers.add_parser(model_name)
        model_cls.get_parser(sp)
        try_add_arg(sp, "--ckpt", type=Path)
        try_add_arg(sp, "--category", type=str, default="chair")
        try_add_arg(sp, "--batch-size", type=int, default=16)
        try_add_arg(sp, "--num-workers", type=int, default=0)
        try_add_arg(sp, "--db-path", type=Path, default=Path("results.db"))
        try_add_arg(sp, "--key-points", type=int, default=10)
        try_add_arg(sp, "--output-dir", type=Path, default=Path("mmd_out"))

        # --- what gets encoded -------------------------------------------------
        try_add_arg(sp, "--input-type", type=str, default="full",
                    choices=["full", *PARTIAL_VIEW_MODES, "all"],
                    help="Cloud the latent distribution is fit from. `full` encodes the "
                         "complete surface cloud, anything else encodes that partial view profile")
        try_add_arg(sp, "--enumerate-partial-samples", action="store_true",
                    help="Fit over every partial sample on disk instead of one random view "
                         "per object. Dataset length becomes the number of (object, view) pairs")
        try_add_arg(sp, "--fit-split", type=str, default="train",
                    choices=["train", "val", "test"],
                    help="Split the latent distribution is fit on")
        try_add_arg(sp, "--ref-split", type=str, default="val",
                    choices=["train", "val", "test"],
                    help="Split providing the reference shapes the generated set is scored against")
        try_add_arg(sp, "--max-fit-samples", type=int, default=None,
                    help="Cap on shapes encoded when fitting the latent distribution")
        try_add_arg(sp, "--max-ref-samples", type=int, default=None,
                    help="Cap on reference shapes (the pairwise matrices are O(gen x ref))")

        # --- the generator -----------------------------------------------------
        try_add_arg(sp, "--num-gen", type=int, default=None,
                    help="Number of shapes to generate. Defaults to the reference set size")
        try_add_arg(sp, "--pca-dim", type=int, default=16,
                    help="PCA dimensions kept over vec(K) before density fitting")
        try_add_arg(sp, "--sampler", type=str, default="gaussian", choices=["gaussian", "kde"],
                    help="`gaussian`: one diagonal Gaussian over the PCA scores. `kde`: pick a "
                         "training latent and jitter it (a real kernel density estimate)")
        try_add_arg(sp, "--kde-bandwidth", type=float, default=1.0,
                    help="Scale on the per-dimension std used by either sampler")
        try_add_arg(sp, "--zaux-mode", type=str, default="pool", choices=["pool", "prior", "mean"],
                    help="Where z_aux comes from. `pool`: resampled from the encoded latents, "
                         "`prior`: N(0, I) as the KL term trains it to be, `mean`: the pool mean")
        try_add_arg(sp, "--scale-to-keypoints", action="store_true",
                    help="Rescale each decoded cloud so its spread matches its conditioning "
                         "keypoints (the correction get_reconstruction_h5.py applies)")

        # --- metrics -----------------------------------------------------------
        try_add_arg(sp, "--emd", action="store_true",
                    help="Also compute MMD-EMD / COV-EMD. The EMD matrix is far slower than CD")
        try_add_arg(sp, "--nna", action="store_true",
                    help="Also compute 1-NNA, which needs the gen-gen and ref-ref matrices too")
        try_add_arg(sp, "--pair-batch", type=int, default=32,
                    help="Pairs scored per chamfer/EMD call when filling the distance matrices")

        # --- outputs -----------------------------------------------------------
        try_add_arg(sp, "--no-save", action="store_true",
                    help="If set, do not write the generated clouds to disk (only metrics)")
        try_add_arg(sp, "--viz", action="store_true",
                    help="Write an interactive HTML viewer (generated cloud + nearest reference "
                         "+ sampled keypoints) and the MMD/COV distance histograms")
        try_add_arg(sp, "--viz-out", type=Path, default=None,
                    help="Path of the HTML viewer. Defaults to "
                         "<output-dir>/<model>/<category>_mmd_interactive.html")
        try_add_arg(sp, "--viz-max-samples", type=int, default=100,
                    help="Cap on generated shapes embedded in the HTML viewer")
        try_add_arg(sp, "--viz-cloud-points", type=int, default=1500,
                    help="Points kept per cloud for display (stride-subsampled)")

    return p


# ----------------------------
# Dataset
# ----------------------------


def make_loader(opt: argparse.Namespace, split: str) -> DataLoader:
    """
    ShapesPartial over one split, wired exactly like get_reconstruction.make_loader.

    `split` overrides the split config_from_namespace pins, so the fit and reference
    passes can read different halves of the same split file.
    """
    t = build_partial_transform(opt)
    configs_opt = config_from_namespace(opt)
    configs_opt.split = split

    dataset = ShapesPartial(configs_opt, transform=t)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn if opt.model in ("Ours", "Partial") else None,
        drop_last=False,
    )


# ----------------------------
# Latent collection
# ----------------------------


def collect_latents(model, loader, opt, key: str, max_samples=None, collect_clouds=True):
    """
    Encode the split and keep the structured latent of every shape.

    Returns vec(K) as (M, 3*key_points), the posterior mean of z_aux as (M, m), the
    full ground-truth clouds, and the identity of each shape. z_aux is taken as mu
    rather than a reparameterised draw: the pool is a distribution estimate, so the
    sampling noise belongs at generation time, not here.
    """
    if not hasattr(model, "get_network_data"):
        raise RuntimeError(f"model {opt.model} has no get_network_data; undefined model case")

    kps, zauxs, clouds, names, psns, covs = [], [], [], [], [], []

    model.model.eval()
    model.model.cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"Encode ({key})"):
            B = len(batch["target_file"])

            z_kp, mu, _logvar = model.model.encode(model.get_network_data(batch, key=key))
            kps.append(z_kp.reshape(B, -1).detach().cpu())
            zauxs.append(mu.reshape(B, -1).detach().cpu())

            names.extend(batch["target_file"])
            psns.extend(batch["target_partial_sample_name"])
            covs.extend([None if c is None else float(c) for c in batch["target_coverage"]])
            if collect_clouds:
                clouds.append(batch["target_shape"].reshape(B, -1, 3).float().cpu())

            if max_samples is not None and len(names) >= max_samples:
                break

    kps = torch.cat(kps, dim=0)
    zauxs = torch.cat(zauxs, dim=0)
    clouds = torch.cat(clouds, dim=0) if collect_clouds else None

    if max_samples is not None:
        kps, zauxs = kps[:max_samples], zauxs[:max_samples]
        names, psns, covs = names[:max_samples], psns[:max_samples], covs[:max_samples]
        if clouds is not None:
            clouds = clouds[:max_samples]

    return kps, zauxs, clouds, names, psns, covs


def collect_reference_clouds(loader, max_samples=None):
    """Full ground-truth clouds of the reference split - no forward pass needed."""
    clouds, names = [], []
    for batch in tqdm.tqdm(loader, desc="Load reference clouds"):
        B = len(batch["target_file"])
        clouds.append(batch["target_shape"].reshape(B, -1, 3).float().cpu())
        names.extend(batch["target_file"])
        if max_samples is not None and len(names) >= max_samples:
            break

    clouds = torch.cat(clouds, dim=0)
    if max_samples is not None:
        clouds, names = clouds[:max_samples], names[:max_samples]
    return clouds, names


# ----------------------------
# PCA + density fitting
# ----------------------------


def fit_pca(x: torch.Tensor, pca_dim: int):
    """PCA over the flattened keypoint latents. Returns the mean, the components and the scores."""
    x_np = np.asarray(x.numpy(), dtype=np.float64)
    pca_dim = int(min(pca_dim, x_np.shape[0], x_np.shape[1]))
    mu = x_np.mean(axis=0, keepdims=True)
    x_centered = x_np - mu
    _U, _S, Vt = np.linalg.svd(x_centered, full_matrices=False)
    comps = Vt[:pca_dim]
    z_pca = x_centered @ comps.T
    return mu, comps, z_pca


def fit_density(z_pca: np.ndarray, bw_scale: float = 1.0):
    """Per-dimension mean and (scaled) std of the PCA scores."""
    mu = z_pca.mean(axis=0, keepdims=True)
    std = z_pca.std(axis=0, keepdims=True) + 1e-6
    return mu, std * bw_scale


def sample_latents(num, pca_mu, pca_comps, z_pca, dens_mu, dens_std, sampler="gaussian", rng=None):
    """
    Draw `num` keypoint latents and lift them back out of PCA space.

    `gaussian` fits one diagonal Gaussian to the scores - smooth, but it fills the
    space between distinct shape modes. `kde` jitters an actual training latent, so
    every sample stays near a real shape; the bandwidth controls how far.
    """
    rng = rng or np.random
    if sampler == "gaussian":
        z = dens_mu + dens_std * rng.randn(num, dens_mu.shape[1])
    elif sampler == "kde":
        idx = rng.randint(0, z_pca.shape[0], size=num)
        z = z_pca[idx] + dens_std * rng.randn(num, dens_mu.shape[1])
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    return pca_mu + z @ pca_comps


def sample_zaux(num, zaux_pool: torch.Tensor, mode="pool", rng=None):
    """z_aux for the generated shapes: resampled from the encoded pool, from the prior, or fixed."""
    rng = rng or np.random
    m = zaux_pool.shape[1]
    if mode == "pool":
        idx = rng.randint(0, zaux_pool.shape[0], size=num)
        return zaux_pool[torch.from_numpy(idx)].clone()
    if mode == "prior":
        return torch.from_numpy(rng.randn(num, m)).float()
    if mode == "mean":
        return zaux_pool.mean(dim=0, keepdim=True).expand(num, -1).clone()
    raise ValueError(f"unknown zaux mode: {mode}")


# ----------------------------
# Generation
# ----------------------------


def covariance_scale_to_keypoints(points, keypoints, eps=1e-8):
    """
    points:    (B, N, 3)
    keypoints: (B, M, 3)
    returns:   (B, N, 3), (B, 1)
    """
    keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
    p_mean = points.mean(dim=1, keepdim=True)
    k_mean = keypoints.mean(dim=1, keepdim=True)
    p_c = points - p_mean
    k_c = keypoints - k_mean

    p_var = (p_c**2).sum(dim=-1).mean(dim=1)  # (B,)
    k_var = (k_c**2).sum(dim=-1).mean(dim=1)  # (B,)

    s = torch.sqrt((k_var + eps) / (p_var + eps)).view(-1, 1, 1)  # (B,1,1)
    return p_mean + s * p_c, s


def generate_clouds(model, kp_samples, zaux_samples, opt, num_points=2048):
    """Decode sampled latents in batches. Returns (num_gen, num_points, 3) on the cpu."""
    outs = []
    model.model.eval()
    model.model.cuda()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, kp_samples.shape[0], opt.batch_size), desc="Generate"):
            kp = kp_samples[i : i + opt.batch_size].cuda().float()
            za = zaux_samples[i : i + opt.batch_size].cuda().float()
            code = torch.cat([kp.reshape(kp.shape[0], -1), za], dim=1)
            pc = model.model.decode(code, num_points).detach().float()
            if opt.scale_to_keypoints:
                pc = covariance_scale_to_keypoints(pc, kp)[0]
            outs.append(pc.cpu())
    return torch.cat(outs, dim=0)


# ----------------------------
# Distance matrices & metrics
# ----------------------------


def _cd_row(one, others):
    """CD from a single cloud to a stack of clouds, same reduction as EMD_CD_recon."""
    a = one.expand(others.shape[0], -1, -1).contiguous()
    cd, _ = pytorch3d.loss.chamfer_distance(
        a, others, batch_reduction=None, point_reduction="mean"
    )
    return cd


def _emd_row(one, others):
    a = one.expand(others.shape[0], -1, -1).contiguous()
    return emd_approx(a, others)


def pairwise_distances(a, b, metric="cd", pair_batch=32, desc="dist"):
    """
    Full (Na, Nb) distance matrix, filled a row at a time so peak memory stays flat.

    Everything runs on the gpu; the matrix itself comes back on the cpu because the
    metrics below (topk over gen+ref, unique over argmins) are cheap there.
    """
    row_fn = _cd_row if metric == "cd" else _emd_row
    Na, Nb = a.shape[0], b.shape[0]
    out = torch.zeros(Na, Nb)
    b = b.cuda()
    for i in tqdm.tqdm(range(Na), desc=f"{desc} ({metric.upper()})"):
        one = a[i : i + 1].cuda()
        chunks = []
        for j in range(0, Nb, pair_batch):
            chunks.append(row_fn(one, b[j : j + pair_batch]).detach().cpu())
        out[i] = torch.cat(chunks)
    return out


def mmd_cov(M_gen_ref: torch.Tensor):
    """
    PointFlow's MMD / COV from a (num_gen, num_ref) matrix.

    MMD walks the references (each reference's nearest generated shape), COV counts
    how many distinct references are claimed as some generated shape's nearest, so
    the pair catches the two failure modes: bad samples and mode collapse.
    """
    min_from_gen, min_idx = M_gen_ref.min(dim=1)  # per generated shape: nearest ref
    min_from_ref, _ = M_gen_ref.min(dim=0)  # per reference: nearest generated
    return {
        "mmd": float(min_from_ref.mean()),
        "mmd_from_gen": float(min_from_gen.mean()),
        "cov": float(min_idx.unique().numel()) / float(M_gen_ref.shape[1]),
        "per_gen_min": min_from_gen.numpy(),
        "per_gen_argmin": min_idx.numpy(),
        "per_ref_min": min_from_ref.numpy(),
    }


def one_nna(M_rr: torch.Tensor, M_rg: torch.Tensor, M_gg: torch.Tensor, k: int = 1) -> float:
    """
    Leave-one-out 1-NN classifier accuracy over the reference and generated sets.

    0.5 means the two sets are indistinguishable; 1.0 means every shape's nearest
    neighbour comes from its own set, i.e. the generated shapes are off-distribution.
    """
    n_r, n_g = M_rr.shape[0], M_gg.shape[0]
    label = torch.cat([torch.ones(n_r), torch.zeros(n_g)])
    M = torch.cat(
        [torch.cat((M_rr, M_rg), dim=1), torch.cat((M_rg.t(), M_gg), dim=1)], dim=0
    )
    # a shape is never its own neighbour
    M = M + torch.diag(torch.full((n_r + n_g,), float("inf")))
    _val, idx = M.topk(k, dim=0, largest=False)
    count = torch.zeros(n_r + n_g)
    for i in range(k):
        count += label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n_r + n_g)).float()
    return float(torch.eq(label, pred).float().mean())


# ----------------------------
# Saving
# ----------------------------


def save_gen_geoms(gen, kps, matches, ref_names, dists, opt, out_dir: Path):
    """
    One directory per generated shape:
      gen.npy     : the decoded cloud
      gen_kp.npy  : the sampled keypoints it was conditioned on
      meta.json   : nearest reference and its distance
    """
    out_dir = out_dir / opt.model / opt.category / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (pc, kp, m, d) in enumerate(zip(gen, kps, matches, dists)):
        dst = out_dir / f"gen_{i:05d}"
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "gen.npy", np.asarray(pc, dtype=np.float32))
        np.save(dst / "gen_kp.npy", np.asarray(kp, dtype=np.float32).reshape(-1, 3))
        meta = {
            "model": opt.model,
            "category": getattr(opt, "category", None),
            "input_type": opt.input_type,
            "sampler": opt.sampler,
            "zaux_mode": opt.zaux_mode,
            "nearest_ref": ref_names[int(m)] if ref_names else None,
            "nearest_ref_cd": float(d),
        }
        with open(dst / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"[✓] Saved {len(gen)} generated shapes under {out_dir}/")


# ----------------------------
# Visualization
# ----------------------------


# The viewer primitives are shared with the reconstruction scripts; only the
# metric tables below are specific to a generative run - MMD / COV / 1-NNA rather
# than CD / EMD / KVC, and a generated shape is titled by the reference it claims
# instead of by a partial view.
from viz_utils import (  # noqa: E402
    as_xyz as _as_xyz,
    format_run_metrics,
    format_sample_title,
    legend_on_surface,
    style_axis,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

RUN_METRIC_FIELDS = (
    ("model", "{}"),
    ("category", "{}"),
    ("input_type", "input={}"),
    ("fit_split", "fit={}"),
    ("ref_split", "ref={}"),
    ("num_gen", "gen={}"),
    ("num_ref", "ref N={}"),
    ("coverage", "view coverage={:.4f}"),
    ("pca_dim", "PCA={}"),
    ("mmd_cd", "MMD-CD={:.4e}"),
    ("cov_cd", "COV-CD={:.4f}"),
    ("mmd_emd", "MMD-EMD={:.4e}"),
    ("cov_emd", "COV-EMD={:.4f}"),
    ("nna_cd", "1-NNA-CD={:.4f}"),
    ("nna_emd", "1-NNA-EMD={:.4f}"),
)
SAMPLE_METRIC_FIELDS = (
    ("cd", "CD={:.4e}"),
    ("emd", "EMD={:.4e}"),
    ("claimed_refs", "claims {:.0f} refs"),
)
SAMPLE_PREFIX_FIELDS = (("nearest_ref", "nearest ref: {}"),)


def _format_run_metrics(run_metrics: dict) -> str:
    return format_run_metrics(run_metrics, RUN_METRIC_FIELDS)


def _format_sample_title(name: str, meta: dict) -> str:
    return format_sample_title(name, meta, SAMPLE_METRIC_FIELDS, SAMPLE_PREFIX_FIELDS)


def write_mmd_html(entries, out_path: Path, run_metrics: dict = None):
    """
    Interactive viewer: one dropdown entry per generated shape.

    Each entry draws the decoded cloud, the nearest reference shape under CD (the
    pair MMD is actually averaging over), and the sampled keypoints the decoder was
    conditioned on - slot-coloured and direct-labeled, matching get_reconstruction's
    viewer so a slot means the same thing in both. The reference is what tells you
    whether a low CD came from a genuinely matching shape or from a blob that sits
    close to everything.

    `entries` is a list of dicts:
      name       : str, shown in the dropdown
      gen_cloud  : (N,3) decoded cloud
      gen_kp     : (K,3) sampled keypoints, or None
      ref_cloud  : (N,3) nearest reference cloud, or None
      meta       : per-sample metrics for the title
    """
    if go is None:
        print("[!] plotly not installed - skipping HTML viewer.")
        return None
    if not entries:
        print("[!] No samples to visualize.")
        return None

    traces_per_entry = 3
    fig = go.Figure()
    run_line = _format_run_metrics(run_metrics or {})
    buttons = []

    for ei, entry in enumerate(entries):
        first = ei == 0

        # --- generated cloud ----------------------------------------------------
        gen = _as_xyz(entry.get("gen_cloud"))
        fig.add_trace(
            go.Scatter3d(
                x=gen[:, 0], y=gen[:, 1], z=gen[:, 2],
                mode="markers",
                marker={"size": 2.0, "color": "#e34948", "opacity": 0.85},
                name="generated",
                visible=first,
                hoverinfo="skip",
            )
        )

        # --- nearest reference cloud -------------------------------------------
        ref = _as_xyz(entry.get("ref_cloud"))
        ref_name = (entry.get("meta") or {}).get("nearest_ref", "")
        fig.add_trace(
            go.Scatter3d(
                x=ref[:, 0], y=ref[:, 1], z=ref[:, 2],
                mode="markers",
                marker={"size": 1.6, "color": "#c8c7c0"},
                name=f"nearest ref {ref_name}".strip(),
                visible=first,
                hoverinfo="skip",
            )
        )

        # --- the keypoints the decoder was conditioned on ------------------------
        kp = _as_xyz(entry.get("gen_kp"))
        fig.add_trace(
            go.Scatter3d(
                x=kp[:, 0], y=kp[:, 1], z=kp[:, 2],
                mode="markers+text",
                marker={
                    "size": 6,
                    "color": [slot_color(k) for k in range(kp.shape[0])],
                    "symbol": "circle",
                    "line": {"width": 1, "color": "#0b0b0b"},
                },
                text=[str(k) for k in range(kp.shape[0])],
                textposition="top center",
                textfont={"size": 11, "color": "#0b0b0b"},
                hovertext=[f"sampled kp {k}" for k in range(kp.shape[0])],
                hoverinfo="text",
                name="sampled keypoints",
                visible=first,
            )
        )

        visible = [False] * (traces_per_entry * len(entries))
        for t in range(traces_per_entry):
            visible[ei * traces_per_entry + t] = True

        entry_line = _format_sample_title(entry.get("name", f"gen_{ei}"), entry.get("meta"))
        buttons.append(
            {
                "label": entry_line,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{run_line}<br>{entry_line}" if run_line else entry_line},
                ],
            }
        )

    first_line = _format_sample_title(entries[0].get("name", "gen_0"), entries[0].get("meta"))
    fig.update_layout(
        title=f"{run_line}<br>{first_line}" if run_line else first_line,
        template="plotly_white",
        paper_bgcolor="#fcfcfb",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "bgcolor": "#fcfcfb",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"[✓] Saved interactive HTML to {out_path} ({len(entries)} generated shapes)")
    return out_path


def plot_distance_distributions(stats: dict, out_path: Path, metric: str = "CD"):
    """
    The two distributions the scalars hide.

    Left: per-reference nearest-generated distance, whose mean is MMD - a long right
    tail means some references have nothing like them in the generated set.
    Right: how many generated shapes claim each reference as their nearest, whose
    support is COV - a spike at zero is mode collapse.
    """
    per_ref = np.asarray(stats["per_ref_min"], dtype=np.float64)
    argmin = np.asarray(stats["per_gen_argmin"])
    n_ref = per_ref.shape[0]
    claims = np.bincount(argmin, minlength=n_ref)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(per_ref, bins=40, color="#2a78d6", alpha=0.85)
    axes[0].axvline(per_ref.mean(), color="#e34948", linestyle="--",
                    label=f"MMD-{metric}={per_ref.mean():.4e}")
    axes[0].set_title(f"Per-reference nearest generated ({metric})")
    axes[0].set_xlabel(metric)
    axes[0].set_ylabel("references")
    axes[0].legend()

    axes[1].hist(claims, bins=np.arange(claims.max() + 2) - 0.5, color="#1baf7a", alpha=0.85)
    cov = float((claims > 0).sum()) / float(n_ref)
    axes[1].axvline(0.5, color="#e34948", linestyle="--", label=f"COV-{metric}={cov:.4f}")
    axes[1].set_title("Generated shapes claiming each reference")
    axes[1].set_xlabel("times claimed")
    axes[1].set_ylabel("references")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[✓] Saved distance distributions to {out_path}")
    return out_path


# ----------------------------
# DB
# ----------------------------


def init_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mmd_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            ckpt TEXT,
            category TEXT,
            input_type TEXT,
            fit_split TEXT,
            ref_split TEXT,
            sampler TEXT,
            zaux_mode TEXT,
            pca_dim INTEGER,
            num_gen INTEGER,
            num_ref INTEGER,
            coverage REAL,
            mmd_cd REAL,
            cov_cd REAL,
            nna_cd REAL,
            mmd_emd REAL,
            cov_emd REAL,
            nna_emd REAL,
            timestamp TEXT
        )
        """
    )
    con.commit()
    return con


def save_run(db_path: Path, opt: argparse.Namespace, m: dict) -> int:
    con = init_db(db_path)
    cur = con.cursor()

    def _f(key):
        v = m.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return float(v)

    cur.execute(
        """
        INSERT INTO mmd_results
            (model, ckpt, category, input_type, fit_split, ref_split, sampler, zaux_mode,
             pca_dim, num_gen, num_ref, coverage, mmd_cd, cov_cd, nna_cd,
             mmd_emd, cov_emd, nna_emd, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            opt.model,
            str(opt.ckpt) if opt.ckpt is not None else None,
            str(opt.category),
            opt.input_type,
            opt.fit_split,
            opt.ref_split,
            opt.sampler,
            opt.zaux_mode,
            int(opt.pca_dim),
            int(m["num_gen"]),
            int(m["num_ref"]),
            _f("coverage"),
            _f("mmd_cd"),
            _f("cov_cd"),
            _f("nna_cd"),
            _f("mmd_emd"),
            _f("cov_emd"),
            _f("nna_emd"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()
    run_id = cur.lastrowid
    con.close()
    return run_id


# ----------------------------
# Main
# ----------------------------


def main():
    p = build_argparser()
    opt = p.parse_args()

    seed = int(getattr(opt, "seed", 0) or 0)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)

    # the partial view profile the loader hands out, and which cloud gets encoded.
    # 'full' still loads a partial cloud (ShapesPartial always does) - it is just
    # never read, because the encoder is pointed at the "orig" subset
    if opt.input_type == "full":
        opt.partial_view_mode = "full"
        key = "orig"
    else:
        opt.partial_view_mode = opt.input_type
        key = "orig_partial"

    print(f"[✓] MMD evaluation for model={opt.model}, category={opt.category}, "
          f"input={opt.input_type}")
    print(f"[✓] fit split={opt.fit_split}, reference split={opt.ref_split}")
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    out_root = opt.output_dir / opt.model / opt.category
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) encode the fit split under the chosen input mode
    fit_loader = make_loader(opt, opt.fit_split)
    same_split = opt.fit_split == opt.ref_split
    kp_pool, zaux_pool, fit_clouds, _fit_names, _fit_psns, fit_covs = collect_latents(
        model, fit_loader, opt, key=key,
        max_samples=opt.max_fit_samples,
        collect_clouds=same_split,
    )
    valid_covs = [c for c in fit_covs if c is not None]
    cov_mean = float(np.mean(valid_covs)) if valid_covs else None
    print(f"[✓] encoded {kp_pool.shape[0]} shapes -> vec(K) dim {kp_pool.shape[1]}, "
          f"z_aux dim {zaux_pool.shape[1]}")

    # 2) reference clouds. Reusing the fit pass when the splits match keeps this
    #    from reading (and re-normalizing) the same shapes twice
    if same_split:
        ref_clouds, ref_names = fit_clouds, _fit_names
        if opt.max_ref_samples is not None:
            ref_clouds = ref_clouds[: opt.max_ref_samples]
            ref_names = ref_names[: opt.max_ref_samples]
    else:
        ref_loader = make_loader(opt, opt.ref_split)
        ref_clouds, ref_names = collect_reference_clouds(ref_loader, opt.max_ref_samples)
    num_ref = ref_clouds.shape[0]

    # 3) fit the latent distribution and sample from it
    pca_mu, pca_comps, z_pca = fit_pca(kp_pool, opt.pca_dim)
    dens_mu, dens_std = fit_density(z_pca, opt.kde_bandwidth)
    num_gen = int(opt.num_gen) if opt.num_gen else num_ref
    kp_flat = sample_latents(
        num_gen, pca_mu, pca_comps, z_pca, dens_mu, dens_std, sampler=opt.sampler, rng=rng
    )
    kp_samples = torch.from_numpy(kp_flat).float()
    zaux_samples = sample_zaux(num_gen, zaux_pool, mode=opt.zaux_mode, rng=rng)
    print(f"[✓] sampled {num_gen} latents (PCA dim {pca_comps.shape[0]}, "
          f"sampler={opt.sampler}, z_aux={opt.zaux_mode})")

    # 4) decode
    gen_clouds = generate_clouds(model, kp_samples, zaux_samples, opt,
                                 num_points=ref_clouds.shape[1])

    # 5) metrics. The reference clouds are float32 from the loader; matching dtypes
    #    keeps chamfer from silently upcasting one side
    gen_clouds = gen_clouds.float()
    ref_clouds = ref_clouds.float()

    M_gr_cd = pairwise_distances(gen_clouds, ref_clouds, "cd", opt.pair_batch, "gen-ref")
    cd_stats = mmd_cov(M_gr_cd)

    emd_stats = None
    if opt.emd:
        M_gr_emd = pairwise_distances(gen_clouds, ref_clouds, "emd", opt.pair_batch, "gen-ref")
        emd_stats = mmd_cov(M_gr_emd)

    nna_cd = nna_emd = float("nan")
    if opt.nna:
        M_gg_cd = pairwise_distances(gen_clouds, gen_clouds, "cd", opt.pair_batch, "gen-gen")
        M_rr_cd = pairwise_distances(ref_clouds, ref_clouds, "cd", opt.pair_batch, "ref-ref")
        nna_cd = one_nna(M_rr_cd, M_gr_cd.t(), M_gg_cd)
        if opt.emd:
            M_gg_emd = pairwise_distances(gen_clouds, gen_clouds, "emd", opt.pair_batch, "gen-gen")
            M_rr_emd = pairwise_distances(ref_clouds, ref_clouds, "emd", opt.pair_batch, "ref-ref")
            nna_emd = one_nna(M_rr_emd, M_gr_emd.t(), M_gg_emd)

    run_metrics = {
        "model": opt.model,
        "category": getattr(opt, "category", None),
        "input_type": opt.input_type,
        "fit_split": opt.fit_split,
        "ref_split": opt.ref_split,
        "num_gen": num_gen,
        "num_ref": num_ref,
        "coverage": cov_mean,
        "pca_dim": int(pca_comps.shape[0]),
        "mmd_cd": cd_stats["mmd"],
        "cov_cd": cd_stats["cov"],
        "nna_cd": nna_cd,
        "mmd_emd": emd_stats["mmd"] if emd_stats else float("nan"),
        "cov_emd": emd_stats["cov"] if emd_stats else float("nan"),
        "nna_emd": nna_emd,
    }

    # 6) outputs
    kp_np = kp_samples.numpy().reshape(num_gen, -1, 3)
    if not opt.no_save:
        save_gen_geoms(
            gen_clouds.numpy(), kp_np, cd_stats["per_gen_argmin"], ref_names,
            cd_stats["per_gen_min"], opt, out_dir=opt.output_dir,
        )

    if opt.viz:
        max_entries = max(0, int(getattr(opt, "viz_max_samples", 100)))
        cloud_points = int(getattr(opt, "viz_cloud_points", 1500))
        # how many generated shapes claim each reference - the per-sample side of COV
        claims = np.bincount(cd_stats["per_gen_argmin"], minlength=num_ref)

        entries = []
        for i in range(min(num_gen, max_entries)):
            ref_idx = int(cd_stats["per_gen_argmin"][i])
            meta = {
                "nearest_ref": ref_names[ref_idx] if ref_names else None,
                "cd": float(cd_stats["per_gen_min"][i]),
                "claimed_refs": float(claims[ref_idx]),
            }
            if emd_stats is not None:
                meta["emd"] = float(emd_stats["per_gen_min"][i])
            entries.append(
                {
                    "name": f"gen_{i:05d}",
                    "gen_cloud": subsample(gen_clouds[i].numpy(), cloud_points),
                    "gen_kp": kp_np[i],
                    "ref_cloud": subsample(ref_clouds[ref_idx].numpy(), cloud_points),
                    "meta": meta,
                }
            )

        viz_out = getattr(opt, "viz_out", None)
        if viz_out is None:
            viz_out = opt.output_dir / opt.model / f"{opt.category}_mmd_interactive.html"
        write_mmd_html(entries, viz_out, run_metrics=run_metrics)
        plot_distance_distributions(cd_stats, out_root / "mmd_cov_cd.png", metric="CD")
        if emd_stats is not None:
            plot_distance_distributions(emd_stats, out_root / "mmd_cov_emd.png", metric="EMD")

    print(f"MMD-CD   : {run_metrics['mmd_cd']:.8e}")
    print(f"COV-CD   : {run_metrics['cov_cd']:.8f}")
    if opt.emd:
        print(f"MMD-EMD  : {run_metrics['mmd_emd']:.8e}")
        print(f"COV-EMD  : {run_metrics['cov_emd']:.8f}")
    if opt.nna:
        print(f"1-NNA-CD : {nna_cd:.8f}")
        if opt.emd:
            print(f"1-NNA-EMD: {nna_emd:.8f}")

    if not opt.no_save:
        run_id = save_run(opt.db_path, opt, run_metrics)
        print(f"[✓] saved to {opt.db_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
