#!/usr/bin/env python3
"""
Reconstruction evaluation using H5Dataset and EMD / Chamfer metrics.

- Loads *.h5 files from --dataset (recursively) filtered by --category via H5Dataset
- Runs model.get_reconstruction on batches
- Computes Chamfer Distance (CD) and an entropic-regularized EMD (Sinkhorn) per sample
- Logs averages and a combined metric `EMD_CD_recon` to SQLite

Assumptions:
- `MODEL_CLASSES` maps model names -> classes compatible with TestBase
- Each model implements `.load_model(ckpt_path, opt)` and `.get_reconstruction(batch)`
- `H5Dataset` signature (provided by your codebase):
    H5Dataset(h5_files, normalize=True, include_label=False, object_name=args.category)

Run:
  python recon_eval_emd_cd.py <model> --ckpt checkpoints/foo.pth \
         --dataset /path/to/ShapeNetH5/ --category chair --batch-size 16 \
         --db-path results.db
"""
import argparse
import contextlib
import json
import sqlite3
from glob import glob
from pathlib import Path

import numpy as np
import torch
import tqdm
from classes import MODEL_CLASSES
from datasets.H5Datset import H5Dataset
from keypoint_diffuser.utils.eval_metrics import EMD_CD_recon
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)
from torch.utils.data import DataLoader
from torchvision import transforms


TESTSET = "/app/shapenetcorev2_hdf5_2048/val"

# ----------------------------
# CLI
# ----------------------------


def save_recon_geoms(
    best_recons, best_gts, cds, emds, opt, out_dir: Path = Path("output")
):
    """
    Save best reconstructions and ground truths as numpy arrays.
      - recon.npy : best reconstruction (normalized point cloud)
      - gt.npy    : ground truth point cloud
      - meta.json : per-sample metadata (with metrics)
    """

    out_dir = out_dir / opt.model / opt.category
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (recon, gt, cd, emd) in enumerate(zip(best_recons, best_gts, cds, emds)):
        # recon: [M, 3], gt: [N, 3]
        dst = out_dir / f"sample_{idx:05d}"
        dst.mkdir(parents=True, exist_ok=True)

        np.save(dst / "recon.npy", np.asarray(recon, dtype=np.float32))
        np.save(dst / "gt.npy", np.asarray(gt, dtype=np.float32))

        meta = {
            "model": opt.model,
            "category": getattr(opt, "category", None),
            "sample_idx": idx,
            "num_recon": int(recon.shape[0]),
            "num_gt": int(gt.shape[0]),
            "cd": float(cd),
            "emd": float(emd),
        }
        with open(dst / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved += 1

    print(f"[✓] Saved {saved} reconstructions under {out_dir}/")


def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        p.add_argument(*names, **kwargs)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate reconstruction quality (EMD & CD) on H5Dataset and log to SQLite."
    )

    subparsers = p.add_subparsers(dest="model", required=True)

    for model_name, model_cls in MODEL_CLASSES.items():
        sp = subparsers.add_parser(model_name)
        model_cls.get_parser(sp)
        try_add_arg(sp, "--ckpt", type=Path)
        try_add_arg(sp, "--category", type=str, default="chair")
        try_add_arg(sp, "--batch-size", type=int, default=16)
        try_add_arg(sp, "--num-workers", type=int, default=4)
        try_add_arg(sp, "--db-path", type=Path, default=Path("results.db"))
        try_add_arg(sp, "--key-points", type=int, default=10)

    return p


# ----------------------------
# Dataset & Prediction
# ----------------------------


def make_loader(opt: argparse.Namespace) -> DataLoader:
    h5_files = glob(f"{TESTSET}**/*.h5", recursive=True)
    t = (
        transforms.Compose(
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
        if opt.model == "Ours"
        else None
    )

    dataset = H5Dataset(
        h5_files,
        normalize=True,
        get_two=opt.model == "KPD",
        include_label=False,
        object_name=opt.category,
        transform=t,
    )
    # if your project has a custom collate_fn for dicts, reuse it
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_fn if opt.model == "Ours" else None,
        drop_last=False,
    )


def run_reconstruction(model, loader, opt=None, save=True, out_dir=Path("output")):
    cds, emds = [], []
    best_recons, best_gts = [], []

    model.model.eval()
    model.model.cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Reconstruct (best per item)"):
            # recon_list: list of length B; each item is a list of [2048,3] tensors
            recon_list, pc = model.get_reconstruction(batch)
            gt_batch = pc.float()
            B = gt_batch.shape[0]

            for b in range(B):
                cands = recon_list[b]
                if len(cands) == 0:  # no valid candidates for this item
                    continue
                # Stack candidates -> [C, 2048, 3]
                preds = (
                    torch.stack(cands, dim=0).float() if type(cands) == list else cands
                )

                C = preds.shape[0]

                # Repeat GT -> [C, N, 3]
                gt_b = gt_batch[b].unsqueeze(0).expand(C, -1, -1)

                # One call for all candidates
                res = EMD_CD_recon(preds, gt_b, reduced=False)
                cd_all = res["CD"].detach().cpu().numpy()  # shape [C] or [C,]
                emd_all = res["EMD"].detach().cpu().numpy()  # shape [C] or [C,]

                # Pick best by CD
                i = int(np.argmin(cd_all))
                cds.append(float(cd_all[i]))
                emds.append(float(emd_all[i]))

                best_recons.append(preds[i].cpu().numpy())  # [2048,3]
                best_gts.append(gt_batch[b].cpu().numpy())  # [N,3]

    save_recon_geoms(best_recons, best_gts, cds, emds, opt, out_dir=out_dir)

    cd_mean = np.mean(cds)
    emd_mean = np.mean(emds)
    return cd_mean, emd_mean


# ----------------------------
# DB
# ----------------------------


def init_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reconstruction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            ckpt TEXT,
            category TEXT,
            batch_size INTEGER,
            cd_recon REAL,
            emd_recon REAL
        )
        """
    )
    con.commit()
    return con


def save_run(db_path: Path, opt: argparse.Namespace, cd: float, emd: float) -> int:
    con = init_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO reconstruction (model, ckpt, category, batch_size, cd_recon, emd_recon)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            opt.model,
            str(opt.ckpt) if opt.ckpt is not None else None,
            str(opt.category),
            int(opt.batch_size),
            float(cd),
            float(emd),
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

    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()
    model.load_model(opt.ckpt, opt)

    loader = make_loader(opt)

    cd, emd = run_reconstruction(model, loader, opt, out_dir=Path("recons_out"))

    print(f"Chamfer Distance (mean): {cd:.6f}")
    print(f"EMD (Sinkhorn)   (mean): {emd:.6f}")

    run_id = save_run(opt.db_path, opt, cd, emd)
    print(f"[✓] saved to {opt.db_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
