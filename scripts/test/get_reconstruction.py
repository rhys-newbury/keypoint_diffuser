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
import time

import numpy as np
import torch
import tqdm
from classes import MODEL_CLASSES
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions
from keypoint_diffuser.utils.eval_metrics import EMD_CD_recon
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth,
    Collect,
    Deform,
    GridSample,
    ToTensor,
)
from torch.utils.data import DataLoader
from torchvision import transforms


TESTSET = "/mnt/slow/shapenetcorev2-h5/shapenetcorev2_hdf5_2048/test"
PARTIAL_VIEW_MODES = ["default", "myopia", "patch", "tac"]

# ----------------------------
# CLI
# ----------------------------
def save_pred_kp_geoms(
    predicted, names, opt, out_dir: Path = Path("geom_np"), vis_max=1000
):
    """
    Saves raw numpy arrays for each sample:
      - pred_kp.npy   : predicted keypoints (normalized)
    Also dumps meta.json with ids + config.
    """

    out_dir = out_dir / opt.model / opt.category
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (kpcd_norm, name) in enumerate(zip(predicted, names)):
        if saved >= vis_max:
            break

        pred = np.asarray(kpcd_norm, dtype=np.float32)  # normalized preds

        # save per-sample folder
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "pred_kp.npy", pred)

        meta = {
            "model": opt.model,
            "category": opt.category,
            "model_id": name,
            "num_pred": int(pred.shape[0]),
        }
        with open(dst / "kpmeta.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved += 1

    print(f"[✓] Saved {saved} samples as numpy arrays under {out_dir}/")


def save_recon_geoms(
    best_recons, best_gts, best_inputs, best_names, cds, emds, opt, out_dir: Path = Path("output")
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
    for (recon, gt, inpc, name, cd, emd) in zip(best_recons, best_gts, best_inputs, best_names, cds, emds):
        # recon: [M, 3], gt: [N, 3]
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)

        np.save(dst / "recon.npy", np.asarray(recon, dtype=np.float32))
        np.save(dst / "gt.npy", np.asarray(gt, dtype=np.float32))
        np.save(dst / "input.npy", np.asarray(inpc, dtype=np.float32))

        meta = {
            "model": opt.model,
            "category": getattr(opt, "category", None),
            "model_id": name,
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
        try_add_arg(sp, "--batch-size", type=int, default=16)       # on default this is causing batch size to be 48 for some reason
        try_add_arg(sp, "--num-workers", type=int, default=8)
        try_add_arg(sp, "--db-path", type=Path, default=Path("results.db"))
        try_add_arg(sp, "--key-points", type=int, default=10)
        try_add_arg(sp, "--output-dir", type=Path, default=Path("recons_out"))
        try_add_arg(sp, "--input-type", type=str, default="full", choices=["full", *PARTIAL_VIEW_MODES, "all"])
        try_add_arg(sp, "--save-kps", action="store_true")

    return p


# ----------------------------
# Dataset & Prediction
# ----------------------------


def make_loader(opt: argparse.Namespace) -> DataLoader:
    t = (
        transforms.Compose(
        [
            Deform(),  # Forks into two versions: original and deformed
            ApplyToBoth(
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
                            keys=("coord", "grid_coord", "transformation", "shape"),
                            feat_keys=("coord",),
                        ),
                    ]
                )
            ),
        ]
        )
        if opt.model == "Ours"
        else None
    )

    from dataclasses import fields, MISSING
    def _config_from_namespace(ns: argparse.Namespace) -> AEConfig:
        cfg_kwargs = {}
        for f in fields(AEConfig):
            name = f.name
            val = getattr(ns, name, MISSING)

            # missing -> use dataclass default (if provided)
            if val is MISSING:
                if f.default is not MISSING:
                    val = f.default
                elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                    val = f.default_factory()           # type: ignore[misc]
                else:
                    # raise ValueError(f"Missing required field: {name}")
                    # errors will be thrown through the dataset init if it is missing a required arg
                    continue

            # str → Path for Path-typed fields
            if f.type is Path and isinstance(val, str):
                val = Path(val)

            # print(f"adding name-val pair [{name}: {val}]")
            cfg_kwargs[name] = val

        # mimic AEOptions.parse() post-processing
        if cfg_kwargs.get("normalization") == "none":
            cfg_kwargs["normalization"] = None

        # manually add values
        cfg_kwargs["name"] = "recon"
        cfg_kwargs["split"] = "test"
        cfg_kwargs["mesh_dir"] = "/mnt/slow/shapenetcorev2-source/"
        cfg_kwargs["points_dir"] = "/mnt/slow/shapenetcorev2-source/"
        cfg_kwargs["split_file"] = "./data/shapenet_split/splits_out.csv"
        # cfg_kwargs["ckpt_dir"] = "logs/original-50k-iters/"
        # cfg_kwargs["db"] = "airplane-10kpt-50k-iters-train-results.db"
        # cfg_kwargs["db_root"] = "db/train/original-50k-iters/"

        return AEConfig(**cfg_kwargs)

    configs_opt = _config_from_namespace(opt)

    dataset = ShapesPartial(configs_opt, transform=t)
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
    recons, gts, inputs, names = [], [], [], []
    kps = []

    model.model.eval()
    model.model.cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Reconstruct (best per item)"):
            # recon_list: list of length B; each item is a list of [2048,3] tensors
            if opt.input_type == "full":
                key = "orig"
            else:
                key = "partial_orig"
            
            if opt.model == "Ours" or opt.model == "Partial":
                recon_batch, input_pc, full_pc = model.get_reconstruction(batch, key=key)
            else:
                raise RuntimeError("undefined model case")
                
            input_batch = input_pc.float()
            gt_batch = full_pc.float()
            name_batch = batch["target_name"]
            B = gt_batch.shape[0]

            recon_batch = recon_batch.squeeze()    # [B, 1, 2048, 3] to [B, 2048, 3]

            # One call for all candidates
            res = EMD_CD_recon(recon_batch, gt_batch, reduced=False)
            cd_all = res["CD"].detach().cpu().numpy()  # shape [B]
            emd_all = res["EMD"].detach().cpu().numpy()  # shape [B]
            cds.extend(list(cd_all))
            emds.extend(list(emd_all))

            recons.extend(list(recon_batch.cpu().numpy()))
            gts.extend(list(gt_batch.cpu().numpy()))
            inputs.extend(list(input_batch.cpu().numpy()))
            names.extend(name_batch)
            
            if opt.save_kps:
                data_dict = model.get_network_data(batch, key=key)
                kp_batch = model.get_keypoints(data_dict)
                kps.extend(list(kp_batch))

    save_recon_geoms(recons, gts, inputs, names, cds, emds, opt, out_dir=out_dir)
    if opt.save_kps:
        # print("saving prediction keypoints")
        save_pred_kp_geoms(kps, names, opt, out_dir=out_dir, vis_max=1000)

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

    if opt.input_type == "full":
        opt.partial_view_mode = "default"   # hack, the partial pc will not be used
    else:
        opt.partial_view_mode = opt.input_type
    
    loader = make_loader(opt)

    print(f"[✓] Running reconstruction evaluation for model={opt.model}, category={opt.category}")
    print(f"[✓] Output dir: {opt.output_dir}")
    cd, emd = run_reconstruction(model, loader, opt, out_dir=opt.output_dir)

    print(f"Chamfer Distance (mean): {cd:.6f}")
    print(f"EMD (Sinkhorn)   (mean): {emd:.6f}")

    run_id = save_run(opt.db_path, opt, cd, emd)
    print(f"[✓] saved to {opt.db_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
