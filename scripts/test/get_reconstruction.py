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
import json
import sqlite3
from dataclasses import fields, MISSING
from glob import glob
from pathlib import Path
import time
import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from classes import MODEL_CLASSES
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions
from keypoint_diffuser.utils.eval_metrics import EMD_CD_recon
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.utils import normalize_to_box, normalize_scale_min_max
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
PARTIAL_VIEW_MODES = ["default", "myopia", "patch", "tac", "uniform"]

# ----------------------------
# CLI
# ----------------------------

def save_recon_geoms(
    recons, gts, inputs, names, partial_sample_names, kps, gt_kps, cds, emds, coverages, opt, out_dir: Path = Path("output")
):
    """
    Save reconstructions and ground truths as numpy arrays.
      - recon.npy : reconstruction (normalized point cloud)
      - gt.npy    : ground truth point cloud
      - meta.json : per-sample metadata (with metrics)
    """

    out_dir = out_dir / opt.model / opt.category
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for (recon, gt, inpc, name, psn, kp, gt_kp, cd, emd, cov) in zip(
        recons, gts, inputs, names, partial_sample_names, kps, gt_kps, cds, emds, coverages
    ):
        # recon: [M, 3], gt: [N, 3]
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)

        np.save(dst / "recon.npy", np.asarray(recon, dtype=np.float32))
        np.save(dst / "gt.npy", np.asarray(gt, dtype=np.float32))
        np.save(dst / "input.npy", np.asarray(inpc, dtype=np.float32))
        np.save(dst / "pred_kp.npy", np.asarray(kp, dtype=np.float32))
        # gt_kp is None for samples with no human annotation
        if gt_kp is not None:
            np.save(dst / "gt_kp.npy", np.asarray(gt_kp, dtype=np.float32))

        meta = {
            "model": opt.model,
            "category": getattr(opt, "category", None),
            "model_id": name,
            "num_recon": int(recon.shape[0]),
            "num_gt": int(gt.shape[0]),
            "cd": float(cd),
            "emd": float(emd),
            "partial_sample_name": psn,
            # surface fraction covered by the partial view (1 for input-type=full)
            "coverage": float(cov) if cov is not None else None,
            # per-sample keypoint agreement; DAS/mIoU are dataset-level only
            **keypoint_distance_stats(kp, gt_kp),
        }
        with open(dst / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved += 1

    print(f"[✓] Saved {saved} reconstructions under {out_dir}/")


def try_add_arg(p: argparse.ArgumentParser, *names, **kwargs):
    """
    add_argument, but the default declared here wins over one already registered.

    AEOptions.initialize adds one flag per AEConfig field before any of these
    scripts declare their own, so a name collides in two different ways and the
    caller's default was being lost either way:

      * same option string ("--split"): add_argument raises ArgumentError. This
        used to swallow it, leaving the dataclass default - None for split, which
        matches no rows in the split file and fails inside ShapesPartial - and
        dropping any choices= declared at the call site.
      * same dest, different spelling ("--batch-size" here vs "--batch_size" from
        the dataclass): add_argument succeeds, both actions write to batch_size,
        and argparse seeds the default from whichever was registered first, i.e.
        the dataclass one.

    So after adding, the default (and choices) are applied to every action sharing
    the dest. Both spellings still parse; what changes is which default a run gets
    when the flag is omitted.
    """
    dest = kwargs.get("dest") or names[0].lstrip("-").replace("-", "_")

    action = None
    try:
        action = p.add_argument(*names, **kwargs)
    except argparse.ArgumentError:
        pass  # already registered under this exact option string

    same_dest = [a for a in p._actions if a.dest == dest]
    for a in same_dest:
        if "default" in kwargs:
            a.default = kwargs["default"]
        if kwargs.get("choices") is not None:
            a.choices = kwargs["choices"]

    return action or (same_dest[0] if same_dest else None)


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
        try_add_arg(sp, "--split", type=str, default="test", choices=["train", "val", "test"],
                    help="Which rows of --split-file to evaluate. Matched exactly against the "
                         "csv's `split` column, so only these three values select anything. "
                         "`phase` stays at test whatever this is set to: it selects deterministic "
                         "sampling and no augmentation, which is what evaluation wants even on "
                         "the train rows")
        try_add_arg(sp, "--batch-size", type=int, default=16)
        try_add_arg(sp, "--num-workers", type=int, default=8)
        try_add_arg(sp, "--pcd-path", type=Path, default=Path("/mnt/slow/shapenetcorev2-h5/pcds/"))
        try_add_arg(sp, "--db-path", type=Path, default=Path("results.db"))
        try_add_arg(sp, "--key-points", type=int, default=10)
        try_add_arg(sp, "--output-dir", type=Path, default=Path("recons_out_new"))
        try_add_arg(sp, "--input-type", type=str, default="full", choices=["full", *PARTIAL_VIEW_MODES, "all"])
        try_add_arg(sp, "--enumerate-partial-samples", action="store_true", help="Evaluate every partial sample on disk instead of one random view per object. Dataset length becomes the number of (object, view) pairs")
        try_add_arg(sp, "--no-save", action="store_true", help="If set, do not save reconstructions to disk (only compute metrics)")
        try_add_arg(sp, "--gt-kp-mode", type=str, default="extract", choices=["extract", "load", "human"], help="Where to get the ground truth keypoints. `extract`: extract from the full point cloud, `load`: load from pre-extracted files, and `human`: use KeypointNet human annotations.")
        try_add_arg(sp, "--gt-kp-src", type=Path, default=None, help="Where to get the gt kps for das calculation. Loads human annotations if None")
        try_add_arg(sp, "--viz", action="store_true", help="Write an interactive HTML viewer (recon / gt / input clouds + keypoints, with metrics)")
        try_add_arg(sp, "--viz-out", type=Path, default=None, help="Path of the HTML viewer. Defaults to <output-dir>/<model>/<category>_recons_interactive.html")
        try_add_arg(sp, "--viz-max-samples", type=int, default=100, help="Cap on objects embedded in the HTML viewer (every cloud is inlined, so the file grows fast)")
        try_add_arg(sp, "--viz-cloud-points", type=int, default=2048, help="Points kept per cloud for display (stride-subsampled)")
        try_add_arg(sp, "--no-coverage-plot", action="store_true", help="Skip the CD / EMD / KVC vs input-coverage figure. It is written by default (into <output-dir>/<model>/) and skips itself when coverage is constant, as it is for full clouds")
        try_add_arg(sp, "--coverage-bins", type=int, default=12, help="Equal-count bins along the coverage axis for the median / IQR summary")

    return p


# ----------------------------
# KP metrics utils
# ----------------------------
def naive_read_pcd(path):
    with open(path) as f:
        lines = f.readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith("DATA ascii"):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(" ") for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float32)
    return pc

def ordered_kp_distance(a, b) -> float:
    """
    Mean distance between keypoints held in the same slot, i.e. the correspondence
    is kept. NaN when the two sets have different lengths, which is the case for
    the human annotations (they carry no slot correspondence with the prediction).
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 3)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.linalg.norm(a - b, axis=-1).mean())

def chamfer_kp(a, b) -> float:
    """
    Symmetric nearest-neighbour distance between two keypoint sets, with the slot
    correspondence discarded. K is small (8-10), so the full K x K matrix is free.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))

def fwd_alignment_scores_gtn(kpn_ds, gt_nkps, pred_nkps):
    """
    uses already normalised gt keypoints gtn_kps and predicted keypoints pred_nkps produced from normalized point clouds
    """
    preds = []
    assignments = []  # store GT indices for each pred
    for entry, ngt, npred in zip(kpn_ds, gt_nkps, pred_nkps):
        ground_truths = np.array(ngt)
        dist = np.sum(
            (np.expand_dims(npred, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1
        )
        argminfwd = np.argmin(dist, -1)
        preds.append([entry["keypoints"][argm]["semantic_id"] for argm in argminfwd])
        assignments.append(argminfwd)
        
    acc = [np.mean(np.array(pa) == np.array(pb)) for pa in preds for pb in preds]

    return np.mean(acc)


def bwd_alignment_scores_gtn(kpn_ds, gt_nkps, pred_nkps):
    """
    uses already normalised gt keypoints gtn_kps and predicted keypoints pred_nkps produced from normalized point clouds
    """
    preds = collections.defaultdict(list)
    for entry, ngt, npred in zip(kpn_ds, gt_nkps, pred_nkps):
        ground_truths = np.array(ngt)
        dist = np.sum(
            (np.expand_dims(npred, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1
        )
        argminbwd = np.argmin(dist, -2)
        for i, kp in enumerate(entry["keypoints"]):
            preds[kp["semantic_id"]].append(argminbwd[i])
    q = []
    for arr in preds.values():
        arr = np.array(arr)
        q.append(np.mean(arr[:, None] == arr[None, :]))
    
    return np.mean(q)

def mIoU(kpn_ds, gt_nkps, pred_nkps, out_norm_val_0s, out_norm_val_1s, pcd_path):
    """
    calculates the mean Intersection over Union (mIoU) for the predicted keypoints against the ground truth keypoints.
    predicted keypoints are projected back on to the original cloud surface before distance calculation, to measure geometric alignment.
    """
    thresholds = np.linspace(0.0, 0.1)
    kps = []
    gts = []
    for entry, ngt, npred, n0, n1 in zip(kpn_ds, gt_nkps, pred_nkps, out_norm_val_0s, out_norm_val_1s):
        cid = entry["class_id"]
        mid = entry["model_id"]
        pc = torch.Tensor(naive_read_pcd(pcd_path / cid / f"{mid}.pcd"))

        # xyz coordinates for each keypoint in the entry, list of tensors
        # the same as just reading entry["keypoints"][idx]["xyz"] for each keypoint
        ground_truths = [pc[kp["pcd_info"]["point_index"]] for kp in entry["keypoints"]]
        gts.append(ground_truths)

        npc, _, _ = normalize_scale_min_max(pc, dmin=n0, dmax=n1)

        # for each pred kp, find the closest point in the point cloud
        dist = np.sqrt(
            np.sum((np.expand_dims(npred, 1) - np.expand_dims(npc, 0)) ** 2, axis=-1)
        )
        kps.append(npc[np.argmin(dist, -1)])
    for threshold in thresholds:
        npos = fp_sum = fn_sum = 0
        for ground_truths, kpcd in zip(gts, kps):
            gt = torch.stack(ground_truths)
            # kpcd is the points from the point cloud closest to the pred kps collected from above
            dist = np.sqrt(
                np.sum(
                    (np.expand_dims(kpcd, 1) - np.expand_dims(gt, 0)) ** 2,
                    axis=-1,
                )
            )
            npos += dist.shape[1]
            fp_sum += np.sum(np.min(dist, -1) > threshold)
            fn_sum += np.sum(np.min(dist, -2) > threshold)
        yield (npos - fn_sum) / (npos + fp_sum)

def mIoU_curve_plot(kpn_ds, gt_nkps, pred_nkps, norm_val_0s, norm_val_1s, pcd_path, out_root):
    plt.style.use("seaborn")
    miou_curve = list(mIoU(kpn_ds, gt_nkps, pred_nkps, norm_val_0s, norm_val_1s, pcd_path))
    plt.plot(np.linspace(0.0, 0.1), miou_curve)
    plt.title("mIoU Curve")
    plt.xlabel("Distance Threshold")
    plt.ylabel("mIoU")
    plt.grid(True)
    plt.savefig(out_root / "mIoU.png")
    return miou_curve[-1]

# ----------------------------
# Interactive visualization & plots
# ----------------------------
# Extracted to viz_utils so the forks of this script share one copy. Imported by
# bare name, like `classes` above - these scripts all run with scripts/test on
# sys.path.
from viz_utils import (  # noqa: E402
    keypoint_distance_stats,
    subsample,
    write_coverage_outputs,
    write_recon_html,
)


# ----------------------------
# Dataset & Prediction
# ----------------------------
def make_h5loader(opt: argparse.Namespace) -> DataLoader:
    h5_files = glob(f"/mnt/slow/shapenetcorev2-h5/shapenetcorev2_hdf5_2048/test/**/*.h5", recursive=True)
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


def build_partial_transform(opt: argparse.Namespace):
    """Transform pipeline used with the ShapesPartial loader (Ours / Partial only)."""
    if opt.model != "Ours" and opt.model != "Partial":
        return None
    return transforms.Compose(
        [
            Deform(),  # Forks into two versions: original and deformed
            ApplyToBoth(
                transforms.Compose(
                    [
                        # voxelize original clouds
                        GridSample(
                            keys=("coord",),
                            coord_key="coord",
                            hash_type="fnv",
                            mode="train",
                            return_grid_coord=True,
                        ),
                        # voxelize partial clouds
                        GridSample(
                            keys=("partial_coord",),
                            coord_key="partial_coord",
                            prefix="partial_", # prevents overwriting
                            hash_type="fnv",
                            mode="train",
                            return_grid_coord=True
                        ),
                        ToTensor(),
                        Collect(
                            keys=("coord", "grid_coord", "transformation", "shape",
                                  "partial_coord", "partial_grid_coord", "partial_shape"),
                            feat_keys=("coord",),
                            partial_feat_keys=("partial_coord",),
                        ),
                    ]
                )
            ),
        ]
    )


def config_from_namespace(ns: argparse.Namespace) -> AEConfig:
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

    # hard coded: manually added values
    cfg_kwargs["name"] = "recon"
    # --split, when the caller registered it. AEConfig's own default is None, which
    # matches no rows in the split file and trips ShapesPartial's `assert len(names)
    # > 0`, so a caller that never declares the flag still lands on test - which is
    # what this pinned unconditionally before the flag existed.
    cfg_kwargs["split"] = getattr(ns, "split", None) or "test"
    cfg_kwargs["mesh_dir"] = "/mnt/slow/shapenetcorev2-source/"
    cfg_kwargs["points_dir"] = "/mnt/slow/shapenetcorev2-source/"
    cfg_kwargs["split_file"] = "./data/shapenet_split/splits_out.csv"
    # cfg_kwargs["split_file"] = "./data/shapenet_split/all.csv"
    # n_partial_samples is left to the namespace / AEConfig default (2): pinning it
    # to 1 makes ShapesPartial's random.randint(0, n-1) always return 0, so every
    # sample is view "_0" and repeat sightings of an object are the same file
    cfg_kwargs["normalize"] = "minmax_scaling"
    # phase, unlike split, stays at test for every split: it selects deterministic
    # sampling and no augmentation, which is what evaluation wants on train rows too
    cfg_kwargs["phase"] = "test"
    # cfg_kwargs["ckpt_dir"] = "logs/original-50k-iters/"
    # cfg_kwargs["db"] = "airplane-10kpt-50k-iters-train-results.db"
    # cfg_kwargs["db_root"] = "db/train/original-50k-iters/"

    return AEConfig(**cfg_kwargs)


def make_loader(opt: argparse.Namespace) -> DataLoader:
    t = build_partial_transform(opt)
    configs_opt = config_from_namespace(opt)

    dataset = ShapesPartial(configs_opt, transform=t)
    # if your project has a custom collate_fn for dicts, reuse it
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn if (opt.model == "Ours") or (opt.model == "Partial") else None,
        drop_last=False,
    )

def run_reconstruction(model, loader, opt=None, save=True, out_dir=Path("output"), no_save=False, viz=False):
    cds, emds = [], []
    # per-sample keypoint consistency against the gt keypoints:
    #   kvc  - mean slot-wise distance (correspondence kept)
    #   ukvc - the same distance with the correspondence discarded (chamfer)
    # NaN for samples with no gt keypoints, or where the two sets have different
    # lengths so no slot correspondence exists (the human-annotation case)
    kvc, ukvc = [], []
    recons, gts, inputs, names, psns, covs = [], [], [], [], [], []
    kps, gt_kps = [], []
    kpn_ds_ordered = []
    out_norm_val_0, out_norm_val_1 = [], []
    kp_gt_exists = []

    # the clouds are needed both to write the npy dumps and to build the viewer
    collect_clouds = (not no_save) or viz

    model.model.eval()
    model.model.cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Reconstruct (best per item)"):
            # recon_list: list of length B; each item is a list of [2048,3] tensors
            if opt.input_type == "full":
                key = "orig"
            else:
                key = "orig_partial"
            
            if opt.model == "Ours" or opt.model == "Partial":
                recon_batch, input_pc, full_pc, kp_batch = model.get_reconstruction(batch, key=key)
            else:
                raise RuntimeError("undefined model case")
            
            # collect for visualization
            input_batch = input_pc.float()
            gt_batch = full_pc.float()
            name_batch = batch["target_file"]
            psn_batch = batch["target_partial_sample_name"]
            cov_batch = batch["target_coverage"]    # full point clouds should have coverage=1
            B = gt_batch.shape[0]

            recon_batch = recon_batch.squeeze(1)    # [B, 1, 2048, 3] to [B, 2048, 3]

            # ---------------- load gt keypoints ----------------
            # use pre-extracted keypoints from saved files as ground truth keypoints
            # gt_kp_src is a directory containing keypoints produced from the full point clouds
            # using a chosen baseline method. 
            # we use this as a self-referencing "ground truth" for predictions from partial pcs
            if opt.gt_kp_mode == "load" and opt.gt_kp_src is not None:
                # NOTE: actual produce approariate gt keypoints before using this
                
                # for human labelled kps:
                # try to match the original way done with the json file
                # which is to build up the list gt_kps by appending to it normalized kps
                # where each kp is a [3, N] numpy array    *check
                # the src should be the recon_out output dir of an existing reconstruction, using the original paradigm
                # when testing on the sampled data, the generated kps are under something like:
                # whatever_dir/epoch_N/Ours/CATEGORY/OBJECT_ID/gt_kp.npy
                # TODO: if want to use that generated from h5 data, the object id is not contained in the outputs, need to change
                
                kp_gt_exists_batch = [True]*B
                for idx, mid in enumerate(name_batch):
                    # load kp
                    gt_kp_file = opt.gt_kp_src / f"{mid}/gt_kp.npy"
                    if not gt_kp_file.exists():
                        kp_gt_exists_batch[idx] = False
                        continue
                    kp = np.load(gt_kp_file)
                    # normalize
                    norm_val_0 = batch['target_norm_val_0'][idx]
                    norm_val_1 = batch['target_norm_val_1'][idx]
                    # nkp, _, _ = normalize_scale_min_max(torch.Tensor(kp), dmin=norm_val_0, dmax=norm_val_1)
                    # gt_kps.append(nkp.numpy())
                    gt_kps.append(kp)
            # use extraction from the full point cloud as ground truth keypoints
            # NOTE: since this is live extracted it means partial paradigm training also needs to perform on full point clouds
            elif opt.gt_kp_mode == "extract":
                kp_gt_exists_batch = [True]*B
                # get keypoints from the full point cloud
                if hasattr(model, "get_keypoints_from_batch"):
                    gt_kp_batch = model.get_keypoints_from_batch(batch, key="orig")
                else:
                    # models without the shortcut pay for the diffusion decode
                    _, _, _, gt_kp_batch = model.get_reconstruction(batch, key="orig")

                # numpy on cpu, to match what the other two branches append
                gt_kp_batch = gt_kp_batch.reshape(B, opt.key_points, 3).cpu().numpy()
                gt_kps.extend(list(gt_kp_batch))
            # use keypoints from KeypointNet human annotations as ground truth keypoints
            # TODO: not checked after changes and probably busted
            elif opt.gt_kp_mode == "human":
                # load the kp annotation file
                annotation_json = Path(f"/mnt/slow/shapenetcorev2-h5/annotations/{batch['target_cat'][0]}.json")
                # build up input list of dicts
                kpn_ds = json.load(open(annotation_json))
                kpn_ds_batch = []
                mids = batch['target_file']
                cids = batch['target_category']
                
                kp_gt_exists_batch = [False]*B
                # loop to find the matching entries, can't think of a better way
                # the full kp list will be filled to calculate kp metrics together later
                out_norm_val_0.extend(batch['target_norm_val_0'])
                out_norm_val_1.extend(batch['target_norm_val_1'])
                for idx, (cid, mid) in enumerate(zip(cids, mids)):
                    for entry in kpn_ds:
                        # print(entry["model_id"], mid)
                        # print(entry["class_id"], str(cid.numpy()))
                        # input()
                        scid = str(cid.numpy()).zfill(8)
                        if entry["model_id"] == mid and entry["class_id"] == scid:
                            kpn_ds_batch.append(entry)
                            kpn_ds_ordered.append(entry)
                            kp_gt_exists_batch[idx] = True
                            
                            # gt kp per entry
                            ground_truths = []
                            for kp in entry["keypoints"]:
                                # normalize as well
                                norm_val_0 = batch['target_norm_val_0'][idx]
                                norm_val_1 = batch['target_norm_val_1'][idx]
                                nkp, _, _ = normalize_scale_min_max(torch.Tensor(kp["xyz"]), dmin=norm_val_0, dmax=norm_val_1)
                                ground_truths.append(nkp.numpy())
                            
                            gt_kps.append(ground_truths)
                            break
            else:
                raise ValueError(f"Unsupported combination of gt_kp_mode: {opt.gt_kp_mode} and gt_kp_src: {opt.gt_kp_src}")
            
            kp_gt_exists.extend(kp_gt_exists_batch)
            # ---------------------------------------------------


            # ---------------- hacky filter for the recon ----------------
            # dist_thresh = 3
            # mean_pt = torch.mean(recon_batch.cpu().numpy(), 1).unsqueeze(1)  # [B, 1, 3]
            # dist_from_mean = np.linalg.norm(recon_batch - mean_pt, axis=2)
            # in_mask = dist_from_mean < dist_thresh
            # while not np.all(in_mask):
            #     recon_batch = recon_batch.cpu().numpy()[in_mask]
            #     mean_pt = torch.mean(recon_batch, 1).unsqueeze(1)  # [B, 1, 3]
            #     dist_from_mean = np.linalg.norm(recon_batch - mean_pt, axis=2)
            #     in_mask = dist_from_mean < dist_thresh
                
            # # make sure recon_batch still has the same number of points as gt_batch
            # n_pts = recon_batch.shape[1]
            # gt_batch = gt_batch[:, :n_pts, :]
            # -----------------------------------------------------------

            # One call for all candidates
            if recon_batch.dtype != gt_batch.dtype:
                gt_batch = gt_batch.to(recon_batch.dtype)
            res = EMD_CD_recon(recon_batch, gt_batch, reduced=False)
            cd_all = res["CD"].detach().cpu().numpy()  # shape [B]
            emd_all = res["EMD"].detach().cpu().numpy()  # shape [B]
            cds.extend(list(cd_all))
            emds.extend(list(emd_all))
            
            # store kps for metrics
            kp_batch = torch.reshape(kp_batch, (B, opt.key_points, 3))
            kp_np = kp_batch.cpu().numpy()
            kps.extend(list(kp_np))

            # calculate the view consistency
            # the gt branches above append one entry per annotated sample in sample order,
            # so this batch's gt keypoints are the tail of gt_kps
            n_new = sum(kp_gt_exists_batch)
            gt_kps_batch = gt_kps[len(gt_kps) - n_new:]
            gi = 0
            for i in range(B):
                if not kp_gt_exists_batch[i]:
                    kvc.append(float("nan"))
                    ukvc.append(float("nan"))
                    continue
                gt_kp_i = gt_kps_batch[gi]
                gi += 1
                kvc.append(ordered_kp_distance(kp_np[i], gt_kp_i))
                ukvc.append(chamfer_kp(kp_np[i], gt_kp_i))

            names.extend(name_batch)
            psns.extend(psn_batch)
            covs.extend([None if c is None else float(c) for c in cov_batch])

            if collect_clouds:
                recons.extend(list(recon_batch.cpu().numpy()))
                gts.extend(list(gt_batch.cpu().numpy()))
                inputs.extend(list(input_batch.cpu().numpy()))
            
            # for testing
            # if len(cds) >= 20:
                # break

    # calculate kp metrics together for all valid samples
    # mask out samples without kp annotation
    # das and miou are calculated pair-wise, so they can only be calculated as a average for the overall dataset
    # thus they will not be labeled or attached to any specific object instances
    
    assert len(kp_gt_exists) == len(kps) == len(gt_kps) + kp_gt_exists.count(False)

    # DAS and mIoU need the semantic_id of each keypoint and the .pcd point index,
    # which only the human annotations carry, so they are undefined in the other
    # gt_kp_mode branches (those keypoints have no semantic labels to agree on)
    # TODO: not checked after changes and probably busted
    if opt.gt_kp_mode == "human":
        assert len(kpn_ds_ordered) == sum(kp_gt_exists) == len(gt_kps)
        # use the bool list to pick out the batch instances that have gt information
        kps_filtered = np.array(kps)[kp_gt_exists]
        out_norm_val_0_filtered = torch.stack(out_norm_val_0, dim=0)[kp_gt_exists].numpy()
        out_norm_val_1_filtered = torch.stack(out_norm_val_1, dim=0)[kp_gt_exists].numpy()
        fwd = fwd_alignment_scores_gtn(kpn_ds_ordered, gt_kps, kps_filtered)
        bwd = bwd_alignment_scores_gtn(kpn_ds_ordered, gt_kps, kps_filtered)
        dual = (fwd + bwd) / 2.0
        miou_at_01 = mIoU_curve_plot(kpn_ds, gt_kps, kps_filtered, out_norm_val_0_filtered, out_norm_val_1_filtered, opt.pcd_path, out_root=out_dir)
    else:
        fwd = bwd = dual = miou_at_01 = float("nan")

    # extend gt_kps to match the full list length
    # each populated element in gt_kps_extended should be a np.ndarray of keypoints ()
    # since gt kps can have different number of kps per instance in a batch, the outcome array may be staggered
    # annoying to do it with np masked assignment
    # gt_kps is only populated for annotated samples, so walk it with an iterator
    # rather than zipping it against the full-length mask (which would mispair)
    gt_kps_iter = iter(gt_kps)
    gt_kps_extended = [np.array(next(gt_kps_iter)) if m else None for m in kp_gt_exists]

    if not no_save:
        save_recon_geoms(recons, gts, inputs, names, psns, kps, gt_kps_extended, cds, emds, covs, opt, out_dir=out_dir)

    cd_mean = np.mean(cds)
    emd_mean = np.mean(emds)
    valid_covs = [c for c in covs if c is not None]
    cov_mean = float(np.mean(valid_covs)) if valid_covs else None

    # samples with no gt keypoints (or no slot correspondence) are NaN, so skip them
    kvc_mean = float(np.nanmean(kvc)) if np.any(np.isfinite(kvc)) else float("nan")
    ukvc_mean = float(np.nanmean(ukvc)) if np.any(np.isfinite(ukvc)) else float("nan")
    # near zero means the keypoints genuinely drift; large means they land in the
    # right places but get assigned to the wrong slots
    swap_gap = kvc_mean - ukvc_mean

    # how the metrics track the amount of the object the input actually showed.
    # Cheap - every array is already in memory - and self-skipping when coverage
    # is constant, which is the full-cloud case
    if not getattr(opt, "no_coverage_plot", False):
        write_coverage_outputs(
            names, psns, covs, cds, emds, kvc, ukvc,
            out_dir=out_dir / opt.model,
            run_label=f"{opt.model} / {getattr(opt, 'category', '?')} / "
                      f"{getattr(opt, 'split', '?')} / "
                      f"input={getattr(opt, 'input_type', '?')}",
            n_bins=getattr(opt, "coverage_bins", 12),
            category=getattr(opt, "category", None),
        )

    if viz:
        run_metrics = {
            "model": opt.model,
            "category": getattr(opt, "category", None),
            "split": getattr(opt, "split", None),
            "input_type": getattr(opt, "input_type", None),
            "n_samples": len(cds),
            "coverage": cov_mean,
            "cd": float(cd_mean),
            "emd": float(emd_mean),
            "kvc": kvc_mean,
            "ukvc": ukvc_mean,
            "swap_gap": swap_gap,
            "fwd": float(fwd),
            "bwd": float(bwd),
            "das": float(dual),
            "miou_at_0_1": float(miou_at_01),
        }
        # one entry per object: the loader enumerates the split exactly once, so each
        # object is seen under a single partial view. write_recon_html still takes a
        # list of views per object, so this hands it a one-element list - the spokes
        # it draws then run from each gt keypoint to the prediction for that slot,
        # i.e. the per-slot error vector.
        # every cloud is inlined into the HTML, so cap how many objects go in
        max_objects = max(0, getattr(opt, "viz_max_samples", 20))
        cloud_points = getattr(opt, "viz_cloud_points", 1500)

        objects = []
        for i in range(min(len(recons), max_objects)):
            per_view_meta = {
                "cd": float(cds[i]),
                "emd": float(emds[i]),
                "coverage": covs[i],
                "kvc": kvc[i],
                "ukvc": ukvc[i],
            }
            objects.append(
                {
                    "name": names[i],
                    "gt_cloud": subsample(gts[i], cloud_points),
                    "gt_kp": gt_kps_extended[i],
                    "views": [
                        {
                            "label": psns[i],
                            "recon": subsample(recons[i], cloud_points),
                            "input": subsample(inputs[i], cloud_points),
                            "pred_kp": kps[i],
                            "meta": per_view_meta,
                        }
                    ],
                    "meta": {
                        "partial_sample_name": psns[i],
                        **per_view_meta,
                        **keypoint_distance_stats(kps[i], gt_kps_extended[i]),
                    },
                }
            )

        viz_out = getattr(opt, "viz_out", None)
        if viz_out is None:
            viz_out = out_dir / opt.model / f"{opt.category}_recons_interactive.html"
        write_recon_html(objects, viz_out, run_metrics=run_metrics, key_points=opt.key_points)

    return cd_mean, emd_mean, dual, miou_at_01, kvc_mean, ukvc_mean, swap_gap


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
        opt.partial_view_mode = "full"   # hack, the partial pc will not be used
    else:
        opt.partial_view_mode = opt.input_type
    
    # TODO: maybe automate choosing between loaders (requires changes to other files)
    loader = make_loader(opt)
    # loader = make_h5loader(opt)

    print(f"[✓] Running reconstruction evaluation for model={opt.model}, "
          f"category={opt.category}, split={opt.split}")
    print(f"[✓] Output dir: {opt.output_dir}")
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    cd, emd, das, miou, kvc, ukvc, swap_gap = run_reconstruction(
        model, loader, opt, out_dir=opt.output_dir, no_save=opt.no_save, viz=opt.viz
    )

    print(f"Chamfer Distance (mean): {cd:.8f}")
    print(f"EMD (Sinkhorn)   (mean): {emd:.8f}")
    print(f"Dual Alignment Score: {das:.6f}")
    print(f"MIoU: {miou:.6f}")
    print(f"KVC (ordered, mean):     {kvc:.6f}")
    print(f"uKVC (chamfer, mean):    {ukvc:.6f}")
    print(f"Slot-swap gap:           {swap_gap:.6f}")

    if not opt.no_save:
        run_id = save_run(opt.db_path, opt, cd, emd)
        print(f"[✓] saved to {opt.db_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
