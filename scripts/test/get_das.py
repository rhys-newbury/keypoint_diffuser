import argparse
import collections
import contextlib
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from baselines.test_base import TestBase
from classes import MODEL_CLASSES
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)
from torchvision import transforms


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


def try_add_arg(p: argparse.ArgumentParser, arg, type=None, default=None, **kwargs):
    with contextlib.suppress(argparse.ArgumentError):
        if type:
            p.add_argument(arg, type=type, default=default, **kwargs)
        else:
            p.add_argument(arg, default=default, **kwargs)


p = argparse.ArgumentParser()
subparsers = p.add_subparsers(dest="model", required=True)

# Create a subparser for each model
for model_name, model_cls in MODEL_CLASSES.items():
    subparser = subparsers.add_parser(model_name)
    model_cls.get_parser(
        subparser
    )  # pass the subparser in instead of creating inside get_parser
    try_add_arg(subparser, "--ckpt", type=Path)
    try_add_arg(
        subparser,
        "--annotation-json",
        type=Path,
        default="/app/annotations/airplane.json",
    )
    try_add_arg(subparser, "--pcd-path", type=Path, default="/app/pcds")
    try_add_arg(subparser, "--batch-size", type=int, default=32)
    try_add_arg(subparser, "--key-points", type=int, default=10)

    try_add_arg(
        subparser, "--category", type=str, help="Category of objects", default="chair"
    )
    try_add_arg(subparser, "--db-path", type=Path, default=Path("results.db"))
    try_add_arg(subparser, "--save", action="store_true")

# ----------------------------
# Utilities
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


# ----------------------------
# Prediction Function
# ----------------------------


def run_prediction(model: TestBase, opt: argparse.Namespace):
    model.model.eval()
    model.model.cuda()

    t = transforms.Compose(
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

    kpn_ds = json.load(open(opt.annotation_json))
    out_kpcd = []
    out_nfact = []
    out_Q = []

    for i in tqdm.tqdm(
        range(0, len(kpn_ds), opt.batch_size), unit_scale=opt.batch_size
    ):
        Q = []
        for j in range(opt.batch_size):
            if i + j >= len(kpn_ds):
                continue
            entry = kpn_ds[i + j]
            cid = entry["class_id"]
            mid = entry["model_id"]
            pc_path = opt.pcd_path / cid / f"{mid}.pcd"
            pc = naive_read_pcd(pc_path)

            pcmax = pc.max()
            pcmin = pc.min()
            pcn = (pc - pcmin) / (pcmax - pcmin)
            pcn = 2.0 * (pcn - 0.5)
            out_nfact.append([pcmax, pcmin])
            Q.append(pcn)

        if len(Q) == 1:
            Q.append(Q[-1])
            out_nfact.append(out_nfact[-1])
        with torch.no_grad():
            Q = np.array(Q)
            out_Q.append(Q)
            T_nb = []
            for i in range(Q.shape[0]):
                data = {"coord": Q[i]}
                T_nb.append(t(data))

            batch = collate_fn(
                T_nb
            )  # assumes collate_fn knows how to stack dictionaries correctly
            batch = {k: v.cuda() for k, v in batch.items()}
            batch["orig"] = Q

            with torch.no_grad():
                key_points = model.get_keypoints(batch)

            for kp in key_points:
                out_kpcd.append(kp)

    predicted = {"kpcd": out_kpcd, "nfact": out_nfact}

    print(f"[✓] Prediction completed for {len(out_kpcd)} samples.")
    return kpn_ds, predicted, out_Q


# ----------------------------
# Evaluation Functions
# ----------------------------


def fwd_alignment_scores(kpn_ds, predicted):
    preds = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted["kpcd"], predicted["nfact"]):
        dmax, dmin = nfact

        ground_truths = []
        for kp in entry["keypoints"]:
            nkp = (kp["xyz"] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)

            ground_truths.append(nkp)
        ground_truths = np.array(ground_truths)
        dist = np.sum(
            (np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1
        )
        argminfwd = np.argmin(dist, -1)
        preds.append([entry["keypoints"][argm]["semantic_id"] for argm in argminfwd])
    acc = [np.mean(np.array(pa) == np.array(pb)) for pa in preds for pb in preds]
    return np.mean(acc)


def bwd_alignment_scores(kpn_ds, predicted):
    preds = collections.defaultdict(list)
    for entry, kpcd, nfact in zip(kpn_ds, predicted["kpcd"], predicted["nfact"]):
        dmax, dmin = nfact
        ground_truths = []
        for kp in entry["keypoints"]:
            nkp = (kp["xyz"] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)

            ground_truths.append(nkp)
        ground_truths = np.array(ground_truths)
        dist = np.sum(
            (np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1
        )
        argminbwd = np.argmin(dist, -2)
        for i, kp in enumerate(entry["keypoints"]):
            preds[kp["semantic_id"]].append(argminbwd[i])
    q = []
    for arr in preds.values():
        arr = np.array(arr)
        q.append(np.mean(arr[:, None] == arr[None, :]))
    return np.mean(q)


def mIoU(kpn_ds, predicted, pcd_path):
    thresholds = np.linspace(0.0, 0.1)
    kps = []
    gts = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted["kpcd"], predicted["nfact"]):
        cid = entry["class_id"]
        mid = entry["model_id"]
        pc = naive_read_pcd(pcd_path / cid / f"{mid}.pcd")
        dmax, dmin = nfact

        ground_truths = [pc[kp["pcd_info"]["point_index"]] for kp in entry["keypoints"]]
        gts.append(ground_truths)

        npc = (pc - dmin) / (dmax - dmin)
        npc = 2.0 * (npc - 0.5)

        dist = np.sqrt(
            np.sum((np.expand_dims(kpcd, 1) - np.expand_dims(npc, 0)) ** 2, axis=-1)
        )
        kps.append(pc[np.argmin(dist, -1)])
    for threshold in thresholds:
        npos = fp_sum = fn_sum = 0
        for ground_truths, kpcd in zip(gts, kps):
            dist = np.sqrt(
                np.sum(
                    (np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2,
                    axis=-1,
                )
            )
            npos += dist.shape[1]
            fp_sum += np.sum(np.min(dist, -1) > threshold)
            fn_sum += np.sum(np.min(dist, -2) > threshold)
        yield (npos - fn_sum) / (npos + fp_sum)


def mIoU_curve_plot(kpn_ds, predicted, pcd_path):
    plt.style.use("seaborn")
    miou_curve = list(mIoU(kpn_ds, predicted, pcd_path))
    plt.plot(np.linspace(0.0, 0.1), miou_curve)
    plt.title("mIoU Curve")
    plt.xlabel("Distance Threshold")
    plt.ylabel("mIoU")
    plt.grid(True)
    plt.savefig("mIoU.png")
    return miou_curve[-1]


def save_numpy_geoms(
    kpn_ds, predicted, out_Q, opt, out_dir: Path = Path("geom_np"), vis_max=1000
):
    """
    Saves raw numpy arrays for each sample:
      - pc.npy        : normalized point cloud
      - pred_kp.npy   : predicted keypoints (normalized)
      - gt_kp.npy     : ground truth keypoints (normalized)
    Also dumps meta.json with ids + config.
    """

    out_dir.mkdir(exist_ok=True)
    flat_Q = [q for qb in out_Q for q in qb]  # flatten batches

    saved = 0
    for idx, (entry, kpcd_norm, nfact) in enumerate(
        zip(kpn_ds, predicted["kpcd"], predicted["nfact"])
    ):
        if saved >= vis_max or idx >= len(flat_Q):
            break

        pc = np.asarray(flat_Q[idx], dtype=np.float32)  # normalized pc
        pred = np.asarray(kpcd_norm, dtype=np.float32)  # normalized preds
        # normalize GT with nfact
        gt = []
        for kp in entry["keypoints"]:
            xyz = np.asarray(kp["xyz"], dtype=np.float32)
            pcmax, pcmin = nfact
            nkp = (xyz - pcmin) / (pcmax - pcmin)
            nkp = 2.0 * (nkp - 0.5)

            gt.append(nkp)
        gt = np.asarray(gt, dtype=np.float32)

        # save per-sample folder
        dst = out_dir / opt.category / entry["model_id"]
        dst.mkdir(exist_ok=True)
        np.save(dst / "pc.npy", pc)
        np.save(dst / "pred_kp.npy", pred)
        np.save(dst / "gt_kp.npy", gt)

        meta = {
            "model": opt.model,
            "category": opt.category,
            "class_id": entry["class_id"],
            "model_id": entry["model_id"],
            "num_pc": int(pc.shape[0]),
            "num_pred": int(pred.shape[0]),
            "num_gt": int(gt.shape[0]),
        }
        with open(dst / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved += 1

    print(f"[✓] Saved {saved} samples as numpy arrays under {out_dir}/")


# ----------------------------
# Database
# ----------------------------
def init_db(db_path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    # single flat table with your config + metrics
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            ckpt TEXT,
            annotation_json TEXT,
            pcd_path TEXT,
            batch_size INTEGER,
            key_points INTEGER,
            category TEXT,
            fwd REAL,
            bwd REAL,
            das REAL,
            miou_at_0_1 REAL
        )
    """
    )
    con.commit()
    return con


def save_run(db_path, opt, fwd, bwd, das, miou_at_0_1):
    con = init_db(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO runs (model, ckpt, annotation_json, pcd_path, batch_size, key_points, category,
                          fwd, bwd, das, miou_at_0_1)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            opt.model,
            str(opt.ckpt) if opt.ckpt is not None else None,
            str(opt.annotation_json),
            str(opt.pcd_path),
            int(opt.batch_size),
            int(opt.key_points),
            str(opt.category),
            float(fwd),
            float(bwd),
            float(das),
            float(miou_at_0_1),
        ),
    )
    con.commit()
    run_id = cur.lastrowid
    con.close()
    return run_id


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    opt = p.parse_args()

    model_cls = MODEL_CLASSES[opt.model]
    model = model_cls()  # or model_cls(opt) if your ctor expects args
    model.load_model(opt.ckpt, opt)

    kpn_ds, predicted, out_Q = run_prediction(model, opt)
    if opt.save:
        save_numpy_geoms(
            kpn_ds, predicted, out_Q, opt, out_dir=Path("output") / opt.model
        )

    fwd = fwd_alignment_scores(kpn_ds, predicted)
    bwd = bwd_alignment_scores(kpn_ds, predicted)
    dual = (fwd + bwd) / 2.0
    miou_at_01 = mIoU_curve_plot(kpn_ds, predicted, opt.pcd_path)

    print("Forward Alignment Score:", fwd)
    print("Backward Alignment Score:", bwd)
    print("DUAL ALIGNMENT SCORE: ", (fwd + bwd) / 2)
    print("mIoU at threshold 0.1: ", miou_at_01 * 100)
    run_id = save_run(
        db_path=opt.db_path, opt=opt, fwd=fwd, bwd=bwd, das=dual, miou_at_0_1=miou_at_01
    )
    print(f"[✓] saved to {opt.db_path} (run_id={run_id})")
