import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plotlib
import numpy as np
import torch
import tqdm
from baselines.sc3k.test_sc3k import SC3K
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)
from torchvision import transforms


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"

# ----------------------------
# Argument Parser
# ----------------------------
p = SC3K.get_parser()
p.add_argument("--ckpt", type=Path)
p.add_argument("--annotation-json", type=Path, default="/app/annotations/chair.json")
p.add_argument("--pcd-path", type=Path, default="/app/pcds")

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


def run_prediction(model, opt):
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
            pc_path = os.path.join(opt.pcd_path, cid, f"{mid}.pcd")
            pc = naive_read_pcd(pc_path)
            pcmax = pc.max()
            pcmin = pc.min()
            pcn = (pc - pcmin) / (pcmax - pcmin)
            pcn = 2.0 * (pcn - 0.5)
            Q.append(pcn)
            out_nfact.append([pcmax, pcmin])

        if len(Q) == 1:
            Q.append(Q[-1])
            out_nfact.append(out_nfact[-1])
        with torch.no_grad():
            Q = np.array(Q)
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
    return kpn_ds, predicted


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
        pc = naive_read_pcd(os.path.join(pcd_path, cid, f"{mid}.pcd"))
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
    plotlib.style.use("seaborn")
    miou_curve = list(mIoU(kpn_ds, predicted, pcd_path))
    plotlib.plot(np.linspace(0.0, 0.1), miou_curve)
    plotlib.title("mIoU Curve")
    plotlib.xlabel("Distance Threshold")
    plotlib.ylabel("mIoU")
    plotlib.grid(True)
    plotlib.savefig("mIoU.png")
    return miou_curve[-1]


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    opt = p.parse_args()

    model = SC3K()
    model.load_model(opt.ckpt, opt)

    kpn_ds, predicted = run_prediction(model, opt)

    fwd = fwd_alignment_scores(kpn_ds, predicted)
    bwd = bwd_alignment_scores(kpn_ds, predicted)
    print("Forward Alignment Score:", fwd)
    print("Backward Alignment Score:", bwd)
    print("DUAL ALIGNMENT SCORE: ", (fwd + bwd) / 2)
    print(
        "mIoU at threshold 0.1: ",
        mIoU_curve_plot(kpn_ds, predicted, opt.pcd_path) * 100,
    )
