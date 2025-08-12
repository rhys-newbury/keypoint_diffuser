"""
Created on Aug 1, 2025
Combined Predictor and Evaluator for Skeleton Merger on KeypointNet dataset.

@author: eliphat
"""

import argparse
import collections
import json
import os

import matplotlib.pyplot as plotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEOptions
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
arg_parser = argparse.ArgumentParser(
    description="Combined prediction and evaluation for Skeleton Merger.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Prediction args
arg_parser.add_argument(
    "-a",
    "--annotation-json",
    type=str,
    default="annotations/cap.json",
    help="Annotation JSON file path from KeypointNet dataset.",
)
arg_parser.add_argument(
    "-i",
    "--pcd-path",
    type=str,
    default="pcds",
    help="Point cloud file folder path from KeypointNet dataset.",
)
arg_parser.add_argument(
    "-m",
    "--checkpoint-path",
    type=str,
    default="trained_merger.pt",
    help="Model checkpoint file path.",
)
arg_parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cuda",
    help='PyTorch device (e.g., "cuda" or "cpu").',
)
arg_parser.add_argument(
    "-k", "--n-keypoint", type=int, default=10, help="Number of keypoints to detect."
)
arg_parser.add_argument(
    "-b", "--batch", type=int, default=8, help="Batch size for prediction."
)
arg_parser.add_argument(
    "--max-points",
    type=int,
    default=2048,
    help="Max number of points in each point cloud.",
)
arg_parser.add_argument(
    "-p",
    "--prediction-output",
    type=str,
    default="merger_prediction.npz",
    help="File path for saving predictions.",
)

# Evaluation flags
arg_parser.add_argument(
    "--op-align-fwd", action="store_true", help="Compute forward alignment score."
)
arg_parser.add_argument(
    "--op-align-bwd", action="store_true", help="Compute backward alignment score."
)
arg_parser.add_argument("--op-miou", action="store_true", help="Plot mIoU curve.")

# ----------------------------
# Utilities
# ----------------------------


def viz_pointcloud_and_keypoints(pc, kps, title, idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pc = np.array(pc)
    kps = np.array(kps)
    ax.scatter(
        pc[:, 0], pc[:, 1], pc[:, 2], c="gray", s=1, alpha=0.5, label="Point Cloud"
    )
    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c="red", s=30, label="Keypoints")
    ax.set_title(title)
    ax.legend()
    plt.savefig(f"{idx}.png")


def _normalize_to_unit_box(points):
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (max_xyz + min_xyz) / 2
    scale = (max_xyz - min_xyz).max()
    return (points - center) / scale, center, scale


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


def run_prediction(opt):
    net = AutoEncoder(opt)
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)

    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt["states"])
    net.eval()
    net.cuda()

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
    out_unitbox_meta = []  # To store unit box center and scale

    for i in tqdm.tqdm(
        range(0, len(kpn_ds), opt.batch_size), unit_scale=opt.batch_size
    ):
        Q = []
        Q_nb = []
        Q_orig = []
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

            pc[:, :3], center, scale = _normalize_to_unit_box(pc[:, :3])

            Q_nb.append(pc)
            Q.append(pcn)
            Q_orig.append(naive_read_pcd(pc_path))
            out_nfact.append([pcmax, pcmin])
            out_unitbox_meta.append((center, scale))

        if len(Q) == 1:
            Q.append(Q[-1])
            out_nfact.append(out_nfact[-1])
        with torch.no_grad():
            Q_nb = np.array(Q_nb)
            T_nb = []
            for i in range(Q_nb.shape[0]):
                data = {"coord": Q_nb[i, ...]}
                T_nb.append(t(data))

            batch = collate_fn(
                T_nb
            )  # assumes collate_fn knows how to stack dictionaries correctly
            batch = {k: v.cuda() for k, v in batch.items()}

            z0, _, _ = net.encode(batch)
            key_points = z0.reshape(-1, 10, 3)
            start_idx = len(out_kpcd)

            for idx in range(key_points.shape[0]):
                kp_unitbox = key_points[idx].cpu().numpy()  # (K, 3)
                center, scale = out_unitbox_meta[start_idx + idx]
                pcmax, pcmin = out_nfact[start_idx + idx]

                # Unnormalize from unit box to original
                kp_abs = kp_unitbox * scale + center  # (K, 3)

                # Renormalize to pcn format
                kp_pcn = (kp_abs - pcmin) / (pcmax - pcmin)
                kp_pcn = 2.0 * (kp_pcn - 0.5)

                out_kpcd.append(kp_pcn)

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
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    kpn_ds, predicted = run_prediction(opt)

    # Run prediction

    # Run evaluations

    fwd = fwd_alignment_scores(kpn_ds, predicted)
    bwd = bwd_alignment_scores(kpn_ds, predicted)
    print("Forward Alignment Score:", fwd)
    print("Backward Alignment Score:", bwd)
    print("DUAL ALIGNMENT SCORE: ", (fwd + bwd) / 2)
    print(
        "mIoU at threshold 0.1: ",
        mIoU_curve_plot(kpn_ds, predicted, opt.pcd_path) * 100,
    )
    # e.g.,
    # python3 /app/scripts/test_das.py --latent_dim=10 --n_iterations=20000 -c=../configs/object.yaml -t=../configs/test.yaml --category=03001627 --extra_latent=5 --use_old=True --use_edm=True --ckpt=/app/data2/keypoints/logs/dandy-sun-228/checkpoints/net_20000.pth
#  python3 /app/scripts/test_das.py --latent_dim=10 --n_iterations=20000 -c=../configs/object.yaml -t=../configs/test.yaml --category=03001627 --extra_latent=5 --use_old=True --use_edm=True --ckpt=/app/data2/keypoints/logs/dandy-sun-228/checkpoints/net_20000.pth
