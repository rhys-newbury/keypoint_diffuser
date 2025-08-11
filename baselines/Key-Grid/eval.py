# -*- coding: utf-8 -*-
"""
Created on Aug 1, 2025
Combined Predictor and Evaluator for Skeleton Merger on KeypointNet dataset.

@author: eliphat
"""

import argparse
import json
import numpy as np
import collections
import torch
import tqdm
import os
import matplotlib.pyplot as plotlib
import merger.merger_net as merger_net

from sklearn.decomposition import PCA
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"

# ----------------------------
# Argument Parser
# ----------------------------
arg_parser = argparse.ArgumentParser(description="Combined prediction and evaluation for Skeleton Merger.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Prediction args
arg_parser.add_argument('-a', '--annotation-json', type=str, default='/data/keypointnet/annotations/cap.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='/data/keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', type=str, default='txt_chair31.pth',
                        help='Model checkpoint file path.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='PyTorch device (e.g., "cuda" or "cpu").')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size for prediction.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Max number of points in each point cloud.')
arg_parser.add_argument('-p', '--prediction-output', type=str, default='merger_prediction.npz',
                        help='File path for saving predictions.')

# Evaluation flags
arg_parser.add_argument('--op-align-fwd', action='store_true', help='Compute forward alignment score.')
arg_parser.add_argument('--op-align-bwd', action='store_true', help='Compute backward alignment score.')
arg_parser.add_argument('--op-miou', action='store_true', help='Plot mIoU curve.')

# ----------------------------
# Utilities
# ----------------------------

def viz_pointcloud_and_keypoints(pc, kps, title, idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = np.array(pc)
    kps = np.array(kps)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='gray', s=1, alpha=0.5, label='Point Cloud')
    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c='red', s=30, label='Keypoints')
    ax.set_title(title)
    ax.legend()
    plt.savefig(f"{idx}.png")



def naive_read_pcd(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float32)
    return pc

# ----------------------------
# Prediction Function
# ----------------------------

def run_prediction(opt):
    net = merger_net.Net(2048, 10).to("cuda")
    net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])

    net.eval()
    net.cuda()

    kpn_ds = json.load(open(opt.annotation_json))
    out_kpcd = []
    out_pcd = []
    out_nfact = []
    recons = []

    for i in tqdm.tqdm(range(0, len(kpn_ds), opt.batch), unit_scale=opt.batch):
        Q = []
        for j in range(opt.batch):
            if i + j >= len(kpn_ds):
                continue
            entry = kpn_ds[i + j]
            cid = entry['class_id']
            mid = entry['model_id']
            pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
            pcmax = pc.max()
            pcmin = pc.min()
            pcn = (pc - pcmin) / (pcmax - pcmin)
            pcn = 2.0 * (pcn - 0.5)

            Q.append(pcn)

            out_nfact.append([pcmax, pcmin])
            out_pcd.append(pcn)

        if len(Q) == 1:
            Q.append(Q[-1])
            out_nfact.append(out_nfact[-1])
        with torch.no_grad():
            key_points, reconstruct = net(torch.Tensor(np.array(Q)).to(ns.device), "False")  # [B, K, 3]
        for kp, r in zip(key_points, reconstruct):
            recons.append(r.cpu())
            out_kpcd.append(kp.cpu())

    predicted = {
        'kpcd': out_kpcd,
        'nfact': out_nfact,
        'pcd': out_pcd,
        'recons': recons
    }

    print(f"[✓] Prediction completed for {len(out_kpcd)} samples.")
    return kpn_ds, predicted

# ----------------------------
# Evaluation Functions
# ----------------------------

def fwd_alignment_scores(kpn_ds, predicted):
    preds = []
    idx = 0
    for entry, kpcd, nfact, npcd, recons in zip(kpn_ds, predicted['kpcd'], predicted['nfact'], predicted["pcd"], predicted["recons"]):
        idx += 1
        dmax, dmin = nfact

        ground_truths = []
        for kp in entry['keypoints']:
            nkp = (kp['xyz'] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)
            ground_truths.append(nkp)

        ground_truths = np.array(ground_truths)
        dist = np.sum((np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1)
        argminfwd = np.argmin(dist, -1)

        preds.append([entry['keypoints'][argm]['semantic_id'] for argm in argminfwd])
    acc = [np.mean(np.array(pa) == np.array(pb)) for pa in preds for pb in preds]
    return np.mean(acc)

def bwd_alignment_scores(kpn_ds, predicted):
    preds = collections.defaultdict(list)
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        dmax, dmin = nfact

        ground_truths = []
        for kp in entry['keypoints']:
            nkp = (kp['xyz'] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)

            ground_truths.append(nkp)
        ground_truths = np.array(ground_truths)
        dist = np.sum((np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1)
        argminbwd = np.argmin(dist, -2)
        for i, kp in enumerate(entry['keypoints']):
            preds[kp['semantic_id']].append(argminbwd[i])
    q = []
    for arr in preds.values():
        arr = np.array(arr)
        q.append(np.mean(arr[:, None] == arr[None, :]))
    return np.mean(q)

def mIoU(kpn_ds, predicted, pcd_path):
    thresholds = np.linspace(0., 0.1)
    kps = []
    gts = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(os.path.join(pcd_path, cid, f"{mid}.pcd"))
        dmax, dmin = nfact

        ground_truths = [pc[kp['pcd_info']['point_index']] for kp in entry['keypoints']]
        gts.append(ground_truths)
        npc = (pc - dmin) / (dmax - dmin)
        npc = 2.0 * (npc - 0.5)
        dist = np.sqrt(np.sum((np.expand_dims(kpcd, 1) - np.expand_dims(npc, 0)) ** 2, axis=-1))
        kps.append(pc[np.argmin(dist, -1)])
    for threshold in thresholds:
        npos = fp_sum = fn_sum = 0
        for ground_truths, kpcd in zip(gts, kps):
            dist = np.sqrt(np.sum((np.expand_dims(kpcd, 1) - np.expand_dims(ground_truths, 0)) ** 2, axis=-1))
            npos += dist.shape[1]
            fp_sum += np.sum(np.min(dist, -1) > threshold)
            fn_sum += np.sum(np.min(dist, -2) > threshold)
        yield (npos - fn_sum) / (npos + fp_sum)

def mIoU_curve_plot(kpn_ds, predicted, pcd_path):
    miou_curve = list(mIoU(kpn_ds, predicted, pcd_path))
    plotlib.plot(np.linspace(0., 0.1), miou_curve)
    plotlib.title("mIoU Curve")
    plotlib.xlabel("Distance Threshold")
    plotlib.ylabel("mIoU")
    plotlib.grid(True)
    plotlib.savefig("mIoU.png")
    return miou_curve[-1]  

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ns = arg_parser.parse_args()
    
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    kpn_ds, predicted = run_prediction(ns)

    # Run prediction
    # kpn_ds, predicted = run_prediction(args)

    # Run evaluations
    
    fwd =fwd_alignment_scores(kpn_ds, predicted)
    bwd =bwd_alignment_scores(kpn_ds, predicted)
    print("Forward Alignment Score:", fwd)
    print("Backward Alignment Score:", bwd)
    print("DUAL ALIGNMENT SCORE: ", (fwd + bwd) / 2)
    print(f"mIoU at threshold 0.1: ", mIoU_curve_plot(kpn_ds, predicted, ns.pcd_path) * 100)
    # e.g.,