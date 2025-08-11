# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:48:13 2020

@author: eliphat
"""
import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse
import open3d as o3d
import pytorch3d.loss
from emd_loss.emd_module import EMDModule


def downsample_batched_point_cloud(point_clouds, num_samples=2048):
    """
    Vectorized downsampling of batched point clouds using random sampling.

    Args:
        point_clouds (torch.Tensor): Batched point clouds of shape (B, N, 3)
        num_samples (int): Number of points to sample per batch

    Returns:
        torch.Tensor: Downsampled batched point clouds of shape (B, num_samples, 3)
    """
    B, N, C = point_clouds.shape
    if num_samples > N:
        raise ValueError("Cannot sample more points than exist in the point cloud.")

    # Generate random indices for each batch element
    rand_vals = torch.rand(B, N, device=point_clouds.device)
    _, indices = rand_vals.topk(num_samples, dim=1, largest=False, sorted=False)

    # Expand batch indices to match indices shape
    batch_indices = (
        torch.arange(B, device=point_clouds.device).view(-1, 1).expand(-1, num_samples)
    )

    # Gather points using advanced indexing
    downsampled = point_clouds[batch_indices, indices]  # Shape: (B, num_samples, 3)

    return downsampled


def emd_approx(sample, ref):
    emd = EMDModule()
    N, N_ref = sample.size(1), ref.size(1)

    min_pts = min(N, N_ref)

    if min_pts >= 2048:
        target = 2048
    elif min_pts >= 1024:
        target = 1024

    sample_ = (downsample_batched_point_cloud(sample, num_samples=target))
    ref_ = (downsample_batched_point_cloud(ref, num_samples=target))
    N, N_ref = sample_.size(1), ref_.size(1)
    assert N_ref == N, "Not sure what would EMD do in this case"

    dis, _ = emd(sample_, ref_, 0.002, 10000)  # 0.005, 50 for training
    emd_norm = torch.sqrt(dis).mean(dim=1) / N
    return emd_norm


arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='annotations/chair.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='chairmerger.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-p', '--prediction-output', type=str, default='merger_prediction.npz',
                        help='Output file where prediction results are written.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
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


kpn_ds = json.load(open(ns.annotation_json))
out_kpcd = []
out_nfact = []
merged_cds = []
hull_cds = []
merged_emds = []
hull_emds = []

for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
    Q = []
    for j in range(ns.batch):
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
    if len(Q) == 1:
        Q.append(Q[-1])
        out_nfact.append(out_nfact[-1])
    with torch.no_grad():
        Q_tensor = torch.Tensor(np.array(Q)).to(ns.device)

        recon, key_points, kpa, emb, null_activation = net(Q_tensor)
        
        activation = null_activation[0].detach().cpu().numpy()  # [45]
        parts = [r[0].detach().cpu().numpy() for r in recon]  # list of [N_i, 3]
        
        # Normalize activation to [0, 1]
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

        best_cd_merged = float('inf')
        best_cd_hull = float('inf')
        best_emd_merged = float('inf')
        best_emd_hull = float('inf')
        best_data = {}

        for ACTIVATION_THRESHOLD in np.linspace(0, 1, 11):
            filtered_parts = [p for p, a in zip(parts, activation) if a > ACTIVATION_THRESHOLD]
            if len(filtered_parts) == 0:
                continue

            merged_points = np.concatenate(filtered_parts, axis=0)
            pcd_merged = o3d.geometry.PointCloud()
            pcd_merged.points = o3d.utility.Vector3dVector(merged_points)

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                hull, _ = pcd_merged.compute_convex_hull()
            sampled_pcd = hull.sample_points_uniformly(number_of_points=2048)

            # Convert to tensors
            np_input = Q_tensor[0].cpu().numpy()
            t_input = torch.tensor(np_input, dtype=torch.float32, device=ns.device)[None, ...]
            t_merged = torch.tensor(np.asarray(pcd_merged.points), dtype=torch.float32, device=ns.device)[None, ...]
            t_sampled = torch.tensor(np.asarray(sampled_pcd.points), dtype=torch.float32, device=ns.device)[None, ...]

            cd_merged = pytorch3d.loss.chamfer_distance(t_input, t_merged)[0].item()
            cd_hull = pytorch3d.loss.chamfer_distance(t_input, t_sampled)[0].item()


            emd_merged = emd_approx(t_input, t_merged)
            emd_hull = emd_approx(t_input, t_sampled)

            emd_merged = emd_approx(t_input, t_merged).item()
            emd_hull = emd_approx(t_input, t_sampled).item()

            best_cd_merged = min(best_cd_merged, cd_merged)
            best_cd_hull = min(best_cd_hull, cd_hull)
            best_emd_merged = min(best_emd_merged, emd_merged)
            best_emd_hull = min(best_emd_hull, emd_hull)

        merged_cds.append(best_cd_merged)
        hull_cds.append(best_cd_hull)
        merged_emds.append(best_emd_merged)
        hull_emds.append(best_emd_hull)

    avg_cd_merged = sum(merged_cds) / len(merged_cds)
    avg_cd_hull = sum(hull_cds) / len(hull_cds)
    avg_emd_merged = sum(merged_emds) / len(merged_emds)
    avg_emd_hull = sum(hull_emds) / len(hull_emds)

print(f"📊 Average Best EMD (merged parts): {avg_emd_merged:.6f}")
print(f"📊 Average Best EMD (convex hull): {avg_emd_hull:.6f}")

print(f"\n📊 Average Best Chamfer Distance (merged parts): {avg_cd_merged:.6f}")
print(f"📊 Average Best Chamfer Distance (convex hull): {avg_cd_hull:.6f}")

