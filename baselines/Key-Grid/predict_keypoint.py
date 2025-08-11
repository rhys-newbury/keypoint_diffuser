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
import os
from glob import glob
from H5Dataset import H5Dataset
import open3d as o3d


def save_batch_keypoints_npz(x, keypoints, save_dir="viz_npz"):
    """
    Save each sample in the batch to a .npz file for later visualization.

    Args:
        x: [B, N, 3] torch.Tensor (point clouds)
        keypoints: [B, K, 3] torch.Tensor (keypoints)
        save_dir: where to save .npz files
    """
    os.makedirs(save_dir, exist_ok=True)
    x = x.cpu().numpy()
    keypoints = keypoints.cpu().numpy()

    for i in range(x.shape[0]):
        save_path = os.path.join(save_dir, f"sample_{i:04d}.npz")
        np.savez_compressed(save_path, pointcloud=x[i], keypoints=keypoints[i])
        print(f"[✓] Saved: {save_path}")


arg_parser = argparse.ArgumentParser(description="Predictor for Keypoint on Clothesnet dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='/app/99.pth',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

ns = arg_parser.parse_args()
net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()
out_kpcd = []
net.cuda()
# x = np.loadtxt("Key_Grid/dataset/fold/pant.txt")
# x = x.reshape(int(x.shape[0]/2048), 2048, 3)
TESTSET="/data/shapenetcorev2_hdf5_2048/val"


h5_files_test = glob(f'{TESTSET}**/*.h5', recursive=True)
dataset_test = H5Dataset(h5_files_test, normalize=True, include_label=False,subclasses=(14,), )
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=4)

all_keypoints = []

with torch.no_grad():
    for batch in tqdm.tqdm(loader_test, desc="Generating keypoints", unit="batch"):
        # Make sure the input tensor is on the correct device
        x = batch.float().to(ns.device)  # [B, N, 3] or [B*N, 3]
        # If your model expects batch input as a dict, pass `batch` directly
        key_points, _ = net(x, "False")  # [B, K, 3]

        # import pdb; pdb.set_trace()
        save_batch_keypoints_npz(x, key_points)

        

        all_keypoints.append(key_points.cpu())  # Store on CPU to avoid OOM

# Combine into single tensor [Total_B, K, 3]
all_keypoints = torch.cat(all_keypoints, dim=0)

np.savetxt('chair_keypoint.txt', np.array(all_keypoints))    
