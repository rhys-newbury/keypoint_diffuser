# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""
import random
import math
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy
import numpy as np
from merger.data_flower import all_h5
from merger.merger_net import Net
from merger.composed_chamfer import loss_all
from newdataset import PointCloudDataset
from torch.utils.data import DataLoader
from glob import glob
from H5Dataset import H5Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description="Training Key_Grid for the PointNet++ on the ClothesNet dataset.")
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='Key_Grid/model/chair.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument("--keynumber", type=int, help="", default=14)
arg_parser.add_argument("--chamfer", type=int, help="", default=20)
arg_parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)

loss_history = {
    "init_points": [],
    "chamfer": [],
}

def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, loader, train, shuffle, batch, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    running_smoothing = 0.0
    net.train(train)
    net.cuda()

    # Global or external storage for losses (initialize somewhere before training loop)

    with contextlib.suppress() if train else torch.no_grad():
        for i, batch_x in enumerate(tqdm(loader, total=len(loader), desc="Training", unit="batch")):
            batch_x = batch_x.to(next(net.parameters()).device)

            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x, 'True')
            loss = loss_all(batch_x, keypoint, reconstruct, epoch, ns)
            running_init_points += loss['init_points']
            if epoch> ns.chamfer:
                running_chamfer += loss['chamfer']
            loss = sum(loss.values())
        
            if train:
                loss.backward()
                optimizer.step()
    
      
            # print('[%s%d, %4d] init_point: %.4f chamfer: %.4f smooth: %.4f '%
            #       ('VT'[train], epoch, i, running_init_points / (i + 1), running_chamfer / (i + 1), running_smoothing/(i+1)))

    num_batches = len(loader)
    avg_init = running_init_points / num_batches
    avg_chamfer = running_chamfer / num_batches

    loss_history["init_points"].append(avg_init.item())
    try:
        loss_history["chamfer"].append(avg_chamfer.item())
    except:
        loss_history["chamfer"].append(0)

    # Plot
    plt.figure()
    plt.plot(loss_history["init_points"], label='Init Points Loss')
    plt.plot(loss_history["chamfer"], label='Chamfer Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_plots/loss_epoch_{epoch}.png")
    plt.close()

            
if __name__ == '__main__':
    ns = arg_parser.parse_args()
    batch = ns.batch
    
    DATASET = "/data/shapenetcorev2_hdf5_2048/train"
    # class_id="02691156"
    TESTSET="/data/shapenetcorev2_hdf5_2048/val"


    h5_files = glob(f'{DATASET}**/*.h5', recursive=True)
    dataset = H5Dataset(h5_files, normalize=True, include_label=False, subclasses=(12,),)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)

    # h5_files_test = glob(f'{TESTSET}**/*.h5', recursive=True)
    # dataset_test = H5Dataset(h5_files_test, normalize=True, include_label=False,subclasses=(14,), )
    # loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch, shuffle=True, num_workers=4)

    net = Net(ns.max_points, ns.n_keypoint).cuda()
    optimizer = optim.Adam(net.parameters(),lr=0.1)
    
    for epoch in range(ns.epochs):
        feed(net, optimizer, loader, True, False, batch, epoch, ns)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, f"{epoch}.pth")
       
        
        
            
