"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""

import argparse
from contextlib import nullcontext
from glob import glob
from pathlib import Path

import torch
from baselines.key_grid.composed_chamfer import loss_all
from baselines.key_grid.merger_net import Net
from db_utils import save_train_run
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from torch import optim
from tqdm import tqdm
from utils import DATASET

import wandb


arg_parser = argparse.ArgumentParser(
    description="Training Key_Grid for the PointNet++ on the ClothesNet dataset."
)

arg_parser.add_argument(
    "-k",
    "--key_points",
    type=int,
    default=10,
    help="Requested number of keypoints to detect.",
)
arg_parser.add_argument("--db", type=Path, help="Database path")

arg_parser.add_argument("-b", "--batch", type=int, default=8, help="Batch size.")

arg_parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="Number of epochs to train."
)
arg_parser.add_argument(
    "--max-points",
    type=int,
    default=2048,
    help="Indicates maximum points in each input point cloud.",
)
arg_parser.add_argument("--keynumber", type=int, help="", default=14)
arg_parser.add_argument("--chamfer", type=int, help="", default=20)
arg_parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
arg_parser.add_argument(
    "--category", type=str, help="Category of objects", default="chair"
)
arg_parser.add_argument("--ckpt-dir", type=Path, default=Path("."))


def feed(net, optimizer, loader, train, shuffle, batch, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    net.train(train)
    net.cuda()

    # Global or external storage for losses (initialize somewhere before training loop)

    with nullcontext() if train else torch.no_grad():
        for _i, batch_x in enumerate(
            tqdm(loader, total=len(loader), desc="Training", unit="batch")
        ):
            batch_x = batch_x.to(next(net.parameters()).device)

            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x, "True")
            loss = loss_all(batch_x, keypoint, reconstruct, epoch, ns)
            running_init_points += loss["init_points"]
            if epoch > cfg.chamfer:
                running_chamfer += loss["chamfer"]

            wandb.log(loss)
            loss = sum(loss.values())

            if train:
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    cfg = arg_parser.parse_args()
    batch = cfg.batch

    wandb.init(project=f"keygrid_{cfg.category}_train", config=cfg)

    ckpt_dir = cfg.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()
    save_train_run(cfg.db, "SC3K", cfg.category, ckpt_dir, cfg.key_points)

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files,
        normalize=True,
        include_label=False,
        object_name=cfg.category,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=0
    )

    net = Net(cfg.max_points, cfg.key_points).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(cfg.epochs):
        feed(net, optimizer, loader, True, False, batch, epoch, cfg)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            },
            f"{ckpt_dir!s}/{cfg.key_points}kp_{epoch}.pth",
        )
