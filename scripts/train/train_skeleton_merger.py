"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""
import argparse
import contextlib
import time
from glob import glob
from pathlib import Path

import torch
from baselines.skeleton_merger.composed_chamfer import composed_sqrt_chamfer
from baselines.skeleton_merger.merger_net import Net
from db_utils import save_train_run
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from torch import optim
from tqdm import tqdm
from utils import DATASET, TESTSET

import wandb


arg_parser = argparse.ArgumentParser(
    description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

arg_parser.add_argument("--category", type=str, help="Category of objects")
arg_parser.add_argument("--db", type=Path, help="Database path")

arg_parser.add_argument(
    "-k",
    "--key-points",
    type=int,
    default=10,
    help="Requested number of keypoints to detect.",
)
arg_parser.add_argument(
    "-d", "--device", type=str, default="cuda", help="Pytorch device for training."
)
arg_parser.add_argument("-b", "--batch", type=int, default=8, help="Batch size.")
arg_parser.add_argument(
    "-e", "--epochs", type=int, default=80, help="Number of epochs to train."
)
arg_parser.add_argument(
    "--max-points",
    type=int,
    default=2048,
    help="Indicates maximum points in each input point cloud.",
)
arg_parser.add_argument("--ckpt-dir", type=Path, default=Path("."))


def L2(embed):
    return 0.01 * (torch.sum(embed**2))


def feed(net, optimizer, loader, train, epoch):
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(train)

    with contextlib.suppress() if train else torch.no_grad():
        running_lrc = 0.0
        running_ldiv = 0.0
        running_loss = 0.0
        total_time = 0.0

        loop = tqdm(
            enumerate(loader), total=len(loader), desc="Train" if train else "Eval"
        )

        for i, batch_x in loop:
            start = time.time()

            batch_x = batch_x.to(next(net.parameters()).device)

            if train:
                optimizer.zero_grad()
            try:
                RPCD, KPCD, KPA, LF, MA = net(batch_x)
            except ValueError:
                continue

            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            if train:
                loss.backward()
                optimizer.step()

            running_lrc += blrc.item()
            running_ldiv += bldiv.item()
            running_loss += loss.item()

            batch_time = time.time() - start
            total_time += batch_time

            prefix = "train" if train else "val"

            wandb.log(
                {
                    f"{prefix}_loss": running_loss / (i + 1),
                    f"{prefix}_Lrc": running_lrc / (i + 1),
                    f"{prefix}_Ldiv": running_ldiv / (i + 1),
                    f"{prefix}_batch_time(s)": batch_time,
                },
                step=epoch,
            )
    return running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)


if __name__ == "__main__":
    ns = arg_parser.parse_args()

    batch = ns.batch
    wandb.init(project=f"skeleton_merger_{ns.category}_train", config=ns)

    ckpt_dir = ns.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()

    save_train_run(ns.db, "SM", ns.category, ckpt_dir, ns.key_points)

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files,
        normalize=True,
        include_label=False,
        object_name=ns.category,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
    )

    h5_files_test = glob(f"{TESTSET}**/*.h5", recursive=True)
    dataset_test = H5Dataset(
        h5_files_test,
        normalize=True,
        include_label=False,
        object_name=ns.category,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
    )

    net = Net(ns.max_points, ns.key_points).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)
    for epoch in range(ns.epochs):
        feed(net, optimizer, loader, True, epoch)
        feed(net, optimizer, loader_test, False, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            },
            f"{str(ckpt_dir)}/{ns.key_points}kp_{epoch}.pth",
        )
