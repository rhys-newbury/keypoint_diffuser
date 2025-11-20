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
from datasets.discovery import discover_datasets
from db_utils import save_train_run
from torch import optim
from tqdm import tqdm
from utils import DATA_DIR, DATASET

import wandb


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())

parser = argparse.ArgumentParser(
    description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=dataset_choices,
    default=dataset_choices[0] if dataset_choices else None,
    help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
)

parser.add_argument("--category", type=str, help="Category of objects")
parser.add_argument("--db", type=Path, help="Database path")

parser.add_argument(
    "-k",
    "--key_point",
    type=int,
    default=10,
    help="Requested number of keypoints to detect.",
)
parser.add_argument(
    "-d", "--device", type=str, default="cuda", help="Pytorch device for training."
)
parser.add_argument("-b", "--batch", type=int, default=8, help="Batch size.")
parser.add_argument(
    "-e", "--epochs", type=int, default=80, help="Number of epochs to train."
)
parser.add_argument(
    "--max-points",
    type=int,
    default=2048,
    help="Indicates maximum points in each input point cloud.",
)
parser.add_argument("--ckpt_dir", type=Path, default=Path("."))


def L2(embed):
    return 0.01 * (torch.sum(embed**2))


def feed(net, optimizer, loader, epoch):
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(True)

    with contextlib.nullcontext():
        running_lrc = 0.0
        running_ldiv = 0.0
        running_loss = 0.0
        total_time = 0.0

        loop = tqdm(enumerate(loader), total=len(loader), desc="Train")

        for i, batch_x in loop:
            start = time.time()

            batch_x = batch_x.to(next(net.parameters()).device)

            optimizer.zero_grad()
            try:
                RPCD, _KPCD, _KPA, LF, MA = net(batch_x)
            except ValueError:
                continue

            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            loss.backward()
            optimizer.step()

            running_lrc += blrc.item()
            running_ldiv += bldiv.item()
            running_loss += loss.item()

            batch_time = time.time() - start
            total_time += batch_time

            prefix = "train"

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
    ns = parser.parse_args()

    batch = ns.batch
    wandb.init(
        project=f"skeleton_merger_{ns.category if ns.dataset == 'H5Dataset' else 'People'}_train",
        config=ns,
    )

    ckpt_dir = ns.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()

    save_train_run(ns.db, "SM", ns.category, ckpt_dir, ns.key_point)

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    DatasetClass = AVAILABLE_DATASETS[ns.dataset]

    dataset = DatasetClass(
        h5_files=h5_files,
        root_dir=DATA_DIR,
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

    net = Net(ns.max_points, ns.key_point).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)
    for epoch in range(ns.epochs):
        feed(net, optimizer, loader, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            },
            f"{ckpt_dir!s}/{ns.key_point}kp_{epoch}.pth",
        )
