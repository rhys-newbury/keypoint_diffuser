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
from datasets.discovery import discover_datasets
from db_utils import save_train_run
from torch import optim
from tqdm import tqdm
from utils import DATA_DIR, DATASET

import wandb


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())


parser = argparse.ArgumentParser(
    description="Training Key_Grid for the PointNet++ on the ClothesNet dataset."
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=dataset_choices,
    default=dataset_choices[0] if dataset_choices else None,
    help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
)


parser.add_argument(
    "-k",
    "--key_points",
    type=int,
    default=10,
    help="Requested number of keypoints to detect.",
)
parser.add_argument("--db", type=Path, help="Database path")

parser.add_argument("-b", "--batch", type=int, default=8, help="Batch size.")

parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="Number of epochs to train."
)
parser.add_argument(
    "--max-points",
    type=int,
    default=2048,
    help="Indicates maximum points in each input point cloud.",
)
parser.add_argument("--keynumber", type=int, help="", default=14)
parser.add_argument("--chamfer", type=int, help="", default=20)
parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
parser.add_argument("--category", type=str, help="Category of objects", default="chair")
parser.add_argument("--ckpt-dir", type=Path, default=Path("."))


def feed(net, optimizer, loader, train, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    net.train(train)
    net.cuda()

    with nullcontext() if train else torch.no_grad():
        for _i, batch_x in enumerate(
            tqdm(loader, total=len(loader), desc="Training", unit="batch")
        ):
            batch_x = batch_x.to(next(net.parameters()).device)

            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x, True)
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
    cfg = parser.parse_args()
    batch = cfg.batch

    wandb.init(project=f"keygridOrig_{cfg.category}_train", config=cfg)

    ckpt_dir = cfg.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()
    save_train_run(cfg.db, "KeyGridOrig", cfg.category, ckpt_dir, cfg.key_points)

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    DatasetClass = AVAILABLE_DATASETS[cfg.dataset]

    dataset = DatasetClass(
        h5_files=h5_files,
        root_dir=DATA_DIR,
        normalize=True,
        include_label=False,
        object_name=cfg.category,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=0, drop_last=True
    )

    net = Net(cfg.max_points, cfg.key_points).cuda()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1, eps=1e-2)

    for epoch in range(cfg.epochs):
        feed(net, optimizer, loader, True, epoch, cfg)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            },
            f"{ckpt_dir!s}/{cfg.key_points}kp_{epoch}.pth",
        )
