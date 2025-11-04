import argparse
from glob import glob
from pathlib import Path

import torch
from db_utils import save_train_run
from tqdm import tqdm
from utils import DATA_DIR, DATASET

import wandb
from baselines.sc3k import network
from baselines.sc3k.utils import AverageMeter, compute_loss
from datasets.discovery import discover_datasets


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())


parser = argparse.ArgumentParser(description="Train SC3K (argparse version)")

# Core training args

parser.add_argument(
    "--dataset",
    type=str,
    choices=dataset_choices,
    default=dataset_choices[0] if dataset_choices else None,
    help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
)

parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--max-epoch", type=int, default=80)
parser.add_argument("--db", type=Path, help="Database path")

parser.add_argument("--lr", type=float, default=1e-3)

# Domain/task specifics you reference
parser.add_argument("--key-points", type=int, default=10)
parser.add_argument("--log-path", type=str, default=None)
parser.add_argument("--category", type=str, help="Category of objects", default="chair")

parser.add_argument("--overlap-threshold", type=float, default=0.05)

# Loss term weights
parser.add_argument("--separation", type=float, default=0.5)
parser.add_argument("--shape", type=float, default=6.0)
parser.add_argument("--volume", type=float, default=1.0)
parser.add_argument("--overlap", type=float, default=0.07)

parser.add_argument("--consist", type=float, default=1.0)
parser.add_argument("--pose", type=float, default=0.05)


parser.add_argument("--lamda", type=float, default=0.0)
parser.add_argument("--lamda2", type=float, default=0.0)
parser.add_argument("--sample-points", type=int, default=2048)

parser.add_argument("--ckpt-dir", type=Path, default=Path("."))


def train(cfg, ckpt_dir):
    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    DatasetClass = AVAILABLE_DATASETS[cfg.dataset]

    dataset = DatasetClass(
        h5_files=h5_files,
        root_dir=DATA_DIR,
        normalize=True,
        include_label=False,
        object_name=cfg.category,
        random_rotate=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.sc3k(cfg).to(device)  # cuda()   # unsupervised network
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    meter = AverageMeter()
    train_step = 0

    for epoch in range(cfg.max_epoch):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for data in train_iter:
            kp1, kp2 = model(data)
            loss, values = compute_loss(kp1, kp2, data, cfg, split="train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())

            wandb.log(
                {**values, "train_loss/overall": loss}, step=train_step
            )  # write training loss
            train_step += 1  # increment in train_step

        train_loss = meter.avg

        torch.save(
            model.state_dict(),
            f"{ckpt_dir!s}/{cfg.key_points}kp_{epoch}.pth",
        )

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)


if __name__ == "__main__":
    cfg = parser.parse_args()
    cfg.task = "generic"
    cfg.split = "train"

    wandb.init(project=f"sc3k_{cfg.category}_train", config=cfg)

    ckpt_dir = cfg.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()

    save_train_run(cfg.db, "SC3K", cfg.category, ckpt_dir, cfg.key_points)

    train(cfg, ckpt_dir)
