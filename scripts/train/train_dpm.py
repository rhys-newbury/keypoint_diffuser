import argparse
from glob import glob
from pathlib import Path

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from db_utils import save_train_run
from discovery import discover_datasets
from utils import DATA_DIR, DATASET

import wandb
from baselines.diffusion_point_cloud.autoencoder import AutoEncoder
from baselines.diffusion_point_cloud.common import get_linear_scheduler


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument("--key-points", type=int, default=10)

parser.add_argument(
    "--dataset",
    type=str,
    choices=dataset_choices,
    default=dataset_choices[0] if dataset_choices else None,
    help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
)


parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--beta_1", type=float, default=1e-4)
parser.add_argument("--beta_T", type=float, default=0.05)
parser.add_argument("--sched_mode", type=str, default="linear")
parser.add_argument("--flexibility", type=float, default=0.0)
parser.add_argument("--residual", type=eval, default=True, choices=[True, False])
parser.add_argument("--resume", type=str, default=None)

# Datasets and loaders
parser.add_argument("--category", type=str, help="Category of objects")
parser.add_argument("--db", type=Path, help="Database path")
parser.add_argument(
    "-e", "--epochs", type=int, default=80, help="Number of epochs to train."
)

parser.add_argument("--train_batch_size", type=int, default=128)

# Optimizer and scheduler
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--max_grad_norm", type=float, default=10)
parser.add_argument("--end_lr", type=float, default=1e-4)
parser.add_argument("--sched_start_epoch", type=int, default=150 * 1000)
parser.add_argument("--sched_end_epoch", type=int, default=300 * 1000)

# Training
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--ckpt-dir", type=Path, default=Path("."))

# Main loop
if __name__ == "__main__":
    args = parser.parse_args()

    args.latent_dim = args.key_points + 3 * 5

    wandb.init(project=f"dpm_{args.category}_train", config=args)
    ckpt_dir = args.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()
    save_train_run(args.db, "DPM", args.category, ckpt_dir, args.key_points)

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)

    DatasetClass = AVAILABLE_DATASETS[args.dataset]
    dataset = DatasetClass(
        h5_files=h5_files,
        root_dir=DATA_DIR,
        normalize=True,
        include_label=False,
        object_name=args.category,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = AutoEncoder(args).to(args.device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr,
    )

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        for batch in loader:
            # Load data

            x = batch.to(args.device)

            # Reset grad and model state
            optimizer.zero_grad()
            model.train()

            # Forward
            loss = model.get_loss(x)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss})
        print(loss)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            },
            f"{ckpt_dir!s}/{args.key_points}kp_{epoch}.pth",
        )
