import argparse
from glob import glob

import torch
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from sc3k import network
from sc3k.utils import AverageMeter, compute_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


p = argparse.ArgumentParser(description="Train SC3K (argparse version)")

# Core training args
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--num-workers", type=int, default=4)
p.add_argument("--max-epoch", type=int, default=200)

p.add_argument("--lr", type=float, default=1e-3)

# Domain/task specifics you reference
p.add_argument("--class-name", type=str, default="default_class")
p.add_argument("--key-points", type=int, default=10)
p.add_argument("--log-path", type=str, default=None)
p.add_argument("--category", type=str, help="Category of objects", default="chair")

# Loss term weights
p.add_argument("--separation", type=float, default=1.0)
p.add_argument("--overlap", type=float, default=1.0)
p.add_argument("--overlap-threshold", type=float, default=0.05)

p.add_argument("--shape", type=float, default=6.0)
p.add_argument("--consist", type=float, default=1.0)
p.add_argument("--volume", type=float, default=1.0)
p.add_argument("--pose", type=float, default=0.07)
p.add_argument("--lamda", type=float, default=0.0)
p.add_argument("--lamda2", type=float, default=0.0)
p.add_argument("--sample-points", type=int, default=2048)


def train(cfg):
    writer = SummaryWriter("train_summary")

    DATASET = "/app/shapenetcorev2_hdf5_2048/train"
    TESTSET = "/app/shapenetcorev2_hdf5_2048/val"

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files,
        normalize=True,
        random_rotate=True,
        include_label=False,
        object_name=cfg.category,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    h5_files_test = glob(f"{TESTSET}**/*.h5", recursive=True)
    dataset_test = H5Dataset(
        h5_files_test,
        normalize=True,
        random_rotate=True,
        include_label=False,
        object_name=cfg.category,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.sc3k(cfg).to(device)  # cuda()   # unsupervised network
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    meter = AverageMeter()
    best_loss = 1e10
    train_step = 0
    val_step = 0
    for epoch in range(cfg.max_epoch):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for _i, data in enumerate(train_iter):
            kp1, kp2 = model(data)
            loss = compute_loss(kp1, kp2, data, writer, train_step, cfg, split="train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
            writer.add_scalar(
                "train_loss/overall", loss, train_step
            )  # write training loss
            train_step += 1  # increment in train_step

        train_loss = meter.avg

        # validation loss
        model.eval()
        meter.reset()
        val_iter = tqdm(val_dataloader)
        for _i, data in enumerate(val_iter):
            with torch.no_grad():
                kp1, kp2 = model(data)
                loss = compute_loss(kp1, kp2, data, writer, val_step, cfg, split="val")

                writer.add_scalar(
                    "val_loss/overall", loss, val_step
                )  # write validation loss
                val_step += 1  # increment in val_step

            val_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
        val_loss = meter.avg
        if val_loss < best_loss:
            best_loss = meter.avg
            torch.save(
                model.state_dict(),
                f"Best_{cfg.class_name}_{cfg.key_points}kp.pth",
            )

        writer.add_scalars(
            "loss_per_epoch", {"train_loss": train_loss, "val_loss": val_loss}, epoch
        )  # write validation loss

    writer.close()  # close the summary writer
    torch.save(
        model.state_dict(),
        f"{cfg.class_name}_{cfg.key_points}kp_{cfg.max_epoch}.pth",
    )


if __name__ == "__main__":
    cfg = p.parse_args()
    cfg.task = "generic"
    cfg.split = "train"
    train(cfg)
