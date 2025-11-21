import time
from glob import glob
from typing import Any

import numpy as np
import pytorch3d.io
import pytorch3d.loss
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from datasets.discovery import discover_datasets
from db_utils import save_train_run
from einops import repeat
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.models.encoder_models.common import get_linear_scheduler
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions
from keypoint_diffuser.utils.nn import save_network
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from utils import DATA_DIR, DATASET

import wandb


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())


torch.autograd.set_detect_anomaly(True)
RUN = None

from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth,
    Collect,
    Deform,
    GridSample,
    ToTensor,
)
from torchvision import transforms


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


def cosine_schedule(step, total_steps, start, end):
    """
    Cosine annealing from start -> end over total_steps using torch ops.
    Returns a torch scalar (same device as inputs).
    """
    step = torch.clamp(torch.tensor(step, dtype=torch.float32), 0, total_steps)
    t = step / total_steps
    value = end + 0.5 * (start - end) * (1 + torch.cos(torch.pi * t))
    return value


def get_data(dataset, data):
    data = dataset.uncollate(data)

    target_shape = data["target_shape"]

    target_shape_t = target_shape.transpose(1, 2)

    return None, target_shape_t


def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=points.device)

    gather_index = repeat(index, "b -> b c 1", c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, "b -> b c 1", c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(
            dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1)
        )

    if return_index:
        return sampled, indexes
    else:
        return sampled


def kp_proximity_constraints(
    code_: torch.Tensor,
    target_bn3: torch.Tensor,
    t: int,
    eps: float = 0.05,
    w_hinge: float = 100.0,
):
    """
    Enforce every keypoint in `code_` to be within `eps` of `target_bn3`.

    code_:  (B, K, 3)   predicted keypoints
    target_bn3: (B, N, 3) target points
    eps:   distance tolerance
    w_hinge: weight for hinge penalty beyond eps (per-kp)
    w_max:   extra weight on stragglers (max / worst-k)
    worst_frac: fraction of worst KPs penalized (e.g., 0.2 = worst 20%)
    """
    # L2 distances to nearest target point per keypoint
    d = torch.cdist(code_, target_bn3)  # (B, K, N)
    dmin = d.min(dim=2).values  # (B, K)

    # Hinge penalty: only violations contribute
    viol = (dmin - eps).clamp_min_(0.0)  # (B, K)
    loss_hinge = (viol[viol > 0] ** 2).mean()

    # Total constraint loss
    constraint_loss = w_hinge * loss_hinge

    # Useful diagnostics
    diag = {
        "kp/viol_min": viol.min().detach(),
        "kp/viol_med": viol.median().detach(),
        "kp/viol_max": viol.max().detach(),
        "kp/far_frac(@eps)": (dmin > eps).float().mean().detach(),
        "kp/dmin_rms": torch.sqrt((dmin**2).mean().detach()),
    }

    wandb.log(
        {
            "kp_constraint/loss": constraint_loss,
            **{k: v.item() for k, v in diag.items()},
        },
        step=t,
    )

    return constraint_loss


def get_network_data(data: dict[str, Any], key="orig"):
    opp = "deformed" if key == "orig" else "orig"

    d = {}
    for k, v in data.items():
        if k.startswith(opp):
            continue
        elif k.startswith(key):
            d[k[len(key) + 1 :]] = v
        else:
            d[k] = v
    return d


def train(opt: AEConfig):
    ckpt_dir = opt.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()
    save_train_run(opt.db, "Ours2", opt.category, ckpt_dir, opt.key_point)

    t = transforms.Compose(
        [
            Deform(
                max_stretch_factor=opt.max_stretch_factor,
                max_bending_factor=opt.max_bending_factor,
                max_twist_factor=opt.max_twist_factor,
                max_taper_factor=opt.max_taper_factor,
                max_rotation_angle=opt.max_rotation_angle,
            ),  # Forks into two versions: original and deformed
            ApplyToBoth(
                transforms.Compose(
                    [
                        GridSample(
                            keys=("coord",),
                            hash_type="fnv",
                            mode="train",
                            return_grid_coord=True,
                        ),
                        ToTensor(),
                        Collect(
                            keys=("coord", "grid_coord", "transformation", "shape"),
                            feat_keys=("coord",),
                        ),
                    ]
                )
            ),
        ]
    )

    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    DatasetClass = AVAILABLE_DATASETS[opt.dataset]

    dataset = DatasetClass(
        h5_files=h5_files,
        root_dir=DATA_DIR,
        normalize=True,
        include_label=False,
        object_name=opt.category,
        transform=t,
    )

    train_sampler = None
    shuffle = True  # Only shuffle when not using DistributedSampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    max_steps = len(dataloader) * opt.epochs
    print(f"training with max_steps=  {max_steps}")
    net = AutoEncoder(opt, max_steps=max_steps).cuda()

    net.train()
    t = 0

    optimizer = torch.optim.Adam(
        net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )

    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=opt.sched_start_epoch,
        end_epoch=opt.sched_end_epoch,
        start_lr=opt.lr,
        end_lr=opt.end_lr,
    )

    accumulation_steps = int(128 / opt.batch_size)

    cur_nimg = 0

    iter_time_start = time.time()

    lambda_0 = opt.lambda_0
    lambda_1 = opt.lambda_1
    lambda_2 = opt.lambda_2
    lambda_3 = opt.lambda_3
    lambda_4 = opt.lambda_4

    kl_warmup_steps = opt.kl_warmup_steps

    for e in range(opt.epochs):
        for _, data in enumerate(dataloader):
            target_shape_t = (
                data["target_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )
            temp_t = cosine_schedule(t, max_steps, start=0.2, end=0.01)

            diffusion_loss, code, mu, logvar = net.get_loss(
                get_network_data(data), step=t, temperature=temp_t
            )

            code_ = code[:, : opt.key_point * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )

            q = Normal(mu, torch.exp(0.5 * logvar))
            p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))

            lambda_4 = 0 if opt.lambda_4 == 0 else min(1.0, t / kl_warmup_steps)
            kl = kl_divergence(q, p).sum(dim=1).mean()

            wandb.log({"diffusion_loss": diffusion_loss}, step=t)
            wandb.log({"kl_divergence": kl}, step=t)

            fps = sample_farthest_points(target_shape_t, opt.key_point + 10).transpose(
                2, 1
            )

            fps_loss, _ = pytorch3d.loss.chamfer_distance(fps, code_)

            wandb.log({"fps_loss": fps_loss}, step=t)

            max_schedule = opt.max_schedule
            chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                code_, target_shape_t.transpose(2, 1), single_directional=True
            )

            data["deformed_shape"].view(data["orig_offset"].shape[0], -1, 3)

            deformed_matrix = data["deformed_transformation"].view(
                data["orig_offset"].shape[0], -1, 3
            )

            kp_orig = code_.reshape(code.shape[0], -1, 3)
            deformed_code, _, _ = net(get_network_data(data, "deformed"))
            kp_deformed = deformed_code.reshape(code.shape[0], -1, 3)
            kp_transformed = torch.bmm(kp_orig, deformed_matrix.transpose(1, 2))
            mse_loss = torch.mean((kp_transformed - kp_deformed) ** 2)

            loss_ = (
                lambda_0 * fps_loss
                + lambda_1 * diffusion_loss
                + lambda_2 * chamfer_loss
                + lambda_3 * mse_loss
                + lambda_4 * kl
            )

            wandb.log({"chamfer_loss": chamfer_loss, "mse_loss": mse_loss}, step=t)

            loss = max(0, max_schedule - t) / max_schedule * loss_

            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (t + 1) % accumulation_steps == 0:
                print(f"Gradient step at iteration {t + 1}")
                clip_grad_norm_(net.parameters(), opt.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            cur_nimg += opt.batch_size

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()

            if t % opt.log_interval == 0:
                samples_sec = opt.batch_size / iter_time
                losses_str = str(loss)
                log_str = (
                    f"{t:d}: iter {iter_time:.1f} sec"
                    f"{samples_sec:.1f} samples/sec {losses_str}"
                )
                print(log_str)

            t += 1

        save_network(net, ckpt_dir, network_label=f"{opt.key_point}kp", epoch_label=e)
    save_network(net, ckpt_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "train":
        wandb.init(
            project=f"ours_{opt.category if opt.dataset == 'H5Dataset' else 'People'}_train",
            config=opt,
        )
        wandb.run.log_code(".")
        train(opt)

    else:
        raise ValueError()
