import os
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
from db_utils import save_train_run, find_checkpoint_from_db_dir
from einops import repeat
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.models.encoder_models.common import get_linear_scheduler
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions
from keypoint_diffuser.utils.nn import save_network
from keypoint_diffuser.utils.utils import reparameterize
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_

import wandb


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


def get_network_data(data: dict[str, Any], key="orig"):
    # key is the subset 
    # original keys were (orig, deformed)
    # add new with (orig, deformed, partial_orig, partial_deformed)
    opplist = ("orig", "deformed", "partial_orig", "partial_deformed")
    opp = tuple(o for o in opplist if o != key)

    d = {}
    for k, v in data.items():
        # skips if it belongs to another subset
        if k.startswith(opp):
            continue
        # subset specific data, keep and remove subset prefix str
        elif k.startswith(key):
            d[k[len(key) + 1 :]] = v.cuda()
        # shared data, keep
        elif type(v) == list:
            d[k] = v
        else:
            d[k] = v.cuda()
    return d


def train(opt: AEConfig):
    ckpt_dir = opt.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()
    opt.db.parent.mkdir(exist_ok=True)
    save_train_run(opt.db, "Partial", opt.category, ckpt_dir, opt.key_points)

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

    dataset = ShapesPartial(opt, transform=t)

    print("Using regular DataLoader (no DistributedSampler).")
    train_sampler = None
    shuffle = True  # Only shuffle when not using DistributedSampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    pretrained_root = opt.pretrained_root
    if str(pretrained_root).endswith(".pth"):
        # path provided is a pth file, read directly
        pretrained_dir = pretrained_root
    else:
        # path provided is a dir to training db files, which contains the ckpt_dir
        pretrained_dir = find_checkpoint_from_db_dir(pretrained_root, opt.category, opt.pretrained_epoch)
    # db files save (opt.ckpt_dir / wandb.run.name)
    pretrained = torch.load(pretrained_dir)

    net = AutoEncoder(opt).cuda()
    net.load_state_dict(pretrained["states"])
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
    epoch_time_start = time.time()

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

            # START modified partial loss ####################################################
            partial_target_shape_t = (
                data["target_partial_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            # diffusion loss (full point cloud)
            diffusion_loss, code, mu, logvar = net.get_loss(
                get_network_data(data), step=t
            )
            
            # code is z0 (entire latent including kp and aux), extract kp only to code_
            code_ = code[:, : opt.key_points * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )
            
            # diffusion loss (partial point cloud)
            # autoencoder loss encodes the input point cloud to get the latents, then uses point cloud as reconstruction target
            # should have different keypoints source and target point cloud, i.e.:
            # get the partial keypoints, but with the full point cloud as the reconstruction target
            
            # extracted from the autoencoder get_loss method: #######
            partial_data = get_network_data(data, "partial_orig")

            z0, partial_mu, partial_logvar = net.encode(partial_data)
            z_aux = reparameterize(partial_mu, partial_logvar)
            partial_code = torch.cat([z0.reshape((z0.shape[0], -1)), z_aux], dim=1)
            #########################################################

            partial_code_ = partial_code[:, : opt.key_points * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )

            # kl divergence (full point cloud)
            q = Normal(mu, torch.exp(0.5 * logvar))
            p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
            # kl divergence (partial point cloud)
            partial_q = Normal(partial_mu, torch.exp(0.5 * logvar))
            partial_p = Normal(torch.zeros_like(partial_mu), torch.ones_like(partial_logvar))

            # linear warmup until t=kl_warmup_steps
            lambda_4 = 0 if opt.lambda_4 == 0 else min(1.0, t / kl_warmup_steps)
            kl = kl_divergence(q, p).sum(dim=1).mean()
            partial_kl = kl_divergence(partial_q, partial_p).sum(dim=1).mean()
            wandb.log({"diffusion_loss": diffusion_loss}, step=t)
            wandb.log({"kl_divergence": kl}, step=t)
            wandb.log({"partial_kl_divergence": partial_kl}, step=t)
            
            # fps (full point cloud)
            if t > opt.fps_steps and lambda_0 > 0:
                print("turning off FPS loss")
                lambda_0 = 0
            fps = sample_farthest_points(target_shape_t, opt.key_points + 10).transpose(
                2, 1
            )
            fps_loss, _ = pytorch3d.loss.chamfer_distance(fps, code_)
            # fps (partial point cloud, target fps points generated from the original point cloud)
            partial_fps_loss, _ = pytorch3d.loss.chamfer_distance(fps, partial_code_)
            wandb.log({"fps_loss": fps_loss}, step=t)
            wandb.log({"partial_fps_loss": partial_fps_loss}, step=t)
            
            # chamfer loss (full point cloud)
            max_schedule = opt.max_schedule
            chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                code_, target_shape_t.transpose(2, 1)
            )
            # chamfer loss (partial point cloud keypoints with the full point cloud as target)
            partial_chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                partial_code_, target_shape_t.transpose(2, 1)
            )
            
            # deformation consistency mse (full point cloud & differentialble transformation)
            data["deformed_shape"].view(data["orig_offset"].shape[0], -1, 3)

            deformed_matrix = data["deformed_transformation"].view(
                data["orig_offset"].shape[0], -1, 3
            ).cuda()

            kp_orig = code_.reshape(code.shape[0], -1, 3).cuda()
            deformed_code, _, _ = net(get_network_data(data, "deformed"))
            kp_deformed = deformed_code.reshape(code.shape[0], -1, 3)
            kp_transformed = torch.bmm(kp_orig, deformed_matrix.transpose(1, 2))
            mse_loss = torch.mean((kp_transformed - kp_deformed) ** 2)
            
            # partial view consistency mse (full point cloud & partial view)
            # use keypoint predictions from the encoder process
            kp_partial = partial_code_.reshape(partial_code.shape[0], -1, 3)
            partial_mse_loss = torch.mean((kp_orig - kp_partial) ** 2)
            
            # partial warmup
            if opt.lambda_p == 0:
                lambda_p = 0
            else:
                lambda_p = 0 if t < opt.partial_start_steps else min(1.0, (t - opt.partial_start_steps) / opt.partial_warmup_steps)
            
            # combine all losses
            total_diffusion_loss = diffusion_loss
            total_fps_loss = fps_loss + lambda_p * partial_fps_loss
            total_kl_loss = kl + lambda_p * partial_kl
            total_chamfer_loss = chamfer_loss + lambda_p * partial_chamfer_loss
            total_mse_loss = mse_loss + lambda_p * partial_mse_loss
            
            loss_ = (
                lambda_0 * total_fps_loss
                + lambda_1 * total_diffusion_loss
                + lambda_2 * total_chamfer_loss
                + lambda_3 * total_mse_loss
                + lambda_4 * total_kl_loss
            )
            
            wandb.log({"chamfer_loss": chamfer_loss, "mse_loss": mse_loss}, step=t)
            wandb.log({"partial_chamfer_loss": partial_chamfer_loss, "partial_mse_loss": partial_mse_loss}, step=t)

            # END modified partial loss ######################################################

            loss = max(0, max_schedule - t) / max_schedule * loss_

            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (t + 1) % accumulation_steps == 0:
                # print(f"Gradient step at iteration {t + 1}")
                clip_grad_norm_(net.parameters(), opt.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            cur_nimg += opt.batch_size

            # if t % opt.save_interval == 0:
                # os.path.join(ckpt_dir, "outputs", "%07d" % t)
                # save_network(
                    # net, ckpt_dir, network_label=f"{opt.key_points}kp", epoch_label=t
                # )

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()

            # if t % opt.log_interval == 0:
            #     samples_sec = opt.batch_size / iter_time
            #     losses_str = str(loss)
            #     log_str = f"{t:d}: iter {iter_time:.1f} sec, {samples_sec:.1f} samples/sec {losses_str}"
            #     print(log_str)

            t += 1
            
        if e % opt.save_interval == 0 and e != 0:
            os.path.join(ckpt_dir, "outputs", "%07d" % t)
            save_network(
                net, ckpt_dir, network_label=f"{opt.key_points}kp", epoch_label=e
            )
            
        epoch_time = time.time() - epoch_time_start
        epoch_time_start = time.time()
        if e % opt.log_interval == 0:
            losses_str = str(loss)
            log_str = f"epoch {e:d}, iter {t:d}: epoch {epoch_time:.1f} sec"
            print(log_str)

    save_network(net, ckpt_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "train":
        wandb.init(project=f"partial_{opt.category}_train", config=opt)
        wandb.run.log_code(".")
        train(opt)

    else:
        raise ValueError()
