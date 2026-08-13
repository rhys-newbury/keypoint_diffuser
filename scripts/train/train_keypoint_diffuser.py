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
from pathlib import Path
# from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypoint_diffuser.datasets.shapespartial import ShapesPartial
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.models.encoder_models.common import get_linear_scheduler
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions
from keypoint_diffuser.utils.nn import save_network
from keypoint_diffuser.utils.utils import simple_plot
from keypoint_diffuser.utils.loss import local_repulsion
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


# --- NEW VISUALIZATION FUNCTION ---
def plot_instances_with_dropdown(instances_points: list[list[np.ndarray]], instances_names: list[list[str]], out_html: str | Path):
    """
    Saves an interactive Plotly HTML file containing multiple dataloader instances.
    Uses a dropdown menu to toggle visibility between instances (acting as tabs).
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    traces_per_instance = [len(pts) for pts in instances_points]
    total_traces = sum(traces_per_instance)
    
    for i, (points_list, name_list) in enumerate(zip(instances_points, instances_names)):
        for points, name in zip(points_list, name_list):
            shape = points.shape
            # assume it is object point cloud if a lot of points
            if np.max(shape) > 1000:
                marker = {"size": 2, "opacity": 1.0}
            else:
                # generate a random color string for Plotly
                color_str = f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
                marker = {"size": 3, "opacity": 1.0, "color": color_str, "symbol": "diamond", "line": {"width": 2, "color": "black"}}
                
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers",
                    marker=marker,
                    name=f"Inst {i} - {name}",
                    visible=(i == 0) # Only the first instance is visible by default
                )
            )
            
    # Create dropdown menu buttons
    buttons = []
    start_idx = 0
    for i in range(len(instances_points)):
        num_traces = traces_per_instance[i]
        
        # Boolean array where only the traces for the current instance are True
        visible_array = [False] * total_traces
        for j in range(start_idx, start_idx + num_traces):
            visible_array[j] = True
            
        button = dict(
            label=f"Instance {i}",
            method="update",
            args=[{"visible": visible_array},
                  {"title": f"Viewing Dataloader Instance {i}"}]
        )
        buttons.append(button)
        start_idx += num_traces

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ],
        title="Viewing Dataloader Instance 0",
        scene=dict(aspectmode="data")
    )
    
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    print(f"[✓] Saved interactive multi-tab HTML to {out_html}")
# ----------------------------------


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
    if wandb.run.name is None:
        run_name = "debug_run"
    else:
        run_name = wandb.run.name
    ckpt_dir = opt.ckpt_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt.db.parent.mkdir(parents=True, exist_ok=True)
    save_train_run(opt.db, "Ours", opt.category, ckpt_dir, opt.key_points)

    t = transforms.Compose(
        [
            Deform(
                max_stretch_factor=opt.max_stretch_factor,
                max_bending_factor=opt.max_bending_factor,
                max_twist_factor=opt.max_twist_factor,
                max_taper_factor=opt.max_taper_factor,
                max_rotation_angle=opt.max_rotation_angle,
                max_scaling_factor=opt.max_scaling_factor,
                max_translation_offset=opt.max_translation_offset
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
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    net = AutoEncoder(opt).cuda()
    # to continue on previous training
    if opt.pretrained_root is not None:
        pretrained_root = opt.pretrained_root
        if str(pretrained_root).endswith(".pth"):
            # path provided is a pth file, read directly
            pretrained_dir = pretrained_root
        else:
            # path provided is a dir to training db files, which contains the ckpt_dir
            pretrained_dir = find_checkpoint_from_db_dir(pretrained_root, opt.category, opt.pretrained_iterations, use_max_epoch=opt.pretrained_use_final)
        # db files save (opt.ckpt_dir / wandb.run.name)
        pretrained = torch.load(pretrained_dir)
        
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

            diffusion_loss, code, mu, logvar = net.get_loss(
                get_network_data(data, key="orig"), step=t,
            )

            code_ = code[:, : opt.key_points * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )

            q = Normal(mu, torch.exp(0.5 * logvar))
            p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))

            lambda_4 = 0 if opt.lambda_4 == 0 else min(1.0, t / kl_warmup_steps)
            kl = kl_divergence(q, p).sum(dim=1).mean()

            wandb.log({"diffusion_loss": diffusion_loss}, step=t)
            wandb.log({"kl_divergence": kl}, step=t)

            fps = sample_farthest_points(target_shape_t, opt.key_points + 2).transpose(
                2, 1
            )

            # drives keypoints closer to fps points
            fps_to_pred, _ = pytorch3d.loss.chamfer_distance(fps, code_, single_directional=True, batch_reduction=None)
            pred_to_fps, _ = pytorch3d.loss.chamfer_distance(code_, fps, single_directional=True, batch_reduction=None)
            
            fps_loss = 1.5 * fps_to_pred.mean() + 0.5 * pred_to_fps.mean()  # make the full surface better covered

            wandb.log({"fps_loss": fps_loss}, step=t)

            # chamfer loss (full point cloud)
            # drives keypoints closer to surface
            max_schedule = opt.max_schedule
            chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                code_, target_shape_t.transpose(2, 1), single_directional=True, batch_reduction=None
            )
            # keypoint coverage loss
            kp_coverage_loss, _ = pytorch3d.loss.chamfer_distance(
                    target_shape_t.transpose(2, 1), code_, single_directional=True, batch_reduction=None)
            chamfer_loss = 0.5 * chamfer_loss.mean() + 1.5 * kp_coverage_loss.mean()
            # also add repulsion factor
            chamfer_loss = chamfer_loss + 0.1 * local_repulsion(code_, k=8, margin=0.01).mean()            
            
            data["deformed_shape"].view(data["orig_offset"].shape[0], -1, 3)

            deformed_matrix = data["deformed_transformation"].view(
                data["orig_offset"].shape[0], -1, 4
            )

            kp_orig = code_.reshape(code.shape[0], -1, 3)
            # extend to homogeneous form
            homogeneous_kp = torch.cat(
                [kp_orig, torch.ones(kp_orig.shape[0], kp_orig.shape[1], 1, device=kp_orig.device)], dim=-1
            )  # (B, N, 3) -> (B, N, 4)
            # transformed keypoints
            kp_transformed = torch.bmm(homogeneous_kp, deformed_matrix.transpose(1, 2))
            # convert back
            kp_transformed = kp_transformed[:, :, :3] / kp_transformed[:, :, 3:]
            
            # output keypoints from deformed inputs
            deformed_code, _, _ = net(get_network_data(data, "deformed"))
            kp_deformed = deformed_code.reshape(code.shape[0], -1, 3)
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
            
        # save a quick html visualization of the various used pcs
        if e % opt.log_interval == 0: # Adjust this condition as needed (e.g. t % opt.log_interval == 0)
            num_instances_to_plot = min(4, opt.batch_size) # Extract a few items from the batch
            instances_points = []
            instances_names = []
            
            for b in range(num_instances_to_plot):
                # --- USER TODO: Define your point clouds and names for instance 'b' here ---
                # Ensure tensors are detached, moved to CPU, and converted to NumPy arrays.
                #
                # Example:
                # pc_target = target_shape_t[b].transpose(0, 1).detach().cpu().numpy()
                # pc_kp = kp_orig[b].detach().cpu().numpy()
                #
                # points_list = [pc_target, pc_kp]
                # name_list = ["Target Shape", "Keypoints"]
                
                # full point cloud
                vog = data["target_shape"].view(data["orig_offset"].shape[0], -1, 3)[b].detach().cpu().numpy()
                # deformed point cloud
                vdf = data["deformed_shape"].view(data["orig_offset"].shape[0], -1, 3)[b].detach().cpu().numpy()
                # full keypoints
                vkp = kp_orig[b].detach().cpu().numpy()
                # transformed keypoints
                vtk = kp_transformed[b].detach().cpu().numpy()
                # keypoints from deformed inputs
                vkd = kp_deformed[b].detach().cpu().numpy()
                # fps keypoints
                vkf = fps[b].detach().cpu().numpy()
                
                points_list = [vog, vdf, vkp, vtk, vkd, vkf]
                name_list = ["Target Shape", "Deformed Shape", "Keypoints", "Transformed Keypoints", "Keypoints from Deformed Input", "FPS Keypoints"]
                # -------------------------------------------------------------------------
                
                if len(points_list) > 0:
                    instances_points.append(points_list)
                    instances_names.append(name_list)

            if len(instances_points) > 0:
                html_path = ckpt_dir / "outputs" / f"visualization_step_{t}.html"
                plot_instances_with_dropdown(instances_points, instances_names, html_path)
                    
        if e % opt.save_interval == 0:
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
        wandb.init(project=f"ours_{opt.category}_train", config=opt)
        wandb.run.log_code(".")
        train(opt)

    else:
        raise ValueError()
