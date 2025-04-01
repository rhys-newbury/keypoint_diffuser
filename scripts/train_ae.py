import os
import sys
from pathlib import Path


kp_path = Path(__file__).resolve().absolute().parent.parent
sys.path.append(str(kp_path))

import copy
import json
import time
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pytorch3d.io
import pytorch3d.loss
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from einops import repeat
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from keypointdeformer.datasets import get_dataset
from keypointdeformer.models.encoder_models.autoencoder import AutoEncoder
from keypointdeformer.options.ae_options import AEOptions
from keypointdeformer.utils import io
from keypointdeformer.utils.cages import deform_with_MVC
from keypointdeformer.utils.eval_metrics import EMD_CD
from keypointdeformer.utils.nn import load_network, save_network
from keypointdeformer.utils.utils import Timer


torch.autograd.set_detect_anomaly(True)
RUN = None

from torchvision import transforms
from transforms import ApplyToBoth, Collect, Deform, GridSample, ToTensor
from utils import collate_fn


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


# def write_losses(writer, losses, step):
#     for name, value in losses.items():
#         writer.add_scalar('loss/' + name, value, global_step=step)


def save_normalization(file_path, center, scale):
    with open(file_path, "w") as f:
        json.dump(
            {
                "center": [str(x) for x in center.cpu().numpy()],
                "scale": str(scale.cpu().numpy()[0]),
            },
            f,
        )


# Initialize distributed environment
def setup(rank, world_size):
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        print("local_rank", local_rank, "rank: ", rank, "world_size: ", world_size)
        torch.cuda.set_device(local_rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def save_data_keypoints(data, save_dir, name):
    if name in data:
        io.save_keypoints(os.path.join(save_dir, name + ".txt"), data[name])


def save_data_txt(f, data, fmt):
    np.savetxt(f, data.cpu().detach().numpy(), fmt=fmt)


def save_pts(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    io.save_pts(f, points.cpu().detach().numpy(), normals=normals)


def save_ply(f, verts, faces):
    pytorch3d.io.save_ply(f, verts.cpu(), faces=faces.cpu())


def save_output(save_dir_root, data, outputs, save_mesh=True, save_auxilary=True):
    name = data["source_file"]
    save_dir = os.path.join(save_dir_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # save meshes
    if save_mesh and "source_mesh" in data:
        io.save_mesh(
            os.path.join(save_dir, "source_mesh.obj"),
            data["source_mesh"],
            data["source_face"],
        )

        if save_auxilary:
            save_data_txt(
                os.path.join(save_dir, "source_vertices.txt"),
                data["source_mesh"],
                "%0.6f",
            )
            save_data_txt(
                os.path.join(save_dir, "source_faces.txt"), data["source_face"], "%d"
            )

        io.save_mesh(
            os.path.join(save_dir, "target_mesh.obj"),
            data["target_mesh"],
            data["target_face"],
        )

        if outputs is not None:
            deformed, weights, _ = deform_with_MVC(
                outputs["cage"][None],
                outputs["new_cage"][None],
                outputs["cage_face"][None],
                data["source_mesh"][None],
                verbose=True,
            )
            io.save_mesh(
                os.path.join(save_dir, "deformed_mesh.obj"),
                deformed[0],
                data["source_face"],
            )
            if save_auxilary:
                save_data_txt(
                    os.path.join(save_dir, "weights.txt"), weights[0], "%0.6f"
                )

    # save pointclouds
    save_pts(
        os.path.join(save_dir, "source_pointcloud.pts"),
        data["source_shape"],
        normals=data["source_normals"],
    )
    if outputs is not None:
        save_pts(os.path.join(save_dir, "deformed_pointcloud.pts"), outputs["deformed"])
        if save_auxilary:
            save_data_txt(
                os.path.join(save_dir, "influence.txt"), outputs["influence"], "%0.6f"
            )

    save_pts(
        os.path.join(save_dir, "target_pointcloud.pts"),
        data["target_shape"],
        normals=data["target_normals"],
    )

    # save cages
    if outputs is not None:
        save_ply(
            os.path.join(save_dir, "cage.ply"), outputs["cage"], outputs["cage_face"]
        )
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, "cage.txt"), outputs["cage"], "%0.6f")
        save_ply(
            os.path.join(save_dir, "deformed_cage.ply"),
            outputs["new_cage"],
            outputs["cage_face"],
        )

        if outputs is not None:
            io.save_keypoints(
                os.path.join(save_dir, "source_keypoints.txt"),
                outputs["source_keypoints"].transpose(0, 1),
            )
            io.save_keypoints(
                os.path.join(save_dir, "target_keypoints.txt"),
                outputs["target_keypoints"].transpose(0, 1),
            )

        save_data_keypoints(data, save_dir, "source_keypoints_gt")
        save_data_keypoints(data, save_dir, "target_keypoints_gt")

        io.save_keypoints(
            os.path.join(save_dir, "source_init_keypoints.txt"),
            outputs["source_init_keypoints"].transpose(0, 1),
        )
        io.save_keypoints(
            os.path.join(save_dir, "target_init_keypoints.txt"),
            outputs["target_init_keypoints"].transpose(0, 1),
        )

        if "source_keypoints_gt_center" in data:
            save_normalization(
                os.path.join(save_dir, "source_keypoints_gt_normalization.txt"),
                data["source_keypoints_gt_center"],
                data["source_keypoints_gt_scale"],
            )

        if "source_seg_points" in data:
            io.save_labelled_pointcloud(
                os.path.join(save_dir, "source_seg_points.xyzrgb"),
                data["source_seg_points"].detach().cpu().numpy(),
                data["source_seg_labels"].detach().cpu().numpy(),
            )
            save_data_txt(
                os.path.join(save_dir, "source_seg_labels.txt"),
                data["source_seg_labels"],
                "%d",
            )


def split_batch(data, b, singleton_keys=None):
    return {
        k: v[b] if k not in (singleton_keys or []) else v[0] for k, v in data.items()
    }


def save_outputs(outputs_save_dir, data, outputs, save_mesh=True):
    for b in range(data["source_shape"].shape[0]):
        save_output(
            outputs_save_dir,
            split_batch(data, b, singleton_keys=["cage_face"]),
            split_batch(outputs, b, singleton_keys=["cage_face"]),
            save_mesh=save_mesh,
        )


def normalize_point_clouds(pcs, mode):
    if mode is None:
        print("Will not normalize point clouds.")
        return pcs
    print(f"Normalization mode: {mode}")
    for i in tqdm(range(pcs.size(0)), desc="Normalize"):
        pc = pcs[i]
        if mode == "shape_unit":
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == "shape_bbox":
            pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


def get_data(dataset, data):
    data = dataset.uncollate(data)

    # source_shape, target_shape = data["source_shape"], data["target_shape"]
    target_shape = data["target_shape"]

    # source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)

    return None, target_shape_t


def visualize_point_cloud(
    points, labels, keypoints, orig_shape, visual=False, icp=True
):
    """
    Visualize a 3D point cloud with color based on labels.

    Args:
    - points (torch.Tensor): Shape [N, 3], point cloud data.
    - keypoints (torch.Tensor): Shape [M, 3], point cloud data, which are bigger and blue

    - labels (torch.Tensor): Shape [N], labels for each point.
    """
    # Convert tensors to NumPy arrays
    points_np = points.cpu().numpy()  # Shape: [N, 3]
    labels_np = labels.cpu().numpy().astype(np.int32)  # Shape: [N]
    orig_shape = orig_shape.cpu().numpy()
    keypoints_np = keypoints.cpu().numpy().T  # Shape: [M, 3]

    # Normalize labels to be in range [0, 1] for color mapping
    max_label = labels_np.max() + 1  # Avoid division by 0

    colors = plt.cm.get_cmap("tab10", max_label)(labels_np / max_label)[
        :, :3
    ]  # RGB from colormap

    # Create Open3D point cloud
    orig_pcd = o3d.geometry.PointCloud()
    orig_pcd.points = o3d.utility.Vector3dVector(orig_shape)
    orig_pcd.paint_uniform_color([1, 0, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if icp:
        threshold = 0.2  # Distance threshold for ICP
        np.eye(4)  # Initial transformation (identity matrix)

        theta = 0  # 90 degrees in radians
        initial_rotation_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        # print("Running ICP...")
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd,
            orig_pcd,
            threshold,
            initial_rotation_y,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        pcd.transform(reg_icp.transformation)

    if visual:
        keypoint_spheres = []
        for keypoint in keypoints_np:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05
            )  # Adjust radius as needed

            sphere.translate(keypoint)
            sphere.paint_uniform_color([0, 0, 1])  # Blue color
            keypoint_spheres.append(sphere)
            break

        # Visualize
        o3d.visualization.draw_geometries(
            [pcd, orig_pcd, *keypoint_spheres],
            window_name="Point Cloud with Labels and Keypoints",
        )

    return np.asarray(pcd.points)


def visualize_point_clouds(
    original, deformed, kp_orig, kp_transformed, kp_deformed, save_dir="./"
):
    """
    Visualizes the first point cloud in the batch before and after deformation.

    Args:
        original: (B, N, 3) tensor of original point cloud.
        deformed: (B, N, 3) tensor of deformed point cloud.
    """
    original_np = original[0].cpu().numpy()  # Extract first sample, convert to NumPy
    deformed_np = deformed[0].cpu().numpy()  # Extract first sample, convert to NumPy
    kp_orig_np = kp_orig[0].cpu().numpy()  # Extract first sample's keypoints
    kp_transformed_np = kp_transformed[0].cpu().numpy()  # Transformed keypoints
    kp_deformed_np = kp_deformed[0].cpu().numpy()  # Keypoints from deformed shape

    # Save NumPy arrays
    np.save(f"{save_dir}/original_point_cloud.npy", original_np)
    np.save(f"{save_dir}/deformed_point_cloud.npy", deformed_np)
    np.save(f"{save_dir}/kp_orig.npy", kp_orig_np)
    np.save(f"{save_dir}/kp_transformed.npy", kp_transformed_np)
    np.save(f"{save_dir}/kp_deformed.npy", kp_deformed_np)


def test(opt, save_subdir="test"):
    t = transforms.Compose(
        [
            Deform(),  # Forks into two versions: original and deformed
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

    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)
    # /app/data/keypoints/logs/autumn-waterfall-200/checkpoints/net_final.pth
    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt, transform=t)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)

    ckpt = torch.load(ckpt)

    ae_model = AutoEncoder(opt).cuda()
    ae_model.load_state_dict(ckpt["states"])
    ae_model.eval()
    all_ref = []
    all_recons = []
    Timer("step")
    with torch.no_grad():
        closest_labels_ = []

        total_batches = len(dataloader)

        # Wrap the dataloader with tqdm
        for data in tqdm(
            dataloader, desc="Processing data", unit="batch", total=total_batches
        ):
            # import pdb; pdb.set_trace()
            # data_ = dataset.uncollate(data)
            # target_shape_t = data_["target_shape"].transpose(1, 2).cuda()
            target_shape_t = (
                data["target_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            code = ae_model.encode(get_network_data(data))
            code[:, :-5].reshape(code.shape[0], -1, 3)

            # recons = ae_model.decode(
            #     code, target_shape_t.size(2), flexibility=opt.flexibility
            # ).detach()

            recons = ae_model.decode_edm(code).detach()

            all_ref.append(target_shape_t.detach().cpu())
            all_recons.append(recons.detach().cpu())

            target_sampled_points = data["target_sampled_points"].view(
                data["orig_offset"].shape[0], -1, 4
            )

            for i in range(code.shape[0]):
                kp = code[i, :-5].reshape(-1, 3)

                # import pdb; pdb.set_trace()

                points = target_shape_t[i, ...].T

                seg_labels = target_sampled_points[i, :, -1].int().cuda()
                seg_points = target_sampled_points[i, :, :3]

                # import pdb; pdb.set_trace()

                seg_points = visualize_point_cloud(
                    seg_points,
                    seg_labels,
                    kp,
                    points,
                    visual=False,
                )

                distances = torch.cdist(kp.double(), torch.tensor(seg_points).cuda())
                threshold = 0.05

                within_threshold_mask = (
                    distances <= threshold
                )  # True where distance <= 0.05

                keypoint_indices, seg_point_indices = torch.nonzero(
                    within_threshold_mask, as_tuple=True
                )
                # import pdb; pdb.set_trace()
                valid_seg_labels = seg_labels[
                    seg_point_indices
                ]  # The labels for valid segmentation points

                max_label = 5  # Ensure it includes the highest label

                # Create a Boolean matrix: (num_keypoints, max_label)
                label_presence_matrix = torch.zeros(
                    (distances.size(0), max_label),
                    dtype=torch.bool,
                    device=seg_labels.device,
                )

                # Mark True for each label that is present for each keypoint
                label_presence_matrix[keypoint_indices, valid_seg_labels.long()] = True

                closest_labels_.append(label_presence_matrix)
            # break

        # import pdb; pdb.set_trace()
        closest_labels_tensor = torch.stack(closest_labels_)

        average_correlation_per_keypoint = (
            closest_labels_tensor[:, :, :].sum(dim=0).max(dim=1)[0]
            / closest_labels_tensor.shape[0]
        ).mean()
        # print()
        wandb.log(
            {"average_correlation_per_keypoint": average_correlation_per_keypoint}
        )

        # import pdb; pdb.set_trace()
        print(average_correlation_per_keypoint)

        all_ref = torch.cat(all_ref, dim=0).permute(0, 2, 1)
        all_ref = normalize_point_clouds(all_ref, "shape_bbox")
        all_recons = torch.cat(all_recons, dim=0)
        all_recons = normalize_point_clouds(all_recons, "shape_bbox")
        print(
            EMD_CD(
                all_recons.to("cuda").double(),
                all_ref.to("cuda").double(),
                opt.batch_size,
            )
        )


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)


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


def train(opt, rank, world_size):
    if rank == 0:
        log_dir = os.path.join(opt.log_dir, RUN.name)
        checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    ema_halflife_kimg = (
        500  # Half-life of the exponential moving average (EMA) of model weights.
    )
    ema_rampup_ratio = 0.05  # EMA ramp-up coefficient, None = no rampup.

    t = transforms.Compose(
        [
            Deform(),  # Forks into two versions: original and deformed
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

    dataset = get_dataset(opt.dataset)(opt, transform=t)

    if torch.cuda.device_count() > 1 and world_size > 1:
        print("Using DistributedSampler for multiple GPUs.")
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
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
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    opt_test = copy.deepcopy(opt)
    opt_test.phase = "test"
    test_dataset = get_dataset(opt_test.dataset)(opt_test, transform=t)

    torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    net = AutoEncoder(opt).cuda()

    if torch.cuda.device_count() > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        ema = copy.deepcopy(net).eval().requires_grad_(False)

        print(f"Using DistributedDataParallel on {torch.cuda.device_count()}")
        net = DDP(
            net, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
    else:
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    # train
    net.train()
    t = 0

    # train
    if rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        log_path = os.path.join(checkpoints_dir, "training_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write(str(net) + "\n")
        summary_dir = datetime.now().strftime("%y%m%d-%H%M%S")
        writer = SummaryWriter(
            logdir=os.path.join(checkpoints_dir, "logs", summary_dir), flush_secs=5
        )

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

    accumulation_steps = int(64 / opt.batch_size)

    cur_nimg = 0

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    epoch = 0

    while t <= opt.n_iterations:
        print(t)
        epoch += 1
        iter_time_start = time.time()  # Start iteration timer
        if torch.cuda.device_count() > 1 and world_size > 1:
            dataloader.sampler.set_epoch(epoch)

        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            target_shape_t = (
                data["target_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            module = (
                net.module if torch.cuda.device_count() > 1 and world_size > 1 else net
            )

            loss, code = module.get_loss(
                get_network_data(data), opt.use_perceptual_loss
            )
            code_ = code[:, : opt.latent_dim * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )

            if rank == 0:
                wandb.log({"diffusion_loss": loss}, step=t)

            if t < 1000:
                fps = sample_farthest_points(target_shape_t, opt.latent_dim).transpose(
                    2, 1
                )

                loss, _ = pytorch3d.loss.chamfer_distance(fps, code_)

                if rank == 0:
                    wandb.log({"fps_loss": loss}, step=t)

            else:
                max_schedule = 100000
                chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                    code_, target_shape_t.transpose(2, 1)
                )
                data["deformed_shape"].view(data["orig_offset"].shape[0], -1, 3)

                deformed_matrix = data["deformed_transformation"].view(
                    data["orig_offset"].shape[0], -1, 3
                )

                kp_orig = code[:, :-5].reshape(code.shape[0], -1, 3)
                deformed_code = net(get_network_data(data, "deformed"))
                kp_deformed = deformed_code[:, :-5].reshape(code.shape[0], -1, 3)
                kp_transformed = torch.bmm(kp_orig, deformed_matrix.transpose(1, 2))
                mse_loss = torch.mean((kp_transformed - kp_deformed) ** 2)

                loss += mse_loss

                if rank == 0:
                    wandb.log(
                        {"chamfer_loss": chamfer_loss, "mse_loss": mse_loss}, step=t
                    )

                loss += max(0, max_schedule - t) / max_schedule * chamfer_loss

            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (t + 1) % accumulation_steps == 0:
                print(f"Rank {rank}: Gradient step at iteration {t+1}")
                clip_grad_norm_(net.parameters(), opt.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (opt.batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            cur_nimg += opt.batch_size

            if t % opt.save_interval == 0 and rank == 0:
                os.path.join(checkpoints_dir, "outputs", "%07d" % t)
                save_network(ema, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()

            if t % opt.log_interval == 0 and rank == 0:
                samples_sec = opt.batch_size / iter_time
                losses_str = str(loss)
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str
                )
                print(log_str)
                with open(log_path, "a") as log_file:
                    log_file.write(log_str + "\n")
                writer.add_scalar("train/loss", loss, t)

            t += 1

        # torch.cuda.empty_cache()
        # test_loss = 0
        # net.eval()
        # with torch.no_grad():
        #     for _, data in enumerate(test_dataloader):
        #         # target_shape_t = data["target_shape"].transpose(1, 2).cuda()
        #         # print(target_shape_t.shape)

        #         loss, code = net.get_loss(
        #             get_network_data(data), opt.use_perceptual_loss
        #         )
        #         test_loss += loss
        # wandb.log({"mse_test_loss": test_loss}, step=t)
    if rank == 0:
        save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)
    else:
        rank = 0
        world_size = 1

    print("SETUP IS COMPLETE!!!!!!!!!!!!!!!!")

    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        RUN = wandb.init(project="diffuse_keypoints_test")
        wandb.log({"ckpt": opt.ckpt, "n_keypoints": opt.latent_dim, "type": "ours"})

        test(opt, save_subdir=opt.subdir)
    elif opt.phase == "train":
        print(f"Rank: {rank}, World size: {world_size}")

        if rank == 0:
            RUN = wandb.init(project="diffuse_keypoints_lamp")
        train(opt, rank, world_size)
    else:
        raise ValueError()
