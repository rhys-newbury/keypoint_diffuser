import os
import sys
from pathlib import Path


kp_path = Path(__file__).resolve().absolute().parent.parent
sys.path.append(str(kp_path))

import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pytorch3d.io
import pytorch3d.loss
import torch
import torch.nn.parallel
import torch.utils.data
from einops import repeat
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from keypointdeformer.datasets import get_dataset
from keypointdeformer.models.encoder_models.autoencoder import AutoEncoder
from keypointdeformer.options.ae_options import AEOptions
from keypointdeformer.utils import io
from keypointdeformer.utils.cages import deform_with_MVC
from keypointdeformer.utils.nn import load_network, save_network
from keypointdeformer.utils.utils import Timer


wandb.init(project="diffuse_keypoints")

from deform import apply_general_deformation


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
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate,
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

    Timer("step")
    with torch.no_grad():
        closest_labels_ = []

        total_batches = len(dataloader)

        # Wrap the dataloader with tqdm
        for data in tqdm(
            dataloader, desc="Processing data", unit="batch", total=total_batches
        ):
            data = dataset.uncollate(data)
            target_shape_t = data["target_shape"].transpose(1, 2).cuda()

            code = ae_model.encode(target_shape_t)
            code[:, :-5].reshape(code.shape[0], -1, 3)

            ae_model.decode(
                code, target_shape_t.size(2), flexibility=opt.flexibility
            ).detach()

            for i in range(code.shape[0]):
                kp = code[i, :-5].reshape(-1, 3)

                points = data["target_shape"][i, ...]

                seg_labels = data["target_sampled_points"][i, :, -1].int()
                seg_points = data["target_sampled_points"][i, :, :3]
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

        closest_labels_tensor = torch.stack(closest_labels_)

        average_correlation_per_keypoint = (
            closest_labels_tensor[:, :, :].sum(dim=0).max(dim=1)[0]
            / closest_labels_tensor.shape[0]
        ).mean()
        print(average_correlation_per_keypoint)


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


def train(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    net = AutoEncoder(opt).cuda()

    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    # train
    net.train()
    t = 0

    # train
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

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    while t <= opt.n_iterations:
        iter_time_start = time.time()  # Start iteration timer

        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            if t % 200 == 0:
                time.time()
                net.diffusion.freeze_network()
                # print(f"Freeze network time: {time.time() - start:.4f} sec")

            time.time()
            target_shape_t = data["target_shape"].transpose(1, 2).cuda()
            # print(f"Data transfer to CUDA time: {time.time() - start:.4f} sec")

            optimizer.zero_grad()
            net.train()

            time.time()
            loss, code = net.get_loss(target_shape_t, opt.use_perceptual_loss)
            code_ = code[:, : opt.latent_dim * 3].reshape(
                target_shape_t.shape[0], -1, 3
            )
            # print(f"Forward pass time: {time.time() - start:.4f} sec")

            time.time()
            wandb.log({"diffusion_loss": loss}, step=t)
            # print(f"Wandb logging time: {time.time() - start:.4f} sec")

            if t < 1000:
                time.time()
                fps = sample_farthest_points(target_shape_t, opt.latent_dim).transpose(
                    2, 1
                )
                # print(f"FPS sampling time: {time.time() - start:.4f} sec")

                time.time()
                loss, _ = pytorch3d.loss.chamfer_distance(fps, code_)
                # print(f"Chamfer distance (FPS) time: {time.time() - start:.4f} sec")

                time.time()
                wandb.log({"fps_loss": loss}, step=t)
                # print(f"Wandb FPS logging time: {time.time() - start:.4f} sec")

            else:
                time.time()
                max_schedule = 100000
                chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
                    code_, target_shape_t.transpose(2, 1)
                )
                # print(f"Chamfer distance time: {time.time() - start:.4f} sec")

                time.time()
                deformed_shape, deformed_matrix = apply_general_deformation(
                    target_shape_t.permute(0, 2, 1)
                )
                deformed_shape = deformed_shape.permute(0, 2, 1)
                # print(f"Deformation time: {time.time() - start:.4f} sec")

                time.time()
                kp_orig = code[:, :-5].reshape(code.shape[0], -1, 3)
                deformed_code = net.encode(deformed_shape)
                kp_deformed = deformed_code[:, :-5].reshape(code.shape[0], -1, 3)
                kp_transformed = torch.bmm(kp_orig, deformed_matrix.transpose(1, 2))
                mse_loss = torch.mean((kp_transformed - kp_deformed) ** 2)
                # print(f"Keypoint transformation time: {time.time() - start:.4f} sec")

                loss += mse_loss

                time.time()
                wandb.log({"chamfer_loss": chamfer_loss, "mse_loss": mse_loss}, step=t)
                # print(f"Wandb logging (chamfer & mse) time: {time.time() - start:.4f} sec")

                loss += max(0, max_schedule - t) / max_schedule * chamfer_loss

            time.time()
            loss.backward()
            clip_grad_norm_(net.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # print(f"Backward + optimizer step time: {time.time() - start:.4f} sec")

            if t % opt.save_interval == 0:
                time.time()
                os.path.join(checkpoints_dir, "outputs", "%07d" % t)
                save_network(net, checkpoints_dir, network_label="net", epoch_label=t)
                # print(f"Saving network time: {time.time() - start:.4f} sec")

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()

            if t % opt.log_interval == 0:
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

    save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        test(opt, save_subdir=opt.subdir)
    elif opt.phase == "train":
        train(opt)
    else:
        raise ValueError()
