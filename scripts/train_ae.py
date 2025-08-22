import copy
import os
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
from keypoint_diffuser.datasets import get_dataset
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.models.encoder_models.common import get_linear_scheduler
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.utils.eval_metrics import EMD_CD
from keypoint_diffuser.utils.nn import load_network, save_network
from keypoint_diffuser.utils.utils import Timer, reparameterize
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import wandb


torch.autograd.set_detect_anomaly(True)
RUN = None

from keypoint_diffuser.utils.pc_utils import collate_fn, normalize_point_clouds
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth,
    Collect,
    Deform,
    GridSample,
    ToTensor,
    # ApplyToAll,
    # DeformWithPartial,
)
from torchvision import transforms


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


# Initialize distributed environment
def setup(rank, world_size):
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        print("local_rank", local_rank, "rank: ", rank, "world_size: ", world_size)
        torch.cuda.set_device(local_rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_data(dataset, data):
    data = dataset.uncollate(data)

    target_shape = data["target_shape"]

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

    # try to align the two point clouds with ICP
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
        for keypoint in keypoints_np.T:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05
            )  # Adjust radius as needed

            sphere.translate(keypoint)
            sphere.paint_uniform_color([0, 0, 1])  # Blue color
            keypoint_spheres.append(sphere)

        # Visualize
        o3d.visualization.draw_geometries(
            [pcd, orig_pcd, *keypoint_spheres],
            window_name="Point Cloud with Labels and Keypoints",
        )

    return np.asarray(pcd.points)

def visualize_reconstructed_point_cloud(
    recon_shape, ref_shape, keypoints, orig_shape, visual=False, icp=False
):
    """
    Visualize input point cloud, the encoded keypoints and reconstructed point cloud

    Args:
    - points (torch.Tensor): Shape [N, 3], point cloud data.
    - keypoints (torch.Tensor): Shape [M, 3], point cloud data, which are bigger and blue
    """
    # Convert tensors to NumPy arrays
    recon_shape_np = recon_shape.cpu().numpy()  # Shape: [N, 3]
    ref_shape_np = ref_shape.cpu().numpy()  # Shape: [N, 3]
    orig_shape_np = orig_shape.cpu().numpy()
    keypoints_np = keypoints.cpu().numpy().T  # Shape: [M, 3]

    # Create Open3D point cloud
    orig_pcd = o3d.geometry.PointCloud()
    orig_pcd.points = o3d.utility.Vector3dVector(orig_shape_np)
    # orig_pcd.paint_uniform_color([1, 0, 0])
    orig_pcd.paint_uniform_color([0, 1, 0])

    recon_pcd = o3d.geometry.PointCloud()
    recon_pcd.points = o3d.utility.Vector3dVector(recon_shape_np)
    recon_pcd.paint_uniform_color([1, .7, .7])

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_shape_np)
    ref_pcd.paint_uniform_color([0, 0, 0])
    
    # try to align the two point clouds with ICP
    # for reconstructions the two point clouds should be in the same frame, default to false
    # if icp:
    #     threshold = 0.2  # Distance threshold for ICP
    #     np.eye(4)  # Initial transformation (identity matrix)

    #     theta = 0  # 90 degrees in radians
    #     initial_rotation_y = np.array(
    #         [
    #             [np.cos(theta), 0, np.sin(theta), 0],
    #             [0, 1, 0, 0],
    #             [-np.sin(theta), 0, np.cos(theta), 0],
    #             [0, 0, 0, 1],
    #         ]
    #     )
    #     reg_icp = o3d.pipelines.registration.registration_icp(
    #         ref_pcd,
    #         orig_pcd,
    #         threshold,
    #         initial_rotation_y,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     )

    #     ref_pcd.transform(reg_icp.transformation)

    # get the path for the source of the partial point cloud, and load it in
    # loaded_path = "data/shape_data_eric/03797390/10f6e09036350e92b3f21f1137c3c347/models/partial_samples_myopia_2.npy"
    # loaded_points = np.load(loaded_path)
    # loaded_pcd = o3d.geometry.PointCloud()
    # loaded_pcd.points = o3d.utility.Vector3dVector(loaded_points[:, :3])
    # loaded_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for loaded points

    if visual:
        keypoint_spheres = []
        for keypoint in keypoints_np.T:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05
            )  # Adjust radius as needed

            sphere.translate(keypoint)
            # sphere.paint_uniform_color([1, 0, 1])  # Pink color
            sphere.paint_uniform_color([0, 0, 1])  # Blue color
            keypoint_spheres.append(sphere)

        # Visualize
        o3d.visualization.draw_geometries(
            # [recon_pcd, orig_pcd, ref_pcd, *keypoint_spheres, loaded_pcd],
            [recon_pcd, orig_pcd, ref_pcd, *keypoint_spheres],
            window_name="R: Input point cloud; G: Reconstructed point cloud; B: Reference point cloud; Keypoints in pink",
        )


def test(opt):
    t = transforms.Compose(
        [
            # DeformWithPartial(),  # Forks into two versions: original and deformed
            # ApplyToAll(
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
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
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
            # get input point cloud
            if not opt.test_partial_samples:
                target_shape_t = (
                    data["target_shape"]
                    .view(data["orig_offset"].shape[0], -1, 3)
                    .transpose(1, 2)
                    .cuda()
                )
            else:
                target_shape_t = (
                    data["target_partial_shape"]
                    .view(data["orig_offset"].shape[0], -1, 3)
                    .transpose(1, 2)
                    .cuda()
                )

            # get reference point cloud (original full points regardless of input type)
            ref_shape_t = (
                data["target_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            # encode the input point cloud (original/partial depending on opt)
            if not opt.test_partial_samples:
                z0, mu, logvar = ae_model.encode(get_network_data(data))
            else:
                z0, mu, logvar = ae_model.encode(get_network_data(data, "partial_orig"))
            z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)

            # concatenate the keypoint latent z0 and the auxiliary latent z_aux
            z_full = torch.cat([z0, z_aux], dim=1)

            # decode for the reconstructed point cloud
            recons = ae_model.decode(z_full, 5000).detach()

            all_ref.append(ref_shape_t.detach().cpu())
            all_recons.append(recons.detach().cpu())

            # from point_resampled_labeled.npy
            target_sampled_points = data["target_sampled_points"].view(
                data["orig_offset"].shape[0], -1, 4
            )

            for i in range(z0.shape[0]):
                kp = z0[i, :].reshape(-1, 3)

                # from new_samples.npy / partial_samples.npy
                points = target_shape_t[i, ...].T
                ref_points = ref_shape_t[i, ...].T
                recon_points = recons[i, ...]

                seg_labels = target_sampled_points[i, :, -1].int().cuda()
                seg_points = target_sampled_points[i, :, :3]

                # visualise the input point cloud and the original point cloud with labelled parts
                seg_points = visualize_point_cloud(
                    seg_points,     # from point_resampled_labeled.npy
                    seg_labels,     # labels for the seg_points
                    kp,     # keypoints from the latent z0
                    points,     # input point cloud
                    # visual=True,
                )

                # visualise the input point cloud with keypoints, the reconstructed point cloud, and the ref point cloud
                visualize_reconstructed_point_cloud(
                    recon_points,
                    ref_points, 
                    kp,
                    points,
                    # visual=True,
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

                max_label = 7  # Ensure it includes the highest label

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
        wandb.log(
            {"average_correlation_per_keypoint": average_correlation_per_keypoint}
        )

        print(average_correlation_per_keypoint)

        all_ref = torch.cat(all_ref, dim=0).permute(0, 2, 1)
        all_ref = normalize_point_clouds(all_ref, "shape_bbox")
        all_recons = torch.cat(all_recons, dim=0)
        all_recons = normalize_point_clouds(all_recons, "shape_bbox")
        # test metrics calculation
        metrics = EMD_CD(
            all_recons.to("cuda").double(),
            all_ref.to("cuda").double(),
            opt.batch_size,
        )
        wandb.log(metrics)
        for key, value in metrics.items():
            print(f"{key}: {value.item():.10f}")


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
            d[k[len(key) + 1 :]] = v
        # shared data, keep
        else:
            d[k] = v
    return d


def train(opt, rank, world_size):
    if rank == 0:
        print(opt.log_dir, RUN.name)
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
        train_sampler = dist.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
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
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
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
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    net = AutoEncoder(opt).cuda()

    if torch.cuda.device_count() > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        ema = copy.deepcopy(net).eval().requires_grad_(False)

        print(f"Using DistributedDataParallel on {torch.cuda.device_count()}")
        net = DistributedDataParallel(
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

    # freeze decoder if specified
    if opt.freeze_decoder:
        for param in net.diffusion.parameters():
            param.requires_grad = False

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

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    epoch = 0

    lambda_0 = opt.lambda_0
    lambda_1 = opt.lambda_1
    lambda_2 = opt.lambda_2
    lambda_3 = opt.lambda_3
    lambda_4 = opt.lambda_4

    kl_warmup_steps = opt.kl_warmup_steps

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
            
            partial_target_shape_t = (
                data["target_partial_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            # diffusion loss (full point cloud)
            module = (
                net.module if torch.cuda.device_count() > 1 and world_size > 1 else net
            )

            diffusion_loss, code, mu, logvar = module.get_loss(
                get_network_data(data), step=t
            )

            # code is z0 (entire latent including kp and aux), extract kp only to code_
            code_ = code[:, : opt.latent_dim * 3].reshape(
                data["orig_offset"].shape[0], -1, 3
            )
            
            # diffusion loss (partial point cloud)
            # autoencoder loss encodes the input point cloud to get the latents, then uses point cloud as reconstruction target
            # should have different keypoints source and target point cloud, i.e.:
            # get the partial keypoints, but with the full point cloud as the reconstruction target
            
            # extracted from the autoencoder get_loss method: ##############################################
            # partial_diffusion_loss, partial_code, partial_mu, partial_logvar = module.get_loss(
            #     get_network_data(data, "partial_orig"), step=t
            # )
            
            # orig_data = get_network_data(data)
            partial_data = get_network_data(data, "partial_orig")

            z0, partial_mu, partial_logvar = module.encode(partial_data)
            z_aux = reparameterize(partial_mu, partial_logvar)
            partial_code = torch.cat([z0, z_aux], dim=1)

            # orig_ts = orig_data["target_shape"].view(-1, 5000, 3).cuda()
            # if module.use_edm:
            #     partial_diffusion_loss = module.loss(
            #         net=module.diffusion, data=orig_ts, code=partial_code.detach(), step=t
            #     ).mean()
            # else:
            #     partial_diffusion_loss = module.diffusion.get_loss(orig_ts.transpose(1, 2), partial_code)

            ################################################################################################

            partial_code_ = partial_code[:, : opt.latent_dim * 3].reshape(
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

            if rank == 0:
                wandb.log({"diffusion_loss": diffusion_loss}, step=t)
                # wandb.log({"partial_diffusion_loss": partial_diffusion_loss}, step=t)
                wandb.log({"kl_divergence": kl}, step=t)
                wandb.log({"partial_kl_divergence": partial_kl}, step=t)

            # fps (full point cloud)
            if t > opt.fps_steps and lambda_0 > 0:
                print("turing off FPS loss")
                lambda_0 = 0

            fps = sample_farthest_points(target_shape_t, opt.latent_dim).transpose(2, 1)

            fps_loss, _ = pytorch3d.loss.chamfer_distance(fps, code_)
            
            # fps (partial point cloud, target fps points generated from the original point cloud)
            partial_fps_loss, _ = pytorch3d.loss.chamfer_distance(fps, partial_code_)
            
            if rank == 0:
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
            )

            kp_orig = code_.reshape(code.shape[0], -1, 3)
            deformed_code, _, _ = net(get_network_data(data, "deformed"))
            kp_deformed = deformed_code.reshape(code.shape[0], -1, 3)
            kp_transformed = torch.bmm(kp_orig, deformed_matrix.transpose(1, 2))
            mse_loss = torch.mean((kp_transformed - kp_deformed) ** 2)

            # partial view consistency mse (full point cloud & partial view)
            # use keypoint predictions from the encoder process
            kp_partial = partial_code_.reshape(partial_code.shape[0], -1, 3)
            partial_mse_loss = torch.mean((kp_orig - kp_partial) ** 2)
            
            # partial view consistency mse (partial view point cloud & deformed partial view)
            # data["partial_deformed_shape"].view(data["partial_orig_offset"].shape[0], -1, 3)

            # partial_deformed_matrix = data["partial_deformed_transformation"].view(
            #     data["partial_orig_offset"].shape[0], -1, 3
            # )

            # partial_deformed_code, _, _ = net(get_network_data(data, "partial_deformed"))
            # kp_partial_deformed = partial_deformed_code.reshape(code.shape[0], -1, 3)
            # kp_partial_transformed = torch.bmm(kp_partial, partial_deformed_matrix.transpose(1, 2))
            # partial_deformed_mse_loss = torch.mean((kp_partial_transformed - kp_partial_deformed) ** 2)
            
            
            # partial warmup
            if opt.lambda_p == 0:
                lambda_p = 0
            else:
                lambda_p = 0 if t < opt.partial_start_steps else min(1.0, (t - opt.partial_start_steps) / opt.partial_warmup_steps)
            
            # combine all losses
            total_diffusion_loss = diffusion_loss #+ partial_diffusion_loss
            total_fps_loss = fps_loss + lambda_p * partial_fps_loss
            total_kl_loss = kl + lambda_p * partial_kl
            total_chamfer_loss = chamfer_loss + lambda_p * partial_chamfer_loss
            total_mse_loss = mse_loss + lambda_p * partial_mse_loss #+ partial_deformed_mse_loss
            

            loss_ = (
                lambda_0 * total_fps_loss
                + lambda_1 * total_diffusion_loss
                + lambda_2 * total_chamfer_loss
                + lambda_3 * total_mse_loss
                + lambda_4 * total_kl_loss
            )

            if rank == 0:
                wandb.log({"chamfer_loss": chamfer_loss, "mse_loss": mse_loss}, step=t)
                wandb.log({"partial_chamfer_loss": partial_chamfer_loss, "partial_mse_loss": partial_mse_loss}, step=t)

            loss = max(0, max_schedule - t) / max_schedule * loss_

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
            for p_ema, p_net in zip(ema.parameters(), net.parameters(), strict=False):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            cur_nimg += opt.batch_size

            # record options and parameters at the start of training
            if t == 0 and rank == 0:
                message = "--------------------------- Options -------------------------\n"
                message += "------------- All options not shown are default -------------\n"
                for k, v in sorted(vars(opt).items()):
                    v = v if v is not None else ""
                    message += f"{k:>25}: {v:<30}\n"
                message += "--------------------------- End -----------------------------\n"
                with open(log_path, "a") as log_file:
                    log_file.write(message)
                    
            if t % opt.save_interval == 0 and rank == 0:
                os.path.join(checkpoints_dir, "outputs", "%07d" % t)
                save_network(ema, checkpoints_dir, network_label="ema", epoch_label=t)
                save_network(net, checkpoints_dir, network_label="net", epoch_label=t)

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

        # with torch.no_grad():
        #     for _, data in enumerate(test_dataloader):

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
        RUN = wandb.init(project="diffuse_keypoints_test_fr")
        wandb.log(
            {
                "ckpt": opt.ckpt,
                "n_keypoints": opt.latent_dim,
                "type": "ours",
                "category": opt.category,
            }
        )

        test(opt)
    elif opt.phase == "train":
        print(f"Rank: {rank}, World size: {world_size}")

        if rank == 0:
            RUN = wandb.init(project="diffuse_keypoints_lamp_fr")
            wandb.run.log_code(".")
        train(opt, rank, world_size)
        print(f"Run name: {wandb.run.name}")

    else:
        raise ValueError()
