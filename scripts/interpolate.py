import os
from typing import Any

import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from keypoint_diffuser.datasets import get_dataset
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.utils.utils import Timer, reparameterize
from tqdm import tqdm


torch.autograd.set_detect_anomaly(True)

from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    ApplyToBoth,
    Collect,
    Deform,
    GridSample,
    ToTensor,
)
from sklearn.decomposition import PCA
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


def interpolate(opt):
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
    Timer("step")
    all_z0 = []  # flatten z0 per batch
    all_z_aux = []

    with torch.no_grad():
        total_batches = len(dataloader)

        # Wrap the dataloader with tqdm
        for data in tqdm(
            dataloader, desc="Processing data", unit="batch", total=total_batches
        ):
            z0, mu, logvar = ae_model.encode(get_network_data(data))
            z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)

            # Step 3: Concatenate
            z_full = torch.cat([z0, z_aux], dim=1)

            all_z0.append(z0.detach().cpu())
            all_z_aux.append(z_aux.cpu())

    all_z0 = torch.cat(all_z0, dim=0)  # [N, 3d]
    all_z_aux = torch.cat(all_z_aux, dim=0)  # [N, m]
    mean_z_aux = all_z_aux.mean(dim=0, keepdim=True)  # [1, m]

    # Step 2: Fit PCA on z0
    pca = PCA(n_components=10)
    pca.fit(all_z0.numpy())

    # Step 3: Pick two real samples (airplanes)
    data_iter = iter(dataloader)
    data = next(data_iter)

    target_shape_t = (
        data["target_shape"]
        .view(data["orig_offset"].shape[0], -1, 3)
        .transpose(1, 2)
        .cuda()
    )

    pcd1 = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(target_shape_t[0, :, :].T.cpu().numpy())
    )
    pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd1.orient_normals_consistent_tangent_plane(k=10)

    pcd2 = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(target_shape_t[1, :, :].T.cpu().numpy())
    )
    pcd2.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd2.orient_normals_consistent_tangent_plane(k=10)

    o3d.io.write_point_cloud("interp_pca_airplane_0.ply", pcd1)

    o3d.io.write_point_cloud("interp_pca_airplane_7.ply", pcd2)

    encoded = ae_model.encode(get_network_data(data))
    keypoints_a = encoded[0][0].detach().cpu().numpy().reshape(8, 3)
    keypoints_b = encoded[0][1].detach().cpu().numpy().reshape(8, 3)

    # Step 3: Interpolate directly in keypoint space
    alphas = np.linspace(0, 1, num=6)
    interp_keypoints = np.array(
        [(1 - a) * keypoints_a + a * keypoints_b for a in alphas]
    )  # [6, 8, 3]

    # Step 4: Flatten and convert to tensor
    interp_keypoints_tensor = (
        torch.tensor(interp_keypoints.reshape(6, -1)).float().cuda()
    )

    # Step 5: Concatenate mean z_aux
    z_aux_tensor = mean_z_aux.repeat(len(interp_keypoints_tensor), 1).cuda()
    z_full = torch.cat([interp_keypoints_tensor, z_aux_tensor], dim=1)

    recons_interps = []

    with torch.no_grad():
        for idx in range(6):
            print(idx)
            recons = ae_model.decode(z_full[idx : idx + 1, ...], num_points=5000)
            recons_interps.append(recons.cpu())
            torch.cuda.empty_cache()

    # Step 7: Save point clouds
    for i, pc in enumerate(recons_interps):
        pcd = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(pc.squeeze().numpy())
        )
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd_clean.orient_normals_consistent_tangent_plane(k=10)

        o3d.io.write_point_cloud(f"interp_pca_airplane_{i+1}.ply", pcd_clean)


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    interpolate(opt)
