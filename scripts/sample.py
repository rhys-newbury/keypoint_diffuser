import os
from enum import Enum
from glob import glob
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypoint_diffuser.models import get_model
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.models.encoder_models.autoencoder_orig import AutoEncoderOrig
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.options.base_options import BaseOptions
from keypoint_diffuser.utils.eval_metrics import EMD_CD, EMD_CD_recon
from keypoint_diffuser.utils.nn import load_network
from keypoint_diffuser.utils.pc_utils import collate_fn, normalize_point_clouds
from keypoint_diffuser.utils.utils import Timer, reparameterize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


torch.autograd.set_detect_anomaly(True)

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


class Algos(Enum):
    Ours = 0
    KPD = 1
    DPM = 2


# Initialize distributed environment
def setup(rank, world_size):
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        print("local_rank", local_rank, "rank: ", rank, "world_size: ", world_size)
        torch.cuda.set_device(local_rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_data(dataset, data):
    data = dataset.uncollate(data)

    source_shape = data["source_shape"]

    target_shape = data["target_shape"]

    target_shape_t = target_shape.transpose(1, 2)
    source_shape_t = source_shape.transpose(1, 2)

    return source_shape_t, target_shape_t


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


def sample(opt):
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
    opt.phase = "train"
    opt.split = "train"

    DATASET = "/app/shapenetcorev2_hdf5_2048/train/"
    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files, normalize=True, include_label=False, subclasses=(12,), transform=t
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn if Algos.Ours == CURRENT_EVAL else dataset.collate,
        num_workers=0,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    opt.phase = "test"
    opt.split = "test"
    DATASET = "/app/shapenetcorev2_hdf5_2048/val/"
    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    test_dataset = H5Dataset(
        h5_files, normalize=True, include_label=False, subclasses=(12,), transform=t
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn if Algos.Ours == CURRENT_EVAL else dataset.collate,
        num_workers=0,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)

    if Algos.Ours == CURRENT_EVAL:
        ckpt = torch.load(ckpt)

        ae_model = AutoEncoder(opt).cuda()
        ae_model.load_state_dict(ckpt["states"])
        ae_model.eval()
    elif Algos.KPD == CURRENT_EVAL:
        net = get_model(opt.model)(opt).cuda()
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)
    elif Algos.DPM == CURRENT_EVAL:
        ae_model = AutoEncoderOrig(opt).cuda()
        ckpt = torch.load(ckpt)
        ae_model.load_state_dict(ckpt["states"])

    Timer("step")
    all_z0 = []  # flatten z0 per batch
    all_recons = []
    all_z_aux = []
    models = []

    with torch.no_grad():
        total_batches = len(train_dataloader)

        # Wrap the train_dataloader with tqdm
        for data in tqdm(
            train_dataloader,
            desc="Processing train data",
            unit="batch",
            total=total_batches,
        ):
            if Algos.Ours == CURRENT_EVAL:
                z0, mu, logvar = ae_model.encode(get_network_data(data))
                z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)

                # Step 3: Concatenate
                z_full = torch.cat([z0.reshape(z0.shape[0], -1), z_aux], dim=1)

                all_z0.append(z0.detach().cpu())
                all_z_aux.append(z_aux.cpu())
            elif Algos.KPD == CURRENT_EVAL:
                data = dataset.uncollate(data)

                source_shape_t, target_shape_t = get_data(dataset, data)
                outputs = net(source_shape_t, target_shape=target_shape_t)
                # B, 3, 8
                all_z0.append(
                    outputs["target_keypoints"]
                    .reshape(source_shape_t.shape[0], -1)
                    .detach()
                    .cpu()
                )
            elif Algos.DPM == CURRENT_EVAL:
                x = data["target_shape"].view(len(data["target_cat"]), -1, 3).cuda()
                z0 = ae_model.encode(x)
                all_z0.append(z0.detach().cpu())

        for data in tqdm(
            test_dataloader,
            desc="Processing test data",
            unit="batch",
            total=len(test_dataloader),
        ):
            if Algos.Ours == CURRENT_EVAL:
                target_shape_t = (
                    data["target_shape"]
                    .view(data["orig_offset"].shape[0], -1, 3)
                    .transpose(1, 2)
                )

                z0, mu, logvar = ae_model.encode(get_network_data(data))
                z_aux = reparameterize(mu, logvar)  # sampled from q(z|x)

                # Step 3: Concatenate
                z_full = torch.cat([z0.reshape(z0.shape[0], -1), z_aux], dim=1)

                recons = ae_model.decode(z_full, 2048).detach()
                all_recons.append(recons)

            elif CURRENT_EVAL == Algos.KPD or CURRENT_EVAL == Algos.DPM:
                _, target_shape_t = get_data(test_dataset, data)
                target_shape_t = target_shape_t.cpu()

            models.append(target_shape_t)

    all_z0 = torch.cat(all_z0, dim=0).reshape(-1, 30)  # [N, 3d]
    if Algos.Ours == CURRENT_EVAL:
        all_z_aux = torch.cat(all_z_aux, dim=0)  # [N, m]
        mean_z_aux = all_z_aux.mean(dim=0, keepdim=True)  # [1, m]

    # Step 2: Fit PCA on z0
    pca = PCA(n_components=10)
    z0_pca = pca.fit_transform(all_z0.numpy())
    print("fit pca")

    # Fit KDE in PCA space
    kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(z0_pca)
    print("fit kde")

    # Sample new points
    models = torch.cat(models).transpose(1, 2)
    num_samples = models.shape[0]
    if Algos.Ours == CURRENT_EVAL:
        z_aux_tensor = mean_z_aux.repeat(num_samples, 1).cuda()

    samples = kde.sample(num_samples)
    new_z0 = torch.tensor(pca.inverse_transform(samples)).float().cuda()
    if Algos.Ours == CURRENT_EVAL:
        z_full = torch.cat([new_z0, z_aux_tensor], dim=1)
    elif Algos.KPD == CURRENT_EVAL:
        z_full = new_z0.reshape(-1, 3, 8)
    elif Algos.DPM == CURRENT_EVAL:
        z_full = new_z0

    new = []

    with torch.no_grad():
        for idx in tqdm(range(num_samples), total=num_samples):
            if Algos.Ours == CURRENT_EVAL:
                recons = ae_model.decode(z_full[idx : idx + 1, ...], num_points=2048)
            elif Algos.KPD == CURRENT_EVAL:
                data = dataset.get_sample(np.random.randint(dataset.get_real_length()))
                recons = net.decode(
                    data["source_shape"].T[None, :, :].cuda(),
                    z_full[idx : idx + 1, ...],
                )
                recons = recons["deformed"]  # torch.Size([1, 2048, 3])
            elif Algos.DPM == CURRENT_EVAL:
                recons = ae_model.decode(z_full[idx : idx + 1, ...], 2048).detach()

            new.append(recons.cpu())
            torch.cuda.empty_cache()

    generated = torch.cat(new).float().cpu()  # shape: (N, 2048, 3)
    np.save("generated_ppl.npy", generated)

    generated = normalize_point_clouds(generated, "shape_bbox")
    models = normalize_point_clouds(models, "shape_bbox")
    recons = normalize_point_clouds(torch.cat(all_recons), "shape_bbox")

    # Get min and max from generated reconstructions

    # Generate random noise in [0, 1]

    # Scale noise to match the bounding box of the generated point clouds

    print(
        "Generated vs Ground Truth:",
        EMD_CD(generated.float(), models.cpu().float(), batch_size=8),
    )
    print(
        "reconstruction errors: ",
        EMD_CD_recon(recons.cuda().float(), models.cuda().float(), batch_size=8),
    )

    # print(
    # for i, pc in enumerate(new):
    #     pcd_clean.estimate_normals(


if __name__ == "__main__":
    CURRENT_EVAL = Algos.Ours

    if CURRENT_EVAL == Algos.KPD:
        parser = BaseOptions()
    elif Algos.Ours == CURRENT_EVAL or CURRENT_EVAL == Algos.DPM:
        parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    sample(opt)
