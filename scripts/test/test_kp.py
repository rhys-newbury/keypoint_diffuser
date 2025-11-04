from glob import glob
from typing import Any

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from datasets.H5Datset import H5Dataset
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEConfig, AEOptions


torch.autograd.set_detect_anomaly(True)
RUN = None

from torchvision import transforms

from keypoint_diffuser.utils.nn import load_network
from keypoint_diffuser.utils.pc_utils import collate_fn
from keypoint_diffuser.utils.transforms import (
    Collect,
    GridSample,
    ToTensor,
)


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


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

def project_to_surface(pred, surface_points, eps=0.05):
    """
    Hard project keypoints so they lie within `eps` distance of surface.
    If already closer than eps, no change.

    pred: (K, 3)
    surface_points: (N, 3)
    eps: allowed max distance
    """
    with torch.no_grad():
        # Compute nearest surface point for each keypoint
        dists = torch.cdist(pred[None, ...], surface_points[None, ...])  # (1, K, N)
        idx = dists.argmin(dim=-1)  # (1, K)
        nearest = surface_points[0, idx.squeeze(0)]  # (K, 3)

        # Vector from keypoint → nearest surface
        vec = nearest - pred
        dist = vec.norm(dim=-1, keepdim=True) + 1e-8

        # If farther than eps, pull it just enough to reach eps
        overshoot = (dist > eps)
        scale = torch.where(overshoot, (dist - eps) / dist, torch.zeros_like(dist))
        pred_clamped = pred + vec * scale

        return pred_clamped


def train(opt: AEConfig):

    t = transforms.Compose(
        [
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
        ]
    )

    DATASET = "/app/shapenetcorev2_hdf5_2048/test/"
    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files,
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

    net = AutoEncoder(opt).cuda()
    load_network(net, "/mnt/slow2/skeleton_merger/light-blaze-22/10kp_99.pth")

    net.eval()


    for _e in range(opt.epochs):
        for _, data in enumerate(dataloader):
            target_shape_t = (
                data["target_shape"]
                .view(data["orig_offset"].shape[0], -1, 3)
                .cuda()
            )
            batch = {k: v.cuda() for k, v in data.items()}

            _, code, _, _ = net.get_loss(
                get_network_data(batch), step=0
            )
            code = project_to_surface(code, target_shape_t)
            torch.cdist(torch.tensor(code)[0, ...], torch.tensor(target_shape_t)[0, ...]).min(axis=1)[0]

            import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train(opt)
