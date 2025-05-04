import os
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from keypoint_diffuser.datasets import get_dataset
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEOptions
from tqdm import tqdm

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


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def test(opt):
    t = transforms.Compose(
        [
            Deform(),
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
    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt, transform=t)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
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

    temporal_dists = []  # [K x T] list of distances

    for idx, data in tqdm(
        enumerate(dataloader), desc="Evaluating temporal consistency", total=100
    ):
        frame_count = 10  # Should be 10

        # Load joints for each frame
        joints_per_frame = []
        for _i in range(frame_count):
            pkl_path = f"/app/data/smplx/{data['smplx_path'][_i][0]}"
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            joints = torch.tensor(pkl_data["joints"])  # Shape: [J, 3]
            joints_per_frame.append(joints)

        # Process entire sequence through encoder

        skip_keys = {"smplx_path"}  # or any set of keys you want to skip

        # Filter out keys you want to include
        include_keys = [k for k in data if k not in skip_keys]

        list_of_dicts = []
        for values in zip(*(data[k] for k in include_keys), strict=True):
            item = {}
            for k, v in zip(include_keys, values, strict=True):
                if isinstance(v, torch.Tensor):
                    if v.dim() > 0 and v.size(0) == 1:
                        v = v.squeeze(0)
                    v = v.cuda()
                item[k] = v
            list_of_dicts.append(item)

        inp = collate_fn(list_of_dicts)

        z0, mu, logvar = ae_model.encode(get_network_data(inp))
        keypoints = z0.reshape(10, -1, 3).cpu()  # Shape: [B, K, 3]

        # Step 3: Concatenate

        # Associate each keypoint to closest joint at frame 0
        frame0_kp = keypoints[0, ...]  # [K, 3]
        frame0_joints = joints_per_frame[0]  # [J, 3]
        dists = torch.cdist(frame0_kp.unsqueeze(0), frame0_joints.unsqueeze(0))[
            0
        ]  # [K, J]
        assigned_joint_idx = torch.argmin(dists, dim=1)  # [K]

        # Track distance between each keypoint and same joint across all frames
        per_keypoint_dists = []  # List of [T] tensors
        for k in range(frame0_kp.shape[0]):
            joint_idx = assigned_joint_idx[k]
            dist_over_time = torch.stack(
                [
                    torch.norm(keypoints[t, k] - joints_per_frame[t][joint_idx])
                    for t in range(frame_count)
                ]
            )
            per_keypoint_dists.append(dist_over_time)

        per_keypoint_dists = torch.stack(per_keypoint_dists)  # [K, T]
        temporal_dists.append(per_keypoint_dists)

        mean_dist = per_keypoint_dists.mean().item()
        # if mean_dist < best_result["mean_dist"]:

        result = {
            "mean_dist": mean_dist,
            "data": {
                "keypoints": keypoints.detach().numpy(),
                "assigned_joints": torch.stack(
                    [
                        joints_per_frame[t][assigned_joint_idx]
                        for t in range(frame_count)
                    ]
                ).numpy(),  # [T, K, 3]
                "target_shape": inp["point_clouds"]
                .reshape(10, -1, 3)
                .cpu()
                .numpy(),  # [T, N, 3]
                # "reconstructed": recons.cpu().numpy(),  # [T, N, 3]
            },
        }
        np.savez(
            f"./scripts/result_{idx}.npz",
            keypoints=result["data"]["keypoints"],
            assigned_joints=result["data"]["assigned_joints"],
            target_shape=result["data"]["target_shape"],
        )

    temporal_dists = torch.cat(temporal_dists, dim=0)  # [N*K, T]

    mean_per_kp = temporal_dists.mean(dim=1)  # [N*K]
    std_per_kp = temporal_dists.std(dim=1)  # [N*K]

    mean_mean = mean_per_kp.mean().item()
    mean_std = std_per_kp.mean().item()

    # np.savez(
    #     "best_result.npz",

    wandb.log(
        {
            "temporal_consistency_mean_distance": mean_mean,
            "temporal_consistency_std_distance": mean_std,
        }
    )

    print(f"Temporal Consistency - Mean Distance: {mean_mean:.6f}, Std: {mean_std:.6f}")


if __name__ == "__main__":
    parser = AEOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        RUN = wandb.init(project="smplx")
        print(opt)
        wandb.log(
            {
                "ckpt": opt.ckpt,
                "n_keypoints": opt.latent_dim,
                "type": "ours",
                "category": opt.category,
            }
        )

        test(opt)

    else:
        raise ValueError()
