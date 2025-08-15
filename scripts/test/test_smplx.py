import csv
import os
import pickle
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from keypoint_diffuser.datasets import get_dataset
from keypoint_diffuser.models import get_model
from keypoint_diffuser.models.encoder_models.autoencoder import AutoEncoder
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.options.base_options import BaseOptions
from keypoint_diffuser.utils.nn import load_network
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


class Algos(Enum):
    Ours = 0
    KPD = 1


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


def get_data(dataset, data):
    data = dataset.uncollate(data)
    source_shape = data["point_clouds"]

    source_shape_t = torch.cat(source_shape).transpose(1, 2)

    return source_shape_t, source_shape_t


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

    if Algos.Ours == CURRENT_EVAL:
        ckpt = torch.load(ckpt)

        ae_model = AutoEncoder(opt).cuda()
        ae_model.load_state_dict(ckpt["states"])
        ae_model.eval()
    else:
        net = get_model(opt.model)(opt).cuda()
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    temporal_dists = []  # [K x T] list of distances

    for idx, data in tqdm(
        enumerate(dataloader), desc="Evaluating temporal consistency", total=100
    ):
        frame_count = 10  # Should be 10

        # Load joints for each frame
        joints_per_frame = []
        vertices_per_frame = []
        faces_per_frame = []
        for _i in range(frame_count):
            pkl_path = f"/app/data/smplx/{data['smplx_path'][_i][0]}"
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            joints = torch.tensor(pkl_data["joints"])  # Shape: [J, 3]
            vertices = torch.tensor(pkl_data["vertices"])
            faces = pkl_data["body_model.faces"]

            joints_per_frame.append(joints)
            vertices_per_frame.append(vertices)
            faces_per_frame.append(faces)

        # Process entire sequence through encoder

        if Algos.Ours == CURRENT_EVAL:
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
            pc = inp["point_clouds"]
            keypoints = z0.reshape(10, -1, 3).cpu()  # Shape: [B, K, 3]
        elif CURRENT_EVAL == Algos.KPD:
            data = dataset.uncollate(data)
            pc = torch.cat(data["point_clouds"])

            source_shape_t, target_shape_t = get_data(dataset, data)
            outputs = net(source_shape_t, target_shape=target_shape_t)
            keypoints = outputs["target_keypoints"].transpose(1, 2).cpu()

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
                "target_shape": pc.reshape(10, -1, 3).cpu().numpy(),  # [T, N, 3]
                "vertices": torch.stack(vertices_per_frame),
                "faces_per_frame": faces_per_frame,
                # "reconstructed": recons.cpu().numpy(),  # [T, N, 3]
            },
        }
        np.savez(
            f"./scripts/{CURRENT_EVAL.name}_{idx}.npz",
            mean_dist=result["mean_dist"],
            keypoints=result["data"]["keypoints"],
            assigned_joints=result["data"]["assigned_joints"],
            target_shape=result["data"]["target_shape"],
            vertices=result["data"]["vertices"],
            faces_per_frame=result["data"]["faces_per_frame"],
        )

    len(temporal_dists)  # number of sequences
    K = temporal_dists[0].shape[0]
    T = temporal_dists[0].shape[1]

    temporal_dists = torch.stack(temporal_dists, dim=0)  # [N, K, T]

    # Compute mean and std over time
    mean_over_time = temporal_dists.mean(dim=2)  # [N, K]
    std_over_time = temporal_dists.std(dim=2)  # [N, K]

    # Now average over samples → per-keypoint metrics
    mean_per_kp = mean_over_time.mean(dim=0)  # [K]
    std_per_kp = std_over_time.mean(dim=0)  # [K]

    # np.savez(
    #     "best_result.npz",

    for i, (mean_i, std_i) in enumerate(zip(mean_per_kp, std_per_kp, strict=True)):
        wandb.log(
            {
                f"temporal_consistency/per_kp/mean_distance/kp_{i}": mean_i.item(),
                f"temporal_consistency/per_kp/std_distance/kp_{i}": std_i.item(),
            }
        )

    mean_per_kp_per_frame = temporal_dists.mean(dim=0)  # [K, T]

    K, T = mean_per_kp_per_frame.shape

    wandb.log(
        {
            "temporal_consistency/keypoint_distance_over_time": wandb.plot.line_series(
                xs=list(range(T)),
                ys=[mean_per_kp_per_frame[k].detach().cpu().numpy() for k in range(K)],
                keys=[f"KPT {k}" for k in range(K)],
                title="Keypoint-Joint Distance Over Time",
                xname="Frame Index",
            )
        }
    )

    # Compute mean and std over sequences
    mean_per_kp_per_frame = temporal_dists.mean(dim=0)  # [K, T]
    temporal_dists.std(dim=0)  # [K, T]

    # Average over sequences and keypoints: [T]
    mean_per_frame = temporal_dists.mean(dim=(0, 1))  # [T]
    std_per_frame = temporal_dists.std(dim=(0, 1))  # [T]

    # Output path
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"temporal_consistency_summary_{CURRENT_EVAL.name}.csv"
    )

    # Write CSV: Frame, Mean, Std
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Mean", "Std"])
        for t in range(mean_per_frame.shape[0]):
            writer.writerow([t, mean_per_frame[t].item(), std_per_frame[t].item()])

    print(f"Saved simplified mean/std CSV to {output_path}")


if __name__ == "__main__":
    CURRENT_EVAL = Algos.KPD

    if CURRENT_EVAL == Algos.KPD:
        parser = BaseOptions()
    elif Algos.Ours == CURRENT_EVAL:
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
                "key_points": opt.latent_dim,
                "type": CURRENT_EVAL.name,
                "category": opt.category,
            }
        )

        test(opt)

    else:
        raise ValueError()
