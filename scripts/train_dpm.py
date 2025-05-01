import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from keypoint_diffuser.datasets import get_dataset
from keypoint_diffuser.models.encoder_models.autoencoder_orig import AutoEncoderOrig
from keypoint_diffuser.models.encoder_models.common import get_linear_scheduler
from keypoint_diffuser.options.ae_options import AEOptions
from keypoint_diffuser.utils.eval_metrics import EMD_CD
from keypoint_diffuser.utils.nn import load_network, save_network
from keypoint_diffuser.utils.pc_utils import collate_fn, normalize_point_clouds
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import wandb


torch.autograd.set_detect_anomaly(True)
RUN = None


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


# Initialize distributed environment
def setup(rank, world_size):
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        print("local_rank", local_rank, "rank: ", rank, "world_size: ", world_size)
        torch.cuda.set_device(local_rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def test(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)
    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt)

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

    ae_model = AutoEncoderOrig(opt).cuda()
    ae_model.load_state_dict(ckpt["states"])
    ae_model.eval()
    all_ref = []
    all_recons = []
    with torch.no_grad():
        total_batches = len(dataloader)

        # Wrap the dataloader with tqdm
        for data in tqdm(
            dataloader, desc="Processing data", unit="batch", total=total_batches
        ):
            target_shape_t = (
                data["target_shape"]
                .view(len(data["target_cat"]), -1, 3)
                .transpose(1, 2)
                .cuda()
            )

            x = data["target_shape"].view(len(data["target_cat"]), -1, 3).cuda()

            z0 = ae_model.encode(x)
            recons = ae_model.decode(z0, 5000).detach()

            all_ref.append(target_shape_t.detach().cpu())
            all_recons.append(recons.detach().cpu())

        all_ref = torch.cat(all_ref, dim=0).permute(0, 2, 1)
        all_ref = normalize_point_clouds(all_ref, "shape_bbox")
        all_recons = torch.cat(all_recons, dim=0)
        all_recons = normalize_point_clouds(all_recons, "shape_bbox")
        metrics = EMD_CD(
            all_recons.to("cuda").double(),
            all_ref.to("cuda").double(),
            opt.batch_size,
        )
        wandb.log(metrics)
        for key, value in metrics.items():
            print(f"{key}: {value.item():.10f}")


def train(opt, rank, world_size):
    if rank == 0:
        log_dir = os.path.join(opt.log_dir, RUN.name)
        checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)

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
    test_dataset = get_dataset(opt_test.dataset)(opt_test)

    torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id_: np.random.seed(np.random.get_state()[1][0] + id_),
    )

    net = AutoEncoderOrig(opt).cuda()

    if torch.cuda.device_count() > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        copy.deepcopy(net).eval().requires_grad_(False)

        print(f"Using DistributedDataParallel on {torch.cuda.device_count()}")
        net = DistributedDataParallel(
            net, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
    else:
        copy.deepcopy(net).eval().requires_grad_(False)

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

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    epoch = 0

    while t <= opt.n_iterations:
        epoch += 1
        iter_time_start = time.time()  # Start iteration timer
        if torch.cuda.device_count() > 1 and world_size > 1:
            dataloader.sampler.set_epoch(epoch)

        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            x = data["target_shape"].view(len(data["target_cat"]), -1, 3).cuda()
            loss = net.get_loss(x)
            loss.backward()

            clip_grad_norm_(net.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if t % opt.save_interval == 0 and rank == 0:
                os.path.join(checkpoints_dir, "outputs", "%07d" % t)
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
                "type": "dpm",
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
