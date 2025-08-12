import os
import time
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from keypointdeformer.models.cage_skinning import CageSkinning
from keypointdeformer.options.base_options import BaseOptions
from keypointdeformer.utils.nn import load_network, save_network, weights_init
from tensorboardX import SummaryWriter


CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_EXT = ".pth"


def get_data(dataset, data):
    source_shape, target_shape = data["source_shape"], data["target_shape"]

    source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)

    return source_shape_t.cuda(), target_shape_t.cuda()


def train(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    DATASET = "/app/shapenetcorev2_hdf5_2048/train/"
    h5_files = glob(f"{DATASET}**/*.h5", recursive=True)
    dataset = H5Dataset(
        h5_files,
        normalize=True,
        include_label=False,
        object_name=opt.category,
        get_two=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.n_workers,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    # network
    net = CageSkinning(opt).cuda()
    net.apply(weights_init)
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

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")
    summary_dir = datetime.now().strftime("%y%m%d-%H%M%S")
    SummaryWriter(
        logdir=os.path.join(checkpoints_dir, "logs", summary_dir), flush_secs=5
    )

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    while t <= opt.n_iterations:
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            source_shape_t, target_shape_t = get_data(dataset, data)
            net(source_shape_t, target_shape=target_shape_t)
            current_loss = net.compute_loss(t)
            net.optimize(current_loss, t)

            if t % opt.save_interval == 0:
                os.path.join(checkpoints_dir, "outputs", "%07d" % t)
                save_network(net, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()
            if t % opt.log_interval == 0:
                log_str = ""
                samples_sec = opt.batch_size / iter_time
                losses_str = ", ".join(
                    [f"{k} {v.mean().item():.3g}" for k, v in current_loss.items()]
                )
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str
                )

                print(log_str)
                log_file.write(log_str + "\n")

            t += 1

    log_file.close()
    save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "train":
        train(opt)
    else:
        raise ValueError()
