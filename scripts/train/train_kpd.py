from glob import glob

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from baselines.keypointdeformer.models.cage_skinning import CageSkinning
from baselines.keypointdeformer.options.base_options import BaseOptions
from baselines.keypointdeformer.utils.nn import save_network, weights_init
from keypoint_diffuser.datasets.H5Datset import H5Dataset
from utils import DATASET

import wandb


def get_data(data):
    source_shape, target_shape = data["source_shape"], data["target_shape"]

    source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)

    return source_shape_t.cuda(), target_shape_t.cuda()


def train(opt):
    wandb.init(project=f"keygrid_{opt.category}_train", config=opt)

    ckpt_dir = opt.ckpt_dir / wandb.run.name
    ckpt_dir.mkdir()

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

    # train
    net.train()
    t = 0

    if opt.iteration:
        t = opt.iteration
    epoch_idx = 0

    while t <= opt.n_iterations:
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            source_shape_t, target_shape_t = get_data(data)
            net(source_shape_t, target_shape=target_shape_t)
            current_loss = net.compute_loss(t)
            net.optimize(current_loss, t)

            wandb.log(current_loss)
            t += 1

        epoch_idx += 1

        save_network(net, ckpt_dir, network_label="net", epoch_label=t)

    save_network(net, ckpt_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train(opt)
