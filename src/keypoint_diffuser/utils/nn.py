import os
from collections import OrderedDict

import numpy as np
import torch


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step

    From https://github.com/yifita/deep_cage
    """
    if isinstance(path, str):
        if path[-3:] == "pth":
            loaded_state = torch.load(path)
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
        else:
            loaded_state = np.load(path).item()
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if len(missingkeys) > 0:
        print(
            f"load_network {len(missingkeys)} missing keys",
            "\n".join(missingkeys),
        )
    if len(unexpectedkeys) > 0:
        print(
            f"load_network {len(unexpectedkeys)} unexpected keys",
            "\n".join(unexpectedkeys),
        )


def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included

    From https://github.com/yifita/deep_cage
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers

    From https://github.com/yifita/deep_cage
    """
    if isinstance(m, torch.nn.modules.conv._ConvNd | torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)
