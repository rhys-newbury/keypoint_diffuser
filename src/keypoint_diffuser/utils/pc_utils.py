from collections.abc import Mapping, Sequence

import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


def normalize_point_clouds(pcs, mode):
    if mode is None:
        print("Will not normalize point clouds.")
        return pcs
    print(f"Normalization mode: {mode}")
    for i in tqdm(range(pcs.size(0)), desc="Normalize"):
        pc = pcs[i]
        if mode == "shape_unit":
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == "shape_bbox":
            pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch, strict=False)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch:
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)
