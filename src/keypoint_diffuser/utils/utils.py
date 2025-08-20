
import numpy as np
import pytorch3d.io
import torch
from einops import repeat


def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=points.device)

    gather_index = repeat(index, "b -> b c 1", c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, "b -> b c 1", c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(
            dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1)
        )

    if return_index:
        return sampled, indexes
    else:
        return sampled


def resample_mesh(mesh, n_points):
    points, normals = pytorch3d.ops.sample_points_from_meshes(
        mesh, n_points, return_normals=True
    )
    points = torch.cat([points[0], normals[0]], dim=-1)
    return points


def normalize_to_box(inp):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    inp: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(inp.shape) == 2:
        axis = 0
        P = inp.shape[0]
        D = inp.shape[1]
    elif len(inp.shape) == 3:
        axis = 1
        P = inp.shape[1]
        D = inp.shape[2]
    else:
        raise ValueError()

    if isinstance(inp, np.ndarray):
        maxP = np.amax(inp, axis=axis, keepdims=True)
        minP = np.amin(inp, axis=axis, keepdims=True)
        centroid = (maxP + minP) / 2
        inp = inp - centroid
        furthest_distance = np.amax(np.abs(inp), axis=(axis, -1), keepdims=True)
        inp = inp / furthest_distance
    elif isinstance(inp, torch.Tensor):
        maxP = torch.max(inp, dim=axis, keepdim=True)[0]
        minP = torch.min(inp, dim=axis, keepdim=True)[0]
        centroid = (maxP + minP) / 2
        inp = inp - centroid
        in_shape = [*list(inp.shape[:axis]), P * D]
        furthest_distance = torch.max(
            torch.abs(inp).view(in_shape), dim=axis, keepdim=True
        )[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        inp = inp / furthest_distance
    else:
        raise ValueError()

    return inp, centroid, furthest_distance


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
