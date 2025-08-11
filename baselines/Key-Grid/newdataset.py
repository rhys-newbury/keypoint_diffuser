import os
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=1))
    return pc


def farthest_point_sample(point, npoint):
    N, _ = point.shape
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = point[farthest, :]
        dist = np.sum((point - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return point[centroids]


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, class_id, split_csv, split='train', sample_points=2048,
                 normalize=True, downsample=True):
        """
        Args:
            root_dir (str): Root directory containing <class_id>/<model_id>/models/*.npy
            class_id (str): e.g., "02691156"
            split_csv (str): CSV file with columns [synsetId, modelId, split]
            split (str): "train" or "val"
            sample_points (int): Number of points to sample (via farthest point)
            normalize (bool): Whether to normalize point clouds to [-1, 1]
            downsample (bool): Whether to apply FPS to sample points
        """
        self.sample_points = sample_points
        self.normalize = normalize
        self.downsample = downsample
        self.model_paths = []

        # Load split file
        import pandas as pd
        df = pd.read_csv(split_csv, dtype={"synsetId": str})
        df["synsetId"] = df["synsetId"].str.zfill(8)
        df = df[(df['split'] == split) & (df['synsetId'] == class_id)]

        model_ids = df['modelId'].tolist()

        for model_id in tqdm(model_ids, desc=f"Indexing {split} samples"):
            model_dir = os.path.join(root_dir, class_id, model_id, "models")
            if not os.path.exists(model_dir):
                continue
            npy_files = sorted(glob(os.path.join(model_dir, 'new_samples_*.npy')))
            if npy_files:
                self.model_paths.append(npy_files)

        print(f"Loaded {len(self.model_paths)} models with point clouds.")

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        file_list = self.model_paths[idx]
        file_path = np.random.choice(file_list)
        pc = np.load(file_path)

        if self.normalize:
            pc = normalize_pc(pc)
        if self.downsample:
            pc = farthest_point_sample(pc, self.sample_points)

        return torch.from_numpy(pc.astype(np.float32)).float() 
