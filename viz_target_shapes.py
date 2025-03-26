import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# Load your point cloud data
x = np.load("target_sampled_points.npy")

# Get the unique segmentation labels for color mapping
all_labels = np.unique(x[..., 3].astype(int))
num_labels = len(all_labels)

# Generate a colormap
cmap = plt.get_cmap("tab20", num_labels)
colors_map = {label: cmap(i)[:3] for i, label in enumerate(all_labels)}


unique_labels, counts = np.unique(x[..., 3], return_counts=True)
print(unique_labels, counts)


for i in range(x.shape[0]):
    y = x[i, ...]  # Shape: (5000, 4)

    # Extract xyz and labels
    xyz = y[:, 0:3]
    labels = y[:, 3].astype(int)

    # Map labels to colors
    colors = np.array([colors_map[label] for label in labels])

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Visualizing sample {i+1}/{x.shape[0]}")
    o3d.visualization.draw_geometries([pcd])
