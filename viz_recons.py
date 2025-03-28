import numpy as np
import open3d as o3d


def np_to_pcd(y, color=(0.5, 0.5, 0.5)):
    """
    Convert a numpy array to an Open3D point cloud with uniform color.

    Args:
        y: numpy array of shape (N, 4), where y[:, :3] are xyz and y[:, 3] are labels (ignored).
        color: tuple of 3 floats in [0, 1], specifying RGB color.

    Returns:
        Open3D PointCloud object with uniform color.
    """
    xyz = y[:, 0:3]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Set uniform color
    pcd.paint_uniform_color(color)

    return pcd


data = np.load("recons.npy")

for t in range(800):
    pcd = np_to_pcd(data[t, ...])
    o3d.visualization.draw_geometries([pcd])
