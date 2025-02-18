import numpy as np
import open3d as o3d


def load_and_view_point_clouds(
    original_path="original_point_cloud.npy",
    deformed_path="deformed_point_cloud.npy",
    kp_orig_path="kp_orig.npy",
    kp_transformed_path="kp_transformed.npy",
    kp_deformed_path="kp_deformed.npy",
):
    """
    Loads NumPy saved point clouds and keypoints, and visualizes them using Open3D.

    Args:
        original_path: Path to the original point cloud `.npy` file.
        deformed_path: Path to the deformed point cloud `.npy` file.
        kp_orig_path: Path to original keypoints `.npy` file.
        kp_transformed_path: Path to transformed keypoints `.npy` file.
        kp_deformed_path: Path to keypoints extracted from deformed shape `.npy` file.
    """

    # Load the point clouds
    original_np = np.load(original_path).T  # (N, 3)
    deformed_np = np.load(deformed_path).T  # (N, 3)

    # Load keypoints
    kp_orig_np = np.load(kp_orig_path)  # (K, 3)
    kp_transformed_np = np.load(kp_transformed_path)  # (K, 3)
    kp_deformed_np = np.load(kp_deformed_path)  # (K, 3)

    # Convert to Open3D point cloud objects
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_np)
    pcd_original.paint_uniform_color([1, 0, 0])  # Red for original

    pcd_deformed = o3d.geometry.PointCloud()
    pcd_deformed.points = o3d.utility.Vector3dVector(deformed_np)
    pcd_deformed.paint_uniform_color([0, 0, 1])  # Blue for deformed

    # Convert keypoints to Open3D point clouds
    kp_orig_pcd = o3d.geometry.PointCloud()
    kp_orig_pcd.points = o3d.utility.Vector3dVector(kp_orig_np)
    kp_orig_pcd.paint_uniform_color([0, 1, 0])  # Green for original keypoints

    kp_transformed_pcd = o3d.geometry.PointCloud()
    kp_transformed_pcd.points = o3d.utility.Vector3dVector(kp_transformed_np)
    kp_transformed_pcd.paint_uniform_color(
        [1, 0, 0]
    )  # Yellow for transformed keypoints

    kp_deformed_pcd = o3d.geometry.PointCloud()
    kp_deformed_pcd.points = o3d.utility.Vector3dVector(kp_deformed_np)
    kp_deformed_pcd.paint_uniform_color([1, 0, 1])  # Magenta for deformed keypoints

    # Visualize everything together

    o3d.visualization.draw_geometries([pcd_deformed, pcd_original])

    o3d.visualization.draw_geometries([pcd_original, kp_orig_pcd])
    o3d.visualization.draw_geometries([pcd_deformed, kp_deformed_pcd])

    o3d.visualization.draw_geometries([pcd_deformed, kp_transformed_pcd])


if __name__ == "__main__":
    load_and_view_point_clouds()
