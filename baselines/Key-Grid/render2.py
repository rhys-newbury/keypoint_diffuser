import numpy as np
import open3d as o3d

# Load the data
data = np.load("output_6.npz")
points = data["pointcloud"]       # Nx3
recons = data["recons"]       # Nx3

keypoints = data["keypoints"]     # 10x3
ground_truths = data["ground_truths"]  # 17x3

# Create point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

pcd_recons = o3d.geometry.PointCloud()
pcd_recons.points = o3d.utility.Vector3dVector(recons)
pcd_recons.paint_uniform_color([0, 0, 0])  # light gray


# Function to create keypoints as small spheres
def create_spheres(centers, color, radius=0.03):
    spheres = []
    for c in centers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(c)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres

# Create keypoint and ground truth spheres
kp_spheres = create_spheres(keypoints, color=[1, 0, 0])       # red
gt_spheres = create_spheres(ground_truths, color=[0, 1, 0])   # green

# Combine all geometries
geometries = [pcd_recons] + kp_spheres + gt_spheres

# Visualize
o3d.visualization.draw_geometries(geometries)
