# render_npz_open3d.py
import numpy as np
import open3d as o3d
import sys
import os

def render_npz_open3d(npz_path, radius=0.02):
    data = np.load(npz_path)
    pc = data["pointcloud"]
    kps = data["keypoints"]

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # light gray

    # Keypoints as red spheres
    spheres = []
    print(kps.shape)
    for kp in kps:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(kp)
        sphere.paint_uniform_color([1, 0, 0])
        spheres.append(sphere)

    o3d.visualization.draw_geometries([pcd] + spheres)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "viz_npz/sample_0000.npz"
    render_npz_open3d(path)
