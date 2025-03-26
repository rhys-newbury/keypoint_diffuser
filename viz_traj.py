import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def chamfer_distance_numpy(pts1, pts2):
    print(pts1.shape, pts2.shape)
    kdtree_1 = cKDTree(pts1)
    kdtree_2 = cKDTree(pts2)

    dist1, _ = kdtree_1.query(pts2)
    dist2, _ = kdtree_2.query(pts1)

    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer


def look_top_down(vis):
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([1, -1, -1])  # Camera is up and to the right of the object
    ctr.set_up([0, 0, 1])  # Z is vertical in the view
    ctr.set_zoom(0.6)
    return False


def convert_green_to_transparent_opencv(input_path, output_path):
    # Load BGR image
    bgr = cv2.imread(input_path)

    # Convert to HSV for more robust green detection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Define loose green range
    lower_green = np.array([40, 40, 40])  # H, S, V
    upper_green = np.array([80, 255, 255])

    # Create mask where green is present
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask for alpha: green becomes transparent
    alpha = cv2.bitwise_not(green_mask)

    # Convert BGR to BGRA and apply new alpha
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    # Save output image
    cv2.imwrite(output_path, bgra)
    print(f"Saved cleaned RGBA image to: {output_path}")


def spheres_from_points(points, radius=0.01, color=(1, 0, 0)):
    spheres = []
    for pt in points:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
        mesh.translate(pt)
        mesh.paint_uniform_color(color)
        spheres.append(mesh)
    return spheres


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


# Load your point cloud data

data = np.load("diffusion_taj.npy")

kp_orig = np.load("viz_code.npy")[:, :-5].reshape(16, -1, 3)


import sys


t = int(sys.argv[1])
# for t in range(0, data.shape[0], 5):
print(t)
x = data[t, ...]
# import pdb; pdb.set_trace()
cd = []
i = 7
# for i in range(x.shape[0]):
pcd = np_to_pcd(x[i, ...])

R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  # 90° around X
pcd.rotate(R, center=(0, 0, 0))

vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)  # Offscreen render
opt = vis.get_render_option()
opt.background_color = np.array([0.0, 1.0, 0.0])


kpts = o3d.geometry.PointCloud()
kpts.points = o3d.utility.Vector3dVector(kp_orig[i])
kpts.paint_uniform_color([1.0, 0.0, 0.0])  # Red keypoints

kpts.rotate(R, center=(0, 0, 0))

keypoint_spheres = spheres_from_points(np.asarray(kpts.points), radius=0.04)
for sph in keypoint_spheres:
    vis.add_geometry(sph)


vis.add_geometry(pcd)


ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])
ctr.set_front([1, -1, -1])  # from top-front-right
ctr.set_up([0, 0, 1])
ctr.set_zoom(0.6)


for _ in range(10):
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)  # small delay to ensure rendering

# Save screenshot
vis.capture_screen_image(f"traj/plane_view_{t}.png")
print("Saved image as plane_view.png")

# Optionally keep window open
time.sleep(1)
vis.destroy_window()


convert_green_to_transparent_opencv(
    f"traj/plane_view_{t}.png", f"traj/plane_view_traj_{t}.png"
)
