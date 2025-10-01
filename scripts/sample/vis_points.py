import os
os.environ["PYGLET_HEADLESS"] = "True"      # make pyglet headless
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # this works better than egl
# os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
# os.environ.pop("DISPLAY", None)

# Point directly to the system OSMesa .so:
# Common locations; pick the one that exists on your system:
for candidate in (
    "/usr/lib/x86_64-linux-gnu/libOSMesa.so.8",
    "/usr/lib/x86_64-linux-gnu/libOSMesa.so",
    "/usr/local/lib/libOSMesa.so",
):
    if os.path.exists(candidate):
        os.environ["PYOPENGL_OSMESA_LIBRARY"] = candidate
        break
import pyrender
from pathlib import Path
from random import shuffle
from tqdm import tqdm
import open3d as o3d
import trimesh
import numpy as np
import pymeshlab
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
import yaml
import argparse
import re
from typing import List, Optional

# Use shared taxonomy/synset utilities
from synset_utils import (
    load_taxonomy_maps,
    names_to_synsets,
    synsets_to_names,
)

def look_at(eye, center, up=np.array([0, 0, 1])):
    """
    Create a 4x4 view matrix looking from eye to center.
    """
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rotation_matrix = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])
    # 5. Construct the translation part of the matrix
    translation_matrix = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])
    # 6. Combine rotation and translation
    # Note: In OpenGL-style, translation is applied after rotation
    model_view_matrix = translation_matrix @ rotation_matrix
    return model_view_matrix

def sample_visible_points_from_single_view(mesh, num_samples, fov_degrees=75, start_image_size=512, radius=2.0, max_image_size=2048, max_viewpoint_retries=10):
    """
    Sample at least `num_samples` visible points from a single random view.
    If not enough points are captured, try new views up to `max_viewpoint_retries`.
    """
    # Center and scale the mesh
    mesh_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-mesh_center)
    bbox_size = mesh.bounds[1] - mesh.bounds[0]
    scale = 2.0 / np.max(bbox_size)
    mesh.apply_scale(scale)
    for attempt in range(max_viewpoint_retries):
        # Generate a random viewpoint
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        viewpoint = np.array([x, y, z])
        
        # Setup scene
        scene = pyrender.Scene()
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh_pyrender)
        camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_degrees))
        camera_pose = look_at(eye=viewpoint, center=np.array([0, 0, 0]), up=np.array([0, 0, 1]))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)  # Add light from same direction as the camera
        scene.add(camera, pose=camera_pose)
        image_size = start_image_size
        while image_size <= max_image_size:
            # Render images in the scene
            r = pyrender.OffscreenRenderer(image_size, image_size)
            color, depth = r.render(scene)
            color_img = Image.fromarray((color * 255).astype(np.uint8))
            color_img.save(f"render_color_{attempt}.png")
            print(f"Rendered color image saved as render_color_{attempt}.png")
            r.delete()
            
            # Convert depth to 3D points
            fx = fy = image_size / (2 * np.tan(np.radians(fov_degrees / 2)))
            cx = cy = image_size / 2
            i, j = np.meshgrid(np.arange(image_size), np.arange(image_size))
            i = i.flatten()
            j = j.flatten()
            z = depth.flatten()
            valid = z > 0
            x = (i[valid] - cx) * z[valid] / fx
            y = (j[valid] - cy) * z[valid] / fy
            visible_points = np.vstack((x, y, z[valid])).T
            visible_points_hom = np.hstack((visible_points, np.ones((visible_points.shape[0], 1))))
            visible_points_world = (np.linalg.inv(camera_pose) @ visible_points_hom.T).T[:, :3]
            visible_points_world += mesh_center
            
            # Check if we have enough points
            if visible_points_world.shape[0] >= num_samples:
                # Downsample if needed
                if visible_points_world.shape[0] > num_samples:
                    indices = np.random.choice(visible_points_world.shape[0], num_samples, replace=False)
                    visible_points_world = visible_points_world[indices]
                return visible_points_world
            image_size *= 2  # Increase resolution
        print(f"Attempt {attempt+1}: Max resolution reached, trying new viewpoint...")
    raise RuntimeError(f"Failed to sample {num_samples} points from any viewpoint after {max_viewpoint_retries} tries.")

def generate_labels_for_sampled_points(sampled_points, seg_points, seg_labels):
    """
    Generate labels for sampled points based on the nearest neighbor in the labeled segmentation points.
    Args:
        sampled_points (ndarray): Sampled points from the mesh (shape: [N, 3]).
        seg_points (ndarray): Points with known labels (shape: [M, 3]).
        seg_labels (ndarray): Labels for `seg_points` (shape: [M]).
    Returns:
        sampled_labels (ndarray): Labels for the sampled points (shape: [N]).
    """
    # Build KDTree from segmentation points
    kdtree = cKDTree(seg_points)
    # Find the nearest neighbor for each sampled point
    distances, indices = kdtree.query(sampled_points, k=5)
    # Assign labels based on the nearest neighbor
    sampled_labels = seg_labels[indices]
    return sampled_labels


def visualise_file(i: str, data_root_dir: Path, mode: str, different_n: bool = False, nin: Optional[int] = None):
    """
    Visualise original vs. resampled/partial point clouds for a given instance path like
    '02691156/<instance_id>/' under data_root_dir.
    If nin is provided, use that single n; otherwise try n in range(5).
    """
    for n in range(5):
        if nin is not None:
            n = nin
        # file_path = data_root_dir / Path(f"{i}/models/new_samples_{n}.npy")
        file_path = data_root_dir / Path(f"{i}/models/surface_samples.npy")
        # file_path = data_root_dir / Path(f"{i}/models/partial_samples_{mode}_{n}.npy")

        if not file_path.is_file():
            print(f"File {file_path} does not exist, skipping...")
            if nin is not None:
                return
            continue

        if different_n:
            pn = np.random.randint(1, 5)
            while pn == n:
                pn = np.random.randint(1, 5)
        else:
            pn = n
        
        if mode == "full":
            resampled_path = data_root_dir / Path(f"{i}/models/new_samples_{pn}.npy")
        elif mode == "surface":
            resampled_path = data_root_dir / Path(f"{i}/models/surface_samples.npy")
        else:
            resampled_path = data_root_dir / Path(f"{i}/models/partial_samples_{mode}_{pn}.npy")
        
        if not resampled_path.is_file():
            print(f"Resampled file {resampled_path} does not exist, skipping...")
            if nin is not None:
                return
            continue
        
        pc = np.load(file_path)
        resampled_pc = np.load(resampled_path)

        print(f"Visualising:")
        print(f"Original point cloud: {file_path}")
        print(f"Resampled point cloud: {resampled_path}")
        
        # Visualise with open3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points)
        # o3d.visualization.draw_geometries([pcd])
        
        # input()
        # Create Open3D point cloud
        
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(pc)
        orig_pcd.paint_uniform_color([0, 0, 0])
        # orig_pcd.paint_uniform_color([1, 0, 1])

        resampled_pcd = o3d.geometry.PointCloud()
        resampled_pcd.points = o3d.utility.Vector3dVector(resampled_pc)
        resampled_pcd.paint_uniform_color([0, 1, 0])

        # try to align the two point clouds with ICP
        # for reconstructions the two point clouds should be in the same frame, default to false
        # threshold = 0.2  # Distance threshold for ICP
        # np.eye(4)  # Initial transformation (identity matrix)

        # theta = 0  # 90 degrees in radians
        # initial_rotation_y = np.array(
        #     [
        #         [np.cos(theta), 0, np.sin(theta), 0],
        #         [0, 1, 0, 0],
        #         [-np.sin(theta), 0, np.cos(theta), 0],
        #         [0, 0, 0, 1],
        #     ]
        # )
        # reg_icp = o3d.pipelines.registration.registration_icp(
        #     resampled_pcd,
        #     orig_pcd,
        #     threshold,
        #     initial_rotation_y,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # )

        # resampled_pcd.transform(reg_icp.transformation)


        # Visualize
        o3d.visualization.draw_geometries(
            [resampled_pcd, orig_pcd],
            window_name="Point Cloud with Labels and Keypoints",
        )
        
        if nin is not None:
            return

def _parse_classes_arg(classes_arg: str) -> List[str]:
    """Split classes by comma/whitespace; synset_utils handles normalization/aliases."""
    if not classes_arg:
        return []
    return [p for chunk in classes_arg.split(",") for p in chunk.split() if p]

def _filter_instances_by_synsets(list_file: Path, allowed_synsets: Optional[List[str]]) -> List[str]:
    """Read lines like '02691156/<instance>/' and filter by synset ID prefix if provided."""
    out = []
    if not list_file.is_file():
        print(f"[WARN] list.txt not found at {list_file}. No instances to visualise.")
        return out
    with open(list_file, "r") as f:
        for l in f:
            entry = l.strip()
            if not entry:
                continue
            synset_id = entry.split("/")[1]
            if allowed_synsets is None or synset_id in allowed_synsets:
                out.append(entry)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise ShapeNet PCs with class/mode/n filters.")
    parser.add_argument("--root", type=str, default="/mnt/slow/shapenetcorev2-source",
                        help="Dataset root containing list.txt and <synset>/<instance>/models.")
    parser.add_argument("--taxonomy", type=str, default="filtered_taxonomy.json",
                        help="Taxonomy JSON path (absolute or relative to --root).")
    parser.add_argument("--classes", type=str, default="",
                        help="Comma/space-separated class names to include (use taxonomy). Blank=ALL.")
    parser.add_argument("--mode", type=str, default="default",
                        help="Mode to visualise (e.g., default, tac, surface, full).")
    parser.add_argument("--n", type=int, default=None,
                        help="Specific stored index n to show for partial/new_samples (e.g., 0..4). If omitted, iterate n in range(5).")
    parser.add_argument("--different_n", action="store_true",
                        help="Visualise resampled PCs at a different random n than the original.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional limit on number of instances to visualise (0=all).")
    args = parser.parse_args()

    data_root_dir = Path(args.root).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.is_absolute():
        taxonomy_path = (data_root_dir / taxonomy_path).resolve()

    classes_list = _parse_classes_arg(args.classes)
    if classes_list:
        try:
            allowed_synsets = names_to_synsets(classes_list, str(taxonomy_path))
        except Exception as e:
            print(f"[ERROR] Failed to resolve classes via taxonomy: {e}")
            exit(1)
    else:
        allowed_synsets = None  # ALL classes

    list_file = data_root_dir / "list.txt"
    print(f"[INFO] Reading instances from {list_file}, for classes: {allowed_synsets if allowed_synsets else 'ALL'}")
    folders_to_run = _filter_instances_by_synsets(list_file, allowed_synsets)
    if not folders_to_run:
        print("[INFO] No matching instances found to visualise.")
        exit(0)

    shuffle(folders_to_run)
    if args.limit and args.limit > 0:
        folders_to_run = folders_to_run[: args.limit]

    if allowed_synsets is None:
        print(f"[INFO] Classes: ALL")
    else:
        try:
            names = synsets_to_names(allowed_synsets, str(taxonomy_path))
        except Exception:
            names = allowed_synsets
        print(f"[INFO] Classes (synsets): {allowed_synsets}")
        print(f"[INFO] Classes (names): {names}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] n: {args.n if args.n is not None else 'iterate 0..4'}")
    print(f"[INFO] Total instances to visualise: {len(folders_to_run)}")

    for idx, i in enumerate(folders_to_run):
        print(f"[{idx+1}/{len(folders_to_run)}] {i}")
        visualise_file(i, data_root_dir, args.mode, args.different_n, nin=args.n)