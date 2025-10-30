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
import pandas as pd
import random

# Use shared taxonomy/synset utilities
from synset_utils import (
    load_taxonomy_maps,
    names_to_synsets,
    synsets_to_names,
)

TRAINABLE = [
    "airplane",
    "bed",
    "bottle",
    "cap",
    "car",
    "chair",
    "guitar",
    "helmet",
    "knife",
    "motorbike",
    "mug",
    "table",
    "vessel",
]

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
        if not Path(i).is_absolute():
            # file_path = data_root_dir / Path(f"{i}/models/new_samples_{n}.npy")
            file_path = data_root_dir / Path(f"{i}/models/surface_samples.npy")
            # file_path = data_root_dir / Path(f"{i}/models/partial_samples_{mode}_{n}.npy")
        else:
            file_path = Path(i) / "models/surface_samples.npy"

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

def plot_collage(ii, modes_list, max_n=2, grid=(6, 6), 
                 out_path=Path("sampling_collage.png"),
                 surface_point_size=1, partial_point_size=1, 
                 surface_colour="gray", partial_colour="red"):
    # grid size should be one class instance per row, and different samples on the columns
    fig = plt.figure(figsize=(grid[1] * 5, grid[0] * 5), dpi=500)
    # fig = plt.figure(constrained_layout=True, figsize=(grid[1] * 2, grid[0] * 2), dpi=500)
    # fig = plt.figure(constrained_layout=True, figsize=(grid[1] * 5, grid[0] * 5), dpi=500)
    
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    
    
    # looping through rows of class instances
    print(ii)
    for row, i in enumerate(ii):
        # init the dirs to plot
        surface_pts = np.load(i / Path("models/surface_samples.npy"))
        # downsample for visibility 
        nds = 2048
        surface_pts = surface_pts[np.random.choice(len(surface_pts), nds, replace=False)]
        # get all the different modes up to the max n
        dirs = []
        covs = []
        for m in modes_list:
            for nn in range(max_n):
                # get point cloud dir
                sample_path = i / Path(f"models/partial_samples_{m}_{nn}.npy")
                dirs.append(sample_path)
                # get the coverage value from csv
                p = pd.read_csv(i / Path(f"models/coverage_{m}.csv"), usecols=['coverage'])
                covs.append(p['coverage'][nn])
        # shuffle the lists together
        indices = np.arange(len(dirs))
        random.shuffle(indices)
        dirs_shuffled = [dirs[i] for i in indices]
        covs_shuffled = [covs[i] for i in indices]
        dirs_shuffled = dirs_shuffled[:grid[1]]
        covs_shuffled = covs_shuffled[:grid[1]]
        
        # looping through different partial samples
        for col, dir in enumerate(dirs_shuffled):
            # load point cloud from the directory
            partial_pts = np.load(dir)
            ndsp = 1024
            partial_pts = partial_pts[np.random.choice(len(partial_pts), ndsp, replace=False)]
            
            # build title with extracted class information and coverage metric
            [class_id] = re.findall(r"/(\d{8})(?=/|$)", str(i))
            [class_name] = synsets_to_names([class_id], taxonomy_path="/mnt/slow/shapenetcorev2-source/filtered_taxonomy.json")
            # title = f"Class: {class_name}\nCoverage: {covs_shuffled[col]}"

            # fill the plot
            ax = fig.add_subplot(grid[0], grid[1], row*grid[1]+col+1, projection="3d")
            ax.scatter(
                surface_pts[:, 0], 
                surface_pts[:, 2], 
                surface_pts[:, 1], 
                s=surface_point_size, 
                c=surface_colour, 
                alpha=0.5
            )
            ax.scatter(
                partial_pts[:, 0],
                partial_pts[:, 2],
                partial_pts[:, 1],
                s=partial_point_size,
                c=partial_colour,
                alpha=0.8,
                label=f"coverage: {covs_shuffled[col]:.2f}"
            )
            # ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # ax.view_init(elev=20, azim=30)
            ax.view_init(elev=30, azim=45)
            leg = ax.legend(
                    loc="upper left", 
                    # bbox_to_anchor=(0.02, 0.75),
                    bbox_to_anchor=(0.12, 0.75),
                    # loc="upper center",
                    # bbox_to_anchor=(0, 0),
                    # bbox_transform=ax.transAxes, 
                    frameon=False, fontsize=12)
            for h in leg.legendHandles:
                h.set_visible(False)    # hide marker artists
            
            # size shenanigans
            all_pts = np.vstack([surface_pts])
            max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2.0
            mid = all_pts.mean(0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            
            # pad_frac = 0.01  # 1% padding
            # mins = surface_pts.min(0)
            # maxs = surface_pts.max(0)
            # rng  = maxs - mins
            # pad  = pad_frac * rng.max()
            # ax.set_xlim(mins[0]-pad, maxs[0]+pad)
            # ax.set_ylim(mins[1]-pad, maxs[1]+pad)
            # ax.set_zlim(mins[2]-pad, maxs[2]+pad)
            # # Equal aspect in 3D to avoid extra margins:
            # ax.set_box_aspect(rng)  # (dx, dy, dz)    
            
            
            # shift the axis
            pos = ax.get_position()  # Bbox: [x0, y0, width, height] in figure coords
            # print(f"col {col} moving by dx=-0.02*{(col % grid[1])}")
            shift_frac = 0.065
            dx = -shift_frac * (col % grid[1])   # move later columns left
            ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])
            # reset drawing order
            ax.set_zorder(grid[1]-col)

            # remove the axis background
            ax.axis('off')

    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    # fig.subplots_adjust(wspace=0.01)   # smaller = tighter horizontal gaps
    # fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.02, hspace=0.0, wspace=0.0)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(
        f"[✓] Saved collage with {len(dirs)} samples to {out_path}"
    )
    
    return

def _parse_classes_arg(classes_arg: str) -> List[str]:
    """Split classes by comma/whitespace; synset_utils handles normalization/aliases."""
    if not classes_arg:
        return []
    return [p for chunk in classes_arg.split(",") for p in chunk.split() if p]

def _filter_instances_by_synsets(data_root_dir: Path, list_file: Path, allowed_synsets: Optional[List[str]]) -> List[str]:
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
                idir = (data_root_dir / entry).resolve()
                out.append(idir)
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
                        help="Mode to visualise (e.g., default, tac, surface, full). Ignored for plot-collage")
    parser.add_argument("--n", type=int, default=None,
                        help="Specific stored index n to show for partial/new_samples (e.g., 0..4). If omitted, iterate n in range(5).")
    parser.add_argument("--different_n", action="store_true",
                        help="Visualise resampled PCs at a different random n than the original.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional limit on number of instances to visualise (0=all).")
    
    parser.add_argument("--plot-collage", action="store_true",
                        help="Plot a collage of points from the same perspective instead of interactive visualisation")
    parser.add_argument("--rows", type=int, default=None, 
                        help="Used with plot-collage, the number of different class instances to show in the collage")
    parser.add_argument("--cols", type=int, default=None, 
                        help="Used with plot-collage, the number of different samples to show in the collage per class instance")
    args = parser.parse_args()

    data_root_dir = Path(args.root).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.is_absolute():
        taxonomy_path = (data_root_dir / taxonomy_path).resolve()

    if args.classes == "all":
        classes_list = TRAINABLE
        try:
            allowed_synsets = names_to_synsets(classes_list, str(taxonomy_path))
        except Exception as e:
            print(f"[ERROR] Failed to resolve classes via taxonomy: {e}")
            exit(1)
    elif args.classes:
        classes_list = _parse_classes_arg(args.classes)
        try:
            allowed_synsets = names_to_synsets(classes_list, str(taxonomy_path))
        except Exception as e:
            print(f"[ERROR] Failed to resolve classes via taxonomy: {e}")
            exit(1)
    else:
        allowed_synsets = None  # ALL classes

    list_file = data_root_dir / "list.txt"
    print(f"[INFO] Reading instances from {list_file}, for classes: {classes_list if classes_list else 'ALL'}")
    folders_to_run = _filter_instances_by_synsets(data_root_dir, list_file, allowed_synsets)
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
        
    if not args.plot_collage: 
        print(f"[INFO] Mode: {args.mode}")
        print(f"[INFO] n: {args.n if args.n is not None else 'iterate 0..4'}")
        print(f"[INFO] Total instances to visualise: {len(folders_to_run)}")

        for idx, i in enumerate(folders_to_run):
            print(f"[{idx+1}/{len(folders_to_run)}] {i}")
            visualise_file(i, data_root_dir, args.mode, args.different_n, nin=args.n)
    else: 
        with open(data_root_dir / "resample.yaml", 'r') as stream:
            modes = yaml.safe_load(stream)
        mode_list = list(modes.keys())
        mode_list = ["default", "myopia", "patch"]
        
        r = args.rows
        for idx in range(len(folders_to_run)):
            if r == 1:
                ii = [folders_to_run[idx]]
            else:
                ii = folders_to_run[idx*r:idx*r+r-1]
            plot_collage(ii, modes_list=mode_list, max_n=2, grid=(args.rows, args.cols))
            input("Check map output")