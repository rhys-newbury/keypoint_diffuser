import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
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
import pyrender
import yaml
import argparse

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

def _get_seg_points_path(folder, name):
    return os.path.join("/mnt/slow/Shapenetcore_benchmark", folder, 'points', name + '.pts')

def _get_seg_labels_path(folder, name):
    return os.path.join("/mnt/slow/Shapenetcore_benchmark", folder, 'points_label', name + '.seg')

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

# from concurrent.futures import ThreadPoolExecutor
seg_labels_ = None
# count, total = 0,0
def process_file(i, data_root_dir, save_root_dir, mode="default"):
    # Original pre-processed model file
    file_path = data_root_dir / Path(f"{i}/models/model_normalized2.obj")
    if not file_path.is_file():
        return
    # if all((output_path.parent / f"new_samples_{n}.npy").is_file() for n in range(5)):
    #     return
    simple_path = str(file_path).replace('model_normalized2', "simplified_mesh")
    if Path(simple_path).is_file():
        # print(f"File already exists: {simple_path}")
        pass
    else:       
        input(f"File does not exist: {simple_path}. Press Enter to continue.")
        # Filter down the original model to a simpler version
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))
        # Apply mesh simplification while preserving the boundary
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=2000,  # Target number of faces
                        preservenormal=True,
                        preservetopology=True,  # Prevents the creation of holes
                        boundaryweight=1.0)  # High weight to preserve boundaries
        ms.save_current_mesh(simple_path)
    try:
        # Fill single traingle and single quad holes in the mesh. twice?
        simplified_trimesh = trimesh.load(simple_path)
        
        simplified_trimesh.fill_holes()
        simplified_trimesh.fill_holes()
        
    except Exception:
        print(i)
    sampled_points = None
    
    # Sample 5 total partial meshes
    for n in range(5):
        save_path = save_root_dir / i
        if (save_path / f"partial_samples_{mode}_{n}.npy").is_file():
            continue
        # print(file_path)
        # Load arguments form yaml file
        with open(data_root_dir / "resample.yaml", 'r') as stream:
            args = yaml.safe_load(stream)[mode]
            
        # Sample the mesh
        sampled_points = sample_visible_points_from_single_view(simplified_trimesh,
                                                                num_samples=args['num_samples'],
                                                                fov_degrees=args['fov_degrees'],
                                                                start_image_size=args['start_image_size'],
                                                                radius=args['radius'],
                                                                max_image_size=args['max_image_size'],
                                                                max_viewpoint_retries=args['max_viewpoint_retries'])
        
        # Visualise with open3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points)
        # o3d.visualization.draw_geometries([pcd])
        # input()
        
        
        # num_samples = 1000  # Desired number of points to sample
        # sampled_points = np.empty((0, 3))  # Initialize empty array to store points
        # while len(sampled_points) < num_samples:
        #     # print(len(sampled_points))
        #     new_points = trimesh.sample.volume_mesh(simplified_trimesh, count=1000)  # Always sample 1000 points
        #     sampled_points = np.vstack((sampled_points, new_points))
        # # If we end up with more points than needed, trim the array
        # sampled_points = sampled_points[:num_samples]
        
        # print(f"Saving to: {save_path / f'partial_samples_{mode}_{n}.npy'}')")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        if (save_path / f"partial_samples_{mode}_{n}.npy").is_file():
            continue
        
        np.save(save_path / f"partial_samples_{mode}_{n}.npy", sampled_points)
    
    # if sampled_points is None:
    #     sampled_points = np.load(output_path.parent / f"new_samples_{n}.npy")
    # name =  i.split("/")[2]
    # # print(name)
    # if not Path(_get_seg_points_path(folder,name)).is_file():
    #     # print(_get_seg_points_path(folder,name))
    #     return
    # if not Path(_get_seg_labels_path(folder,name)).is_file():
    #     # print(_get_seg_labels_path(folder,name))
    #     return
    # seg_points = np.loadtxt(_get_seg_points_path(folder,name)).astype(np.float32)
    # seg_labels = np.loadtxt(_get_seg_labels_path(folder,name)).astype(np.int32)
    # max_label = seg_labels.max() + 1  # Avoid division by 0
    # global seg_labels_
    # if seg_labels_ is None:
    #     seg_labels_= seg_labels
    # else:
    #     seg_labels_ = np.concatenate((seg_labels_, seg_labels))
    # # print(_get_seg_labels_path(folder,name), max_label)
    # print(np.unique(seg_labels_, return_counts=True))
    # import pdb; pdb.set_trace()
    # return
    # colors = plt.cm.get_cmap("tab10", max_label)(seg_labels / max_label)[:, :3]  # RGB from colormap
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(seg_points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # # Convert sampled points to Open3D point cloud
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    # threshold = 0.2  # Distance threshold for ICP
    # transformation_init = np.eye(4)  # Initial transformation (identity matrix)
    # theta = np.pi / 2  # 90 degrees in radians
    # initial_rotation_y = np.array([
    #     [np.cos(theta), 0, np.sin(theta), 0],
    #     [0, 1, 0, 0],
    #     [-np.sin(theta), 0, np.cos(theta), 0],
    #     [0, 0, 0, 1]
    # ])
    # # print("Running ICP...")
    # reg_icp = o3d.pipelines.registration.registration_icp(
    #     pcd, point_cloud, threshold, initial_rotation_y,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # pcd.transform(reg_icp.transformation)
    # sampled_labels = generate_labels_for_sampled_points(sampled_points, np.asarray(pcd.points), seg_labels)
    # # import pdb; pdb.set_trace()
    # colors = plt.cm.get_cmap("tab10", seg_labels.max() + 1)(sampled_labels / (seg_labels.max() + 1))[:, :3]
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # output = np.vstack([
    #     np.hstack([np.asarray(point_cloud.points), sampled_labels.reshape(-1, 1)]),
    #     np.hstack([np.asarray(pcd.points), seg_labels.reshape(-1, 1)])
    # ])
    # # Save as 'lamp_<idx>.npy'
    # print(output_path.parent / f"lamp_{name}.npy")
    # np.save(output_path.parent / f"lamp_{name}.npy", output)
    # import pdb; pdb.set_trace()
    # output = np.vstack([np.hstack([np.asarray(point_cloud.points), sampled_labels.reshape(5000,1)]), np.hstack([np.asarray(pcd.points), seg_labels.reshape(-1, 1)])])
    # np.save(output_path.parent / "point_resampled_labeled.npy", output)

# folders = {"02691156", "02773838", "02954340", "02958343", "03001627", "03261776", "03467517", "03624134", "03636649", "03642806", "03790512", "03797390", "03948459", "04099429", "04225987", "04379243"}
folders = {"02691156"}
# folders =   {"02691156", "02954340", "02958343", "03001627", "03467517", "03624134", "03642806", "03790512", "03797390", "04225987", "04379243", "03948459", "02773838", "04099429", "03261776", "03636649"}
if __name__ == "__main__":
    folders_to_run = []
    for l in open("/mnt/shape_data/list.txt"):
        if l.strip().split("/")[1] in folders:
            folders_to_run.append(l.strip())
    print(len(folders_to_run))
    shuffle(folders_to_run)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="/mnt/shape_data", help="Root directory for the source mesh.")
    parser.add_argument("--dst_dir", type=str, default="/mnt/smb/shape_data_eric", help="Root directory for saving the resampled point clouds.")
    parser.add_argument("--view_mode", type=str, default="default", help="View mode to resample the data. See resample.yaml for options.")
    args = parser.parse_args()
    
    data_root_dir = Path(args.src_dir)
    save_root_dir = Path(args.dst_dir)
    
    for idx, i in tqdm(enumerate(folders_to_run), total=len(folders_to_run)):
        process_file(i, data_root_dir, save_root_dir, args.view_mode)