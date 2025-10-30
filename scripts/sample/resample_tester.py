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

from compute_coverage import *

def create_axis_pointcloud(pose=np.eye(4), length=0.1, step=0.01):
    """
    Create an axis-aligned RGB pointcloud centered at the origin, transformed by a pose.
    - pose: 4x4 numpy array (camera or object pose)
    - length: length of each axis (in meters)
    - step: resolution (distance between points)
    """
    pts = []
    colors = []

    # X-axis (red)
    x = np.linspace(0, length, int(length / step))
    for i in x:
        pts.append([i, 0, 0])
        colors.append([1, 0, 0])

    # Y-axis (green)
    y = np.linspace(0, length, int(length / step))
    for i in y:
        pts.append([0, i, 0])
        colors.append([0, 1, 0])

    # Z-axis (blue)
    z = np.linspace(0, length, int(length / step))
    for i in z:
        pts.append([0, 0, i])
        colors.append([0, 0, 1])

    # Convert to Nx3 arrays
    pts = np.array(pts)
    colors = np.array(colors)

    # Apply pose transformation
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 4)
    transformed = (pose @ pts_h.T).T[:, :3]  # drop homogeneous coordinate

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# def look_at(eye, center, up=np.array([0, 0, 1])):
#     """
#     Create a 4x4 view matrix looking from eye to center.
#     """
#     forward = center - eye
#     forward /= np.linalg.norm(forward)
#     right = np.cross(up, forward)
#     right /= np.linalg.norm(right)
#     up = np.cross(forward, right)
#     # special opengl camera coordinates convention: eye is looking down the negative z-axis
#     rotation_matrix = np.array([
#         [right[0], up[0], forward[0], 0],
#         [right[1], up[1], forward[1], 0],
#         [right[2], up[2], forward[2], 0],
#         [0, 0, 0, 1]
#     ])
#     # 5. Construct the translation part of the matrix
#     translation_matrix = np.array([
#         [1, 0, 0, -eye[0]],
#         [0, 1, 0, -eye[1]],
#         [0, 0, 1, -eye[2]],
#         [0, 0, 0, 1]
#     ])
#     # 6. Combine rotation and translation
#     # Note: In OpenGL-style, translation is applied after rotation
#     model_view_matrix = translation_matrix @ rotation_matrix
#     return model_view_matrix

def look_at(eye, center, up=np.array([0, 1, 0])):
    """
    Create a 4x4 view matrix looking from eye to center.
    """
    backward = eye - center  # center to eye
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    # print(f"up: {up}")
    # print(f"forward: {backward}")
    # special pyrender/opengl camera coordinates convention: eye is looking down the negative z-axis
    rotation_matrix = np.array([
        [right[0], up[0], backward[0], 0],
        [right[1], up[1], backward[1], 0],
        [right[2], up[2], backward[2], 0],
        [0, 0, 0, 1]
    ])
    # 5. Construct the translation part of the matrix
    translation_matrix = np.array([
        [1, 0, 0, eye[0]],
        [0, 1, 0, eye[1]],
        [0, 0, 1, eye[2]],
        [0, 0, 0, 1]
    ])
    
    vis_translation_matrix = np.array([
        [1, 0, 0, eye[0]],
        [0, 1, 0, eye[1]],
        [0, 0, 1, eye[2]],
        [0, 0, 0, 1]
    ])
    vis_translation_matrix[:-1, -1] -= .1*backward
    
    # 6. Combine rotation and translation
    # Note: In OpenGL-style, translation is applied after rotation
    # this way the translation is applied in the world frame
    model_view_matrix = translation_matrix @ rotation_matrix
    vis_model_view_matrix = (vis_translation_matrix) @ rotation_matrix
    return model_view_matrix, vis_model_view_matrix


def sample_visible_points_from_single_view(mesh, num_samples, fov_degrees=75, start_image_size=512, radius=2.0, max_image_size=2048, max_viewpoint_retries=10, remove_far_points=False):
    """
    Sample at least `num_samples` visible points from a single random view.
    If not enough points are captured, try new views up to `max_viewpoint_retries`.
    """
    # Center and scale the mesh
    # mesh_center = mesh.bounds.mean(axis=0)
    # mesh.apply_translation(-mesh_center)
    # bbox_size = mesh.bounds[1] - mesh.bounds[0]
    # scale = 2.0 / np.max(bbox_size)
    # mesh.apply_scale(scale)
    for attempt in range(max_viewpoint_retries):
        # Generate a random viewpoint
        # sample from half spherical surface (check coordinate system for which half, probably upper)
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        print(f"radius: {radius}")
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        viewpoint = np.array([x, y, z])
        
        # mesh is a trimesh object
        # for a full point cloud can sample directly from trimesh
        
        # Setup scene
        scene = pyrender.Scene()
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh_pyrender)
        camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_degrees))
        camera_pose, axis_pose = look_at(eye=viewpoint, center=np.array([0, 0, 0]), up=np.array([0, 1, 0]))

        # compute pose with fixed viewpoint
        # vp = np.array([2., 0, 2.])
        # camera_pose, axis_pose = look_at(eye=vp, center=np.array([0, 0, 0]), up=np.array([0, 1, 0]))
        
        # print("camera pose inside sample function:")
        # print(camera_pose)
        
        # # use a fixed camera pose for test
        # camera_pose = np.array([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0.5],
        #     [0, 0, 1, radius],
        #     [0, 0, 0, 1]
        # ])
        # axis_pose = np.array([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0.5],
        #     [0, 0, 1, radius-1],
        #     [0, 0, 0, 1]
        # ])
        
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)  # Add light from same direction as the camera
        scene.add(camera, pose=camera_pose)
        
        # rotation from pyrender/opengl to pinhole camera conventions
        R = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ])
        
        # camera_pose = camera_pose @ R @ np.linalg.inv(R)
        # camera_pose = camera_pose @ R  # rotate camera pose in its own frame to match pinhole convention
        
        
        # visualise the scene
        # axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(axis_length=2, origin_size=0.001), smooth=False)
        # scene.add(axis_mesh, pose=axis_pose)
        # oaxis = trimesh.creation.axis(axis_length=2)
        # oaxis.visual.face_colors = np.array([255, 255, 0, 255])  # yellow
        # origin_mesh = pyrender.Mesh.from_trimesh(oaxis, smooth=False)
        # scene.add(origin_mesh, pose=np.eye(4))  # Add origin axis for reference
        # print("\n\n\ncamera_pose:")
        # print(camera_pose)
        # print("\n\n\naxis_pose:")
        # print(axis_pose)
        
        # pyrender.Viewer(scene)
        # input()
        
        image_size = start_image_size
        while image_size <= max_image_size:
            # Render images in the scene
            r = pyrender.OffscreenRenderer(image_size, image_size)
            color, depth = r.render(scene)
            color_img = Image.fromarray((color * 255).astype(np.uint8))
            color_img.save(f"render_color_{attempt}.png")
            r.delete()
            
            # Convert depth to 3D points
            # import pdb; pdb.set_trace()
            fx = fy = image_size / (2 * np.tan(np.radians(fov_degrees / 2)))
            cx = cy = image_size / 2
            i, j = np.meshgrid(np.arange(image_size), np.arange(image_size))
            # print(f"i shape: {i.shape}, j shape: {j.shape}, depth shape: {depth.shape}")
            i = i.flatten()
            j = j.flatten()
            z = depth.flatten()
                            
            # for tac mode, remove points that are much further away than the closest point
            if remove_far_points:
                print(f"Removing far points with tac mode, threshold: {image_size / 2}")
                min_dist = np.min(z[z > 0])  # Minimum depth value greater than zero
                print(f"Minimum depth value: {min_dist}")
                diff_threshold = 0.05
                far = z > (min_dist + diff_threshold)
                z[far] = 0.0  # Set far points to zero (not considered by depth image)
                
            valid = z > 0
            x = (i[valid] - cx) * z[valid] / fx
            y = (j[valid] - cy) * z[valid] / fy
            # print(f"x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")
            visible_points = np.vstack((x, y, z[valid])).T
            # print(f"visible points: {visible_points.shape}")
            visible_points_hom = np.hstack((visible_points, np.ones((visible_points.shape[0], 1))))
            

            # Extract 3D coordinates and homogeneous scale
            # x, y, z, w = visible_points_hom.T  # Transpose to separate columns
            # x /= w  # Normalize homogeneous coordinates
            # y /= w
            # z /= w

            # # Project back to pixel coordinates
            # i = (x * fx / z) + cx
            # j = (y * fy / z) + cy

            # # Create an empty depth image
            # depth_image = np.zeros((image_size, image_size), dtype=np.float32)

            # # Filter valid points (inside image bounds)
            # valid = (i >= 0) & (i < image_size) & (j >= 0) & (j < image_size) & (z > 0)
            # i_valid = i[valid].astype(int)
            # j_valid = j[valid].astype(int)
            # z_valid = z[valid]

            # # Populate the depth image with valid depth values
            # depth_image[j_valid, i_valid] = z_valid

            # # Visualize the reverted depth image
            # plt.imshow(depth_image, cmap='gray', vmin=0, vmax=np.max(depth))
            # plt.title("Reverted Depth Image")
            # plt.colorbar()
            # plt.show()





            # camera_pose[:, 2] *= -1
            
            ###
            camera_R = np.eye(4)
            camera_R[:3, :3] = camera_pose[:3, :3]
            camera_t = camera_pose[:3, 3]
            camera_R_hom = np.eye(4)
            camera_R_hom[:3, :3] = camera_R[:3, :3]
            camera_t_hom = camera_pose[:, 3].reshape(4, 1)
            cam_inv = np.linalg.inv(camera_pose)
            cam_inv_R_hom = np.eye(4)
            cam_inv_R_hom[:3, :3] = cam_inv[:3, :3]
            cam_inv_t_hom = cam_inv[:, 3].reshape(4, 1)
            
            # visible_points_world_split = np.linalg.inv(camera_R) @ (visible_points_hom[:, :3] - camera_t).T# @ camera_R.T  # Transform points to world coordinates
            # visible_points_world_split = visible_points_world_split.T  # Transpose back to (N, 3)
            # visible_points_world_split_2 = ((visible_points_hom[:, :3] - camera_t) @ camera_R.T)  # Transform points to world coordinates
            # visible_points_world = np.linalg.inv(camera_R) @ visible_points_hom[:, :3].T - camera_t.reshape(3, 1)  # Transform points to world coordinates
            # visible_points_world = visible_points_world.T  # Transpose back to (N, 3)
            # print(f"split vs split2: {np.allclose(visible_points_world_split, visible_points_world_split_2)}")
            # print(f"whole vs split: {np.allclose(visible_points_world_split, visible_points_world)}")
            # print(f"whole vs split2: {np.allclose(visible_points_world_split_2, visible_points_world)}")###
            
            
            # visible_points_world = (np.linalg.inv(camera_pose) @ visible_points_hom.T).T[:, :3]
            # visible_points_world = (np.linalg.inv(camera_R) @ (np.linalg.inv(R) @ visible_points_hom.T)).T[:, :3]
            # visible_points_world = (np.linalg.inv(camera_R) @ (visible_points_hom @ np.linalg.inv(R)).T).T[:, :3]
            # visible_points_world = (np.linalg.inv(camera_R) @ (visible_points_hom @ np.eye(4)).T).T[:, :3]
            # visible_points_world += camera_t  # Add translation
            
            # print(f"camera_pose:\n{camera_pose}")
            # print(f"camera_pose inverse:\n{np.linalg.inv(camera_pose)}")
            # print(f"camera_R:\n{camera_R}")
            # print(f"camera_t:\n{camera_t}")
            # print(f"camera_R inverse:\n{np.linalg.inv(camera_R)}")
            # print(f"cam_inv_t_hom:\n{cam_inv_t_hom}")
            # print(f"R @ inv_pose:\n{R @ np.linalg.inv(camera_pose)}")
            
            # visible_points_world = visible_points_hom[:, :3]
            # visible_points_world = (np.linalg.inv(camera_pose) @ visible_points_hom.T).T[:, :3]
            # visible_points_world = (np.linalg.inv(camera_R) @ (visible_points_hom @ np.linalg.inv(R)).T).T[:, :3]
            # visible_points_world = (((visible_points_hom @ np.linalg.inv(R)).T) + camera_t_hom).T @ np.linalg.inv(camera_R)
            # visible_points_world = visible_points_world[:, :3]
            # visible_points_world += mesh_center
            
            # visible_points_world = ((visible_points_hom.T + cam_inv_t_hom).T @ np.linalg.inv(camera_R)) @ np.linalg.inv(R)
            # visible_points_world = visible_points_world[:, :3]
            
            # points obtained from pinhole camera conventions, need to invert
            pinhole_T = R @ np.linalg.inv(camera_pose)
            pinhole_R_hom = np.eye(4)
            pinhole_R_hom[:3, :3] = pinhole_T[:3, :3]
            pinhole_t_hom = pinhole_T[:, 3].reshape(4, 1)
            
            # visible_points_world = (visible_points_hom.T + pinhole_t_hom).T @ pinhole_R_hom
            visible_points_world = camera_R_hom @ ((visible_points_hom @ R).T + pinhole_t_hom)
            visible_points_world = visible_points_world.T[:, :3]  # Drop homogeneous coordinate
            
            # visible_points_world = (((visible_points_hom @ np.linalg.inv(R)).T + cam_inv_t_hom).T)# @ np.linalg.inv(camera_R))
            # visible_points_world = visible_points_world[:, :3]
            
            # visible_points_world = np.linalg.inv(R) @ ((visible_points_hom.T + camera_t_hom).T @ np.linalg.inv(camera_R)).T
            # visible_points_world = visible_points_world.T[:, :3]
            
            
            
            # Check if we have enough points
            if visible_points_world.shape[0] >= num_samples:
                # Downsample if needed
                if visible_points_world.shape[0] > num_samples:
                    indices = np.random.choice(visible_points_world.shape[0], num_samples, replace=False)
                    visible_points_world = visible_points_world[indices]
                return visible_points_world, camera_pose
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
def process_file(i, data_root_dir, save_root_dir, mode="default", visualize=False):
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
    
    # load the surface points
    try:
        # load the surface point clouds
        surface_points = np.load(data_root_dir / i / "models" / "surface_samples.npy")
    except Exception:
        print(i)
        
    coverage_list = []
        
    # Sample 5 total meshes, each with one full point cloud and one partial point cloud
    for n in range(5):
        save_path = save_root_dir / i / "models"
        
        # full_pc_path = save_path / f"full_samples_{n}.npy"
        partial_pc_path = save_path / f"partial_samples_{mode}_{n}.npy"

        # Load partial point cloud sampling options from yaml file
        with open(data_root_dir / "resample.yaml", 'r') as stream:
            args = yaml.safe_load(stream)[mode]
            
        # Sample the mesh for partial point clouds
        sampled_points, camera_pose = sample_visible_points_from_single_view(simplified_trimesh,
                                                                num_samples=args['num_samples'],
                                                                fov_degrees=args['fov_degrees'],
                                                                start_image_size=args['start_image_size'],
                                                                radius=args['radius'],
                                                                max_image_size=args['max_image_size'],
                                                                max_viewpoint_retries=args['max_viewpoint_retries'],
                                                                remove_far_points=args['remove_far_points'])
        
        # Sample the mesh for full point clouds, from the same frame
        
        # calculate the coverage of the partial point cloud
        coverage, radius, covered_points, not_covered_points = surface_coverage_pointwise(surface_points, sampled_points)
        coverage_list.append(coverage)
        
        # Visualise with open3d
        # visualise the resampled points and the original full point cloud
        og_full_pc_path = partial_pc_path.parent / f"new_samples_{n}.npy"
        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = o3d.utility.Vector3dVector(np.load(og_full_pc_path))
        full_pcd.paint_uniform_color([0, 1, 0])  # Green for full point cloud
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_points)
        pcd.paint_uniform_color([1, 0, 0])  # Red for sampled point cloud
        
        axis_pcd = create_axis_pointcloud(pose=camera_pose, length=1, step=0.01)
        origin_pcd = create_axis_pointcloud(pose=np.eye(4), length=.2, step=0.01)
        
        ############## figure out what the actual transform we are seeing between camera poses is
        ##### everything is in the open3d coordinate frame
        # print(f"pyrender camera location is max z of the sampled pcd: {np.max(np.asarray(pcd.points)[:, 2])}")
        # print(f"open3d camera location is min z of the axis_pcd: {np.min(np.asarray(axis_pcd.points)[:, 2])}")
        # print("camera pose when visualising:")
        # print(camera_pose)
        
        y_tip_y = np.max(np.asarray(pcd.points)[:, 1])
        x_tip_x = np.max(np.asarray(pcd.points)[:, 0])
        origin_y = np.min(np.asarray(pcd.points)[:, 1])
        # print(f"pyrender cloud max y (tip of y axis): {np.max(np.asarray(pcd.points)[:, 1])}")
        # print(f"pyrender cloud max x (tip of x axis): {np.max(np.asarray(pcd.points)[:, 0])}")
        # print(f"pyrender cloud min y (centre of axes): {np.min(np.asarray(pcd.points)[:, 1])}")
        
        # find the matching coodinates for the points
        # find the point that corresponds to the y coordinate of y_tip_y
        y_tip_point = np.asarray(pcd.points)[np.argmax(np.asarray(pcd.points)[:, 1] == y_tip_y)]
        x_tip_point = np.asarray(pcd.points)[np.argmax(np.asarray(pcd.points)[:, 0] == x_tip_x)]
        axis_origin = np.asarray(pcd.points)[np.argmax(np.asarray(pcd.points)[:, 1] == origin_y)]
        
        # print(f"y axis tip: {y_tip_point}")
        # print(f"x axis tip: {x_tip_point}")
        # print(f"axis_origin : {axis_origin}")

        # cx is unit vector from camera axis origin in x direction
        cx = x_tip_point - axis_origin
        cx = cx / np.linalg.norm(cx)
        cy = y_tip_point - axis_origin
        cy = cy / np.linalg.norm(cy)
        cz = np.cross(cx, cy)  # Cross product to get z-axis
        cz = cz / np.linalg.norm(cz)
        camera_axis = np.vstack((cx, cy, cz)).T
        
        # print(f"x axis: {cx}")
        # print(f"y axis: {cy}")
        # print(f"z axis: {cz}")
        # print(f"camera axis in the open3d frame:")
        # print(camera_axis)
        
        # get the full transform by getting the translation between the two as well
        # the actual origin of the camera axis is shifted 1 unit in the camera's z direction from axis_origin
        axis_t = axis_origin + cz
        full_cam_axis = np.eye(4)
        full_cam_axis[:3, :3] = camera_axis
        full_cam_axis[:3, 3] = axis_t
        # print(f"full camera axis pose in the open3d frame:")
        np.set_printoptions(precision=4, suppress=True)
        # print(full_cam_axis)
        
        # transform from open3d to pyrender camera poses
        o3Tpyr = np.linalg.inv(camera_pose) @ full_cam_axis
        # print(f"The transform: ")
        # print(o3Tpyr)
        
        ############################

        if visualize:
            o3d.visualization.draw_geometries([pcd, full_pcd, axis_pcd, origin_pcd])

        
        
        # num_samples = 1000  # Desired number of points to sample
        # sampled_points = np.empty((0, 3))  # Initialize empty array to store points
        # while len(sampled_points) < num_samples:
        #     # print(len(sampled_points))
        #     new_points = trimesh.sample.volume_mesh(simplified_trimesh, count=1000)  # Always sample 1000 points
        #     sampled_points = np.vstack((sampled_points, new_points))
        # # If we end up with more points than needed, trim the array
        # sampled_points = sampled_points[:num_samples]
        
        # print(f"Saving to: {save_path / f'partial_samples_{mode}_{n}.npy'}')")
        
        # save the points
        save_path.mkdir(parents=True, exist_ok=True)
        # np.save(save_path / f"partial_samples_{mode}_{n}.npy", sampled_points)
        
    return coverage_list
    
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
# folders = {"02691156", "03636649", "03467517", "02954340", "02958343"}    # airplane, lamp, guitar, cap, car
folders = {"02691156"}
# folders =   {"02691156", "02954340", "02958343", "03001627", "03467517", "03624134", "03642806", "03790512", "03797390", "04225987", "04379243", "03948459", "02773838", "04099429", "03261776", "03636649"}
# folders =   {"03636649", "03467517", "02954340", "02958343"}  # lamp, guitar, cap, car
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for the source mesh.")
    parser.add_argument("--dst_dir", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for saving the resampled point clouds.")
    parser.add_argument("--mode", type=str, default="default", help="View mode to resample the data. See resample.yaml for options.")
    parser.add_argument("--visualize", action='store_true', help="Visualize sampled point clouds")
    args = parser.parse_args()
    
    data_root_dir = Path(args.src_dir)
    save_root_dir = Path(args.dst_dir)
    
    folders_to_run = []
    for l in open(data_root_dir / "list.txt"):
        if l.strip().split("/")[1] in folders:
            folders_to_run.append(l.strip())
    print(len(folders_to_run))
    shuffle(folders_to_run)
    
    coverage_list_list = []
    for idx, i in tqdm(enumerate(folders_to_run), total=len(folders_to_run)):
        coverage_list = process_file(i, data_root_dir, save_root_dir, args.mode)
        coverage_list_list.append(coverage_list)
        if idx == 1000:
            break
        
    coverage_list_list = np.array([c for l in coverage_list_list for c in l])
    
    plt.figure(figsize=(6,4))
    plt.hist(coverage_list_list, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.title("Distribution of Coverage Values")
    plt.grid(alpha=0.3)
    plt.yscale("log")
    plt.savefig(f"coverage_distribution_{args.mode}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean coverage for profile {args.mode}: {np.mean(coverage_list_list):.4f} ± {np.std(coverage_list_list):.4f}")