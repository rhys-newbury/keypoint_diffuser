import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYGLET_HEADLESS"] = "True"      # make pyglet headless
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # or "egl" if you have EGL/NVIDIA
from pathlib import Path
from random import shuffle
from tqdm import tqdm
import open3d as o3d
import trimesh
import numpy as np
import pymeshlab
from PIL import Image
import pyrender
import yaml
import argparse
from synset_utils import load_taxonomy_maps, names_to_synsets, synsets_to_names
from coverage_utils import compute_coverage_for_profile, plot_coverage_histogram

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

# the added classes compared to the initially used subset
TRAINABLE_DIFF = [
    "bed",
    "bottle",
    "helmet",
    "vessel"
]

# map taxonomy names <-> KEYS names (when they differ)
KEYS_ALIASES = {
    "motorcycle": "motorbike",
    "loudspeaker": "speaker",
    "cell phone": "cellphone",
    "computer keyboard": "keyboard",
    "display": "monitor",
    "can": "tin_can",
    # also the other direction
    "motorbike": "motorcycle",
    "speaker": "loudspeaker",
    "cellphone": "cell phone",
    "keyboard": "computer keyboard",
    "monitor": "display",
    "tin_can": "can",
}

def create_axis_pointcloud(pose=np.eye(4), length=0.1, step=0.01):
    """
    Create an Open3D axis-aligned RGB pointcloud centered at the origin, transformed by a pose.
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

def look_at(eye, center, up=np.array([0, 1, 0])):
    """
    Create a 4x4 view matrix looking from eye to center.
    Using the OpenGL convention where the camera looks down the negative z-axis.
    """
    backward = eye - center  # center to eye
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    
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
    
    # axis that is slightly offset from the camera, for visualising camera position as points
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

    min_radius = 0.1
    while radius >= min_radius:
        for attempt in range(max_viewpoint_retries):
            # Generate a random viewpoint
            # sample from half spherical surface
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
            camera_pose, axis_pose = look_at(eye=viewpoint, center=np.array([0, 0, 0]), up=np.array([0, 1, 0]))
            
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)  # Add light from same direction as the camera
            scene.add(camera, pose=camera_pose)
            
            # visualise the pyrender scene with additional axes for camera and origin ######################################
            # axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(axis_length=2, origin_size=0.001), smooth=False)
            # scene.add(axis_mesh, pose=axis_pose)
            
            # oaxis = trimesh.creation.axis(axis_length=2)
            # oaxis.visual.face_colors = np.array([255, 255, 0, 255])  # yellow
            # origin_mesh = pyrender.Mesh.from_trimesh(oaxis, smooth=False)
            # scene.add(origin_mesh, pose=np.eye(4))  # Add origin axis for reference
            # visualise the pyrender scene with additional axes for camera and origin ######################################
            
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
                            
                # for simulating contact, remove points that are much further away than the closest point
                if remove_far_points:
                    try:
                        # print(f"Removing far points with tac mode")
                        min_dist = np.min(z[z > 0])  # Minimum depth value greater than zero
                        # print(f"Minimum depth value: {min_dist}")
                        diff_threshold = 0.05
                        far = z > (min_dist + diff_threshold)
                        z[far] = 0.0  # Set far points to zero (not considered by depth image)
                    except Exception as e:
                        # print(e)
                        # print("No valid points to reduce")
                        pass
                    
                valid = z > 0
                x = (i[valid] - cx) * z[valid] / fx
                y = (j[valid] - cy) * z[valid] / fy
                visible_points = np.vstack((x, y, z[valid])).T
                visible_points_hom = np.hstack((visible_points, np.ones((visible_points.shape[0], 1))))
                
                # camera matrices for transforms
                camera_R_hom = np.eye(4)
                camera_R_hom[:3, :3] = camera_pose[:3, :3]
                camera_t_hom = camera_pose[:, 3].reshape(4, 1)
                
                cam_inv = np.linalg.inv(camera_pose)
                cam_inv_R_hom = np.eye(4)
                cam_inv_R_hom[:3, :3] = cam_inv[:3, :3]
                cam_inv_t_hom = cam_inv[:, 3].reshape(4, 1)
                
                # since the points were obtained from pinhole camera conventions, 
                # need to apply the rotation to the camera pose, 
                # then apply inverse to get the points into the world frame
                
                # rotation matrix for conversion between pyrender/opengl and pinhole camera conventions
                cam_conv_R = np.array([
                        [1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]
                    ])
                
                # get the inverse transform in pinhole camera conventions
                pinhole_T = cam_conv_R @ np.linalg.inv(camera_pose)
                pinhole_R_hom = np.eye(4)
                pinhole_R_hom[:3, :3] = pinhole_T[:3, :3]
                pinhole_t_hom = pinhole_T[:, 3].reshape(4, 1)
                
                # not too sure why the rotation is applied as original non-inverse form, but it works
                visible_points_world = camera_R_hom @ ((visible_points_hom @ cam_conv_R).T + pinhole_t_hom)
                visible_points_world = visible_points_world.T[:, :3]  # Drop homogeneous coordinate
                
                # Check if we have enough points
                if visible_points_world.shape[0] >= num_samples:
                    # Downsample if needed
                    if visible_points_world.shape[0] > num_samples:
                        indices = np.random.choice(visible_points_world.shape[0], num_samples, replace=False)
                        visible_points_world = visible_points_world[indices]
                    return visible_points_world, camera_pose
                image_size *= 2  # Increase resolution
            tqdm.write(f"Attempt {attempt+1}: Max resolution reached, trying new viewpoint...")
            
        # some objects are really small for some reason, sample from a smaller radius
        radius *= 0.5  # Reduce radius for next attempt
        tqdm.write(f"Failed to sample {num_samples} points from any viewpoint after {max_viewpoint_retries} tries. Trying smaller radius: {radius}")
        
    # failed to sample after all attempts. skip and log the model path causing the error
    tqdm.write(f"Failed to sample {num_samples} points from any viewpoint after {max_viewpoint_retries} tries.")
    return None, None
    # raise RuntimeError(f"Failed to sample {num_samples} points from any viewpoint after {max_viewpoint_retries} tries.")

def sample_surface_points(mesh, num_points):
    # Get the triangles and vertices from the mesh
    triangles = mesh.triangles  # (n, 3, 3) array
    vertices = mesh.vertices     # (n, 3) array
    
    # Calculate areas for each triangle
    triangle_areas = []
    for i in range(len(triangles)):
        v0, v1, v2 = triangles[i]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        triangle_areas.append(area)
    
    total_area = sum(triangle_areas)
    
    # Normalize areas to get probability distribution
    triangle_areas = np.array(triangle_areas)
    probabilities = triangle_areas / total_area
    
    # Sample points based on area probabilities
    sampled_points = []
    while len(sampled_points) < num_points:
        # Choose a triangle with probability proportional to its area
        triangle_idx = np.random.choice(len(triangles), p=probabilities)
        v0, v1, v2 = triangles[triangle_idx]
        
        # Sample point inside the triangle using barycentric coordinates
        r1, r2 = np.random.random(), np.random.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        sampled_point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
        
        sampled_points.append(sampled_point)
    
    return np.array(sampled_points)

def process_file(i, data_root_dir, save_root_dir, mode="default", num_samples=5000, max_n=5, overwrite=False, debug=False):
    failed_list = []
    
    # Original pre-processed model file
    file_path = data_root_dir / Path(f"{i}/models/model_normalized2.obj")
    if not file_path.is_file():
        tqdm.write(f"{file_path} not found")
        return
    # if all((output_path.parent / f"new_samples_{n}.npy").is_file() for n in range(5)):
    #     return
    simple_path = str(file_path).replace('model_normalized2', "simplified_mesh")
    if Path(simple_path).is_file():
        # print(f"File already exists: {simple_path}")
        pass
    else:       
        # input(f"File does not exist: {simple_path}. Press Enter to continue.")
        tqdm.write(f"File does not exist: {simple_path}")
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
        if mode == "surface":
            mesh = trimesh.load(file_path)
            mesh.fill_holes()
            mesh.fill_holes()
        else:
            # Fill single traingle and single quad holes in the mesh. twice?
            simplified_trimesh = trimesh.load(simple_path)
            
            simplified_trimesh.fill_holes()
            simplified_trimesh.fill_holes()
        
    except Exception:
        tqdm.write(f"Failed to load simplified mesh under {i}")
        # redo the mesh simplification then try again
        # Filter down the original model to a simpler version
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))
        tqdm.write(f"type of {file_path} is {type(ms)}")
        # Apply mesh simplification while preserving the boundary
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=2000,  # Target number of faces
                        preservenormal=True,
                        preservetopology=True,  # Prevents the creation of holes
                        boundaryweight=1.0)  # High weight to preserve boundaries
        ms.save_current_mesh(simple_path)
        
        simplified_trimesh = trimesh.load(simple_path)
        tqdm.write(f"type of simplified mesh is {type(simplified_trimesh)}")
            
        simplified_trimesh.fill_holes()
        simplified_trimesh.fill_holes()
    sampled_points = None
    
    # processing for surface sampling
    if mode == "surface":
        save_path = save_root_dir / i / "models"
        surface_pc_path = save_path / f"surface_samples.npy"
        # skip if the file already exists
        if surface_pc_path.is_file():
            tqdm.write(f"File already exists: {surface_pc_path}")
            if not overwrite:
                return
        # sample points from the original mesh instead of the simplified mesh
        surface_points = sample_surface_points(mesh, num_points=10000)
        tqdm.write(f"Processing: {surface_pc_path}")
        # save the points
        save_path.mkdir(parents=True, exist_ok=True)
        if not debug:
            np.save(surface_pc_path, surface_points)
        return
            
    # Load partial point cloud sampling options from yaml file
    with open(data_root_dir / "resample.yaml", 'r') as stream:
        args = yaml.safe_load(stream)[mode]
    
    for n in range(max_n):
        save_path = save_root_dir / i / "models"
        
        partial_pc_path = save_path / f"partial_samples_{mode}_{num_samples}_{n}.npy"
        
        # skip if the file already exists
        if partial_pc_path.is_file():
            # print(f"File already exists: {partial_pc_path}")
            if not overwrite:
                continue
            
        # print(f"Processing: {save_path / f'partial_samples_{mode}_{n}.npy'}")
        
        # Sample the mesh for partial point clouds
        sampled_points, camera_pose = sample_visible_points_from_single_view(simplified_trimesh,
                                                                num_samples=num_samples,
                                                                fov_degrees=args['fov_degrees'],
                                                                start_image_size=args['start_image_size'],
                                                                radius=args['radius'],
                                                                max_image_size=args['max_image_size'],
                                                                max_viewpoint_retries=args['max_viewpoint_retries'], 
                                                                remove_far_points= args['remove_far_points'])
        
        # if no points were sampled successfully then both outputs will be None
        if sampled_points is None or camera_pose is None:
            tqdm.write(f"Failed to sample points for {save_root_dir / i}. Skipping...")
            failed_list.append(save_root_dir / i)
            break
        
        # Visualise with open3d all the point clouds together ##########################
        # # visualise the full point cloud (sampled separately)
        # og_full_pc_path = partial_pc_path.parent / f"new_samples_{n}.npy"
        # full_pcd = o3d.geometry.PointCloud()
        # full_pcd.points = o3d.utility.Vector3dVector(np.load(og_full_pc_path))
        # full_pcd.paint_uniform_color([0, 1, 0])  # Green for full point cloud
        
        # # visualise the sampled partial point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points)
        # pcd.paint_uniform_color([1, 0, 0])  # Red for sampled point cloud
        
        # o3d.visualization.draw_geometries([pcd, full_pcd])
        #################################################################################
        # # visualise the camera pose and origin axes as point clouds
        # axis_pcd = create_axis_pointcloud(pose=camera_pose, length=1, step=0.01)
        # origin_pcd = create_axis_pointcloud(pose=np.eye(4), length=.2, step=0.01)
        # o3d.visualization.draw_geometries([pcd, full_pcd, axis_pcd, origin_pcd])
        #################################################################################
        
        # print(f"Saving to: {save_path / f'partial_samples_{mode}_{n}.npy'}')")
        
        # save the points
        if not debug:
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / f"partial_samples_{mode}_{n}.npy", sampled_points)
        
    # save the failed list to a file
    if failed_list:
        with open(f"failed_list_resample_{mode}.txt", 'a') as f:
            # log time
            dt = np.datetime64('now') + np.timedelta64(10, 'h')
            f.write(f"{dt}\n")
            for item in failed_list:
                f.write(f"{item}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for the source mesh.")
    parser.add_argument("--dst", type=str, default="", help="Root directory for saving the resampled point clouds. Same as src_dir if not specified.")
    parser.add_argument("--mode", type=str, default="default", help="View mode to resample the data. See resample.yaml for options. 'all' to process all available modes.")
    parser.add_argument("--taxonomy", type=str, default="filtered_taxonomy.json",
                    help="Path to taxonomy JSON (has 'synsetId' and 'name').")
    parser.add_argument("--classes", type=str, default="",
                    help="Class names to include (comma/space-separated). Example: 'airplane, mug car', leave blank for all default trainable classes.")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of points to sample for each point cloud")
    parser.add_argument("--n", type=int, default=5, 
                    help="Number of point clouds to sample for each instance")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite any existing files.")
    parser.add_argument("--debug", action='store_true', help="Run without saving for debug.")

    args = parser.parse_args()
    
    data_root_dir = Path(args.src)
    save_root_dir = Path(args.dst) if args.dst != "" else Path(args.src)
    tax_dir = data_root_dir / args.taxonomy
    
    # determine which classes to run
    if args.classes == "":
        classes_list = TRAINABLE
    elif "," in args.classes:
        classes_list = [s.strip() for s in args.classes.split(",") if s.strip()]
    else:
        classes_list = [args.classes]
    name2synset, synset2name = load_taxonomy_maps(tax_dir)
        
    synset_ids = names_to_synsets(classes_list, tax_dir, aliases=KEYS_ALIASES)
    folders = synset_ids   # <-- list of synset IDs
    
    # gather all folders to run
    folders_dict = {entry: [] for entry in classes_list}
    # list contains dirs using synset ids
    for l in open(data_root_dir / "list.txt"):
        if l.strip().split("/")[1] in folders:
            # get index of id and convert to class name
            i = folders.index(l.strip().split("/")[1])
            # write dir into list with class name as key
            folders_dict[classes_list[i]].append(l.strip())
            
    # check all instances of the selected classes exist in folders_to_run
    instance_count = {entry: 0 for entry in classes_list}
    for entry in folders_dict.keys():
        if entry in classes_list:
            instance_count[entry] += 1
    if any(count == 0 for count in instance_count.values()):
        print("Warning: Some classes have zero instances in the dataset:")
        for entry, count in instance_count.items():
            if count == 0:
                print(f" - {entry}")
    
    total_len = 0
    for l in folders_dict.values():
        total_len += len(l)
    print(f"Classes to run: {list(folders_dict.keys())}")
    print(f"Total object instances: {total_len}")
    
    # determine which modes to run
    mode_list = []
    if args.mode == "all":
        with open(data_root_dir / "resample.yaml", 'r') as stream:
            modes = yaml.safe_load(stream)
        mode_list = list(modes.keys())
    elif ',' in args.mode:
        mode_list = [s.strip() for s in args.mode.split(",") if s.strip()]
    else:
        mode_list = [args.mode]
        # note "surface" is an exception not in the yaml file, it samples from the original mesh surface
    
    print(f"Modes to run: {mode_list if mode_list else [args.mode]}")
    
    # input("Press Enter to continue...")
    for c in classes_list:
        print(f"Processing class: {c}")
        folders_to_run = folders_dict[c]
        for mode in mode_list:
            coverage_list_list = []
            print(f"Processing mode: {mode}")
            # loop through each object folder
            for idx, i in tqdm(enumerate(folders_to_run), total=len(folders_to_run)):
                # sample partial point clouds
                process_file(i, data_root_dir, save_root_dir, mode, max_n=args.n, overwrite=args.overwrite, debug=args.debug)
                if mode != "surface":
                    # calculate coverage for the sampled point clouds
                    coverage_list = compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode, max_n=args.n, output_csv=(not args.debug))
                    # collate for plotting
                    if not coverage_list is None:
                        coverage_list_list.append(coverage_list)
                
            if mode != "surface":
                # plot histograms on a per-mode basis (?)
                try:
                    coverage_values = np.array([c for l in coverage_list_list for c in l])
                    plot_coverage_histogram(save_root_dir, coverage_values, mode)
                except Exception as e:
                    tqdm.write(str(e))
                    tqdm.write(f"Skipping plotting for class {c} mode {mode}")
