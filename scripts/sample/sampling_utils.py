import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYGLET_HEADLESS"] = "True"      # make pyglet headless
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # or "egl" if you have EGL/NVIDIA
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import trimesh
import numpy as np
from PIL import Image
import pyrender

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
                # Increase resolution and try again with same view point if not enough points
                image_size *= 2
            # if not enough points with max resolution, try another viewpoint
            # tqdm.write(f"Attempt {attempt+1}: Max resolution reached, trying new viewpoint...")
            
        # some objects are really small for some reason, sample from a smaller radius
        # this is a failsafe to prevent skipping and shouldn't be triggered normally
        radius *= 0.5  # Reduce radius for next attempt
        # tqdm.write(f"Failed to sample {num_samples} points from any viewpoint after {max_viewpoint_retries} tries. Trying smaller radius: {radius}")
        
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

def sample_full_pc_trimesh(mesh, num_points, max_attempts_factor=50, seed=None, debug=False):
    """
    Uniformly sample points from the *volume* of a closed watertight mesh
    using rejection sampling in the mesh's axis-aligned bounding box.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Should be a closed, watertight mesh for best results.
    num_points : int
        Number of points to sample.
    max_attempts_factor : int
        Upper bound on proposals = max_attempts_factor * num_points.
        If the mesh is thin relative to its AABB, increase this.
    seed : int | None
        RNG seed.

    Returns
    -------
    (num_points, 3) np.ndarray
        Points uniformly distributed inside the mesh volume.

    Raises
    ------
    RuntimeError if unable to sample enough points within the attempt budget.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # AABB
    bounds = mesh.bounds  # shape (2, 3): [min, max]
    bmin, bmax = bounds[0], bounds[1]

    # Quick sanity check (optional, but helpful)
    # If mesh isn't watertight, containment tests can be unreliable.
    # You can choose to warn or raise.
    if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
        # Not raising because you may have meshes that still work okay,
        # but this is important to know.
        # You could change to: raise ValueError(...)
        pass

    collected = []
    pts = np.zeros(0)
    attempts = 0
    max_attempts = int(max_attempts_factor * num_points)

    # Batch proposals to be efficient
    # Start with a batch size; adapt if needed.
    batch = max(1024, num_points * 2)

    while len(pts) < num_points and attempts < max_attempts:
        remaining = num_points - len(collected)
        cur_batch = max(batch, remaining * 2)

        # Uniform proposals in AABB
        proposals = rng.uniform(bmin, bmax, size=(cur_batch, 3))

        # Inside test: trimesh has fast containment checks
        inside = mesh.contains(proposals)  # boolean mask

        accepted = proposals[inside]
        if accepted.size > 0:
            collected.append(accepted)
        
        pts = np.vstack(collected)

        attempts += cur_batch

        if debug:
            print(f"Current attempt: {attempts}")
            print(f"length of collected: {len(collected)}")
            print(f"accepted: {type(accepted), accepted.shape}")
            print(f"length of pts: {len(np.vstack(collected))}")
            print(f"shape of pts: {np.vstack(collected).shape}")

    if len(collected) == 0:
        raise RuntimeError("No points were accepted. Is the mesh closed / does contains() work for it?")

    if pts.shape[0] < num_points:
        raise RuntimeError(
            f"Failed to sample {num_points} points within budget. "
            f"Got {pts.shape[0]}. Try increasing max_attempts_factor "
            f"or use a tighter bounding volume / tetrahedral sampling."
        )

    if debug:
        print(f"Finished sampling after {attempts} (batched) attempts")

    return pts[:num_points]
