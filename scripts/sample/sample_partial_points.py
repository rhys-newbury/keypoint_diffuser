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
from keypoint_diffuser.utils.synset_utils import load_taxonomy_maps, names_to_synsets, synsets_to_names
from coverage_utils import compute_coverage_for_profile, plot_coverage_histogram
import sampling_utils
from sampling_utils import sample_visible_points_from_single_view, sample_surface_points, create_axis_pointcloud

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
        
        partial_pc_path = save_path / f"partial_samples_{mode}_{n}.npy"
        # partial_pc_path = save_path / f"partial_samples_{mode}_{num_samples}_{n}.npy"
        
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
            np.save(partial_pc_path, sampled_points)
        
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
    parser.add_argument("--coverage_only", action='store_true', help="Only calculate the coverage value from saved samples")

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
                if not args.coverage_only:
                    process_file(i, data_root_dir, save_root_dir, mode, max_n=args.n, overwrite=args.overwrite, debug=args.debug)
                    
                if mode != "surface":
                    # calculate coverage for the sampled point clouds
                    coverage_list = compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode, max_n=args.n, overwrite=args.overwrite, output_csv=(not args.debug))
                    # collate for plotting
                    if not coverage_list is None:
                        coverage_list_list.append(coverage_list)
                # early escape for debugging
                # if idx == 100:
                #     break
                
            if mode != "surface" and not args.coverage_only:
                # plot histograms on a per-mode basis (?)
                try:
                    coverage_values = np.array([c for l in coverage_list_list for c in l])
                    plot_coverage_histogram(save_root_dir, coverage_values, mode)
                except Exception as e:
                    tqdm.write(str(e))
                    tqdm.write(f"Skipping plotting for class {c} mode {mode}")
