import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYGLET_HEADLESS"] = "True"      # make pyglet headless
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # or "egl" if you have EGL/NVIDIA
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import trimesh
import numpy as np
import pymeshlab
from PIL import Image
import yaml
import argparse
from keypoint_diffuser.utils.synset_utils import load_taxonomy_maps, names_to_synsets, synsets_to_names
from coverage_utils import compute_coverage_for_profile, plot_coverage_histogram, surface_coverage_pointwise
from sampling_utils import sample_visible_points_from_single_view, sample_surface_points
from sampling_utils import TRAINABLE, KEYS_ALIASES
import csv

def init_folders(args):
    data_root_dir = Path(args.src)
    save_root_dir = Path(args.dst) if args.dst != "" else Path(args.src)
    tax_dir = data_root_dir / args.taxonomy
    
    # collate folders to run
    # determine which classes to run
    if args.classes == "":
        classes_list = TRAINABLE
    elif "," in args.classes:
        classes_list = [s.strip() for s in args.classes.split(",") if s.strip()]
    else:
        classes_list = [args.classes]
    name2synset, synset2name = load_taxonomy_maps(tax_dir)
        
    synset_ids = names_to_synsets(classes_list, tax_dir, aliases=KEYS_ALIASES)  # a list of synset ids
    
    # gather all folders to run
    # folders_dict = {c: [] for c in classes_list}
    splits = ["train", "val", "test"]
    folders_dict = {c: {s: [] for s in splits} for c in classes_list}
    
    # extract directories from list.txt
    # list.txt contains all directories of object instances using synset ids
    # for l in open(data_root_dir / "list.txt"):
    #     if l.strip().split("/")[1] in synset_ids:
    #         # get index of id and convert to class name
    #         i = synset_ids.index(l.strip().split("/")[1])
    #         # write dir into list with class name as key
    #         folders_dict[classes_list[i]].append(l.strip())
            
    # extract and build directories from splits file (only contains classes from TRAINABLE)
    with open(args.split_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            synset_id, model_id, split = row
            if synset_id in synset_ids:
                i = synset_ids.index(synset_id)
                folders_dict[classes_list[i]][split].append(f"./{synset_id}/{model_id}")
    
            
    # check all instances of the selected classes exist in folders_to_run
    instance_count = {c: 0 for c in classes_list}
    for class_entry in folders_dict.keys():
        if class_entry in classes_list:
            instance_count[class_entry] += 1
    if any(count == 0 for count in instance_count.values()):
        print("Warning: Some classes have zero instances in the dataset:")
        for class_entry, count in instance_count.items():
            if count == 0:
                print(f" - {class_entry}")
    
    total_len = 0
    for l in folders_dict.values():
        total_len += len(l)
    print(f"Classes to run: {list(folders_dict.keys())}")
    print(f"Total object instances: {total_len}")
    
    return folders_dict, classes_list

# uniform sampling functions ####################################################################################
def k_most_underrepresented_bins(counts, k=3):
    counts = np.asarray(counts)
    K = len(counts)
    if K == 0:
        return []
    k = min(k, K)
    # indices sorted by count ascending
    sorted_idx = np.argsort(counts)
    # return the first k indices (smallest counts)
    return sorted_idx[:k].tolist()

def online_uniform_sampling(folders_list, data_root_dir, save_root_dir, csv_path, class_name, split="", max_n=2, num_bins=10, overwrite=False, debug=False):
    # 1. inits
    # set up coverage histogram
    bin_counts = np.zeros(num_bins, dtype=int)
    # try to sample each bin of coverage values up to max_size
    bin_max_size = np.floor(1.2 * max_n * len(folders_list) / num_bins)
    
    # coverage values list for plotting    
    sampled_coverages = []
    
    # for bin update printing
    prev = None
    
    # 2. start sampling, loop until all mesh instances have been processed
    with tqdm(folders_list, total=len(folders_list)) as pbar:
        for idx, i in enumerate(pbar):
            # early escape for debugging
            # if idx == 100:
            #     break
            
            pbar.set_postfix_str(
                "bins=" + ",".join(str(c) for c in bin_counts)
            )
            
            # load mesh files
            # Original pre-processed model file
            file_path = data_root_dir / Path(f"{i}/models/model_normalized2.obj")
            if not file_path.is_file():
                tqdm.write(f"{file_path} not found")
                # if not found should skip this sample and go to the next
                continue
            # simplified mesh path
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
            # load
            simplified_trimesh = trimesh.load(simple_path)
            simplified_trimesh.fill_holes()
            simplified_trimesh.fill_holes()
            
            # init the surface points of this mesh for coverage calculation
            surface_points = np.load(data_root_dir / i / "models" / "surface_samples.npy")
            for n in range(max_n):
                save_path = save_root_dir / i / "models"
                partial_pc_path = save_path / f"partial_samples_uniform_dist_{n}.npy"
                    
                # skip if the file already exists
                if partial_pc_path.is_file():
                    # print(f"File already exists: {partial_pc_path}")
                    if not overwrite:
                        continue
                # keep sampling until a valid coverage is found for the one of the most underrepresented bins
                max_attempts = 1000
                attempts = 0
                while attempts < max_attempts:
                    attempts += 1
                    
                    # 3. sample with randomised virtual camera parameters
                    camera_radius = np.random.uniform(0.5, 2.0)
                    fov_degrees = np.random.uniform(5, 90)
                    remove_far_points = np.random.choice([True, False])
                    start_image_size = 512  # some parameters can stay constant
                    max_image_size = 2048
                    max_viewpoint_retries = 10
                    num_samples = 5000
                    
                    # Sample the mesh for partial point clouds
                    sampled_points, camera_pose = sample_visible_points_from_single_view(simplified_trimesh,
                                                                            num_samples=num_samples,
                                                                            fov_degrees=fov_degrees,
                                                                            start_image_size=start_image_size,
                                                                            radius=camera_radius,
                                                                            max_image_size=max_image_size,
                                                                            max_viewpoint_retries=max_viewpoint_retries, 
                                                                            remove_far_points=remove_far_points)
                    
                    # if no points were sampled successfully then both outputs will be None
                    if sampled_points is None or camera_pose is None:
                        tqdm.write(f"Failed to sample points for {save_root_dir / i}. Skipping...")
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
                    
                    # 4. calculate the coverage of the sampled point cloud and add to histogram
                    coverage, _, _, _ = surface_coverage_pointwise(surface_points, sampled_points, radius=None)
                    
                    # find the appropriate bin for this coverage value
                    # bin_index = min(int(coverage * num_bins), num_bins - 1)
                    max_coverage = 0.65
                    norm = np.clip(coverage / max_coverage, 0.0, 1.0)
                    bin_index = min(int(norm * num_bins), num_bins - 1)
                    # check if this bin is underrepresented, and if it has reached max size
                    k = num_bins // 3
                    # skip checking if the bin is not yet close to full to speed up initial sampling
                    if bin_counts[bin_index] >= bin_max_size * 0.8:
                        if bin_index not in k_most_underrepresented_bins(bin_counts, k=k) or bin_counts[bin_index] >= bin_max_size:
                            # skip this sample and try again
                            # tqdm.write(f"Skipping sample with coverage {coverage:.4f} in bin {bin_index}, count {bin_counts[bin_index]}/{bin_max_size}")
                            continue
                    # otherwise, accept this sample and update the bin count
                    # tqdm.write(f"Accepted sample with coverage {coverage:.4f} in bin {bin_index}, count {bin_counts[bin_index]}/{bin_max_size}")
                    bin_counts[bin_index] += 1
                    # save the coverage values for plotting
                    sampled_coverages.append(coverage)
                    
                    # populate and save metadata dict
                    row = {
                        "class_id": i.split("/")[1],
                        "model_id": i.split("/")[2],
                        "sample_index": n,
                        "split": split,
                        "num_samples": num_samples,
                        "fov_degrees": fov_degrees,
                        "camera_radius": camera_radius,
                        "remove_far_points": remove_far_points,
                        "coverage": coverage,
                    }
                    with csv_path.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
                        writer.writerow(row)
                    
                    # save the points
                    if not debug:
                        save_path.mkdir(parents=True, exist_ok=True)
                        np.save(partial_pc_path, sampled_points)

                    break  # move on to next n
                
                if attempts >= max_attempts:
                    tqdm.write(f"Max attempts reached for {i}, n={n}; no acceptable sample found.")

    return sampled_coverages

#################################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for the source mesh.")
    parser.add_argument("--dst", type=str, default="", help="Root directory for saving the resampled point clouds. Same as src_dir if not specified.")
    parser.add_argument("--taxonomy", type=str, default="filtered_taxonomy.json",
                    help="Path to taxonomy JSON (has 'synsetId' and 'name').")
    parser.add_argument("--split-file", type=str, default="./data/shapenet_split/splits_out.csv", 
                    help="CSV file containing the splits for each instance.")
    parser.add_argument("--split", type=str, default="train",
                    help="Which split to process: train/val/test/all")
    parser.add_argument("--classes", type=str, default="",
                    help="Class names to include (comma/space-separated). Example: 'airplane, mug, car', leave blank for all default trainable classes.")
    parser.add_argument("--n", type=int, default=5, 
                    help="Number of point clouds to sample for each instance")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite any existing files.")
    parser.add_argument("--debug", action='store_true', help="Run without saving for debug.")

    args = parser.parse_args()
    
    # get the directories and classes to run
    folders_dict, classes_list = init_folders(args)
    data_root_dir = Path(args.src)
    save_root_dir = Path(args.dst) if args.dst != "" else Path(args.src)
    
    # determine which splits to run
    if args.split == "all":
        splits_to_run = ["train", "val", "test"]
    else:
        splits_to_run = [args.split]
    
    
    # input("Press Enter to continue...")
    for c in classes_list:
        print(f"Processing class: {c}")
        classes_to_run = folders_dict[c]
        
        # init the metadata csv, one csv for each class, one for all splits
        META_FIELDS = [
            "class_id", "model_id", "sample_index", "split",  
            "num_samples", "fov_degrees", "camera_radius", "remove_far_points", "coverage", 
        ]
        csv_path = save_root_dir / f"uniform_sampling_metadata_{c}.csv"
        if not csv_path.exists() or args.overwrite:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=META_FIELDS)
                writer.writeheader()
        
        for s in splits_to_run:
            print(f" Processing split: {s}")
            folders_to_run = classes_to_run[s]
            
            # sample partial point clouds
            # the distribution of coverage values should be uniform for each split of each class
            coverage_values = online_uniform_sampling(folders_to_run, data_root_dir, save_root_dir, csv_path, class_name=c,
                                                        split=s, max_n=args.n, num_bins=10, 
                                                        overwrite=args.overwrite, debug=args.debug)
                
            # plot histograms on a per-split basis
            try:
                plot_coverage_histogram(save_root_dir, coverage_values, log_scale=False, file_name=f"coverage_distribution_{c}_{s}.png")
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write(f"Skipping plotting for class {c} split {s}")
