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
from sampling_utils import sample_visible_points_from_single_view, sample_full_pc_trimesh
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


def sample_full_point_clouds(folders_list, data_root_dir, save_root_dir, class_name, max_n=2, overwrite=False, debug=False):
    # 2. start sampling, loop until all mesh instances have been processed
    with tqdm(folders_list, total=len(folders_list)) as pbar:
        for idx, i in enumerate(pbar):
            # early escape for debugging
            # if idx == 100:
            #     break
            
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
            
            for n in range(max_n):
                save_path = save_root_dir / i / "models"
                full_pc_path = save_path / f"new_samples_{n}.npy"
                # skip if the file already exists
                if full_pc_path.is_file():
                    tqdm.write(f"File already exists: {full_pc_path}")
                    if not overwrite:
                        continue
                    
                num_samples = 5000
                # Sample the mesh for full point clouds
                sampled_points = sample_full_pc_trimesh(simplified_trimesh, num_points=num_samples, max_attempts_factor=200, debug=debug)
                    
                # if no points were sampled successfully then both outputs will be None
                if sampled_points is None:
                    tqdm.write(f"Failed to sample points for {save_root_dir / i}. Skipping...")
                    continue
                    
                if debug:
                    # visualise the surface point cloud (sampled separately)
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter3d(
                            x=sampled_points[:, 0], y=sampled_points[:, 1], z=sampled_points[:, 2],
                            mode="markers",
                            marker={"size": 2.5, "color": "red", "opacity": 1.0},
                            name="Full points",
                            visible=(idx == 0),
                        )
                    )

                    surface_pc_path = full_pc_path.parent / f"surface_samples.npy"
                    surface_pts = np.load(surface_pc_path)
                    fig.add_trace(
                        go.Scatter3d(
                            x=surface_pts[:, 0], y=surface_pts[:, 1], z=surface_pts[:, 2],
                            mode="markers",
                            marker={"size": 2, "color": "green", "opacity": 1.0},
                            name="Surface points",
                            visible=(idx == 0),
                        )
                    )
                    out_path = Path("./full_cloud_debug.html")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
                else:
                    # save the points
                    save_path.mkdir(parents=True, exist_ok=True)
                    np.save(full_pc_path, sampled_points)

                continue  # move on to next n
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for the source mesh.")
    parser.add_argument("--dst", type=str, default="", help="Root directory for saving the resampled point clouds. Same as src_dir if not specified.")
    parser.add_argument("--taxonomy", type=str, default="filtered_taxonomy.json",
                    help="Path to taxonomy JSON (has 'synsetId' and 'name').")
    parser.add_argument("--split-file", type=str, default="./data/shapenet_split/splits_out.csv", 
                    help="CSV file containing the splits for each instance.")
    parser.add_argument("--split", type=str, default="all",
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
        
        for s in splits_to_run:
            print(f" Processing split: {s}")
            folders_to_run = classes_to_run[s]
            
            # sample partial point clouds
            # the distribution of coverage values should be uniform for each split of each class
            sample_full_point_clouds(folders_to_run, data_root_dir, save_root_dir, class_name=c,
                                    max_n=args.n, overwrite=args.overwrite, debug=args.debug)
