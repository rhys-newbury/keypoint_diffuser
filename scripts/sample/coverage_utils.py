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

def estimate_spacing(points, k=36):
    """
    Estimate typical point spacing from median 2nd-NN distance.
    points: (N, 3) numpy array of XYZ coords
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)  # shape: (N, k)
    # Take median of distances to the 2nd neighbor (index 1, since index 0 is the point itself)
    return np.max(dists[:, 1])

def surface_coverage_pointwise(surface_points, partial_points, radius=None):
    """
    Calculate fraction of surface_points that are within `radius`
    of any partial_points.
    
    surface_points: (Ns, 3) numpy array
    partial_points: (Np, 3) numpy array
    radius: float or None — if None, automatically estimated
    
    Returns:
        coverage_fraction, radius_used
    """
    if radius is None:
        spacing = estimate_spacing(surface_points)
        radius = 1.5 * spacing if spacing > 0 else 0.01

    tree_partial = cKDTree(partial_points)
    # Query: for each surface point, count neighbors within radius
    neighbors = tree_partial.query_ball_point(surface_points, r=radius)
    covered_mask = np.fromiter((len(n) > 0 for n in neighbors), count=len(surface_points), dtype=bool)
    covered_points = surface_points[covered_mask]
    not_covered_points = surface_points[~covered_mask]
    coverage = covered_mask.mean()

    return coverage, radius, covered_points, not_covered_points



def compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode="default", visualise=False, max_n=5, overwrite=False, output_csv=True):
    """
    Compute the coverage of partial point clouds against the surface point cloud.
    Partial point clouds are read from file.
    Coverage values are calculated on a per-object instance basis.
    """
    # if mode is surface then should just skip
    if mode == "surface":
        return
    
    # check for existing
    coverage_file_path = save_root_dir / i / "models" / f"coverage_{mode}.csv"
    if coverage_file_path.is_file() and not overwrite:
        return
    
    # initialize
    path_list = []
    coverage_list = []
    radius_list = []
    failed_list = []
    file_path = data_root_dir / Path(f"{i}/models/model_normalized2.obj")
    simple_path = str(file_path).replace('model_normalized2', "simplified_mesh")
    
    
    # Original pre-processed model file
    if not file_path.is_file():
        tqdm.write(f"{file_path} not found")
        return
    # if all((output_path.parent / f"new_samples_{n}.npy").is_file() for n in range(5)):
    #     return
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
        # load the surface point clouds
        surface_points = np.load(data_root_dir / i / "models" / "surface_samples.npy")
    except Exception:
        tqdm.write("Failed to load surface points")
        tqdm.write(data_root_dir / i / "models" / "surface_samples.npy")
    
    for n in range(max_n):
        save_path = save_root_dir / i / "models"
        
        partial_pc_path = save_path / f"partial_samples_{mode}_{n}.npy"
        
        # load the partial point cloud
        for idx in range(5):  # try max of 5 times, sometimes the file is still being saved and need to be loaded again
            try:
                partial_points = np.load(partial_pc_path, allow_pickle=True)
                break
            except Exception as e:
                pass
                # tqdm.write(f"Error when loading {partial_pc_path}, trying again")
                # tqdm.write(str(e))
                # continue
            if idx == 4:
                failed_list.append(partial_pc_path)
                tqdm.write(f"Partial point cloud at {partial_pc_path} does not exist.")
                return

        
        # calculate the coverage of the partial point cloud
        coverage, radius, covered_points, not_covered_points = surface_coverage_pointwise(surface_points, partial_points)
        
        # save name and coverage to lists
        path_to_save = Path(i) / "models" / f"partial_samples_{mode}_{n}.npy"
        path_list.append(str(path_to_save))
        coverage_list.append(coverage)
        radius_list.append(radius)
        
        # visualise for testing
        if visualise:
            print(f"\n\nCoverage for {partial_pc_path} using radius {radius:.4f}: {coverage:.4f}\n\n")
            # visualise_file(i, data_root_dir, mode, nin=n, different_n=False)
            
            partial_pcd = o3d.geometry.PointCloud()
            partial_pcd.points = o3d.utility.Vector3dVector(partial_points)
            partial_pcd.paint_uniform_color([0, 0, 0])
            # orig_pcd.paint_uniform_color([1, 0, 1])

            covered_pcd = o3d.geometry.PointCloud()
            covered_pcd.points = o3d.utility.Vector3dVector(covered_points)
            covered_pcd.paint_uniform_color([0, 1, 0])
            
            not_covered_pcd = o3d.geometry.PointCloud()
            not_covered_pcd.points = o3d.utility.Vector3dVector(not_covered_points)
            not_covered_pcd.paint_uniform_color([1, 0, 0])
            
            o3d.visualization.draw_geometries(
                [partial_pcd, covered_pcd, not_covered_pcd],
                window_name="Point Cloud with Labels and Keypoints",
            )
        
    # print mean results
    # mean_coverage = np.mean(coverage_list)
    # print(f"Mean coverage for {i} with mode {mode}: {mean_coverage:.4f}")
    
    # save the results into a csv file
    if output_csv:
        save_path = save_root_dir / i / "models"
        save_path.mkdir(parents=True, exist_ok=True)
        coverage_file = save_path / f"coverage_{mode}.csv"
        with open(coverage_file, "w") as f:
            f.write("partial_pc_path,coverage,radius\n")
            for path, coverage, radius in zip(path_list, coverage_list, radius_list):
                f.write(f"{path},{coverage},{radius}\n")

    # plot histogram of coverage values and save
    # if plot:
    #     coverage_list_list = []
    #     for idx, i in tqdm(enumerate(folders_to_run), total=len(folders_to_run)):
    #         coverage_list, failed_list = compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode=args.mode, overwrite=args.overwrite)
    #         coverage_list_list.append(coverage_list)
            
    #     coverage_list_list = np.array([c for l in coverage_list_list for c in l])
        
    #     plt.figure(figsize=(6,4))
    #     plt.hist(coverage_list_list, bins=20, edgecolor='black', alpha=0.7)
    #     plt.xlabel("Coverage")
    #     plt.ylabel("Frequency")
    #     plt.title("Distribution of Coverage Values")
    #     plt.grid(alpha=0.3)
    #     plt.yscale("log")
    #     plt.savefig(f"coverage_distribution_{args.mode}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    # save the failed list to a file
    if failed_list:
        with open(f"failed_list_coverage_{mode}.txt", 'a') as f:
            # log time
            dt = np.datetime64('now') + np.timedelta64(10, 'h')
            f.write(f"{dt}\n")
            for item in failed_list:
                f.write(f"{item}\n")

    return coverage_list

def plot_coverage_histogram(save_root_dir, coverage_values, mode, log_scale=False):
    """
    Plot histogram of coverage values and save to file.
    Meant for values on a per-mode basis.
    """
    plt.figure(figsize=(6,4))
    plt.hist(coverage_values, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Coverage Values: Mean {np.mean(coverage_values):.4f} ± std {np.std(coverage_values):.4f}")
    plt.grid(alpha=0.3)
    if log_scale:
        plt.yscale("log")
    plt.savefig(save_root_dir / f"coverage_distribution_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean coverage for profile {mode}: {np.mean(coverage_values):.4f} ± {np.std(coverage_values):.4f}")
