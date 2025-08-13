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
from visualise_pc_data import visualise_file

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



def compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode="default", overwrite=False, visualise=False):
    path_list = []
    coverage_list = []
    radius_list = []
    
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
        # load the surface point clouds
        surface_points = np.load(data_root_dir / i / "models" / "surface_samples.npy")
    except Exception:
        print(i)
    sampled_points = None
    
    # Load partial point cloud sampling options from yaml file    
    for n in range(5):
        save_path = save_root_dir / i / "models"
        
        partial_pc_path = save_path / f"partial_samples_{mode}_{n}.npy"
        
        # load the partial point cloud
        partial_points = np.load(partial_pc_path)
        
        # calculate the coverage of the partial point cloud
        coverage, radius, covered_points, not_covered_points = surface_coverage_pointwise(surface_points, partial_points)
        
        # save name and coverage to lists
        path_list.append(partial_pc_path)
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
    save_path = save_root_dir / i / "models"
    save_path.mkdir(parents=True, exist_ok=True)
    coverage_file = save_path / f"coverage_{mode}.csv"
    with open(coverage_file, "w") as f:
        f.write("partial_pc_path,coverage\n")
        for path, coverage, radius in zip(path_list, coverage_list, radius_list):
            f.write(f"{path},{coverage},{radius}\n")    

    return coverage_list

folders = {"02691156"}
# folders = {"02691156", "03636649", "03467517", "02954340", "02958343"}    # airplane, lamp, guitar, cap, car
# folders =   {"03636649", "03467517", "02954340", "02958343"}  # lamp, guitar, cap, car
# folders = {"02691156", "03636649",}
# folders = {"02691156", "03467517", "02954340", "02958343", "03797390", "04225987"}    # airplane, lamp, guitar, cap, car, mug, skateboard
# folders = {"03467517", "02954340", "02958343", "03797390", "04225987"}    # lamp, guitar, cap, car, mug, skateboard
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="data/shape_data_eric", help="Root directory for the source mesh.")
    parser.add_argument("--dst_dir", type=str, default="data/shape_data_eric", help="Root directory for saving the resampled point clouds.")
    parser.add_argument("--mode", type=str, default="default", help="View mode to resample the data. See resample.yaml for options.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files if they exist.")
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
        coverage_list = compute_coverage_for_profile(i, data_root_dir, save_root_dir, mode=args.mode, overwrite=args.overwrite)
        coverage_list_list.append(coverage_list)
        
    print(f"Mean coverage for profile {args.mode}: {np.mean(coverage_list_list):.4f} ± {np.std(coverage_list_list):.4f}")