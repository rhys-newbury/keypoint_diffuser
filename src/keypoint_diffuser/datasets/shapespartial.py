import collections.abc as container_abcs
import itertools
import json
import os
import random
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import csv
import re
from scipy.spatial.transform import Rotation

from ..utils.utils import normalize_to_box_multi, normalize_scale_min_max
from ..utils.synset_utils import synsets_to_names

class ShapesPartial(torch.utils.data.Dataset):
    MESH_FILE_EXT = "obj"
    POINT_CLOUD_FILE_EXT = "pts"
    DO_NOT_BATCH = [
        "source_face",
        "source_mesh",
        "target_face",
        "target_mesh",
        "source_seg_points",
        "target_seg_points",
        "source_seg_labels",
        "target_seg_labels",
        "source_mesh_obj",
        "target_mesh_obj",
    ]
    CATEGORY2SYNSETOFFSET = {
        "airplane": "02691156",
        "bed": "02818832",
        "bottle": "02876657",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "guitar": "03467517",
        "helmet": "03513137",
        "knife": "03624134",
        "motorbike": "03790512",
        "mug": "03797390",
        "table": "04379243",
        "vessel": "04530566",
        }
    # {     # old subset
    #     "airplane": "02691156",
    #     "bag": "02773838",
    #     "cap": "02954340",
    #     "car": "02958343",
    #     "chair": "03001627",
    #     "earphone": "03261776",
    #     "guitar": "03467517",
    #     "knife": "03624134",
    #     "lamp": "03636649",
    #     "laptop": "03642806",
    #     "motorbike": "03790512",
    #     "mug": "03797390",
    #     "pistol": "03948459",
    #     "rocket": "04099429",
    #     "skateboard": "04225987",
    #     "table": "04379243",
    # }
    SYNSETOFFSET2CATEGORY = {v: k for k, v in CATEGORY2SYNSETOFFSET.items()}

    PARTIALSAMPLEMODES = ['full', 'default', 'myopia', 'patch', 'tac', 'all', 'uniform', 'uniform-combined', 'canonical']
    POINTNOISETARGETS = ['both', 'full', 'partial']
    # PARTIALSAMPLEMODES = ['default', 'myopia', 'patch', 'all']

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def normalize(self, x, xp):
        # xp is partial pc, need to normalize using same values as full pc x for consistency
        if self.opt.normalize == "unit_box":
            # keeps spatial and geometric structure
            pc, pcp, center, scale = normalize_to_box_multi(x, xp)
            norm_vals = (center, scale)
        elif self.opt.normalize == "minmax_scaling":
            # translate to zero mean before
            mean_shift = x.mean(axis=0)
            x = x - mean_shift
            xp = xp - mean_shift    # same mean as full pc to keep relative position consistent
            # default normalization by original dataset
            pc, dmin, dmax = normalize_scale_min_max(x)
            pcp, _, _ = normalize_scale_min_max(xp, dmin, dmax)
            norm_vals = (dmin, dmax)
            # translate to zero mean after
            # pc = pc - pc.mean(axis=0)
            # pcp = pcp - pc.mean(axis=0)    # same mean as full pc to keep relative position consistent
        else:
            raise ValueError()
        
        return pc, pcp, norm_vals

    def __init__(self, opt, transform=None):
        self.opt = opt
        self.get_two = self.opt.get_two
        assert (
            self.opt.category == "all"
            and self.opt.test_category is not None
            and self.opt.test_category in self.CATEGORY2SYNSETOFFSET.values()
            or self.opt.test_category in self.SYNSETOFFSET2CATEGORY.values()
        ) or (
            self.opt.category in self.CATEGORY2SYNSETOFFSET.values()
            or self.opt.category in self.SYNSETOFFSET2CATEGORY.values()
            and self.opt.test_category is None
        )

        self.available_modes_list = self.PARTIALSAMPLEMODES
        assert (
            self.opt.partial_view_mode in self.available_modes_list
        )

        self.n_partial_samples = self.opt.n_partial_samples
        self.enumerate_partial_samples = self.opt.enumerate_partial_samples
        
        # build a list of individual view modes to draw from
        # 'all' picks a mode at random out of all available set in the list. (no guarantees)
        # 'uniform-combined' is meant to randomly choose between uniform partial and full
        if self.opt.partial_view_mode == 'uniform-combined':
            self.view_modes = ['uniform', 'full']
        elif self.opt.partial_view_mode == 'all':
            self.view_modes = [
                m for m in self.available_modes_list
                if m not in ('uniform-combined', 'canonical', 'all')
            ]
        else:
            self.view_modes = [self.opt.partial_view_mode]

        self.mesh_dir = opt.mesh_dir

        # cache of all views that exist on disk
        self._views_cache = {}

        self.dataset = self.load_dataset()  # get dict of access info on each model
        self.transform = transform  # Store transform function
        self.dr = opt.domain_randomization
        self.max_scaling_factor = opt.max_scaling_factor
        self.max_translation_offset = opt.max_translation_offset

        # per-point gaussian jitter on the loaded positions (sensor-noise simulation).
        self.point_noise_std = float(self.opt.point_noise_std)
        # which point cloud(s) to apply the noise to: full, partial, or both
        self.point_noise_target = self.opt.point_noise_target
        assert self.point_noise_target in self.POINTNOISETARGETS, (
            f"point_noise_target must be one of {self.POINTNOISETARGETS}, "
            f"got {self.point_noise_target}"
        )

        print(f"dataset size {len(self)}; {len(self.dataset['name'])} unique objects")

    def _load_from_split_file(self, split):
        data_frame = pd.read_csv(self.opt.split_file)

        # Convert category name to synset ID if needed
        if self.opt.category == "all":
            # legacy all categories training
            if split == "train":
                # Use all categories, excluding test category
                data_frame = data_frame.loc[
                    (data_frame.synsetId != int(self.opt.test_category))
                    & (data_frame.split == split)
                ]
            else:
                # Use test category or others depending on eval_on_same
                data_frame = data_frame.loc[
                    (
                        (data_frame.synsetId == int(self.opt.test_category))
                        == self.opt.eval_on_same
                    )
                    & (data_frame.split == split)
                ]

                # Subsample for eval_on_same=False to match test_category size
                if not self.opt.eval_on_same:
                    test_cat_size = len(
                        pd.read_csv(self.opt.split_file).loc[
                            (
                                pd.read_csv(self.opt.split_file).synsetId
                                == int(self.opt.test_category)
                            )
                            & (pd.read_csv(self.opt.split_file).split == split)
                        ]
                    )
                    if test_cat_size > 0 and len(data_frame) > test_cat_size:
                        data_frame = data_frame.sample(
                            n=test_cat_size, random_state=42
                        ).reset_index(drop=True)

        else:
            # Single-category mode
            # Convert category name to synset ID
            if self.opt.category in self.CATEGORY2SYNSETOFFSET:
                category_synset = self.CATEGORY2SYNSETOFFSET[self.opt.category]
            # find instances with corresponding synset ids and splits
            data_frame = data_frame.loc[
                (data_frame.synsetId == int(category_synset))
                & (data_frame.split == split)
            ]

        names, categories = (
            data_frame.modelId.to_numpy(),
            data_frame.synsetId.to_numpy(),
        )
        return names, categories


    def _load_test_pairs(self):
        # legacy function for another setting
        with open(self.opt.test_pairs_file) as f:
            lines = f.read().splitlines()
        names = []
        partners = []
        for line in lines:
            name = line.split(" ")[0]
            partner = line.split(" ")[1]
            names += [name]
            partners += [partner]
        return names, partners

    def _filter_missing(self, names, categories):
        """
        Drop entries whose surface cloud or partial clouds are not on disk.
        This is used in case some paths do not contain point clouds.

        Done once at load time so that every index maps to a loadable object 
        instead of performing checks at access time
        """
        keep = []
        drops = []  # (name, reason), kept for reporting
        for n, c in zip(names, categories, strict=True):
            # check each name category pair for a valid full point cloud path
            if self._get_pointcloud_path(n, c) is None:
                drops.append((n, "no surface cloud"))
                continue
            # check partial point clouds
            reason = self._missing_partial_reason(n, c)
            if reason is not None:
                drops.append((n, reason))
                continue
            keep.append((n, c))

        if drops:
            # print warning message for first three dropped entries
            examples = "; ".join(f"{nm}: {why}" for nm, why in drops[:3])
            print(
                f"[WARN] {len(drops)}/{len(names)} entries have missing clouds under "
                f"{self.opt.points_dir}; dropped from the dataset "
                f"({examples}{'; ...' if len(drops) > 3 else ''})"
            )
        if not keep:
            raise RuntimeError(
                f"no usable entries for category {self.opt.category} under "
                f"{self.opt.points_dir} (checked {len(names)}): every entry is missing "
                f"a surface or a partial cloud - check partial_view_mode "
                f"'{self.opt.partial_view_mode}' and n_partial_samples "
                f"{self.n_partial_samples} against the samples that were generated"
            )
        # filter out only the kept entries
        names, categories = (list(t) for t in zip(*keep))
        return names, categories

    def _missing_partial_reason(self, name, category):
        """
        checking function, returns None if all point clouds can be found,
        otherwise returns a string with reasong for dropping the entry.
        """
        # when enumerating partial samples it will use whatever exists on disk
        if self.enumerate_partial_samples:
            # only need to check that the pair does have partial samples
            if not self._list_views(name, category):
                return f"no partial samples for modes {self.view_modes}"
            return None

        # if index is fixed only check for the specified index
        if self.opt.fixed_partial_sample_index is not None:
            ns = [self.opt.fixed_partial_sample_index]
        else:
            # other wise check all that are required
            ns = range(self.n_partial_samples)
        for mode in self.view_modes:
            for n in ns:
                if mode == 'full':
                    pth = self._models_dir(name, category) / "surface_samples_2048.npy"
                else:
                    pth = self._models_dir(name, category) / f"{self._partial_sample_stem(mode)}{n}.npy"
                if not pth.is_file():
                    return f"no {pth.name}"
        return None

    def _partial_sample_stem(self, mode):
        """
        shorthand for the stem of selected partial sample, excluding index
        change according to the naming scheme used for the partial samples on disk
        """
        # matches the filenames built in _get_partial_pointcloud_path
        if mode == 'uniform':
            return "partial_samples_uniform_dist_"
        return f"partial_samples_{mode}_"

    def _models_dir(self, name, category):
        """
        shorthand for the /models/ directory under the object instance's path, 
        where the point clouds live.
        """
        return Path(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
        )

    def load_dataset(self):
        """
        initial dataloading function
        returns dataset as a dict with individual keys
        with names (model ids) and categories (synset ids), 
        also views modes and indices if enumerating
        """
        dataset = {}
        # should really only be shapenet
        if self.opt.data_type == "shapenet":
            # load from split file csv
            # gets the names (model IDs) and categories (synset IDs)
            names, categories = self._load_from_split_file(self.opt.split)
        else:
            raise ValueError()

        assert len(names) > 0

        # sort together and keep pairing
        # only one category under normal settings
        paired = sorted(zip(names, categories, strict=True))
        names, categories = zip(*paired, strict=True)
        names = list(names)
        categories = list(categories)

        # a legacy setting; safe to ignore
        # if self.opt.load_test_pairs:
        #     names, partners = self._load_test_pairs()
        #     dataset["partners"] = partners
        # else:
        #     # names are already name-sorted by the pairing above
        #     names, categories = self._filter_missing(names, categories)

        # filter out entries where the point clouds are missing on disk
        names, categories = self._filter_missing(names, categories)

        dataset["name"] = names
        dataset["category"] = categories

        # to enumerate over all partial samples, index (object, view) pairs rather than just object,
        # so a pass over the dataset covers every partial sample (view) that exists on disk
        if self.enumerate_partial_samples:
            # list of tuples (model index, mode, sample index)
            dataset["views"] = self._build_view_index(names, categories)

        return dataset

    def _get_pointcloud_path(self, name, category=None):            
        # form path by attaching category and model id names to passed base directory
        # full point cloud path
        full_pc_filename = "surface_samples_2048.npy"   # <- change here for actual name of file on disk
        pth = os.path.join(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            full_pc_filename,
        )
        for _ in range(20): # small hack for issue where file system takes some time to access
        # while not Path(pth).is_file():
            pth = os.path.join(
                self.opt.points_dir,
                (str(category).zfill(8) if category is not None else self.opt.category),
                name,
                "models",
                full_pc_filename,
            )
            if Path(pth).is_file():
                return pth
            else:
                continue
        print(f"[WARN] Point cloud file not found after multiple attempts: {pth}")
        return None
        # raise Exception(f"Point cloud file not found: {pth}")

    def _get_partial_pointcloud_path(self, name, category=None, view=None):
        if view is not None:
            # views are already enumerated off disk, so it needs no random draw
            mode, n = view
        else:
            n = random.randint(0, self.n_partial_samples-1)
            nmode = random.randint(0, len(self.view_modes)-1)
            if self.opt.partial_view_mode == 'all':
                mode = self.view_modes[nmode]
            elif self.opt.partial_view_mode == 'uniform-combined':
                full_sample_thresh = 0.3
                full_sample_rand = random.random()
                mode = 'uniform' if full_sample_rand > full_sample_thresh else 'full'
            else:
                mode = self.opt.partial_view_mode

            # deterministic view selection: pin the sample index so the same object can be
            # loaded again under a known (mode, n) view rather than a random one
            if getattr(self.opt, "fixed_partial_sample_index", None) is not None:
                n = self.opt.fixed_partial_sample_index

        path_root = os.path.join(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
        )
        
        if mode == 'uniform':
            # different naming scheme for uniform partial samples
            partial_pc_path = os.path.join(
                path_root, 
                f"partial_samples_uniform_dist_{n}.npy",
                # different sample numbers are still in the same frame, no need to match
            )
            category_name = synsets_to_names([str(category).zfill(8)], Path(self.opt.points_dir) / "filtered_taxonomy.json")[0]
            partial_coverage_path = os.path.join(
                # self.opt.points_dir, 
                # f"uniform_sampling_metadata_{category_name}.csv",
                path_root, 
                f"coverage_{mode}.csv",
            )
        elif mode == 'full':    # shouldn't really be here, hack case for now
            partial_pc_path = os.path.join(
                path_root, 
                f"surface_samples_2048.npy",
                # f"new_samples_{n}.npy",
            )
            partial_coverage_path = None
        else:
            partial_pc_path = os.path.join(
                path_root, 
                f"partial_samples_{mode}_{n}.npy",
                # different sample numbers are still in the same frame, no need to match
            )
            partial_coverage_path = os.path.join(
                path_root, 
                f"coverage_{mode}.csv",
            )
        
        return partial_pc_path, partial_coverage_path, n, mode
        
    def _get_coverage_csv_path(self, name, category=None):
        if self.opt.partial_view_mode is None:
            raise ValueError("Partial view mode is not set.")
        return os.path.join(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            f"coverage_{self.opt.partial_view_mode}.csv",
        )
    
    def _get_mesh_path(self, name, category=None):
        return os.path.join(
            self.opt.mesh_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            "model_normalized.obj",
        )


    def random_downsample(self, point_cloud, target_num_points):
        """
        Downsamples a point cloud to the target number of points using 
        Farthest Point Sampling (FPS) for even spatial distribution.

        Args:
            point_cloud (ndarray): The input point cloud of shape (N, 3).
            target_num_points (int): The desired number of points.

        Returns:
            downsampled_point_cloud: The downsampled point cloud of shape (target_num_points, 3).
        """
        N = point_cloud.shape[0]
        if N <= target_num_points:
            return point_cloud  # Return original if already small enough

        # Initialize array to hold the indices of the selected points
        farthest_indices = np.zeros(target_num_points, dtype=int)

        # Array to keep track of the shortest distance from every point 
        # to the currently selected set of points. Initialize with infinity.
        distances = np.full(N, np.inf)

        # Select a random initial point to start the process
        farthest_indices[0] = np.random.randint(0, N)

        for i in range(1, target_num_points):
            # Get the coordinates of the last added point
            last_added_point = point_cloud[farthest_indices[i - 1]]

            # Calculate squared distances from the last added point to all other points
            # (Skipping the square root for performance, as we only need relative distances)
            dist_to_last = np.sum((point_cloud - last_added_point) ** 2, axis=1)

            # Update the minimum distance to the selected set for each point
            distances = np.minimum(distances, dist_to_last)

            # The next selected point is the one with the maximum minimum-distance
            farthest_indices[i] = np.argmax(distances)

        return point_cloud[farthest_indices]

    ############################# BEGIN partial enumeration functions ###############################
    def _list_views(self, name, category):
        """
        Finds every partial view of this object specified by view_modes that exists on disk
        returns a list of (mode, n) pairs, where n is the max index for the mode
        used for checking through partial samples and enumerating them
        """
        if (name, category) in self._views_cache:
            # load from cache if it has already been searched before
            return self._views_cache[(name, category)]

        root = self._models_dir(name, category)
        views = []
        # checks over each required view mode
        for mode in self.view_modes:
            # assumes full mode only has one sample
            if mode == 'full':
                if (root / "surface_samples_2048.npy").is_file():
                    views.append((mode, 0))
                    # or can change to still have randomness
                continue
            stem = self._partial_sample_stem(mode)
            # find the number of samples per instance
            ns = sorted(
                int(p.stem[len(stem):])
                for p in root.glob(f"{stem}*.npy")
                if p.stem[len(stem):].isdigit()
            )
            views += [(mode, n) for n in ns]
        # save to a cache to skip expensive glob if it is needed again
        self._views_cache[(name, category)] = views
        return views

    def _build_view_index(self, names, categories):
        """Flatten the split into one entry per (object, view)."""
        views = []  # (model index, mode, sample index)
        for i, (name, category) in enumerate(zip(names, categories, strict=True)):
            views += [(i, mode, n) for mode, n in self._list_views(name, category)]
        if not views:
            raise RuntimeError(
                f"no partial samples found for modes {self.view_modes} under "
                f"{self.opt.points_dir} (checked {len(names)} objects)"
            )
        counts = Counter(mode for _, mode, _ in views)
        print(
            f"[INFO] enumerating {len(views)} views over {len(names)} objects: "
            f"{dict(counts)}"
        )
        return views
    ############################# END partial enumeration functions ###############################

    def add_position_noise(self, shape, partial_shape):
        """
        Add i.i.d. gaussian jitter to the position of every point.

        std is `opt.point_noise_std`, in the units of the normalized cloud - this runs
        after normalization and after domain randomization so the perturbation keeps a
        fixed magnitude instead of being rescaled by the random scaling factor.
        Only the xyz positions are touched; normals and labels are left as loaded.
        """
        if self.point_noise_std <= 0:
            return shape, partial_shape
        if self.point_noise_target in ("both", "full"):
            shape = shape + torch.randn_like(shape) * self.point_noise_std
        if self.point_noise_target in ("both", "partial"):
            partial_shape = (
                partial_shape + torch.randn_like(partial_shape) * self.point_noise_std
            )
        return shape, partial_shape

    def get_item_by_name(
        self, name, category, is_test, sample_mesh=False, load_mesh=False, view=None
    ):
        pc_path = (
            Path(self._get_mesh_path(name, category)).parent
            / "point_resampled_labeled.npy"
        )

        # get point cloud
        # print(f"point_cloud: {self._get_pointcloud_path(name, category)}")
        # print(f"partial_point_cloud: {self._get_partial_pointcloud_path(name, category)}")
        points = np.load(self._get_pointcloud_path(name, category))
        if self.opt.num_point < points.shape[0]:
            points = self.random_downsample(points, self.opt.num_point)
        points = torch.from_numpy(points).float()

        # get partial point cloud
        partial_path, csv_path, n, mode_name = self._get_partial_pointcloud_path(name, category, view=view)
        partial = np.load(partial_path)
        if self.opt.num_point < points.shape[0]:
            points = self.random_downsample(points, self.opt.num_point)
        partial = torch.from_numpy(partial).float()

        # get partial point cloud coverage fraction from metadata csv
        if csv_path is not None:    # mode 'full' does not have a coverage csv, set coverage=1
            with open(csv_path, newline="") as f:
                # format and contents can differ between 'uniform' and other modes
                reader = csv.DictReader(f)
                coverage = None
                radius = None
                for row in reader:
                    c = row["coverage"]
                    # hack for crappy initial tac csv files
                    try:
                        p = row["partial_pc_path"]
                    except KeyError:
                        # 'uniform' mode case, the name of the model instance also needs to be matched
                        p = row["model_id"]
                        
                    # loop until the correct name is found
                    if name not in p:
                        print(name)
                        print(type(name))
                        print(p)
                        print(type(p))
                        continue
                    
                    try:
                        r = row["radius"]   # this is the radius between points used to calculate coverage (not camera radius)
                    except KeyError:
                        r = -1.0
                        
                    # search for the matching sample index, either from the path or as a separate field
                    if len(p) == 1:   # it is sample_index
                        if row["sample_index"] == n:
                            coverage = float(c)
                            radius = float(r)
                            break
                    elif len(p) > 1:  # it is partial_pc_path
                        m = re.search(r"_(\d+)\.npy$", p)
                        if m and int(m.group(1)) == n:
                            coverage = float(c)
                            radius = float(r)
                            break
                    else:
                        raise RuntimeError("Unknown case for searching pattern")
                # didn't find matching case by the end
                if coverage is None or radius is None:
                    raise RuntimeError(f"failed to find matching coverage values for {partial_path} in {csv_path}")
        else:
            coverage = 1    # case for mode 'full'
            radius = 0
        
        # normalize point clouds
        points[:, :3], partial[:, :3], norm_vals = self.normalize(points[:, :3], partial[:, :3])
        points = points.clone()
        partial = partial.clone()

        normals = points[:, 3:6].clone()
        label = points[:, -1].clone()
        shape = points[:, :3].clone()

        partial_normals = partial[:, 3:6].clone()
        partial_label = partial[:, -1].clone()
        partial_shape = partial[:, :3].clone()
        
        norm_val_0 = float(norm_vals[0]) if len(norm_vals[0].shape) <= 1 else norm_vals[0]
        norm_val_1 = float(norm_vals[1])
        
        # domain randomization options
        if self.dr:
            # convert points to homogeneous coordinates for transformation
            device = shape.device
            homogeneous_points = torch.cat(
                [shape, torch.ones(shape.shape[0], 1, device=device)], dim=-1
            )  # (N, 3) -> (N, 4)
            homogeneous_partial = torch.cat(
                [partial_shape, torch.ones(partial_shape.shape[0], 1, device=device)], dim=-1
            )  # (M, 3) -> (M, 4)
            deformation_matrix = torch.eye(4, device=device)
            # random scaling #####################################################################
            # symmetric log-uniform scaling for equal probability of scaling up and down #########
            # logsc = torch.log(torch.tensor(self.max_scaling_factor, device=device))
            # scaling_factors = torch.rand(1, device=device) * logsc * 2 - logsc
            # scaling_factors = torch.exp(scaling_factors)
            # uniform scaling ####################################################################
            scaling_factors = torch.empty(1, device=device).uniform_(
                1.0 / self.max_scaling_factor,
                self.max_scaling_factor,
            )
            ######################################################################################
            # scaling manual disable #############################################################
            scaling_factors = torch.ones(1, device=device)
            ######################################################################################
            # matrices act on column vectors (p' = M p), so the rightmost matrix is applied first:
            # p' = R * s * (p - mean) + mean + t
            centroid = torch.mean(shape, dim=0)
            # translate to zero mean first
            scale_matrix_1 = torch.eye(4, device=device)
            scale_matrix_1[:-1, -1] = -centroid
            # apply scaling
            scale_matrix_2 = torch.eye(4, device=device)
            scale_matrix_2[0, 0] = scaling_factors
            scale_matrix_2[1, 1] = scaling_factors
            scale_matrix_2[2, 2] = scaling_factors
            # random rotation, uniform over SO(3)
            rot_matrix = torch.eye(4, device=device)
            rot_matrix[:3, :3] = torch.tensor(
                Rotation.random().as_matrix(), device=device, dtype=torch.float32
            )
            # translate back to original mean
            scale_matrix_3 = torch.eye(4, device=device)
            scale_matrix_3[:-1, -1] = centroid

            sm = scale_matrix_3 @ rot_matrix @ scale_matrix_2 @ scale_matrix_1

            # random translation
            translation_offsets = torch.rand(3, device=device) * (self.max_translation_offset * 2) - self.max_translation_offset
            # translation manual disable ##########################
            translation_offsets = torch.zeros(3, device=device)
            #######################################################
            trans_matrix = torch.eye(4, device=device)
            trans_matrix[:-1, -1] = translation_offsets
            deformation_matrix = trans_matrix @ sm

            # normals only follow the rotation (uniform scaling/translation leave them unchanged).
            # point files without normals give empty (N, 0) slices, which are left as is
            if normals.shape[-1] == 3:
                normals = normals @ rot_matrix[:3, :3].T
            if partial_normals.shape[-1] == 3:
                partial_normals = partial_normals @ rot_matrix[:3, :3].T

            # Apply transformation
            deformed_cloud = homogeneous_points @ deformation_matrix.transpose(0, 1)
            deformed_partial = homogeneous_partial @ deformation_matrix.transpose(0, 1)
            # convert back to non-homogeneous coordinates
            shape = deformed_cloud[:, :3] / deformed_cloud[:, 3:]
            partial_shape = deformed_partial[:, :3] / deformed_partial[:, 3:]

        # per-point sensor-noise simulation, applied last so its magnitude is unaffected
        # by the domain randomization scaling above.
        # add_position_noise returns new tensors, so shape_clean keeps the pre-noise
        # positions (the same tensor when the jitter is off) and is handed out alongside
        # the noisy cloud: the encoder is trained on the noisy input while the diffusion
        # decoder reconstructs the clean surface. only the full cloud needs a clean copy
        # - the partial branch is already reconstructed/supervised against it
        shape_clean = shape
        shape, partial_shape = self.add_position_noise(shape, partial_shape)

        # build result dict
        result = {
            "shape": shape,
            "shape_clean": shape_clean,
            "normals": normals,
            "label": label,
            "partial_shape": partial_shape,
            "partial_normals": partial_normals,
            "partial_label": partial_label,
            "cat": self.opt.category,
            "file": name,
            "partial_sample_name": f"{mode_name}_{n}",
            "category": category,
            "coverage": coverage,
            "radius": radius,
            "normalization": self.opt.normalize, 
            "norm_val_0": norm_val_0, 
            "norm_val_1": norm_val_1, 
            "scaling_factor": scaling_factors.item() if self.dr else -1,
            "translation_offset": translation_offsets.cpu() if self.dr else -1,
            "rotation_matrix": rot_matrix[:3, :3].cpu() if self.dr else -1,
            "point_noise_std": self.point_noise_std,
        }
        # if pc_path.is_file() and is_test:
        #     pc = np.load(pc_path)
        #     if self.opt.num_point < points.shape[0]:
        #         points = self.random_downsample(points, self.opt.num_point)
        #     pc, _, _ = self.normalize(pc, pc)
        #     result.update({"sampled_points": pc})

        return result

    def get_sample(self, index):
        if self.enumerate_partial_samples:
            # index addresses a view, not an object
            obj_index, mode, n = self.dataset["views"][index]
            view = (mode, n)
        else:
            obj_index, view = index, None
        name = self.dataset["name"][obj_index]
        cat = self.dataset["category"][obj_index]

        sample_mesh = self.opt.sample_mesh or self.opt.points_dir is None
        is_test = self.opt.phase == "test"
        target_data = self.get_item_by_name(
            name,
            cat,
            is_test,
            load_mesh=self.opt.load_mesh,
            sample_mesh=sample_mesh,
            view=view,
        )

        result = {"target_" + k: v for k, v in target_data.items()}

        # only KeypointDeformer baseline requires two shapes, get a source shape
        if self.get_two:
            # draw over objects, not views - get_real_length counts views when enumerating
            index_2 = np.random.randint(len(self.dataset["name"]))
            if self.opt.load_cages_test_pairs or self.opt.load_test_pairs:  # default false
                name_2 = self.dataset["partners"][obj_index]
            else:
                name_2 = self.dataset["name"][index_2]
                cat_2 = self.dataset["category"][index_2]
            source_data = self.get_item_by_name(
                name_2, cat_2, is_test, load_mesh=self.opt.load_mesh, sample_mesh=sample_mesh
            )
            result.update({"source_" + k: v for k, v in source_data.items()})


        return result

    def __getitem__(self, index):
        for _ in range(10): # try 10 times to load a sample, in case of file access errors
            index = index % self.get_real_length()
            try:
                sample = self.get_sample(index)
            except Exception as e:
                print(e)
                print(f"Error loading sample {index}")
                warnings.warn(
                    f"Error loading sample {index}: "
                    + "".join(
                        traceback.format_exception(
                            # etype=type(e), value=e, tb=e.__traceback__
                            type(e), e, e.__traceback__
                        )
                    )
                )
                index += 1
                continue

            if self.transform:
                # original deformation transform
                transformed, deformed = self.transform(
                    {"coord": sample["target_shape"].cpu().numpy(),
                     "partial_coord": sample["target_partial_shape"].cpu().numpy()}
                )  # Apply transform
                
                # unused alt transform function
                # transformed, partial, deformed = self.transform(
                #     sample,
                # )  # Apply transform
                
                # also transform partial view point cloud
                # partial, partial_deformed = self.transform(
                #     {"coord": sample["target_partial_shape"].cpu().numpy()}
                # )
                sample = {
                    **sample,
                    **{
                        f"orig_{key}": value.cuda()
                        for key, value in transformed.items()
                    },
                    **{
                        f"deformed_{key}": value.cuda()
                        for key, value in deformed.items()
                    },
                    # **{
                    #     f"partial_orig_{key}": value.cuda()
                    #     for key, value in partial.items()
                    # },
                    # **{
                    #     f"partial_deformed_{key}": value.cuda()
                    #     for key, value in partial_deformed.items()
                    # },
                }
                
            for k, v in sample.items():
                if v is None:
                    print(k)
                    import pdb; pdb.set_trace()
            return sample

    @staticmethod
    def input_domain_randomization(self, points):
        # points are loaded as 
        # example randomization: random scaling
        scale = random.uniform(0.8, 1.2)
        for key in sample:
            if key.startswith("orig_") or key.startswith("deformed_"):
                sample[key] = sample[key] * scale
        return sample

    @classmethod
    def collate(cls, batch):
        batched = {}
        elem = batch[0]
        for key in elem:
            if key in cls.DO_NOT_BATCH:
                batched[key] = [e[key] for e in batch]
            else:
                try:
                    batched[key] = torch.utils.data.dataloader.default_collate(
                        [e[key] for e in batch]
                    )
                except Exception as e:
                    print(e)
                    continue
        return batched

    @staticmethod
    def uncollate(batched):
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.cuda()
            elif isinstance(v, container_abcs.Sequence) and isinstance(
                v[0], torch.Tensor
            ):
                batched[k] = [e.cuda() for e in v]
        return batched

    def get_real_length(self):
        # when enumerating partial samples, the dataset length is the number of views rather than the number of objects
        if self.enumerate_partial_samples:
            return len(self.dataset["views"])
        return len(self.dataset["name"])

    def __len__(self):
        if self.opt.fixed_target_index is not None:
            return 1000
        else:
            return self.get_real_length() * self.opt.multiply
