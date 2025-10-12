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

from ..utils.utils import normalize_to_box_multi


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

    PARTIALSAMPLEMODES = ['default', 'myopia', 'patch', 'tac', 'all']

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def normalize(self, x, xp):
        if self.opt.normalize == "unit_box":
            pc, pcp, center, scale = normalize_to_box_multi(x, xp)
        else:
            raise ValueError()
        return pc, pcp, center, scale

    def __init__(self, opt, transform=None):
        self.opt = opt
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
        if self.opt.partial_view_mode == 'all':
            self.mode_list = [m for m in self.available_modes_list if m != 'all']
        else:
            self.mode_list = [self.opt.partial_view_mode]
        self.n_partial_samples = self.opt.n_partial_samples

        self.mesh_dir = opt.mesh_dir

        self.dataset = self.load_dataset()
        self.transform = transform  # Store transform function

        print("dataset size %d" % len(self))

    def _load_from_split_file(self, split):
        data_frame = pd.read_csv(self.opt.split_file)

        # Convert category name to synset ID if needed
        if self.opt.category == "all":
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

    def load_dataset(self):
        dataset = {}
        if self.opt.data_type == "shapenet":
            names, categories = self._load_from_split_file(self.opt.split)
        else:
            raise ValueError()

        assert len(names) > 0

        paired = sorted(zip(names, categories, strict=True))
        names, categories = zip(*paired, strict=True)
        names = list(names)
        categories = list(categories)

        if self.opt.load_test_pairs:
            names, partners = self._load_test_pairs()
            dataset["partners"] = partners
        else:
            names = sorted(names)

        dataset["name"] = names
        dataset["category"] = categories

        return dataset

    def _get_pointcloud_path(self, name, category=None):            
        return os.path.join(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            f"new_samples_{random.randint(0, 4)}.npy",
        )

    def _get_partial_pointcloud_path(self, name, category=None):
        n = random.randint(0, self.n_partial_samples)
        nmode = random.randint(0, len(self.mode_list)-1)
        if self.opt.partial_view_mode == 'all':
            mode = self.mode_list[nmode]
        else:
            mode = self.opt.partial_view_mode
        return os.path.join(
            self.opt.points_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            f"partial_samples_{mode}_{n}.npy",
            # different sample numbers are still in the same frame, no need to match
        ), n
    
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
        Randomly downsamples a point cloud to the target number of points.

        Args:
            point_cloud (ndarray): The input point cloud of shape (N, 3).
            target_num_points (int): The desired number of points.

        Returns:
            downsampled_point_cloud: The downsampled point cloud of shape (target_num_points, 3).
        """
        if point_cloud.shape[0] <= target_num_points:
            return point_cloud  # Return original if already small enough
        indices = np.random.choice(
            point_cloud.shape[0], target_num_points, replace=False
        )
        return point_cloud[indices]

    def get_item_by_name(
        self, name, category, is_test, sample_mesh=False, load_mesh=False
    ):
        pc_path = (
            Path(self._get_mesh_path(name, category)).parent
            / "point_resampled_labeled.npy"
        )

        # get point cloud
        # print(f"point_cloud: {self._get_pointcloud_path(name, category)}")
        # print(f"partial_point_cloud: {self._get_partial_pointcloud_path(name, category)}")
        points = np.load(self._get_pointcloud_path(name, category))
        points = self.random_downsample(points, 2048)
        points = torch.from_numpy(points).float()

        # get partial point cloud
        partial_path, n = self._get_partial_pointcloud_path(name, category)
        partial = np.load(partial_path)
        partial = self.random_downsample(partial, 2048)
        partial = torch.from_numpy(partial).float()

        # get partial point cloud coverage fraction
        if is_test:
            csv_path = self._get_coverage_csv_path(name, category)
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        partial_path, coverage, radius = row
                    except:
                        continue
                    m = re.search(r"_(\d+)\.npy$", partial_path)
                    if m and int(m.group(1)) == n:
                        coverage = float(coverage)
                        radius = float(radius)
                        break
        else:
            coverage = 0    # making them 0 instead of None so it works with training
            radius = 0
        
        # normalize point clouds
        points[:, :3], partial[:, :3], center, scale = self.normalize(points[:, :3], partial[:, :3])
        points = points.clone()
        partial = partial.clone()

        normals = points[:, 3:6].clone()
        label = points[:, -1].clone()
        shape = points[:, :3].clone()

        partial_normals = partial[:, 3:6].clone()
        partial_label = partial[:, -1].clone()
        partial_shape = partial[:, :3].clone()

        # build result dict
        result = {
            "shape": shape,
            "normals": normals,
            "label": label,
            "partial_shape": partial_shape,
            "partial_normals": partial_normals,
            "partial_label": partial_label,
            "cat": self.opt.category,
            "file": name,
            "category": category,
            "coverage": coverage,
            "radius": radius,
        }
        if pc_path.is_file() and is_test:
            pc = np.load(pc_path)
            pc = torch.from_numpy(self.random_downsample(pc, 5000))
            pc[:, :3] = (pc[:, :3] - center) / scale
            result.update({"sampled_points": pc})

        return result

    def get_sample(self, index):
        index_2 = np.random.randint(self.get_real_length())

        if self.opt.fixed_source_index is not None:
            index = self.opt.fixed_source_index

        if self.opt.fixed_target_index is not None:
            index_2 = self.opt.fixed_target_index

        name = self.dataset["name"][index]
        cat = self.dataset["category"][index]
        if self.opt.load_cages_test_pairs or self.opt.load_test_pairs:  # default false
            name_2 = self.dataset["partners"][index]
        else:
            name_2 = self.dataset["name"][index_2]
            cat_2 = self.dataset["category"][index_2]

        sample_mesh = self.opt.sample_mesh or self.opt.points_dir is None
        is_test = self.opt.phase == "test"
        target_data = self.get_item_by_name(
            name_2,
            cat_2,
            is_test,
            load_mesh=self.opt.load_mesh,
            sample_mesh=sample_mesh,
        )
        source_data = self.get_item_by_name(
            name, cat, is_test, load_mesh=self.opt.load_mesh, sample_mesh=sample_mesh
        )

        result = {"source_" + k: v for k, v in source_data.items()}
        result.update({"target_" + k: v for k, v in target_data.items()})

        return result

    def __getitem__(self, index):
        for _ in range(10):
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
                    {"coord": sample["target_shape"].cpu().numpy()}
                )  # Apply transform
                
                # unused alt transform function
                # transformed, partial, deformed = self.transform(
                #     sample,
                # )  # Apply transform
                
                # also transform partial view point cloud
                partial, _ = self.transform(
                    {"coord": sample["target_partial_shape"].cpu().numpy()}
                )
                
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
                    **{
                        f"partial_orig_{key}": value.cuda()
                        for key, value in partial.items()
                    },
                    # **{
                    #     f"partial_deformed_{key}": value.cuda()
                    #     for key, value in partial_deformed.items()
                    # },
                }
                
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
        return len(self.dataset["name"])

    def __len__(self):
        if self.opt.fixed_target_index is not None:
            return 1000
        else:
            return self.get_real_length() * self.opt.multiply
