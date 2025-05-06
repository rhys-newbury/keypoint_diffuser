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

from ..utils.io import find_files, read_keypoints, read_pcd
from ..utils.utils import normalize_to_box


class Shapes(torch.utils.data.Dataset):
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
        "bag": "02773838",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "earphone": "03261776",
        "guitar": "03467517",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "motorbike": "03790512",
        "mug": "03797390",
        "pistol": "03948459",
        "rocket": "04099429",
        "skateboard": "04225987",
        "table": "04379243",
    }
    SYNSETOFFSET2CATEGORY = {v: k for k, v in CATEGORY2SYNSETOFFSET.items()}

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def normalize(self, x):
        if self.opt.normalize == "unit_box":
            pc, center, scale = normalize_to_box(x)
        else:
            raise ValueError()
        return pc, center, scale

    def __init__(self, opt, transform=None):
        self.opt = opt
        assert (
            self.opt.category == "all"
            and self.opt.test_category is not None
            and self.opt.test_category in self.CATEGORY2SYNSETOFFSET.values()
        ) or (
            self.opt.category in self.CATEGORY2SYNSETOFFSET.values()
            and self.opt.test_category is None
        )

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
            data_frame = data_frame.loc[
                (data_frame.synsetId == int(self.opt.category))
                & (data_frame.split == split)
            ]

        names, categories = (
            data_frame.modelId.to_numpy(),
            data_frame.synsetId.to_numpy(),
        )
        return names, categories

    def _load_from_files(self, split):
        if self.opt.category == "all":
            categories = self.CATEGORY2SYNSETOFFSET.values()
            test_cat = str(self.opt.test_category)

            if split == "train":
                categories = [c for c in categories if c != test_cat]
            else:
                categories = [test_cat]

            files = []
            for cat in categories:
                cat_files = find_files(
                    os.path.join(self.opt.points_dir, cat),
                    self.POINT_CLOUD_FILE_EXT,
                )
                files.extend(cat_files)
        else:
            files = find_files(
                os.path.join(self.opt.points_dir, self.opt.category),
                self.POINT_CLOUD_FILE_EXT,
            )

        # extract model names from file paths
        names = [x.split(os.path.sep)[-2] for x in files]
        names = sorted(names)
        return names

    def _load_seg_split_file(self, seg_split_file):
        with open(seg_split_file) as f:
            files = json.load(f)
            names = [x.split(os.path.sep)[-2:] for x in files]
            # filter out other categories
            names = [x[1] for x in names if x[0] == self.opt.category]
        return names

    def _load_seg_split(self):
        seg_split_file = os.path.join(
            self.opt.seg_split_dir, "shuffled_%s_file_list.json" % self.opt.split
        )
        return self._load_seg_split_file(seg_split_file)

    def _load_keypointnet_split(self, split_name):
        # load split
        with open(
            os.path.join(self.opt.keypointnet_dir, "splits", split_name + ".txt")
        ) as f:
            lines = f.read().splitlines()
        # line looks like this: 02691156-ecbb6df185a7b260760d31bf9510e4b7
        split = {
            x[len(self.opt.category) + 1 :]
            for x in lines
            if x.startswith(self.opt.category)
        }
        return split

    def _load_keypointnet(self):
        # load keypoints
        file_path = os.path.join(
            self.opt.keypointnet_dir,
            "annotations",
            self.SYNSETOFFSET2CATEGORY[self.opt.category] + ".json",
        )
        with open(file_path) as f:
            data = json.load(f)
        keypoints = {}
        for item in data:
            name = item["model_id"]
            keypoints_sample = [x["xyz"] for x in item["keypoints"]]
            keypoint_ids_sample = [x["semantic_id"] for x in item["keypoints"]]
            keypoints_sample = np.array(keypoints_sample, dtype=np.float32)
            keypoints[name] = (keypoints_sample, keypoint_ids_sample)

        if self.opt.keypointnet_common_keypoints:
            # get most common keypoint ids
            ids = [list(id_) for _, id_ in keypoints.values()]
            max_keypoints = len(set(itertools.chain(*ids)))
            # start with the highest number of keypoints
            success = False
            for n_common_keypoints in range(
                max_keypoints, self.opt.keypointnet_min_n_common_keypoints, -1
            ):
                most_common_ids = sorted(
                    [
                        x[0]
                        for x in Counter(itertools.chain(*ids)).most_common(
                            n_common_keypoints
                        )
                    ]
                )
                # prune keypoints
                pruned_keypoints = {}
                for name, (sample_keypoints, id_) in keypoints.items():
                    if set(most_common_ids).issubset(id_):
                        indices = [id_.index(x) for x in most_common_ids]
                        new_keypoints = sample_keypoints[indices]
                        pruned_keypoints[name] = new_keypoints
                if (
                    len(pruned_keypoints) / len(keypoints)
                    > self.opt.keypointnet_min_samples
                ):
                    success = True
                    break
            if not success:
                raise ValueError()
            keypoints = pruned_keypoints
        else:
            keypoints = {k: v[0] for k, v in keypoints.items()}

        return keypoints

    def _get_shapenet_id_to_model_id(self):
        data_frame = pd.read_csv(self.opt.split_file)
        return {k: v for k, v in zip(data_frame.id, data_frame.modelId, strict=False)}

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
        elif self.opt.data_type == "keypointnet":
            keypoints = self._load_keypointnet()
            names = list(self._load_keypointnet_split(self.opt.split))
            dataset["keypoints"] = keypoints
        elif self.opt.data_type == "shapenetseg":
            names = self._load_seg_split()
        elif self.opt.data_type == "files":
            names = self._load_from_files()
        else:
            raise ValueError()

        if self.opt.keypoints_gt_source == "keypointnet":
            keypoints = self._load_keypointnet()
            names = [x for x in names if x in keypoints]
            dataset["keypoints"] = keypoints
        assert len(names) > 0

        if self.opt.keypointnet_compatible and self.opt.split == "train":
            # remove keypointnet val and test from training
            val_split = self._load_keypointnet_split("val")
            test_split = self._load_keypointnet_split("test")
            names = set(names)
            names -= val_split
            names -= test_split

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

    def _get_mesh_path(self, name, category=None):
        return os.path.join(
            self.opt.mesh_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "models",
            "model_normalized.obj",
        )

    def _get_keypoints_path(self, name, category=None):
        return os.path.join(
            self.opt.keypoints_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            name,
            "keypoints.txt",
        )

    def _get_seg_points_path(self, name, category=None):
        return os.path.join(
            self.opt.segmentations_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            "points",
            name + ".pts",
        )

    def _get_seg_labels_path(self, name, category=None):
        return os.path.join(
            self.opt.segmentations_dir,
            (str(category).zfill(8) if category is not None else self.opt.category),
            "points_label",
            name + ".seg",
        )

    def _read_keypointnet_keypoints(self, name):
        keypoints = torch.from_numpy(self.dataset["keypoints"][name]).float()

        # fix axis
        keypoints = keypoints[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]])

        # compensate for their normalization
        # load associated point cloud
        pcd_path = os.path.join(
            self.opt.keypointnet_dir, "pcds", self.opt.category, name + ".pcd"
        )
        points = read_pcd(pcd_path)
        points = torch.from_numpy(points).float()
        points = points[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]])
        _, center, scale = self.normalize(points)
        keypoints = (keypoints - center) / scale

        return keypoints, center[0], scale[0]

    def _read_txt_keypoints(self, name):
        keypoints = read_keypoints(self._get_keypoints_path(name))
        keypoints = torch.from_numpy(keypoints).float()
        return keypoints

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

        points = np.load(self._get_pointcloud_path(name, category))
        points = torch.from_numpy(points).float()

        points[:, :3], center, scale = self.normalize(points[:, :3])
        points = points.clone()

        normals = points[:, 3:6].clone()
        label = points[:, -1].clone()
        shape = points[:, :3].clone()

        result = {
            "shape": shape,
            "normals": normals,
            "label": label,
            "cat": self.opt.category,
            "file": name,
            "category": category,
        }
        if pc_path.is_file() and is_test:
            pc = np.load(pc_path)
            pc = torch.from_numpy(self.random_downsample(pc, 5000))
            pc[:, :3] = (pc[:, :3] - center) / scale
            result.update({"sampled_points": pc})

        if self.opt.keypoints_gt_source == "keypointnet":
            (
                keypoints_gt,
                keypoints_gt_center,
                keypoints_gt_scale,
            ) = self._read_keypointnet_keypoints(name)
            result["keypoints_gt"] = keypoints_gt
            result["keypoints_gt_center"] = keypoints_gt_center
            result["keypoints_gt_scale"] = keypoints_gt_scale

        return result

    def get_sample(self, index):
        index_2 = np.random.randint(self.get_real_length())

        if self.opt.fixed_source_index is not None:
            index = self.opt.fixed_source_index

        if self.opt.fixed_target_index is not None:
            index_2 = self.opt.fixed_target_index

        name = self.dataset["name"][index]
        cat = self.dataset["category"][index]
        if self.opt.load_cages_test_pairs or self.opt.load_test_pairs:
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
                warnings.warn(
                    f"Error loading sample {index}: "
                    + "".join(
                        traceback.format_exception(
                            etype=type(e), value=e, tb=e.__traceback__
                        )
                    )
                )
                index += 1

            if self.transform:
                transformed, deformed = self.transform(
                    {"coord": sample["target_shape"].cpu().numpy()}
                )  # Apply transform
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
