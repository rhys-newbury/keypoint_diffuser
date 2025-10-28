from dataclasses import dataclass
from pathlib import Path

import configargparse
import numpy as np
from datasets.discovery import discover_datasets


AVAILABLE_DATASETS = discover_datasets()
dataset_choices = sorted(AVAILABLE_DATASETS.keys())


THOUSAND = 1000


@dataclass
class AEConfig:
    name: str
    category: str
    dataset: str = "shapes"
    num_point: int = 2048
    points_dir: str = None
    dim: int = 3
    log_dir: str = "./log"
    subdir: str = "test"
    batch_size: int = 48
    kl_warmup_steps: int = 10000
    fps_steps: int = 1000
    max_schedule: int = 100000
    print_options: bool = False
    phase: str = "train"
    iteration: int = None
    epochs: int = 100
    save_interval: int = 100
    log_interval: int = 10
    key_points: int = 10
    extra_latent: int = 5
    num_steps: int = 200
    beta_1: float = 1e-4
    beta_t: float = 0.05
    sched_mode: str = "linear"
    flexibility: float = 0.0
    residual: bool = False
    resume: str = None
    lr: float = 1e-3
    weight_decay: float = 0
    max_grad_norm: float = 10
    end_lr: float = 1e-3
    sched_start_epoch: int = 150 * THOUSAND
    sched_end_epoch: int = 300 * THOUSAND
    normalization: str = "none"
    seed: int = 0
    n_workers: int = 0
    ckpt: str = None
    mesh_dir: str = None
    keypoints_dir: str = None

    db: Path = Path("results.db")
    ckpt_dir: Path = Path(".")

    use_old: bool = False
    use_edm: bool = True

    segmentations_dir: str = None
    seg_split_dir: str = None
    keypointnet_dir: str = None
    keypointnet_compatible: str = None
    keypointnet_commokey_points: bool = False
    keypointnet_min_n_commokey_points: int = 6
    keypointnet_min_samples: float = 0.8
    keypoints_gt_source: str = None
    data_type: str = "shapenet"
    split_file: str = None
    split: str = None
    fixed_source_index: int = None
    fixed_target_index: int = None
    normalize: str = "unit_box"
    multiply: int = 1
    load_cages_test_pairs: bool = False
    load_test_pairs: bool = False
    load_mesh: bool = False
    sample_mesh: bool = False
    test_pairs_file: str = None

    test_category: str = None
    eval_on_same: bool = False

    lambda_0: int = 1
    lambda_1: int = 1
    lambda_2: int = 1
    lambda_3: int = 10
    lambda_4: int = 1

    max_stretch_factor: float = 1.4
    max_bending_factor: float = 0.3
    max_twist_factor: float = 0.7
    max_taper_factor: float = 0.35
    max_rotation_angle: float = np.pi / 3


class AEOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument(
            "-c",
            "--config",
            required=False,
            is_config_file=True,
            help="config file path",
        )
        parser.add_argument(
            "-t",
            "--test_config",
            required=False,
            is_config_file=True,
            help="test config file path",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            choices=dataset_choices,
            default=dataset_choices[0] if dataset_choices else None,
            help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
        )

        for field_name, field_def in AEConfig.__dataclass_fields__.items():
            if field_name == "dataset":
                continue

            arg = f"--{field_name}"
            ftype = field_def.type
            default = field_def.default

            # special handling for booleans
            if ftype is bool:
                # e.g. --residual / --no-residual
                parser.add_argument(
                    arg,
                    dest=field_name,
                    action="store_true",
                    default=default,
                    help=f"set {field_name}=True",
                )
                parser.add_argument(
                    f"--no-{field_name}",
                    dest=field_name,
                    action="store_false",
                    help=f"set {field_name}=False",
                )
            else:
                parser.add_argument(
                    arg,
                    type=ftype,
                    default=default,
                )

        self.initialized = True
        return parser

    def gather_options(self, args=None, skip_model=False, unknown_ok=False):
        if not self.initialized:
            parser = configargparse.ArgumentParser(
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        else:
            raise RuntimeError()

        opt, _ = parser.parse_known_args(args)

        opt, unknown = (
            parser.parse_known_args(args)
            if unknown_ok
            else (parser.parse_args(args), [])
        )

        self.parser = parser
        return opt, unknown

    def parse(self, args=None, skip_model=False, unknown_ok=False) -> AEConfig:
        opt, unknown = self.gather_options(
            args, skip_model=skip_model, unknown_ok=unknown_ok
        )

        if opt.print_options:
            self.print_options(opt)

        if unknown:
            self.print_unknown(unknown)

        if opt.phase == "test":
            assert opt.ckpt is not None

        if opt.normalization == "none":
            opt.normalization = None

        opt_dict = vars(opt)
        opt_dict.pop("config", None)  # Remove config file path
        opt_dict.pop("test_config", None)  # Remove test config file path

        return AEConfig(**opt_dict)

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            message += f"{k:>25}: {v:<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)

    def print_unknown(self, unknown):
        print("----------------- Unknown options ---------------")
        print(", ".join([item for item in unknown if item.startswith("-")]))
        print("----------------- End -------------------")
