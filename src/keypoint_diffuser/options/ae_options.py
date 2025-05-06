from dataclasses import dataclass

import configargparse

from .. import datasets


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
    n_iterations: int = 200000
    save_interval: int = 100
    log_interval: int = 10
    latent_dim: int = 8
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

    use_old: bool = False
    use_edm: bool = False

    segmentations_dir: str = None
    seg_split_dir: str = None
    keypointnet_dir: str = None
    keypointnet_compatible: str = None
    keypointnet_common_keypoints: bool = False
    keypointnet_min_n_common_keypoints: int = 6
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
    lambda_3: int = 1
    lambda_4: int = 1


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
        for field_name, field_def in AEConfig.__dataclass_fields__.items():
            parser.add_argument(
                f"--{field_name}", type=field_def.type, default=field_def.default
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

        # if not skip_model:

        dataset_name = opt.dataset
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)
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
