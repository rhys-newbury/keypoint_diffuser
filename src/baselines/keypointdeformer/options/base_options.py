from pathlib import Path

import configargparse
from baselines.keypointdeformer.models.cage_skinning import CageSkinning
from datasets.discovery import discover_datasets


class BaseOptions:
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

        # basic parameters
        AVAILABLE_DATASETS = discover_datasets()
        dataset_choices = sorted(AVAILABLE_DATASETS.keys())

        parser.add_argument(
            "--dataset",
            type=str,
            choices=dataset_choices,
            default=dataset_choices[0] if dataset_choices else None,
            help=f"Dataset class to use. Choices: {', '.join(dataset_choices)}",
        )

        parser.add_argument("--category", required=False, type=str, help="object name")
        parser.add_argument("--db", type=Path, help="Database path")

        parser.add_argument("--batch_size", type=int, help="batch size", default=16)
        parser.add_argument("--key_point", type=int, help="", default=10)
        parser.add_argument("--cage_size", type=float, default=1.4, help="")

        parser.add_argument("--lr", type=float, help="learning rate", default=0.001)

        parser.add_argument("--seed", type=int, default=0, help="")
        parser.add_argument("--n_workers", type=int, default=4, help="")
        parser.add_argument("--iteration", type=int, default=None, help="")
        parser.add_argument("--epochs", type=int, default=80, help="")

        parser.add_argument(
            "--bottleneck_size", type=int, help="bottleneck size", default=256
        )
        parser.add_argument("--ckpt_dir", type=Path, default=Path("."), help="")
        parser.add_argument(
            "--normalization",
            type=str,
            choices=["batch", "instance", "none"],
            default="none",
        )

        parser.add_argument(
            "--disable_d_residual", dest="d_residual", action="store_false"
        )
        self.initialized = True
        return parser

    def gather_options(self, args=None, skip_model=False, unknown_ok=False):
        # initialize parser with basic options
        if not self.initialized:
            parser = configargparse.ArgumentParser(
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        else:
            raise RuntimeError()

        # get the basic options
        opt, _ = parser.parse_known_args(args)

        if not skip_model:
            # modify model-related parser options
            parser = CageSkinning.modify_commandline_options(parser)
            opt, _ = parser.parse_known_args(args)  # parse again with the new defaults

        # # modify dataset-related parser options

        if unknown_ok:
            opt, unknown = parser.parse_known_args(args)
        else:
            opt = parser.parse_args(args)
            unknown = []

        self.parser = parser

        return opt, unknown

    def print_unknown(self, unknown):
        message = ""
        message += "----------------- Unknown options ---------------\n"
        for item in unknown:
            if item.startswith("-"):
                message += f"{item}, "
        message += "\n"
        message += "----------------- End -------------------"

    def parse(self, args=None, skip_model=False, unknown_ok=False):
        opt, unknown = self.gather_options(
            args, skip_model=skip_model, unknown_ok=unknown_ok
        )

        if unknown:
            self.print_unknown(unknown)

        if opt.normalization == "none":
            opt.normalization = None

        self.opt = opt

        return self.opt
