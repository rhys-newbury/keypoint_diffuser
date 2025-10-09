import importlib
import inspect
import pkgutil
from pathlib import Path

from torch.utils.data import Dataset

import datasets


def discover_datasets():
    """
    Dynamically scan the `datasets` package for available dataset classes.
    A dataset class is considered valid if it subclasses torch.utils.data.Dataset.
    """

    dataset_classes = {}
    pkg_path = Path(datasets.__file__).parent

    for _, mod_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        try:
            module = importlib.import_module(f"datasets.{mod_name}")
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Dataset) and obj is not Dataset:
                dataset_classes[name] = obj

    return dataset_classes
