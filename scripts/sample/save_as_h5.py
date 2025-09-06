import argparse
import numpy as np
import json, re
import h5py
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from synset_utils import names_to_synsets, load_taxonomy_maps, name_to_label_for_keys

def _ensure_n_points(pc: np.ndarray, n: int = 2048) -> np.ndarray:
    if pc.shape[0] == n:
        return pc.astype(np.float32, copy=False)
    if pc.shape[0] > n:
        idx = np.random.choice(pc.shape[0], n, replace=False)
        return pc[idx].astype(np.float32, copy=False)
    idx = np.random.choice(pc.shape[0], n, replace=True)
    return pc[idx].astype(np.float32, copy=False)

def _model_id_from_rel(rel_path: str) -> str:
    parts = rel_path.strip("/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else rel_path.strip("/")

def pack_partials_to_h5(
    save_root_dir: Path,
    folders: List[str],
    mode: str,
    dataset_name: str = "shapenetcorev2",
    num_point: int = 2048,
    splits_list: List[str],
    shard_size: int = 8192,
    taxonomy_path: str = "/mnt/slow/shapenetcorev2-source/filtered_taxonomy.json",
    include_only_names: Optional[List[str]] = None,  # list of class names to include
    alias_for_names: Optional[Dict[str, str]] = None,  # user-input alias map
):
    """
    Creates <dst>/<dataset_name>_hdf5_{num_point}/{split}{k}.h5 plus sidecar JSONs.
    Filters classes by names if include_only_names is provided.
    Labels are assigned via KEYS to match H5Dataset.
    """
    name2synset, synset2name = load_taxonomy_maps(taxonomy_path)
    allowed_synsets: Optional[set] = None
    if include_only_names:
        sel = names_to_synsets(include_only_names, taxonomy_path, aliases=alias_for_names)
        allowed_synsets = set(sel)

    out_root = save_root_dir / f"{dataset_name}_hdf5_{num_point}"
    out_root.mkdir(parents=True, exist_ok=True)

    data_batch, label_batch, name_batch, file_batch, split_batch = [], [], [], [], []
    shard_idx, total = 0, 0

    def _flush_shard():
        nonlocal shard_idx, total, data_batch, label_batch, name_batch, file_batch
        if not data_batch:
            return
        data = np.stack(data_batch, axis=0).astype(np.float32)
        labels = np.asarray(label_batch, dtype=np.int64).reshape(-1, 1)

        h5_path = out_root / f"{split}{shard_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data) # should be <f4 4 bytes little-endian floating-point
            f.create_dataset("label", data=labels)  # should be <i8 8 bytes little-endian signed integer

        with open(out_root / f"{split}{shard_idx}_id2name.json", "w") as jf:
            json.dump(name_batch, jf)
        with open(out_root / f"{split}{shard_idx}_id2file.json", "w") as jf:
            json.dump(file_batch, jf)

        print(f"[H5] Wrote {h5_path.name}: {len(data_batch)} samples")
        shard_idx += 1
        total += len(data_batch)
        data_batch, label_batch, name_batch, file_batch = [], [], [], []

    for idx, rel in enumerate(tqdm(folders, desc="[H5] Packing", total=len(folders))):
        # rel like "02691156/87d764f7..." (synset/model)
        parts = rel.strip("/").split("/")
        if len(parts) < 2:
            continue
        synset = parts[0]
        if allowed_synsets is not None and synset not in allowed_synsets:
            continue
        if synset not in synset2name:
            print(f"[H5] Skip '{rel}' (synset not in taxonomy)")
            continue

        tax_name = synset2name[synset]            # display name for id2name.json
        try:
            label_id = name_to_label_for_keys(tax_name)  # must match H5Dataset KEYS
        except KeyError as e:
            print(f"[H5] Skip '{rel}': {e}")
            continue

        model_id = _model_id_from_rel(rel)
        models_dir = save_root_dir / rel / "models"
        if not models_dir.is_dir():
            continue

        for n in range(5):
            npy_path = models_dir / f"partial_samples_{mode}_{n}.npy"
            if not npy_path.is_file():
                continue
            pc = np.load(npy_path)
            pc = _ensure_n_points(pc, num_point)

            data_batch.append(pc)
            label_batch.append(label_id)
            name_batch.append(tax_name)   # keep taxonomy display name
            file_batch.append(model_id)
            split_batch.append(splits_list[idx])

            if len(data_batch) >= shard_size:
                _flush_shard()

    _flush_shard()
    print(f"[H5] Done. Total samples packed: {total}")

TRAINABLE = [
    "airplane",
    "bed",
    "bottle",
    "cap",
    "car",
    "chair",
    "guitar",
    "helmet",
    "knife",
    "motorbike",
    "mug",
    "table",
    "vessel",
]

# map taxonomy names <-> KEYS names (when they differ)
KEYS_ALIASES = {
    "motorcycle": "motorbike",
    "loudspeaker": "speaker",
    "cell phone": "cellphone",
    "computer keyboard": "keyboard",
    "display": "monitor",
    "can": "tin_can",
    # also the other direction
    "motorbike": "motorcycle",
    "speaker": "loudspeaker",
    "cellphone": "cell phone",
    "keyboard": "computer keyboard",
    "monitor": "display",
    "tin_can": "can",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/mnt/slow/shapenetcorev2-source", help="Root directory for the source mesh.")
    parser.add_argument("--dst", type=str, default="/mnt/slow/shapenetcorev2-h5", help="Root directory for saving the resampled point clouds.")
    args = parser.parse_args()
    
    data_root_dir = Path(args.src)
    save_root_dir = Path(args.dst)
    tax_dir = data_root_dir / args.taxonomy
    
    synset_ids = names_to_synsets(TRAINABLE, tax_dir)
    folders = set(synset_ids)   # <-- set of synset IDs
    
    # gather all folders to run
    folders_to_run = []
    for l in open(data_root_dir / "list.txt"):
        if l.strip().split("/")[1] in folders:
            folders_to_run.append(l.strip())
            
    # instead of gathering folders from list.txt, build from split.csv which also contains split info
    # each row of split.csv:
    # id,synsetId,subSynsetId,modelId,split
    folders_to_run = []
    splits_list = []
    for line in open(data_root_dir / "split.csv"):
        parts = line.strip().split(",")
        if len(parts) != 5:
            # print(f"Skipping invalid line in split.csv: {line.strip()}")
            continue
        _, synset_id, _, model_id, split = parts
        folders_to_run.append(f"./{synset_id}/{model_id}")
        splits_list.append(split)
    
    pack_partials_to_h5(
        save_root_dir=Path("/mnt/slow/shapenetcorev2-h5"),
        folders=folders_to_run,
        mode=args.mode,
        dataset_name=args.dataset_name,
        splits_list=splits_list,
        shard_size=args.shard_size,
        taxonomy_path=args.taxonomy,
        include_only_names=TRAINABLE,
        alias_for_names=KEYS_ALIASES

    )