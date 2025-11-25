#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import csv
import re
from typing import Iterable, Dict, Any, List, Sequence

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
SYNSETOFFSET2CATEGORY = {v: k for k, v in CATEGORY2SYNSETOFFSET.items()}


DEFAULT_SPLITS = ("test", "train", "val")

def parse_path_entry(s: str) -> Dict[str, Any]:
    """
    Accepts a string like '03001627/b24ed8...6695c.npy' and returns
    {'synsetId': '03001627', 'modelId': 'b24ed8...6695c'}
    """
    s = s.strip().lstrip("./")
    parts = s.split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse synset/model from entry: {s}")
    synset = parts[0]
    fname = Path(parts[1]).name
    model_id = re.sub(r"(\.[A-Za-z0-9]+)+$", "", fname)  # strip one or more extensions
    return {"synsetId": synset, "modelId": model_id}

def row_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts a dict possibly containing id, subSynsetId, synsetId, modelId.
    Drops id and subSynsetId. Falls back to parsing from 'dir' if provided.
    """
    out: Dict[str, Any] = {}
    if "synsetId" in d and d["synsetId"] is not None:
        out["synsetId"] = str(d["synsetId"])
    elif "dir" in d and isinstance(d["dir"], str):
        parsed = parse_path_entry(d["dir"])
        out["synsetId"] = parsed["synsetId"]
        out["modelId"] = parsed["modelId"]

    if "modelId" in d and d["modelId"] is not None:
        out["modelId"] = str(d["modelId"])
    return out

def iter_rows_from_json(js: Any) -> Iterable[Dict[str, Any]]:
    """
    Yields rows with (synsetId, modelId) from either:
      - list[str] of 'synset/model.ext'
      - list[dict] with keys like synsetId/modelId (id/subSynsetId dropped)
    """
    if not isinstance(js, list):
        raise ValueError("Expected top-level JSON array.")
    for item in js:
        if isinstance(item, str):
            yield parse_path_entry(item)
        elif isinstance(item, dict):
            row = row_from_dict(item)
            if "synsetId" in row and "modelId" in row:
                yield row
            else:
                # try to parse from a 'path'-like field if present
                for k in ("path", "file", "filepath"):
                    if k in item and isinstance(item[k], str):
                        yield parse_path_entry(item[k])
                        break
        else:
            continue

def scan_split_dir(root: Path, split: str, pattern: str) -> List[Path]:
    d = root / split
    return sorted(d.glob(pattern)) if d.is_dir() else []

def main():
    ap = argparse.ArgumentParser(
        description="Collect split JSONs from test/train/val into a single CSV (synsetId, modelId, split)."
    )
    ap.add_argument("--root", type=Path, help="Root directory containing split subdirs: test, train, val.", default=Path("/mnt/slow/shapenetcorev2-h5/shapenetcorev2_hdf5_2048"))
    ap.add_argument("-o", "--output", type=Path, default=Path("splits_out.csv"),
                    help="Output CSV path (default: splits_out.csv)")
    ap.add_argument("--glob", type=str, default="*_id2file.json",
                    help="Glob for JSON files inside each split subdir (default: *.json)")
    ap.add_argument("--splits", type=str, default="test,train,val",
                    help="Comma-separated list of split subdirs to scan (default: test,train,val)")
    ap.add_argument("--dedup", action="store_true",
                    help="Drop duplicate (synsetId, modelId, split) rows.")
    args = ap.parse_args()

    # import pdb; pdb.set_trace()
    splits: Sequence[str] = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(DEFAULT_SPLITS)

    all_rows: List[Dict[str, str]] = []
    total_files = 0

    for split in splits:
        json_files = scan_split_dir(args.root, split, args.glob)
        total_files += len(json_files)
        for jf in json_files:
            with jf.open("r") as f:
                data = json.load(f)
            for r in iter_rows_from_json(data):
                synset = r["synsetId"]
                if synset not in SYNSETOFFSET2CATEGORY:
                    continue
                model = r["modelId"]
                all_rows.append({"synsetId": synset, "modelId": model, "split": split})

    if args.dedup:
        seen = set()
        uniq = []
        for r in all_rows:
            key = (r["synsetId"], r["modelId"], r["split"])
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        all_rows = uniq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["synsetId", "modelId", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Scanned {total_files} JSON files across splits {splits}. Wrote {len(all_rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
