import json, re
from typing import Dict, Optional, List, Tuple

KEYS = {
    "table": 18,
    "car": 13,
    "airplane": 0,
    "cabinet": 9,
    "birdhouse": 53,
    "sofa": 48,
    "bus": 8,
    "chair": 14,
    "rifle": 45,
    "pot": 42,
    "vessel": 50,
    "bench": 5,
    "monitor": 17,
    "bathtub": 3,
    "knife": 30,
    "mailbox": 34,
    "faucet": 25,
    "telephone": 19,
    "bottle": 6,
    "lamp": 31,
    "tower": 21,
    "clock": 15,
    "speaker": 33,
    "microwave": 36,
    "bowl": 7,
    "remote_control": 44,
    "skateboard": 47,
    "tin_can": 20,
    "laptop": 32,
    "piano": 39,
    "cellphone": 52,
    "bed": 4,
    "printer": 43,
    "helmet": 28,
    "dishwasher": 16,
    "guitar": 27,
    "can": 10,
    "bookshelf": 54,
    "file": 26,
    "train": 22,
    "jar": 29,
    "mug": 38,
    "washer": 51,
    "motorbike": 37,
    "pistol": 41,
    "stove": 49,
    "camera": 11,
    "pillow": 40,
    "earphone": 24,
    "bag": 1,
    "basket": 2,
    "keyboard": 23,
    "cap": 12,
    "rocket": 46,
    "microphone": 35,
}

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

def name_to_label_for_keys(taxonomy_name: str) -> int:
    """Map taxonomy display name -> KEYS label id (with alias fallback)."""
    nm = _norm(taxonomy_name)
    # first try exact (normalized) key
    for k in KEYS.keys():
        if _norm(k) == nm:
            return KEYS[k]
    # fallback via alias map
    for tname, keys_name in KEYS_ALIASES.items():
        if _norm(tname) == nm:
            return KEYS[keys_name]
    raise KeyError(f"No KEYS label for taxonomy name {taxonomy_name!r}")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("_", " ").replace("-", " ")).strip()

def load_taxonomy_maps(taxonomy_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      name2synset: {normalized_name -> synsetId}
      synset2name: {synsetId -> canonical display name (as in taxonomy)}
    """
    with open(taxonomy_path, "r") as f:
        rows = json.load(f)  # [{"synsetId": "...", "name": "...", ...}, ...]
    name2synset = {_norm(r["name"]): r["synsetId"] for r in rows}
    synset2name = {r["synsetId"]: r["name"] for r in rows}
    return name2synset, synset2name

def names_to_synsets(names: List[str], taxonomy_path: str, 
                     aliases: Optional[Dict[str, str]] = KEYS_ALIASES) -> List[str]:
    """
    Map a list of class names (user input) to synsetIds using taxonomy.
    """
    name2synset, _ = load_taxonomy_maps(taxonomy_path)
    out = []
    for n in names:
        k = _norm(n)
        if aliases and _norm(n) in {_norm(a) for a in aliases.keys()}:
            k = _norm(aliases[n])  # map alias to canonical name for taxonomy lookup
        if k not in name2synset:
            raise KeyError(f"Unknown class name: {n!r}")
        out.append(name2synset[k])
    return out

def synsets_to_names(
    synsets: List[str],
    taxonomy_path: str,
) -> List[str]:
    """
    Reverse of names_to_synsets: map a list of synsetIds to taxonomy display names.
    """
    _, synset2name = load_taxonomy_maps(taxonomy_path)
    out: List[str] = []
    for s in synsets:
        sid = str(s).strip()
        name = synset2name.get(sid)
        if name is None:
            raise KeyError(f"Unknown synsetId: {sid!r}")
        else:
            out.append(name)
    return out