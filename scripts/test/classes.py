try:
    from baselines.keypointdeformer.test_kpd import KPD
except ImportError:
    KPD = None
try:
    from baselines.sc3k.test_sc3k import SC3K
except ImportError:
    SC3K = None
try:
    from baselines.skeleton_merger.test_sm import SM
except ImportError:
    SM = None

try:
    from baselines.key_grid.test_keygrid import KeyGrid
except ImportError:
    KeyGrid = None

try:
    from baselines.diffusion_point_cloud.test_dpm import DPM
except ImportError:
    DPM = None
try:
    from keypoint_diffuser.test_ours import Ours
except ImportError:
    Ours = None

MODEL_CLASSES = {
    "SC3K": SC3K,
    "SM": SM,
    "KPD": KPD,
    "KeyGrid": KeyGrid,
    "KeyGridOrig": KeyGrid,
    "DPM": DPM,
    "Ours": Ours
    # Add more models here:
}
