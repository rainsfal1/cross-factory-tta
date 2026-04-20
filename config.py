from pathlib import Path

_ROOT = Path(__file__).parent

# ── Paths ────────────────────────────────────────────────────────────────────
# Raw dataset sources — relative to repo root data/ folder (or symlink target)
SH17_DIR   = _ROOT / "data" / "sh17"
PICTOR_DIR = _ROOT / "data" / "pictor_ppe"
SHWD_DIR   = _ROOT / "data" / "shwd"
CHV_DIR    = _ROOT / "data" / "chv" / "CHV_dataset"

# Working directories (prepared data, checkpoints, eval outputs)
DATA_DIR = _ROOT / "data" / "prepared"
RUNS_DIR = _ROOT / "runs"

# ── Class mappings ───────────────────────────────────────────────────────────
SH17_CLASSES = [
    "Coverall", "Face Shield", "Gloves", "Goggles", "Hard Hat",
    "No Coverall", "No Face Shield", "No Gloves", "No Goggles", "No Hard Hat",
    "No Safety Boot", "No Safety Vest", "No Vest", "Safety Boot", "Safety Vest",
    "Vest", "person",
]

CANONICAL_CLASSES = ["hard_hat", "no_hard_hat", "person"]

# SH17 indices for the 3 canonical target-domain classes
SH17_EVAL_IDX = {4: "hard_hat", 9: "no_hard_hat", 16: "person"}

# Canonical → SH17 index mapping (used for label remapping during eval)
CANONICAL_TO_SH17 = {0: 4, 1: 9, 2: 16}
