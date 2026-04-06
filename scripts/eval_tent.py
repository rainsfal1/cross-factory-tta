"""
TENT adaptation evaluation on Pictor-PPE.

Usage:
    uv run python scripts/eval_tent.py
    uv run python scripts/eval_tent.py --hparams hparams_fast.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.hparams import load
from src.eval import run_baseline_eval, run_tent_eval

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, help="Path to hparams yaml (default: hparams.yaml)")
args = parser.parse_args()

hp = load(args.hparams)
best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"

if not best_weights.exists():
    print(f"[ERROR] No weights found at {best_weights}. Train first.")
    sys.exit(1)

print("Running baseline on Pictor for comparison...")
baseline_results = run_baseline_eval(best_weights, hp)

run_tent_eval(best_weights, baseline_results, hp)
