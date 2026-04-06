"""
Zero-shot baseline evaluation on all target domains.

Usage:
    uv run python scripts/eval_baseline.py
    uv run python scripts/eval_baseline.py --hparams hparams_fast.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.hparams import load
from src.eval import run_baseline_eval

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, help="Path to hparams yaml (default: hparams.yaml)")
args = parser.parse_args()

hp = load(args.hparams)
best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"

if not best_weights.exists():
    print(f"[ERROR] No weights found at {best_weights}. Train first.")
    sys.exit(1)

results = run_baseline_eval(best_weights, hp)

print(f"\n{'Dataset':<14} {'mAP50':>10} {'mAP50-95':>12}")
print("-" * 38)
for ds, r in results.items():
    print(f"{ds:<14} {r['mAP50']:>10.4f} {r['mAP50_95']:>12.4f}")
