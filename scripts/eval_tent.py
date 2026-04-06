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
parser.add_argument("--hparams", default=None)
args = parser.parse_args()

hp = load(args.hparams)
best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"

if not best_weights.exists():
    print(f"ERROR: no weights at {best_weights}")
    sys.exit(1)

W = 46
print()
print(f"  Baseline (Pictor-PPE)  |  {hp['run_name']}")
print(f"  {'=' * W}")
print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}")
print(f"  {'-' * W}")

baseline_results = run_baseline_eval(best_weights, hp)

print(f"  {'-' * W}")
for ds, r in baseline_results.items():
    print(f"  {ds:<16} {r['mAP50']:>9.4f}  {r['mAP50_95']:>10.4f}")

print()
print(f"  TENT Adaptation  |  pictor_ppe  |  steps={hp['tent']['steps']}  lr={hp['tent']['lr']}")
print(f"  {'=' * W}")
print(f"  {'Method':<16} {'mAP50':>9}  {'mAP50-95':>10}")
print(f"  {'-' * W}")

tent_result = run_tent_eval(best_weights, baseline_results, hp)

b = baseline_results.get("pictor_ppe", {})
print(f"  {'-' * W}")
print(f"  {'Baseline':<16} {b['mAP50']:>9.4f}  {b['mAP50_95']:>10.4f}")
print(f"  {'TENT':<16} {tent_result['mAP50']:>9.4f}  {tent_result['mAP50_95']:>10.4f}")
print(f"  {'-' * W}")
recovery = tent_result["recovery"]
sign = "+" if recovery >= 0 else ""
print(f"  {'Recovery':<16} {sign}{recovery:.4f}")
print(f"  {'=' * W}")
print()
