"""
Zero-shot baseline evaluation on all target domains.

Usage:
    uv run python scripts/eval_baseline.py
    uv run python scripts/eval_baseline.py --hparams hparams_fast.yaml
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.hparams import load
from src.eval import run_baseline_eval, run_sh17_sanity

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
print(f"  Source Domain Verification  |  SH17 val")
print(f"  {'=' * W}")
print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}  {'Status':>8}")
print(f"  {'-' * W}")

sanity = run_sh17_sanity(best_weights, hp)
status = "OK" if sanity["mAP50"] >= hp["eval"].get("sanity_thresh", 0.60) else "WARN: unexpectedly low"
print(f"  {'sh17 (val)':<16} {sanity['mAP50']:>9.4f}  {sanity['mAP50_95']:>10.4f}  {status:>8}")
print(f"  {'=' * W}")

print()
print(f"  Baseline Evaluation  |  {hp['run_name']}")
print(f"  {'=' * W}")
print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}")
print(f"  {'-' * W}")

results = run_baseline_eval(best_weights, hp)

print(f"  {'-' * W}")
for ds, r in results.items():
    print(f"  {ds:<16} {r['mAP50']:>9.4f}  {r['mAP50_95']:>10.4f}")

mean50    = sum(r["mAP50"]    for r in results.values()) / len(results)
mean5095  = sum(r["mAP50_95"] for r in results.values()) / len(results)
print(f"  {'-' * W}")
print(f"  {'Mean':<16} {mean50:>9.4f}  {mean5095:>10.4f}")
print(f"  {'=' * W}")
print()

# ── Persist results ───────────────────────────────────────────────────────────
out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name":   hp["run_name"],
    "timestamp":  datetime.now().isoformat(),
    "source": sanity,
}
for ds, r in results.items():
    payload[ds] = r  # full dict: mAP50, mAP50_95, precision, recall, per-class AP50
out_path = out_dir / "baseline.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
