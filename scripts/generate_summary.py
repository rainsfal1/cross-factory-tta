"""
Aggregate all result JSON files for a run into summary.json.

Usage:
    uv run python scripts/generate_summary.py --run sh17_yolov8m_100ep
    uv run python scripts/generate_summary.py --run all
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DOMAINS = ["pictor_ppe", "shwd", "chv"]

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run_name or 'all'")
args = parser.parse_args()

runs = (
    [d.name for d in (config.RUNS_DIR / "results").iterdir() if d.is_dir()]
    if args.run == "all"
    else [args.run]
)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def generate(run_name: str):
    base_dir = config.RUNS_DIR / "results" / run_name
    summary = {"run_name": run_name, "generated": datetime.now().isoformat()}

    # ── Source domain ─────────────────────────────────────────────────────────
    baseline = _load(base_dir / "baseline.json")
    if baseline:
        src = baseline.get("source", {})
        summary["source_mAP50"]     = src.get("mAP50")
        summary["source_mAP50_95"]  = src.get("mAP50_95")
    else:
        summary["source_mAP50"]     = None
        summary["source_mAP50_95"]  = None

    # ── Per-domain: baseline, best sweep, online TENT ─────────────────────────
    online = _load(base_dir / "online_tent.json")

    for ds in DOMAINS:
        # baseline
        bl_map50 = baseline.get(ds, {}).get("mAP50") if baseline else None
        bl_map5095 = baseline.get(ds, {}).get("mAP50_95") if baseline else None

        # sweep best
        sweep = _load(base_dir / f"sweep_tent_{ds}.json")
        best_steps = best_lr = best_tent = best_rec = None
        if sweep:
            b = sweep.get("best", {})
            best_steps = b.get("steps")
            best_lr    = b.get("lr")
            best_tent  = b.get("mAP50")
            best_rec   = b.get("recovery")
            if bl_map50 is None:
                bl_map50 = sweep.get("baseline_mAP50")

        # online TENT
        ot_tent = ot_rec = None
        if online and ds in online:
            ot_tent = online[ds].get("tent")
            ot_rec  = online[ds].get("recovery")

        summary[f"{ds}_baseline_mAP50"]    = bl_map50
        summary[f"{ds}_baseline_mAP50_95"] = bl_map5095
        summary[f"{ds}_best_tent_mAP50"]   = best_tent
        summary[f"{ds}_best_tent_recovery"] = best_rec
        summary[f"{ds}_best_steps"]        = best_steps
        summary[f"{ds}_best_lr"]           = best_lr
        summary[f"{ds}_online_tent_mAP50"] = ot_tent
        summary[f"{ds}_online_tent_recovery"] = ot_rec

    # ── TENT config used ──────────────────────────────────────────────────────
    if online:
        summary["tent_steps"]       = online.get("steps")
        summary["tent_lr"]          = online.get("lr")
        summary["tent_adapt_batch"] = online.get("adapt_batch")

    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  {run_name}: summary written → {out_path}")
    return summary


for run in runs:
    s = generate(run)
    print(f"  source={s.get('source_mAP50')}  "
          + "  ".join(
              f"{ds}:bl={s.get(f'{ds}_baseline_mAP50')},best={s.get(f'{ds}_best_tent_mAP50')}"
              for ds in DOMAINS
          ))
