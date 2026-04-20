"""
YOLO-World-L confidence threshold sweep on Pictor-PPE and SHWD.

Bulletproofs the zero-shot baseline: confirms conf=0.001 (full mAP-curve
protocol) is optimal or finds the actual best threshold. Tests
conf ∈ {0.001, 0.01, 0.1, 0.25} with the same prompts and NMS settings
used in the main eval_yolo_world.py run.

Usage:
    uv run python scripts/sweep_yolo_world_conf.py
    uv run python scripts/sweep_yolo_world_conf.py --device 1
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("ultralytics").setLevel(logging.ERROR)

import torch
from ultralytics import YOLOWorld

import config
from src.eval import _extract_metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONF_GRID = [0.001, 0.01, 0.1, 0.25]
DOMAINS = [
    ("pictor_ppe", "pictor.yaml"),
    ("shwd", "shwd.yaml"),
]

MODEL_PATH = config._ROOT / "models" / "yolov8l-worldv2.pt"
CLASS_PROMPTS = ["hard hat", "no hard hat", "person"]
WORLD_EVAL_IDX = {0: "hard_hat", 1: "no_hard_hat", 2: "person"}

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

if not MODEL_PATH.exists():
    print(f"ERROR: model not found at {MODEL_PATH}")
    sys.exit(1)

W = 62
print()
print(f"  YOLO-World-L  |  conf threshold sweep  |  prompts: {CLASS_PROMPTS}")
print(f"  conf values: {CONF_GRID}")
print()

all_results: dict[str, dict] = {}

for ds_name, yaml_name in DOMAINS:
    data_yaml = config.DATA_DIR / ds_name / yaml_name
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    print(f"  {ds_name}")
    print(f"  {'=' * W}")
    print(f"  {'conf':>8}  {'mAP50':>9}  {'mAP50-95':>10}  {'hard_hat':>10}  {'no_hh':>8}  {'person':>8}")
    print(f"  {'-' * W}")

    domain_results = []
    for conf in CONF_GRID:
        model = YOLOWorld(str(MODEL_PATH))
        model.set_classes(CLASS_PROMPTS)

        metrics = model.val(
            data=str(data_yaml),
            name=f"yw_conf_sweep_{ds_name}_c{conf}",
            imgsz=640,
            batch=args.batch,
            device=args.device,
            half=True,
            workers=args.workers,
            split="val",
            conf=conf,
            iou=0.6,
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True,
            verbose=False,
        )
        r = _extract_metrics(metrics, WORLD_EVAL_IDX)
        domain_results.append({"conf": conf, **r})

        hh  = r.get("hard_hat_AP50", 0.0)
        nhh = r.get("no_hard_hat_AP50", 0.0)
        prs = r.get("person_AP50", 0.0)
        print(
            f"  {conf:>8.3f}  {r['mAP50']:>9.4f}  {r['mAP50_95']:>10.4f}"
            f"  {hh:>10.4f}  {nhh:>8.4f}  {prs:>8.4f}"
        )
        del model

    print(f"  {'=' * W}")
    best = max(domain_results, key=lambda x: x["mAP50"])
    print(f"  Best conf={best['conf']}  mAP50={best['mAP50']:.4f}")
    print()
    all_results[ds_name] = {"best_conf": best["conf"], "results": domain_results}

out_dir = config.RUNS_DIR / "results" / "yolo_world_L"
out_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "run_name": "yolo_world_L_conf_sweep",
    "model": MODEL_PATH.name,
    "class_prompts": CLASS_PROMPTS,
    "timestamp": datetime.now().isoformat(),
    "domains": DOMAINS,
    "conf_grid": CONF_GRID,
}
payload.update(all_results)

out_path = out_dir / "conf_sweep.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
