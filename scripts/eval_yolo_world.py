"""
Zero-shot YOLO-World-L evaluation on all three target domains.

YOLO-World predicts at canonical class indices 0/1/2, matching the prepared
target-domain labels directly — no remapping needed.

Usage:
    uv run python scripts/eval_yolo_world.py
    uv run python scripts/eval_yolo_world.py --device 1
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import torch
from ultralytics import YOLOWorld

import config
from src.eval import _extract_metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

parser = argparse.ArgumentParser()
parser.add_argument("--device",  type=int, default=0)
parser.add_argument("--batch",   type=int, default=64)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8l-worldv2.pt"
RUN_NAME   = "yolo_world_L"

if not MODEL_PATH.exists():
    print(f"ERROR: model not found at {MODEL_PATH}")
    sys.exit(1)

# Class prompts — order must match canonical label indices 0/1/2
CLASS_PROMPTS = ["hard hat", "no hard hat", "person"]

# Eval index for 3-class YOLO-World output (0→hard_hat, 1→no_hard_hat, 2→person)
WORLD_EVAL_IDX = {0: "hard_hat", 1: "no_hard_hat", 2: "person"}

TARGET_DOMAINS = [
    ("pictor_ppe", "pictor.yaml"),
    ("shwd",       "shwd.yaml"),
    ("chv",        "chv.yaml"),
]

EVAL_KWARGS = dict(
    imgsz   = 640,
    batch   = args.batch,
    device  = args.device,
    half    = True,
    workers = args.workers,
    split   = "val",
    conf    = 0.001,
    iou     = 0.6,
    project = str(config.RUNS_DIR / "eval"),
    exist_ok = True,
    verbose  = False,
)

W = 46
print()
print(f"  YOLO-World-L  |  zero-shot  |  prompts: {CLASS_PROMPTS}")
print(f"  {'=' * W}")
print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}")
print(f"  {'-' * W}")

results = {}

for ds_name, yaml_name in TARGET_DOMAINS:
    model = YOLOWorld(str(MODEL_PATH))
    model.set_classes(CLASS_PROMPTS)

    data_yaml = config.DATA_DIR / ds_name / yaml_name
    print(f"  evaluating {ds_name}...", end=" ", flush=True)

    metrics = model.val(
        data    = str(data_yaml),
        name    = f"yolo_world_{ds_name}",
        **EVAL_KWARGS,
    )

    r = _extract_metrics(metrics, WORLD_EVAL_IDX)
    results[ds_name] = r
    print(f"mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}")
    del model

print(f"  {'-' * W}")
mean50   = sum(r["mAP50"]   for r in results.values()) / len(results)
mean5095 = sum(r["mAP50_95"] for r in results.values()) / len(results)
print(f"  {'Mean':<16} {mean50:>9.4f}  {mean5095:>10.4f}")
print(f"  {'=' * W}")
print()

out_dir = config.RUNS_DIR / "results" / RUN_NAME
out_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "run_name":     RUN_NAME,
    "model":        str(MODEL_PATH.name),
    "class_prompts": CLASS_PROMPTS,
    "timestamp":    datetime.now().isoformat(),
}
for ds, r in results.items():
    payload[ds] = r

out_path = out_dir / "baseline.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
