"""
DATTA evaluation: Detection-Aware Test-Time Adaptation (custom method, Phase 3).

Stage 1 (Soft BN Warm-Up): blends BN running stats toward target domain.
Stage 2 (Detection-Aware Entropy Min): confidence-weighted entropy loss,
  only anchors above conf_threshold contribute — avoids gradient starvation.

Both stages can be toggled independently for ablation.

Usage:
    uv run python scripts/eval_datta.py --hparams hparams_yolov8m.yaml
    uv run python scripts/eval_datta.py --hparams hparams_yolov8m.yaml --s1-only
    uv run python scripts/eval_datta.py --hparams hparams_yolov8m.yaml --s2-only
    uv run python scripts/eval_datta.py --hparams hparams_yolov8m.yaml --domain shwd --device 1
    uv run python scripts/eval_datta.py --hparams hparams_yolov8m.yaml --alpha 0.3 --conf-threshold 0.1
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

import config
from src.datta import DATTA
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all")
parser.add_argument("--device", type=int, default=None)
parser.add_argument("--s1-only", action="store_true", help="Stage 1 only (no gradient step)")
parser.add_argument("--s2-only", action="store_true", help="Stage 2 only (no BN warmup)")
parser.add_argument("--alpha", type=float, default=None, help="Override hparams alpha")
parser.add_argument("--conf-threshold", type=float, default=None, help="Override hparams conf_threshold")
parser.add_argument("--debug", action="store_true",
                    help="print mask stats for first 2 batches of first domain (S2 only)")
args = parser.parse_args()

hp = load(args.hparams)
eval_cfg = hp["eval"]
datta_cfg = hp.get("datta", {})
if args.device is not None:
    eval_cfg["device"] = args.device

imgsz = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]
adapt_batch = hp["tent"].get("adapt_batch", 16)

alpha = args.alpha if args.alpha is not None else datta_cfg.get("alpha", 0.5)
conf_threshold = args.conf_threshold if args.conf_threshold is not None else datta_cfg.get("conf_threshold", 0.25)
lr = datta_cfg.get("lr", 0.005)
steps = datta_cfg.get("steps", 1)

use_stage1 = not args.s2_only
use_stage2 = not args.s1_only

best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
if not best_weights.exists():
    print(f"ERROR: no weights at {best_weights}")
    sys.exit(1)

DOMAINS = ["pictor_ppe", "shwd", "chv"] if args.domain == "all" else [args.domain]
tmp_root = config.RUNS_DIR / "tmp_remap"
lb = LetterBox(new_shape=(imgsz, imgsz))
device = torch.device(
    f"cuda:{eval_cfg['device']}" if isinstance(eval_cfg["device"], int)
    else eval_cfg["device"]
)

stage_label = "S1+S2" if (use_stage1 and use_stage2) else ("S1" if use_stage1 else "S2")
W = 72
print()
print(f"  DATTA Evaluation  |  {hp['run_name']}  |  {stage_label}  "
      f"alpha={alpha}  conf_thr={conf_threshold}  lr={lr}  steps={steps}")
print(f"  {'=' * W}")
print(f"  {'domain':<16}  {'baseline':>10}  {'datta':>10}  {'recovery':>10}  "
      f"{'no_hh_AP50':>12}")
print(f"  {'-' * W}")

domain_results = {}

for ds_name in DOMAINS:
    tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
    img_dir = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"  computing baseline for {ds_name}...", end=" ", flush=True)
    _bm = YOLO(str(best_weights))
    _bmetrics = _bm.val(
        data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"baseline_{ds_name}", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    baseline_r = _extract_metrics(_bmetrics, config.SH17_EVAL_IDX)
    baseline_map = baseline_r["mAP50"]
    del _bm
    print(f"mAP50={baseline_map:.4f}  no_hh={baseline_r.get('no_hard_hat_AP50', 0):.4f}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print(f"  preprocessing {len(img_paths)} images...", end=" ", flush=True)
    tensors_all = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = lb(image=img)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        tensors_all.append(torch.from_numpy(img).float() / 255.0)
    batches = [
        torch.stack(tensors_all[i:i + adapt_batch]).to(device)
        for i in range(0, len(tensors_all), adapt_batch)
    ]
    print(f"done ({len(tensors_all)} images, {len(batches)} batches)")

    # ── DATTA ────────────────────────────────────────────────────────────────
    model = YOLO(str(best_weights))
    model.model.to(device)
    adapter = DATTA(
        model, alpha=alpha, conf_threshold=conf_threshold,
        lr=lr, steps=steps,
        use_stage1=use_stage1, use_stage2=use_stage2,
    )

    if use_stage1:
        print(f"  stage 1 warmup ({len(batches)} batches)...", end=" ", flush=True)
        adapter.warmup(batches)
        print("done")

    if use_stage2:
        print(f"  stage 2 adapting...", end=" ", flush=True)
        for bi, x in enumerate(batches):
            # debug: first batch, first step only — so we get one print per domain
            adapter.adapt(x, _debug=(args.debug and bi == 0))
        print("done")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.model.fuse = lambda verbose=True: model.model
    metrics = model.val(
        data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"datta_{stage_label.lower()}_{ds_name}",
        project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
    datta_map = r["mAP50"]
    recovery = datta_map - baseline_map
    sign = "+" if recovery >= 0 else ""
    no_hh = r.get("no_hard_hat_AP50", 0.0)
    print(f"  {ds_name:<16}  {baseline_map:>10.4f}  {datta_map:>10.4f}  "
          f"{sign}{recovery:>9.4f}  {no_hh:>12.4f}")

    domain_results[ds_name] = {
        "baseline": baseline_map,
        "datta": datta_map,
        "recovery": recovery,
        **{k: v for k, v in r.items() if k != "mAP50"},
    }
    del model

print(f"  {'=' * W}")
print()

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "stage": stage_label,
    "alpha": alpha, "conf_threshold": conf_threshold,
    "lr": lr, "steps": steps,
    "use_stage1": use_stage1, "use_stage2": use_stage2,
    "adapt_batch": adapt_batch,
}
payload.update(domain_results)
suffix = f"datta_{stage_label.lower().replace('+', '_')}"
out_path = out_dir / f"{suffix}.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
