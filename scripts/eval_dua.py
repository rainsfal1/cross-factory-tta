"""
DUA evaluation: Distribution Uncertainty Adaptation on all target domains.

Offline mode: adapts BN running statistics by forwarding each test image
once (gradient-free) with n_augments augmented copies, then calls model.val()
for detection metrics. No optimizer required — faster and more memory-efficient
than TENT.

Paper algorithm: for each test image, create 64 augmented copies, forward
them through the model in train-BN mode with geometrically decaying momentum,
then restore eval mode. Running stats drift toward the target distribution.

Usage:
    uv run python scripts/eval_dua.py --hparams hparams_yolov8m.yaml
    uv run python scripts/eval_dua.py --hparams hparams_yolo11m.yaml
    uv run python scripts/eval_dua.py --hparams hparams_yolov8m.yaml --domain shwd
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
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load
from src.dua import DUA

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
args = parser.parse_args()

hp       = load(args.hparams)
eval_cfg = hp["eval"]
dua_cfg  = hp.get("dua", {})
imgsz    = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]

best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
if not best_weights.exists():
    print(f"ERROR: no weights at {best_weights}")
    sys.exit(1)

DOMAINS  = ["pictor_ppe", "shwd", "chv"] if args.domain == "all" else [args.domain]
tmp_root = config.RUNS_DIR / "tmp_remap"
lb       = LetterBox(new_shape=(imgsz, imgsz))
device   = torch.device(
    f"cuda:{eval_cfg['device']}" if isinstance(eval_cfg["device"], int)
    else eval_cfg["device"]
)

decay_factor          = dua_cfg.get("decay_factor",          0.94)
min_momentum_constant = dua_cfg.get("min_momentum_constant", 0.005)
mom_pre               = dua_cfg.get("mom_pre",               0.1)
n_augments            = dua_cfg.get("n_augments",            64)

W = 58
print()
print(f"  DUA Evaluation  |  {hp['run_name']}  |  "
      f"decay={decay_factor}  min_mom={min_momentum_constant}  "
      f"mom0={mom_pre}  n_aug={n_augments}")
print(f"  {'=' * W}")
print(f"  {'domain':<16}  {'baseline':>10}  {'dua':>10}  {'recovery':>10}")
print(f"  {'-' * W}")

domain_results = {}

for ds_name in DOMAINS:
    tmp_yaml  = make_remapped_yaml(ds_name, tmp_root)
    img_dir   = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))

    # ── Baseline (unadapted) ──────────────────────────────────────────────────
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
    baseline = _extract_metrics(_bmetrics, config.SH17_EVAL_IDX)["mAP50"]
    del _bm
    print(f"mAP50={baseline:.4f}")

    # ── Preprocess test images into per-image tensors ─────────────────────────
    print(f"  preprocessing {len(img_paths)} images...", end=" ", flush=True)
    image_tensors = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = lb(image=img)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        image_tensors.append(torch.from_numpy(img).float() / 255.0)
    print(f"done ({len(image_tensors)} images)")

    # ── DUA adaptation — one image at a time ─────────────────────────────────
    model   = YOLO(str(best_weights))
    model.model.to(device)
    adapter = DUA(
        model,
        decay_factor=decay_factor,
        min_momentum_constant=min_momentum_constant,
        mom_pre=mom_pre,
        n_augments=n_augments,
    )

    print(f"  adapting {len(image_tensors)} images...", end=" ", flush=True)
    for img_t in image_tensors:
        adapter.adapt(img_t.to(device))
    print("done")

    # ── Evaluation after adaptation ───────────────────────────────────────────
    # DUA leaves BN in eval mode with updated running stats.
    # model.val() runs normally — no fuse patch needed (DUA doesn't touch affine params).
    metrics  = model.val(
        data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"dua_{ds_name}", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    r        = _extract_metrics(metrics, config.SH17_EVAL_IDX)
    dua_map  = r["mAP50"]
    recovery = dua_map - baseline
    sign     = "+" if recovery >= 0 else ""
    print(f"  {ds_name:<16}  {baseline:>10.4f}  {dua_map:>10.4f}  {sign}{recovery:>9.4f}")

    domain_results[ds_name] = {
        "baseline": baseline,
        "dua":      dua_map,
        "recovery": recovery,
        **{k: v for k, v in r.items() if k != "mAP50"},
    }

print(f"  {'=' * W}")
print()

# ── Persist results ────────────────────────────────────────────────────────────
out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name":             hp["run_name"],
    "timestamp":            datetime.now().isoformat(),
    "decay_factor":         decay_factor,
    "min_momentum_constant": min_momentum_constant,
    "mom_pre":              mom_pre,
    "n_augments":           n_augments,
}
payload.update(domain_results)
out_path = out_dir / "dua.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
