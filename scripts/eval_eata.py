"""
EATA evaluation: Efficient Anti-Forgetting Test-Time Adaptation (Niu et al., ICML 2022).

Offline mode: optionally pre-computes Fisher diagonal on SH17 source images,
then adapts BN affine params using dual filtering (reliability + redundancy)
with EWC Fisher regularization, then calls model.val() for detection metrics.

Usage:
    uv run python scripts/eval_eata.py --hparams hparams_yolov8m.yaml
    uv run python scripts/eval_eata.py --hparams hparams_yolov8m.yaml --no-fisher
    uv run python scripts/eval_eata.py --hparams hparams_yolov8m.yaml --domain shwd
    uv run python scripts/eval_eata.py --hparams hparams_yolov8m.yaml --device 2
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
from src.eata import EATA, compute_fishers
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
parser.add_argument("--device", type=int, default=None)
parser.add_argument("--no-fisher", action="store_true",
                    help="skip Fisher computation (ablation: filtering only, no EWC)")
args = parser.parse_args()

hp = load(args.hparams)
eval_cfg = hp["eval"]
eata_cfg = hp.get("eata", {})
if args.device is not None:
    eval_cfg["device"] = args.device

imgsz = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]
adapt_batch = hp["tent"].get("adapt_batch", 16)

lr = eata_cfg.get("lr", 0.001)
steps = eata_cfg.get("steps", 1)
e_margin = eata_cfg.get("e_margin", 2.454)
d_margin = eata_cfg.get("d_margin", 0.05)
fisher_alpha = eata_cfg.get("fisher_alpha", 2000.0)
fisher_n = eata_cfg.get("fisher_n_images", 200)

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

W = 60
use_fisher = not args.no_fisher
print()
print(f"  EATA Evaluation  |  {hp['run_name']}  |  lr={lr}  steps={steps}")
print(f"  e_margin={e_margin:.3f}  d_margin={d_margin}  "
      f"fisher={'alpha=' + str(fisher_alpha) if use_fisher else 'disabled'}")
print(f"  {'=' * W}")
print(f"  {'domain':<16}  {'baseline':>10}  {'eata':>10}  {'recovery':>10}")
print(f"  {'-' * W}")

# ── Pre-compute Fisher diagonal once (shared across all domains) ──────────────
fishers = None
if use_fisher:
    source_img_dir = config.SH17_DIR / "images"
    if source_img_dir.is_dir():
        _fisher_model = YOLO(str(best_weights))
        _fisher_model.model.to(device)
        fishers = compute_fishers(
            _fisher_model, source_img_dir, device, imgsz=imgsz, n_images=fisher_n
        )
        del _fisher_model
    else:
        print(f"  WARNING: SH17 source dir not found at {source_img_dir}, running without Fisher")

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
    baseline = _extract_metrics(_bmetrics, config.SH17_EVAL_IDX)["mAP50"]
    del _bm
    print(f"mAP50={baseline:.4f}")

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

    # ── EATA adaptation ───────────────────────────────────────────────────────
    model = YOLO(str(best_weights))
    model.model.to(device)
    adapter = EATA(
        model, lr=lr, steps=steps,
        e_margin=e_margin, d_margin=d_margin,
        fishers=fishers, fisher_alpha=fisher_alpha,
    )

    print(f"  adapting...", end=" ", flush=True)
    for x in batches:
        adapter.adapt(x)
    print("done")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.model.fuse = lambda verbose=True: model.model
    metrics = model.val(
        data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"eata_{ds_name}", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
    eata_map = r["mAP50"]
    recovery = eata_map - baseline
    sign = "+" if recovery >= 0 else ""
    print(f"  {ds_name:<16}  {baseline:>10.4f}  {eata_map:>10.4f}  {sign}{recovery:>9.4f}")

    domain_results[ds_name] = {
        "baseline": baseline,
        "eata": eata_map,
        "recovery": recovery,
        **{k: v for k, v in r.items() if k != "mAP50"},
    }

print(f"  {'=' * W}")
print()

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "lr": lr, "steps": steps,
    "e_margin": e_margin, "d_margin": d_margin,
    "fisher_alpha": fisher_alpha if use_fisher else 0,
    "fisher_n_images": fisher_n if use_fisher else 0,
    "adapt_batch": adapt_batch,
}
payload.update(domain_results)
suffix = "eata" if use_fisher else "eata_no_fisher"
out_path = out_dir / f"{suffix}.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
