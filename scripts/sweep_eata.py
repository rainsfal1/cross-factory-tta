"""
EATA hyperparameter sweep.

Sweeps two axes:
  1. fisher_alpha ∈ {0, 2000, 5000} — 0 = filtering only (no EWC), isolates
     the contribution of Fisher regularization vs. dual filtering alone
  2. e_margin ∈ {2.0, 2.454, 2.763} — paper default is 2.454, observed
     target entropy ~1.75-2.04 so all three should pass the filter

Fisher diagonal is computed once on SH17 source images and reused.

Usage:
    uv run python scripts/sweep_eata.py --hparams hparams_yolov8m.yaml
    uv run python scripts/sweep_eata.py --hparams hparams_yolov8m.yaml --domain pictor_ppe
    uv run python scripts/sweep_eata.py --hparams hparams_yolov8m.yaml --device 2
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

# Observed cos_sim range: 0.90-0.99 — need d_margin > 0.90 to let anything through.
# 1.0 = disable redundancy filter entirely (gives EATA max samples).
# Sweep d_margin at fixed paper defaults for other params, then vary fisher_alpha.
D_MARGIN_GRID = [0.05, 0.95, 1.0]   # 0.05=paper default (silences all), 0.95/1.0=permissive
FISHER_ALPHA_GRID = [0, 2000]        # 0=filtering only, 2000=paper default with Fisher
E_MARGIN = 2.454                     # paper default — all samples pass reliability filter

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all")
parser.add_argument("--device", type=int, default=None)
parser.add_argument("--debug", action="store_true",
                    help="print per-batch filter stats for first domain only")
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
d_margin = eata_cfg.get("d_margin", 0.05)
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

print()
print(f"  EATA Sweep  |  {hp['run_name']}  |  lr={lr}  e_margin={E_MARGIN} (fixed)")
print(f"  d_margin grid:     {D_MARGIN_GRID}  (0.05=paper default silences all, 0.95/1.0=permissive)")
print(f"  fisher_alpha grid: {FISHER_ALPHA_GRID}  (0 = filtering only)")
print()

# ── Pre-compute Fisher diagonal once ─────────────────────────────────────────
source_img_dir = config.SH17_DIR / "images"

fishers = None
if source_img_dir.is_dir():
    _fm = YOLO(str(best_weights))
    _fm.model.to(device)
    fishers = compute_fishers(_fm, source_img_dir, device, imgsz=imgsz, n_images=fisher_n)
    del _fm
else:
    print(f"  WARNING: SH17 source dir not found — Fisher regularization disabled for all runs")

all_results: dict[str, dict] = {}

for ds_name in DOMAINS:
    tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
    img_dir = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))

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

    W = 72
    print()
    print(f"  {ds_name}")
    print(f"  {'=' * W}")
    print(f"  {'d_margin':>10}  {'fisher_α':>10}  {'baseline':>10}  {'eata':>10}  {'recovery':>10}")
    print(f"  {'-' * W}")

    domain_results = []
    for dm in D_MARGIN_GRID:
        for fisher_alpha in FISHER_ALPHA_GRID:
            model = YOLO(str(best_weights))
            model.model.to(device)
            adapter = EATA(
                model, lr=lr, steps=steps,
                e_margin=E_MARGIN, d_margin=dm,
                fishers=fishers if fisher_alpha > 0 else None,
                fisher_alpha=fisher_alpha,
            )
            for bi, x in enumerate(batches):
                adapter.adapt(x, _debug=(args.debug and bi < 2 and ds_name == DOMAINS[0]))

            model.model.fuse = lambda verbose=True: model.model
            metrics = model.val(
                data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
                device=eval_cfg["device"], half=eval_cfg["half"],
                workers=eval_cfg.get("workers", 8),
                split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
                name=f"eata_sweep_{ds_name}_dm{dm}_fa{fisher_alpha}",
                project=str(config.RUNS_DIR / "eval"),
                exist_ok=True, verbose=False,
            )
            r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
            eata_map = r["mAP50"]
            recovery = eata_map - baseline
            sign = "+" if recovery >= 0 else ""
            print(f"  {dm:>10.2f}  {fisher_alpha:>10}  "
                  f"{baseline:>10.4f}  {eata_map:>10.4f}  {sign}{recovery:>9.4f}")
            domain_results.append({
                "d_margin": dm, "fisher_alpha": fisher_alpha,
                "mAP50": eata_map, "recovery": recovery,
            })
            del model

    print(f"  {'=' * W}")
    best = max(domain_results, key=lambda x: x["mAP50"])
    print(f"  Best: d_margin={best['d_margin']}  fisher_alpha={best['fisher_alpha']}  "
          f"mAP50={best['mAP50']:.4f}  recovery={'+' if best['recovery']>=0 else ''}{best['recovery']:.4f}")
    print()
    all_results[ds_name] = {"baseline": baseline, "best": best, "results": domain_results}

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "d_margin_grid": D_MARGIN_GRID,
    "fisher_alpha_grid": FISHER_ALPHA_GRID,
    "e_margin": E_MARGIN, "lr": lr,
}
payload.update(all_results)
out_path = out_dir / "sweep_eata.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
