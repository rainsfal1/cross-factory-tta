"""
TENT hyperparameter sweep: steps x lr, one or all target domains.

Usage:
    uv run python scripts/sweep_tent.py --hparams hparams_yolov8m.yaml
    uv run python scripts/sweep_tent.py --hparams hparams_yolo11m.yaml
    uv run python scripts/sweep_tent.py --hparams hparams_yolov8m.yaml --domain all
    uv run python scripts/sweep_tent.py --hparams hparams_yolov8m.yaml --domain shwd
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
import torch
from ultralytics.data.augment import LetterBox

import config
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load
from src.tent import TENT
from ultralytics import YOLO

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="pictor_ppe",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
parser.add_argument("--device", type=int, default=None,
                    help="override GPU device index from hparams (e.g. 0, 2, 3)")
args = parser.parse_args()

hp          = load(args.hparams)
eval_cfg    = hp["eval"]
if args.device is not None:
    eval_cfg["device"] = args.device
tent_cfg    = hp["tent"]
imgsz       = eval_cfg["imgsz"]
batch_sz    = tent_cfg.get("adapt_batch", eval_cfg["batch"])  # TENT adapt batch
val_batch   = eval_cfg["batch"]                                # val() inference batch
pin_tensors = tent_cfg.get("pin_tensors", False)

best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
if not best_weights.exists():
    print(f"ERROR: no weights at {best_weights}")
    sys.exit(1)

STEPS_LIST = hp.get("sweep", {}).get("steps", [1, 5, 10])
LR_LIST    = hp.get("sweep", {}).get("lr",    [0.001, 0.005, 0.01])
DOMAINS    = ["pictor_ppe", "shwd", "chv"] if args.domain == "all" else [args.domain]

tmp_root  = config.RUNS_DIR / "tmp_remap"
lb        = LetterBox(new_shape=(imgsz, imgsz))
out_dir   = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)

W = 62

for ds_name in DOMAINS:
    tmp_yaml  = make_remapped_yaml(ds_name, tmp_root)
    img_dir   = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))
    n_steps   = len(img_paths) // batch_sz + (1 if len(img_paths) % batch_sz else 0)

    # Compute baseline from this model (no adaptation)
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

    print()
    print(f"  TENT Sweep  |  {ds_name}  |  adapt_batch={batch_sz}  val_batch={val_batch}  ({len(img_paths)} images → {n_steps} grad steps/pass)")
    print(f"  {'='*W}")
    print(f"  {'steps':>6}  {'lr':>7}  {'baseline':>10}  {'tent':>10}  {'recovery':>10}")
    print(f"  {'-'*W}")

    # Preprocess images ONCE per domain, reuse across all combos
    print(f"  preprocessing {len(img_paths)} images...", end=" ", flush=True)
    device = torch.device(f"cuda:{eval_cfg['device']}" if isinstance(eval_cfg['device'], int) else eval_cfg['device'])
    tensors_all = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = lb(image=img)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        tensors_all.append(torch.from_numpy(img).float() / 255.0)
    # Batch on CPU; optionally pin all to GPU if pin_tensors=true and VRAM allows
    if pin_tensors:
        gpu_tensors = torch.stack(tensors_all).to(device)
        batches = [gpu_tensors[i:i+batch_sz] for i in range(0, len(gpu_tensors), batch_sz)]
        print(f"done — pinned {gpu_tensors.shape[0]} tensors to {device} ({gpu_tensors.nbytes/1e9:.2f} GB)")
    else:
        batches = [torch.stack(tensors_all[i:i+batch_sz]) for i in range(0, len(tensors_all), batch_sz)]
        print(f"done ({len(batches)} batches, CPU)")

    results = []

    for steps in STEPS_LIST:
        for lr in LR_LIST:
            model   = YOLO(str(best_weights))
            model.model.to(device)
            adapter = TENT(model, lr=lr, steps=steps)

            for x in batches:
                adapter.step_tensor(x if pin_tensors else x.to(device))

            model.model.fuse = lambda verbose=True: model.model
            metrics = model.val(
                data=str(tmp_yaml),
                imgsz=imgsz,
                batch=val_batch,
                device=eval_cfg["device"],
                half=eval_cfg["half"],
                workers=eval_cfg.get("workers", 8),
                split="val",
                conf=eval_cfg["conf"],
                iou=eval_cfg["iou"],
                name=f"sweep_{ds_name}_s{steps}_lr{lr}",
                project=str(config.RUNS_DIR / "eval"),
                exist_ok=True,
                verbose=False,
            )
            r        = _extract_metrics(metrics, config.SH17_EVAL_IDX)
            tent_map = r["mAP50"]
            recovery = tent_map - baseline
            sign     = "+" if recovery >= 0 else ""
            print(f"  {steps:>6}  {lr:>7.3f}  {baseline:>10.4f}  {tent_map:>10.4f}  {sign}{recovery:>9.4f}")
            results.append({"steps": steps, "lr": lr, "mAP50": tent_map, "recovery": recovery})

    print(f"  {'='*W}")
    best = max(results, key=lambda r: r["mAP50"])
    print(f"  Best: steps={best['steps']} lr={best['lr']} → mAP50={best['mAP50']:.4f}  recovery={'+' if best['recovery']>=0 else ''}{best['recovery']:.4f}")

    payload = {
        "run_name":      hp["run_name"],
        "timestamp":     datetime.now().isoformat(),
        "domain":        ds_name,
        "adapt_batch":   batch_sz,
        "val_batch":     val_batch,
        "baseline_mAP50": baseline,
        "results":       results,
        "best":          best,
    }
    out_path = out_dir / f"sweep_tent_{ds_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {out_path}")

print()
