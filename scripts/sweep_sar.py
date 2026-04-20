"""
SAR e_margin sweep: gives SAR its best chance before reporting.

Default e_margin=0.4*ln(17)=1.145 silences everything because the target-
domain entropy (~1.75-2.04) is above threshold. The official SAR code uses
0.4*ln(1000)=2.763 (ImageNet), which is much more permissive.

Sweeps e_margin ∈ {1.145, 2.0, 2.5, 2.763} to find what actually lets
samples through and whether adaptation helps.

Usage:
    uv run python scripts/sweep_sar.py --hparams hparams_yolov8m.yaml
    uv run python scripts/sweep_sar.py --hparams hparams_yolov8m.yaml --domain pictor_ppe
    uv run python scripts/sweep_sar.py --hparams hparams_yolov8m.yaml --device 2
"""
import argparse
import json
import math
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
from src.sar import SAR

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1.145 = 0.4*ln(17) paper formula, 2.763 = 0.4*ln(1000) official default
E_MARGIN_GRID = [1.145, 2.0, 2.5, 2.763]

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
parser.add_argument("--device", type=int, default=None)
args = parser.parse_args()

hp = load(args.hparams)
eval_cfg = hp["eval"]
sar_cfg = hp.get("sar", {})
if args.device is not None:
    eval_cfg["device"] = args.device

imgsz = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]
adapt_batch = hp["tent"].get("adapt_batch", 16)
lr = sar_cfg.get("lr", 0.00025)
steps = sar_cfg.get("steps", 1)
rho = sar_cfg.get("rho", 0.05)
reset_constant = sar_cfg.get("reset_constant", 0.2)

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

W = 64
print()
print(f"  SAR Sweep  |  {hp['run_name']}  |  lr={lr}  steps={steps}  rho={rho}")
print(f"  e_margin grid: {E_MARGIN_GRID}  (observed target entropy ~1.75-2.04)")
print()

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

    print()
    print(f"  {ds_name}")
    print(f"  {'=' * W}")
    print(f"  {'e_margin':>10}  {'baseline':>10}  {'sar':>10}  {'recovery':>10}  notes")
    print(f"  {'-' * W}")

    domain_results = []
    for e_margin in E_MARGIN_GRID:
        model = YOLO(str(best_weights))
        model.model.to(device)
        adapter = SAR(model, lr=lr, steps=steps, e_margin=e_margin,
                      reset_constant=reset_constant, rho=rho)

        n_updated = 0
        for x in batches:
            from src.sar import _per_sample_entropy
            with torch.no_grad():
                preds = model.model.forward(x, augment=False)
            ent = _per_sample_entropy(preds)
            n_updated += int((ent < e_margin).sum())
            adapter.adapt(x)

        model.model.fuse = lambda verbose=True: model.model
        metrics = model.val(
            data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
            device=eval_cfg["device"], half=eval_cfg["half"],
            workers=eval_cfg.get("workers", 8),
            split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
            name=f"sar_sweep_{ds_name}_em{e_margin:.3f}",
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True, verbose=False,
        )
        r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
        sar_map = r["mAP50"]
        recovery = sar_map - baseline
        sign = "+" if recovery >= 0 else ""
        note = f"{n_updated}/{len(tensors_all)} samples used"
        print(f"  {e_margin:>10.3f}  {baseline:>10.4f}  {sar_map:>10.4f}  {sign}{recovery:>9.4f}  {note}")
        domain_results.append({"e_margin": e_margin, "mAP50": sar_map,
                                "recovery": recovery, "samples_used": n_updated})
        del model

    print(f"  {'=' * W}")
    best = max(domain_results, key=lambda x: x["mAP50"])
    print(f"  Best: e_margin={best['e_margin']}  mAP50={best['mAP50']:.4f}  "
          f"recovery={'+' if best['recovery']>=0 else ''}{best['recovery']:.4f}")
    print()
    all_results[ds_name] = {"baseline": baseline, "best": best, "results": domain_results}

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "e_margin_grid": E_MARGIN_GRID,
    "lr": lr, "steps": steps, "rho": rho,
}
payload.update(all_results)
out_path = out_dir / "sweep_sar.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
