"""
SAR evaluation: Sharpness-Aware and Reliable entropy minimization (Niu et al. 2023).

Offline mode: adapts BN affine params (gamma/beta) over all test images in
batches using SAM + entropy filtering, then calls model.val() for detection
metrics. Same pipeline structure as sweep_tent.py.

Key differences from TENT:
  - SAM two-step update (perturb → compute gradient → restore → step)
  - Unreliable samples filtered out (entropy > e_margin skipped)
  - EMA anchor + model reset if loss diverges

Usage:
    uv run python scripts/eval_sar.py --hparams hparams_yolov8m.yaml
    uv run python scripts/eval_sar.py --hparams hparams_yolov8m.yaml --domain shwd
    uv run python scripts/eval_sar.py --hparams hparams_yolov8m.yaml --device 2
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
from src.sar import SAR

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
parser.add_argument("--device", type=int, default=None,
                    help="override GPU device index from hparams")
parser.add_argument("--debug", action="store_true",
                    help="print entropy values for first batch to diagnose filtering")
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
e_margin = sar_cfg.get("e_margin", 1.145)
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

W = 60
print()
print(f"  SAR Evaluation  |  {hp['run_name']}  |  lr={lr}  steps={steps}  rho={rho}")
print(f"  e_margin={e_margin:.3f}  reset_constant={reset_constant}  adapt_batch={adapt_batch}")
print(f"  {'=' * W}")
print(f"  {'domain':<16}  {'baseline':>10}  {'sar':>10}  {'recovery':>10}")
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

    # ── SAR adaptation ────────────────────────────────────────────────────────
    model = YOLO(str(best_weights))
    model.model.to(device)
    adapter = SAR(model, lr=lr, steps=steps, e_margin=e_margin,
                  reset_constant=reset_constant, rho=rho)

    print(f"  adapting...", end=" ", flush=True)
    for i, x in enumerate(batches):
        adapter.adapt(x, _debug=(args.debug and i == 0))
    print("done")

    # ── Evaluation ────────────────────────────────────────────────────────────
    # SAR modifies affine params — patch fuse so val() doesn't re-fuse Conv+BN
    # (same as sweep_tent.py)
    model.model.fuse = lambda verbose=True: model.model
    metrics = model.val(
        data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"sar_{ds_name}", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
    sar_map = r["mAP50"]
    recovery = sar_map - baseline
    sign = "+" if recovery >= 0 else ""
    print(f"  {ds_name:<16}  {baseline:>10.4f}  {sar_map:>10.4f}  {sign}{recovery:>9.4f}")

    domain_results[ds_name] = {
        "baseline": baseline,
        "sar": sar_map,
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
    "lr": lr,
    "steps": steps,
    "rho": rho,
    "e_margin": e_margin,
    "reset_constant": reset_constant,
    "adapt_batch": adapt_batch,
}
payload.update(domain_results)
out_path = out_dir / "sar.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
