"""
DUA decay_factor sweep: gives DUA its best chance before reporting.

Default hparams (decay=0.94) were tuned for ImageNet-C corruptions, not
catastrophic cross-domain shift. Higher decay = slower momentum decay =
BN stats update more gradually = better for large distribution gaps.

Sweeps decay ∈ {0.94, 0.98, 0.99} with all other params fixed.

Usage:
    uv run python scripts/sweep_dua.py --hparams hparams_yolov8m.yaml
    uv run python scripts/sweep_dua.py --hparams hparams_yolov8m.yaml --domain shwd
    uv run python scripts/sweep_dua.py --hparams hparams_yolov8m.yaml --device 2
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
from src.dua import DUA
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DECAY_GRID = [0.94, 0.98, 0.99]

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
parser.add_argument("--device", type=int, default=None,
                    help="override GPU device index from hparams")
args = parser.parse_args()

hp = load(args.hparams)
eval_cfg = hp["eval"]
dua_cfg = hp.get("dua", {})
if args.device is not None:
    eval_cfg["device"] = args.device

imgsz = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]
min_momentum_constant = dua_cfg.get("min_momentum_constant", 0.005)
mom_pre = dua_cfg.get("mom_pre", 0.1)
n_augments = dua_cfg.get("n_augments", 64)

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

W = 58
print()
print(f"  DUA Sweep  |  {hp['run_name']}  |  min_mom={min_momentum_constant}  mom0={mom_pre}  n_aug={n_augments}")
print(f"  decay grid: {DECAY_GRID}")
print()

all_results: dict[str, dict] = {}

for ds_name in DOMAINS:
    tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
    img_dir = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))

    # Baseline once per domain
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

    # Preprocess images once, reuse across all decay values
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

    print()
    print(f"  {ds_name}")
    print(f"  {'=' * W}")
    print(f"  {'decay':>8}  {'baseline':>10}  {'dua':>10}  {'recovery':>10}")
    print(f"  {'-' * W}")

    domain_results = []
    for decay in DECAY_GRID:
        model = YOLO(str(best_weights))
        model.model.to(device)
        adapter = DUA(
            model,
            decay_factor=decay,
            min_momentum_constant=min_momentum_constant,
            mom_pre=mom_pre,
            n_augments=n_augments,
        )
        for img_t in image_tensors:
            adapter.adapt(img_t.to(device))

        metrics = model.val(
            data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
            device=eval_cfg["device"], half=eval_cfg["half"],
            workers=eval_cfg.get("workers", 8),
            split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
            name=f"dua_sweep_{ds_name}_d{decay}",
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True, verbose=False,
        )
        r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
        dua_map = r["mAP50"]
        recovery = dua_map - baseline
        sign = "+" if recovery >= 0 else ""
        print(f"  {decay:>8.2f}  {baseline:>10.4f}  {dua_map:>10.4f}  {sign}{recovery:>9.4f}")
        domain_results.append({"decay": decay, "mAP50": dua_map, "recovery": recovery})
        del model

    print(f"  {'=' * W}")
    best = max(domain_results, key=lambda x: x["mAP50"])
    print(f"  Best: decay={best['decay']}  mAP50={best['mAP50']:.4f}  recovery={'+' if best['recovery']>=0 else ''}{best['recovery']:.4f}")
    print()
    all_results[ds_name] = {"baseline": baseline, "best": best, "results": domain_results}

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "decay_grid": DECAY_GRID,
    "min_momentum_constant": min_momentum_constant,
    "mom_pre": mom_pre,
    "n_augments": n_augments,
}
payload.update(all_results)
out_path = out_dir / "sweep_dua.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
