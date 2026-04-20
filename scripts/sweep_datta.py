"""
DATTA hyperparameter sweep + ablation.

Runs four groups of experiments:
  1. S1 only  — sweep alpha ∈ {0.3, 0.5, 0.7}   (isolates BN warmup)
  2. S2 only  — sweep conf_threshold ∈ {0.1, 0.25, 0.5}  (isolates detection-aware loss)
  3. S1 + S2  — cross-product alpha × conf_threshold (9 configs, full method)

Together these produce the ablation table (Table 3 in the paper).

Usage:
    uv run python scripts/sweep_datta.py --hparams hparams_yolov8m.yaml
    uv run python scripts/sweep_datta.py --hparams hparams_yolov8m.yaml --domain pictor_ppe
    uv run python scripts/sweep_datta.py --hparams hparams_yolov8m.yaml --device 2
    uv run python scripts/sweep_datta.py --hparams hparams_yolov8m.yaml --group s1
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

ALPHA_GRID = [0.3, 0.5, 0.7]
CONF_GRID = [0.1, 0.25, 0.5]

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain", default="all")
parser.add_argument("--device", type=int, default=None)
parser.add_argument("--group", default="all",
                    choices=["all", "s1", "s2", "full"],
                    help="which ablation group to run (default: all)")
args = parser.parse_args()

hp = load(args.hparams)
eval_cfg = hp["eval"]
datta_cfg = hp.get("datta", {})
if args.device is not None:
    eval_cfg["device"] = args.device

imgsz = eval_cfg["imgsz"]
val_batch = eval_cfg["batch"]
adapt_batch = hp["tent"].get("adapt_batch", 16)
lr = datta_cfg.get("lr", 0.005)
steps = datta_cfg.get("steps", 1)

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

# Build config list for requested group(s)
sweep_configs = []
if args.group in ("all", "s1"):
    for alpha in ALPHA_GRID:
        sweep_configs.append({"use_stage1": True, "use_stage2": False,
                               "alpha": alpha, "conf_threshold": None, "group": "s1"})
if args.group in ("all", "s2"):
    for conf_t in CONF_GRID:
        sweep_configs.append({"use_stage1": False, "use_stage2": True,
                               "alpha": None, "conf_threshold": conf_t, "group": "s2"})
if args.group in ("all", "full"):
    for alpha in ALPHA_GRID:
        for conf_t in CONF_GRID:
            sweep_configs.append({"use_stage1": True, "use_stage2": True,
                                   "alpha": alpha, "conf_threshold": conf_t, "group": "full"})

print()
print(f"  DATTA Sweep  |  {hp['run_name']}  |  lr={lr}  steps={steps}")
print(f"  groups: {args.group}  |  {len(sweep_configs)} configs × {len(DOMAINS)} domains")
print(f"  alpha grid: {ALPHA_GRID}  conf grid: {CONF_GRID}")
print()

all_results: dict[str, dict] = {}

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
    baseline_no_hh = baseline_r.get("no_hard_hat_AP50", 0.0)
    del _bm
    print(f"mAP50={baseline_map:.4f}  no_hh={baseline_no_hh:.4f}")

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
    print()

    W = 84
    print(f"  {ds_name}")
    print(f"  {'=' * W}")
    print(f"  {'group':>6}  {'alpha':>6}  {'conf_t':>7}  "
          f"{'baseline':>10}  {'datta':>10}  {'recovery':>10}  {'no_hh_AP50':>12}")
    print(f"  {'-' * W}")

    domain_results = []
    for cfg in sweep_configs:
        alpha = cfg["alpha"] if cfg["alpha"] is not None else 0.5  # unused in s2-only
        conf_t = cfg["conf_threshold"] if cfg["conf_threshold"] is not None else 0.25

        model = YOLO(str(best_weights))
        model.model.to(device)
        adapter = DATTA(
            model, alpha=alpha, conf_threshold=conf_t,
            lr=lr, steps=steps,
            use_stage1=cfg["use_stage1"], use_stage2=cfg["use_stage2"],
        )

        if cfg["use_stage1"]:
            adapter.warmup(batches)
        if cfg["use_stage2"]:
            for x in batches:
                adapter.adapt(x)

        model.model.fuse = lambda verbose=True: model.model
        label_alpha = f"{alpha:.1f}" if cfg["use_stage1"] else "  —  "
        label_conf = f"{conf_t:.2f}" if cfg["use_stage2"] else "  —   "
        metrics = model.val(
            data=str(tmp_yaml), imgsz=imgsz, batch=val_batch,
            device=eval_cfg["device"], half=eval_cfg["half"],
            workers=eval_cfg.get("workers", 8),
            split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
            name=f"datta_sweep_{ds_name}_{cfg['group']}_a{alpha}_c{conf_t}",
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True, verbose=False,
        )
        r = _extract_metrics(metrics, config.SH17_EVAL_IDX)
        datta_map = r["mAP50"]
        no_hh = r.get("no_hard_hat_AP50", 0.0)
        recovery = datta_map - baseline_map
        sign = "+" if recovery >= 0 else ""
        print(f"  {cfg['group']:>6}  {label_alpha:>6}  {label_conf:>7}  "
              f"{baseline_map:>10.4f}  {datta_map:>10.4f}  "
              f"{sign}{recovery:>9.4f}  {no_hh:>12.4f}")
        domain_results.append({
            "group": cfg["group"],
            "alpha": alpha if cfg["use_stage1"] else None,
            "conf_threshold": conf_t if cfg["use_stage2"] else None,
            "use_stage1": cfg["use_stage1"], "use_stage2": cfg["use_stage2"],
            "mAP50": datta_map, "recovery": recovery,
            "no_hard_hat_AP50": no_hh,
            **{k: v for k, v in r.items() if k not in ("mAP50",)},
        })
        del model

    print(f"  {'=' * W}")
    best = max(domain_results, key=lambda x: x["mAP50"])
    print(f"  Best: group={best['group']}  alpha={best['alpha']}  "
          f"conf_t={best['conf_threshold']}  mAP50={best['mAP50']:.4f}  "
          f"no_hh={best['no_hard_hat_AP50']:.4f}  "
          f"recovery={'+' if best['recovery'] >= 0 else ''}{best['recovery']:.4f}")
    print()
    all_results[ds_name] = {"baseline": baseline_map, "best": best, "results": domain_results}

out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": hp["run_name"],
    "timestamp": datetime.now().isoformat(),
    "alpha_grid": ALPHA_GRID, "conf_grid": CONF_GRID,
    "lr": lr, "steps": steps, "adapt_batch": adapt_batch,
    "group": args.group,
}
payload.update(all_results)
out_path = out_dir / "sweep_datta.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
