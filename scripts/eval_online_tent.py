"""
Online TENT evaluation: per-batch adapt-then-predict, bypassing ultralytics val().

For each batch:
  1. Adapt BN affine params via entropy minimization (train mode, grad enabled)
  2. Switch to eval mode, run model.predict() for detections (no grad)
  3. Switch back to train mode for the next batch

Then compute mAP50 from accumulated predictions using ultralytics' ap_per_class.

Usage:
    uv run python scripts/eval_online_tent.py --hparams hparams_yolov8m.yaml
    uv run python scripts/eval_online_tent.py --hparams hparams_yolo11m.yaml
    uv run python scripts/eval_online_tent.py --hparams hparams_yolov8m.yaml --domain all
    uv run python scripts/eval_online_tent.py --hparams hparams_yolov8m.yaml --domain shwd
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

import config
from src.eval import make_remapped_yaml
from src.hparams import load
from src.tent import TENT

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False



def load_batch_tensors(img_paths, lb, device):
    tensors = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = lb(image=img)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        tensors.append(torch.from_numpy(img).float() / 255.0)
    return torch.stack(tensors).to(device) if tensors else None


def load_gt_labels(img_paths, tmp_root, ds_name, imgsz):
    """
    Load ground-truth YOLO labels (remapped to SH17 indices) for a list of image paths.
    Returns list of [N, 5] arrays (cls, cx, cy, w, h) in pixel-space xyxy.
    """
    label_dir = tmp_root / ds_name / "labels" / "test"
    gt_list = []
    for p in img_paths:
        lbl_path = label_dir / (Path(p).stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert normalized xywh → pixel xyxy
                    x1 = (cx - w / 2) * imgsz
                    y1 = (cy - h / 2) * imgsz
                    x2 = (cx + w / 2) * imgsz
                    y2 = (cy + h / 2) * imgsz
                    boxes.append([cls, x1, y1, x2, y2])
        gt_list.append(np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32))
    return gt_list


def match_predictions(preds_xyxy, pred_conf, pred_cls, gt, iou_thresh=0.5):
    """
    Match predictions to GT boxes.
    preds_xyxy: [M, 4]  pred_conf: [M]  pred_cls: [M]  gt: [N, 5] (cls, x1,y1,x2,y2)
    Returns tp: [M] bool array (True if correctly matched).
    """
    M = len(pred_conf)
    tp = np.zeros(M, dtype=bool)
    if len(gt) == 0 or M == 0:
        return tp

    gt_boxes = gt[:, 1:]   # [N, 4]
    gt_cls   = gt[:, 0]    # [N]

    iou = box_iou(
        torch.tensor(preds_xyxy, dtype=torch.float32),
        torch.tensor(gt_boxes,   dtype=torch.float32),
    ).numpy()  # [M, N]

    matched_gt = set()
    # Sort by confidence descending
    order = np.argsort(-pred_conf)
    for i in order:
        cls_i = pred_cls[i]
        # Candidates: GT boxes with matching class and IoU >= threshold
        candidates = np.where((gt_cls == cls_i) & (iou[i] >= iou_thresh))[0]
        if len(candidates) == 0:
            continue
        # Pick highest IoU candidate not already matched
        best = candidates[np.argmax(iou[i, candidates])]
        if best not in matched_gt:
            tp[i] = True
            matched_gt.add(best)
    return tp


def run_online_tent_eval(best_weights_path, hp, ds_name="pictor_ppe"):
    eval_cfg = hp["eval"]
    tent_cfg = hp["tent"]
    imgsz    = eval_cfg["imgsz"]
    batch_sz = tent_cfg.get("adapt_batch", eval_cfg["batch"])
    tmp_root = config.RUNS_DIR / "tmp_remap"

    tmp_yaml = make_remapped_yaml(ds_name, tmp_root)

    device = torch.device(f"cuda:{eval_cfg['device']}" if isinstance(eval_cfg['device'], int) else eval_cfg['device'])
    model  = YOLO(str(best_weights_path))
    model.model.to(device)
    lb     = LetterBox(new_shape=(imgsz, imgsz))

    adapter = TENT(model, lr=tent_cfg["lr"], steps=tent_cfg["steps"])

    img_dir   = config.DATA_DIR / ds_name / "images" / "test"
    img_paths = sorted(img_dir.glob("*"))

    all_tp, all_conf, all_pred_cls, all_gt_cls = [], [], [], []
    n_gt_per_cls = {}

    n_batches = (len(img_paths) + batch_sz - 1) // batch_sz
    print(f"  [online TENT] {ds_name}: {len(img_paths)} images, {n_batches} batches, steps={tent_cfg['steps']}, lr={tent_cfg['lr']}")

    for batch_idx, i in enumerate(range(0, len(img_paths), batch_sz)):
        batch_paths = img_paths[i : i + batch_sz]
        print(f"\r    batch {batch_idx+1}/{n_batches}", end="", flush=True)

        # ── 1. Adapt on this batch (train mode, gradients enabled) ──────────
        x = load_batch_tensors(batch_paths, lb, device)
        if x is None:
            continue
        adapter.step_tensor(x)

        # ── 2. Sync running stats to test batch, then predict ────────────────
        # Problem: adaptation uses batch stats (train mode, track_running_stats=False),
        # but eval mode uses frozen SH17 running stats → mismatch corrupts activations.
        # Fix: one forward pass with track_running_stats=True to sync running stats
        # to the current batch, then eval mode inference is consistent.
        with torch.no_grad():
            for m in model.model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True
                    m.num_batches_tracked.zero_()  # force momentum=1 → full replacement
            model.model(x, augment=False)          # updates running_mean/var
            for m in model.model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False   # back to batch-stat mode

        model.model.eval()
        with torch.no_grad():
            raw = model.model(x, augment=False)
        model.model.train()

        # ── 3. Decode raw output → boxes ────────────────────────────────────
        # YOLOv8 eval mode returns a tuple: (pred_tensor [B, 4+nc+...], protos?)
        # Use ultralytics NMS to get final detections
        if isinstance(raw, (list, tuple)):
            pred_tensor = raw[0]  # [B, num_anchors, 4+nc]
        else:
            pred_tensor = raw

        # non_max_suppression expects [B, 4+nc, num_anchors] — exactly the eval-mode output shape
        nms_out = non_max_suppression(
            pred_tensor,
            conf_thres=eval_cfg["conf"],
            iou_thres=eval_cfg["iou"],
            max_det=300,
        )  # list of [M, 6] tensors: x1y1x2y2, conf, cls

        # ── 4. Load GT and match ─────────────────────────────────────────────
        gt_list = load_gt_labels(batch_paths, tmp_root, ds_name, imgsz)

        for j, (dets, gt) in enumerate(zip(nms_out, gt_list)):
            # Count GT boxes per class
            for cls in gt[:, 0].astype(int):
                n_gt_per_cls[cls] = n_gt_per_cls.get(cls, 0) + 1
                all_gt_cls.append(cls)

            if len(dets) == 0:
                continue

            dets = dets.cpu().numpy()
            preds_xyxy = dets[:, :4]
            pred_conf  = dets[:, 4]
            pred_cls   = dets[:, 5].astype(int)

            tp = match_predictions(preds_xyxy, pred_conf, pred_cls, gt, iou_thresh=eval_cfg["iou"])
            all_tp.extend(tp)
            all_conf.extend(pred_conf)
            all_pred_cls.extend(pred_cls)

    print()  # newline after \r progress

    if not all_conf:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    # ap_per_class expects tp as [N, num_iou_thresholds] — one column per threshold
    tp_arr       = np.array(all_tp,       dtype=bool).reshape(-1, 1)
    conf_arr     = np.array(all_conf,     dtype=np.float32)
    pred_cls_arr = np.array(all_pred_cls, dtype=int)
    gt_cls_arr   = np.array(all_gt_cls,   dtype=int)

    # ap_per_class returns: tp, fp, p, r, f1, ap, ap_class, ...
    result = ap_per_class(
        tp_arr,
        conf_arr,
        pred_cls_arr,
        gt_cls_arr,
        names={k: v for k, v in config.SH17_EVAL_IDX.items()},
    )
    ap      = result[5]   # [num_classes, num_iou_thresholds] or [num_classes]
    ap_cls  = result[6]   # class indices

    # Filter to eval classes and compute mAP50
    eval_cls = set(config.SH17_EVAL_IDX.keys())
    ap50_vals = []
    per_cls = {}
    for idx, cls in enumerate(ap_cls):
        if cls in eval_cls:
            ap50 = float(ap[idx, 0] if ap.ndim == 2 else ap[idx])
            ap50_vals.append(ap50)
            per_cls[config.SH17_EVAL_IDX[cls]] = ap50

    map50 = np.mean(ap50_vals) if ap50_vals else 0.0
    return {"mAP50": float(map50), "mAP50_95": 0.0, "per_class": per_cls}


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--domain",  default="pictor_ppe",
                    help="dataset name or 'all' for pictor_ppe+shwd+chv")
args = parser.parse_args()

hp = load(args.hparams)
best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"

if not best_weights.exists():
    print(f"ERROR: no weights at {best_weights}")
    sys.exit(1)

domains = ["pictor_ppe", "shwd", "chv"] if args.domain == "all" else [args.domain]

eval_cfg = hp["eval"]

def _compute_baseline(ds_name):
    from src.eval import make_remapped_yaml, _extract_metrics
    tmp_root = config.RUNS_DIR / "tmp_remap"
    tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
    m = YOLO(str(best_weights))
    metrics = m.val(
        data=str(tmp_yaml), imgsz=eval_cfg["imgsz"], batch=eval_cfg["batch"],
        device=eval_cfg["device"], half=eval_cfg["half"],
        workers=eval_cfg.get("workers", 8),
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name=f"baseline_{ds_name}", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    return _extract_metrics(metrics, config.SH17_EVAL_IDX)["mAP50"]

W = 60
print()
_adapt_batch = hp["tent"].get("adapt_batch", hp["eval"]["batch"])
print(f"  Online TENT  |  {hp['run_name']}  |  steps={hp['tent']['steps']}  lr={hp['tent']['lr']}  adapt_batch={_adapt_batch}")
print(f"  {'='*W}")
print(f"  {'domain':<16}  {'baseline':>10}  {'online-tent':>11}  {'recovery':>10}")
print(f"  {'-'*W}")

domain_results = {}

for ds in domains:
    print(f"  computing baseline for {ds}...", end=" ", flush=True)
    base = _compute_baseline(ds)
    print(f"mAP50={base:.4f}")
    r    = run_online_tent_eval(best_weights, hp, ds_name=ds)
    rec  = r["mAP50"] - base
    sign = "+" if rec >= 0 else ""
    print(f"  {ds:<16}  {base:>10.4f}  {r['mAP50']:>11.4f}  {sign}{rec:>9.4f}")
    if r.get("per_class"):
        for cls_name, ap in r["per_class"].items():
            print(f"    {cls_name:<20}  {ap:.4f}")

    entry = {"baseline": base, "tent": r["mAP50"], "recovery": rec}
    entry.update({f"{k}_AP50": v for k, v in (r.get("per_class") or {}).items()})
    domain_results[ds] = entry

print(f"  {'='*W}")
print()

# ── Persist results ───────────────────────────────────────────────────────────
out_dir = config.RUNS_DIR / "results" / hp["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
tent_cfg = hp["tent"]
payload = {
    "run_name":    hp["run_name"],
    "timestamp":   datetime.now().isoformat(),
    "steps":       tent_cfg["steps"],
    "lr":          tent_cfg["lr"],
    "adapt_batch": tent_cfg.get("adapt_batch", hp["eval"]["batch"]),
}
payload.update(domain_results)
out_path = out_dir / "online_tent.json"
out_path.write_text(json.dumps(payload, indent=2))
print(f"  Saved → {out_path}")
