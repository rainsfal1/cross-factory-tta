"""
Zero-shot OWL-ViT evaluation on all three target domains.

Uses `google/owlvit-large-patch14` with text queries matching the canonical
class order. Labels returned by OWL-ViT are integer indices into the text
query list — direct class mapping, no phrase decoding needed.

Usage:
    uv run python scripts/eval_owlvit.py
    uv run python scripts/eval_owlvit.py --device 1
    uv run python scripts/eval_owlvit.py --model google/owlvit-base-patch32
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor

import config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Order must match canonical label indices 0 / 1 / 2
CLASS_PROMPTS = ["hard hat", "no hard hat", "person"]
CANONICAL_NAMES = ["hard_hat", "no_hard_hat", "person"]

RUN_NAME = "owlvit_large"

TARGET_DOMAINS = [
    ("pictor_ppe", "pictor.yaml"),
    ("shwd",       "shwd.yaml"),
    ("chv",        "chv.yaml"),
]

# COCO category ids (1-based)
CAT_IDS = [1, 2, 3]


def _yolo_to_xyxy(cx, cy, w, h, iw, ih):
    cx *= iw; cy *= ih; w *= iw; h *= ih
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _load_gt_for_image(label_path, iw, ih):
    boxes, classes = [], []
    if not label_path.is_file():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xyxy = _yolo_to_xyxy(*map(float, parts[1:5]), iw, ih)
        boxes.append(xyxy)
        classes.append(cls)
    return boxes, classes


def _build_coco_gt(image_paths, label_paths):
    images_json, annotations_json = [], []
    ann_id = 1
    img_ids = []
    for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths, strict=True)):
        im = Image.open(img_path).convert("RGB")
        iw, ih = im.size
        img_id = i + 1
        img_ids.append(img_id)
        images_json.append({"id": img_id, "width": iw, "height": ih, "file_name": img_path.name})
        gt_boxes, gt_cls = _load_gt_for_image(lbl_path, iw, ih)
        for box, cls in zip(gt_boxes, gt_cls, strict=True):
            x, y, w, h = _xyxy_to_xywh(box)
            annotations_json.append({
                "id": ann_id, "image_id": img_id, "category_id": CAT_IDS[cls],
                "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1
    categories_json = [{"id": cid, "name": name} for cid, name in zip(CAT_IDS, CANONICAL_NAMES, strict=True)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images_json, "annotations": annotations_json, "categories": categories_json}
    coco_gt.createIndex()
    return coco_gt, img_ids


def _nms_xyxy(boxes, scores, labels, iou_threshold):
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold)
    return boxes[keep], scores[keep], labels[keep]


def _per_class_ap_from_coco_eval(coco_eval):
    prec = coco_eval.eval["precision"]
    cat_ids = sorted(coco_eval.cocoGt.getCatIds())
    ap50, ap5095 = {}, {}
    K = prec.shape[2]
    for k in range(K):
        cid = int(cat_ids[k])
        s = prec[0, :, k, 0, 2]; s = s[s > -1]
        ap50[cid] = float(np.mean(s)) if len(s) else 0.0
        ap_t = []
        for t in range(prec.shape[0]):
            st = prec[t, :, k, 0, 2]; st = st[st > -1]
            if len(st): ap_t.append(float(np.mean(st)))
        ap5095[cid] = float(np.mean(ap_t)) if ap_t else 0.0
    return ap50, ap5095


def _extract_metrics_payload(coco_eval):
    ap50_by_cid, ap5095_by_cid = _per_class_ap_from_coco_eval(coco_eval)
    map50   = float(np.mean([ap50_by_cid[c]   for c in CAT_IDS]))
    map5095 = float(np.mean([ap5095_by_cid[c] for c in CAT_IDS]))
    result = {"mAP50": map50, "mAP50_95": map5095, "precision": None, "recall": None}
    for cid, name in zip(CAT_IDS, CANONICAL_NAMES, strict=True):
        result[f"{name}_AP50"]    = ap50_by_cid.get(cid, 0.0)
        result[f"{name}_AP50_95"] = ap5095_by_cid.get(cid, 0.0)
    return result


def _list_val_images(ds_root):
    img_dir = ds_root / "images" / "test"
    lbl_dir = ds_root / "labels" / "test"
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    labels = [lbl_dir / (p.stem + ".txt") for p in images]
    return images, labels


def evaluate_domain(processor, model, device, ds_name, yaml_name, threshold, nms_iou):
    ds_root = config.DATA_DIR / ds_name
    if not (ds_root / yaml_name).is_file():
        raise FileNotFoundError(ds_root / yaml_name)

    image_paths, label_paths = _list_val_images(ds_root)
    if not image_paths:
        raise RuntimeError(f"No val images under {ds_root / 'images' / 'test'}")

    coco_gt, img_ids = _build_coco_gt(image_paths, label_paths)
    detections = []

    # OWL-ViT text input: list of lists (one per image in batch — batch=1 here)
    texts = [CLASS_PROMPTS]

    for img_path, img_id in tqdm(list(zip(image_paths, img_ids, strict=True)), desc=f"  {ds_name}", leave=False):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(**inputs)

        target_sizes = [(image.height, image.width)]
        # post_process_grounded_object_detection is the method on OwlViTProcessor;
        # post_process_object_detection exists only on the image_processor sub-object.
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes,
            text_labels=[CLASS_PROMPTS],   # one list per image in batch
        )[0]

        scores = results["scores"]   # (N,)
        labels = results["labels"]   # (N,) — integer indices into CLASS_PROMPTS
        boxes  = results["boxes"]    # (N, 4) xyxy absolute

        if scores.numel() == 0:
            continue

        boxes, scores, labels = _nms_xyxy(boxes, scores, labels.long(), nms_iou)

        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j].tolist()
            cls_idx = int(labels[j])
            if cls_idx >= len(CAT_IDS):
                continue
            detections.append({
                "image_id":   img_id,
                "category_id": CAT_IDS[cls_idx],
                "bbox":       [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":      float(scores[j]),
            })

    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    return _extract_metrics_payload(coco_eval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    type=int,   default=0)
    parser.add_argument("--model",     type=str,   default="google/owlvit-large-patch14")
    parser.add_argument("--threshold", type=float, default=0.001, help="box score threshold — low keeps all detections for mAP")
    parser.add_argument("--nms-iou",   type=float, default=0.6,   dest="nms_iou")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)

    global RUN_NAME
    if "large" in args.model:
        RUN_NAME = "owlvit_large"
    elif "base" in args.model:
        RUN_NAME = "owlvit_base"

    print()
    print(f"  OWL-ViT  |  zero-shot  |  model={args.model}")
    print(f"  prompts: {CLASS_PROMPTS}")
    print(f"  {'=' * 46}")
    print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}")
    print(f"  {'-' * 46}")

    processor = OwlViTProcessor.from_pretrained(args.model)
    model = OwlViTForObjectDetection.from_pretrained(args.model).to(device).eval()

    results = {}
    for ds_name, yaml_name in TARGET_DOMAINS:
        print(f"  evaluating {ds_name}...", end=" ", flush=True)
        r = evaluate_domain(processor, model, device, ds_name, yaml_name, args.threshold, args.nms_iou)
        results[ds_name] = r
        print(f"mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}")

    print(f"  {'-' * 46}")
    mean50   = sum(r["mAP50"]    for r in results.values()) / len(results)
    mean5095 = sum(r["mAP50_95"] for r in results.values()) / len(results)
    print(f"  {'Mean':<16} {mean50:>9.4f}  {mean5095:>10.4f}")
    print(f"  {'=' * 46}")
    print()

    out_dir = config.RUNS_DIR / "results" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name":      RUN_NAME,
        "model":         args.model,
        "class_prompts": CLASS_PROMPTS,
        "timestamp":     datetime.now().isoformat(),
        "threshold":     args.threshold,
        "nms_iou":       args.nms_iou,
    }
    for ds, r in results.items():
        payload[ds] = r
    out_path = out_dir / "baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
