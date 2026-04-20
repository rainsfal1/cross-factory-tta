"""
Zero-shot Grounding DINO (base) evaluation on all three target domains.

Uses Hugging Face `IDEA-Research/grounding-dino-base` with the same text
phrases as YOLO-World (`hard hat`, `no hard hat`, `person`). Metrics are
computed with pycocotools (COCO-style mAP) on the val split (prepared
`images/test` + YOLO labels), aligned with `scripts/eval_yolo_world.py`
result layout.

Usage:
    uv run python scripts/eval_grounding_dino.py
    uv run python scripts/eval_grounding_dino.py --device 1
    uv run python scripts/eval_grounding_dino.py --model IDEA-Research/grounding-dino-tiny
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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Same phrases / order as eval_yolo_world.py (canonical indices 0 / 1 / 2)
CLASS_PROMPTS = ["hard hat", "no hard hat", "person"]
CANONICAL_NAMES = ["hard_hat", "no_hard_hat", "person"]

RUN_NAME = "grounding_dino_base"

TARGET_DOMAINS = [
    ("pictor_ppe", "pictor.yaml"),
    ("shwd", "shwd.yaml"),
    ("chv", "chv.yaml"),
]

# COCO category ids (1-based) — must match GT and predictions
CAT_IDS = [1, 2, 3]  # hard_hat, no_hard_hat, person


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, iw: int, ih: int) -> list[float]:
    cx *= iw
    cy *= ih
    w *= iw
    h *= ih
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def _load_gt_for_image(label_path: Path, iw: int, ih: int) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.is_file():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        xyxy = _yolo_to_xyxy(cx, cy, w, h, iw, ih)
        boxes.append(xyxy)
        classes.append(cls)
    return boxes, classes


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _map_phrase_to_class(text: str) -> int | None:
    t = text.strip().lower()
    # exact match first
    for i, phrase in enumerate(CLASS_PROMPTS):
        if t == phrase.lower():
            return i
    # substring match — longest phrase first so "no hard hat" beats "hard hat"
    for i, phrase in sorted(enumerate(CLASS_PROMPTS), key=lambda x: -len(x[1])):
        p = phrase.lower()
        if p in t or t in p:
            return i
    return None


def _build_coco_gt(
    image_paths: list[Path],
    label_paths: list[Path],
) -> tuple[COCO, list[int]]:
    images_json = []
    annotations_json = []
    ann_id = 1
    img_ids: list[int] = []

    for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths, strict=True)):
        im = Image.open(img_path).convert("RGB")
        iw, ih = im.size
        img_id = i + 1
        img_ids.append(img_id)
        images_json.append({"id": img_id, "width": iw, "height": ih, "file_name": img_path.name})

        gt_boxes, gt_cls = _load_gt_for_image(lbl_path, iw, ih)
        for box, cls in zip(gt_boxes, gt_cls, strict=True):
            x, y, w, h = _xyxy_to_xywh(box)
            annotations_json.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CAT_IDS[cls],
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories_json = [
        {"id": cid, "name": name} for cid, name in zip(CAT_IDS, CANONICAL_NAMES, strict=True)
    ]
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": categories_json,
    }
    coco_gt.createIndex()
    return coco_gt, img_ids


def _coco_detections_from_preds(
    img_id: int,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
) -> list[dict]:
    out = []
    for j in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[j].tolist()
        w, h = x2 - x1, y2 - y1
        out.append(
            {
                "image_id": img_id,
                "category_id": CAT_IDS[int(class_ids[j])],
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scores[j]),
            }
        )
    return out


def _nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold)
    return boxes[keep], scores[keep], labels[keep]


def _per_class_ap_from_coco_eval(coco_eval: COCOeval) -> tuple[dict[int, float], dict[int, float]]:
    """Per-category AP50 and AP50-95 from COCOeval (area=all, maxDets=100)."""
    prec = coco_eval.eval["precision"]
    # [T, R, K, A, M]
    T, _, K, _, _ = prec.shape
    a_idx = 0  # all
    m_idx = 2  # maxDets 100

    cat_ids = sorted(coco_eval.cocoGt.getCatIds())
    ap50: dict[int, float] = {}
    ap5095: dict[int, float] = {}

    for k in range(K):
        cid = int(cat_ids[k])
        s = prec[0, :, k, a_idx, m_idx]
        s = s[s > -1]
        ap50[cid] = float(np.mean(s)) if len(s) else 0.0

        ap_t: list[float] = []
        for t in range(T):
            st = prec[t, :, k, a_idx, m_idx]
            st = st[st > -1]
            if len(st):
                ap_t.append(float(np.mean(st)))
        ap5095[cid] = float(np.mean(ap_t)) if ap_t else 0.0

    return ap50, ap5095


def _macro_mean_from_cat_dict(ap_by_cid: dict[int, float]) -> float:
    vals = [ap_by_cid[cid] for cid in CAT_IDS]
    return float(np.mean(vals)) if vals else 0.0


def _extract_metrics_payload(coco_eval: COCOeval) -> dict:
    ap50_by_cid, ap5095_by_cid = _per_class_ap_from_coco_eval(coco_eval)
    map50 = _macro_mean_from_cat_dict(ap50_by_cid)
    map5095 = _macro_mean_from_cat_dict(ap5095_by_cid)

    result: dict = {
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": None,
        "recall": None,
    }
    for cid, name in zip(CAT_IDS, CANONICAL_NAMES, strict=True):
        result[f"{name}_AP50"] = ap50_by_cid.get(cid, 0.0)
        result[f"{name}_AP50_95"] = ap5095_by_cid.get(cid, 0.0)

    return result


def _list_val_images(ds_root: Path) -> tuple[list[Path], list[Path]]:
    img_dir = ds_root / "images" / "test"
    lbl_dir = ds_root / "labels" / "test"
    images = sorted(img_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    labels = [lbl_dir / (p.stem + ".txt") for p in images]
    return images, labels


def evaluate_domain(
    processor,
    model,
    device: torch.device,
    ds_name: str,
    yaml_name: str,
    threshold: float,
    text_threshold: float,
    nms_iou: float,
) -> dict:
    ds_root = config.DATA_DIR / ds_name
    data_yaml = ds_root / yaml_name
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    image_paths, label_paths = _list_val_images(ds_root)
    if not image_paths:
        raise RuntimeError(f"No val images under {ds_root / 'images' / 'test'}")

    coco_gt, img_ids = _build_coco_gt(image_paths, label_paths)
    detections: list[dict] = []

    model.eval()

    # Candidate labels are merged internally to "hard hat. no hard hat. person."
    text_in = list(CLASS_PROMPTS)

    for img_path, img_id in tqdm(
        list(zip(image_paths, img_ids, strict=True)),
        desc=f"  {ds_name}",
        leave=False,
    ):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=text_in, return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],  # explicit — safer than relying on outputs.input_ids
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

        scores_t = results["scores"]
        boxes_t = results["boxes"]
        text_labels = results.get("text_labels") or results.get("labels") or []

        keep_scores: list[torch.Tensor] = []
        keep_boxes: list[torch.Tensor] = []
        keep_cls: list[int] = []
        for score, box, lbl in zip(scores_t, boxes_t, text_labels, strict=True):
            raw = lbl if isinstance(lbl, str) else processor.tokenizer.decode(lbl)
            cls = _map_phrase_to_class(raw)
            if cls is None:
                continue
            keep_scores.append(score)
            keep_boxes.append(box)
            keep_cls.append(cls)

        if not keep_scores:
            continue

        scores = torch.stack(keep_scores)
        boxes = torch.stack(keep_boxes)
        labels_t = torch.tensor(keep_cls, device=scores.device, dtype=torch.long)
        boxes, scores, labels_t = _nms_xyxy(boxes, scores, labels_t, nms_iou)

        detections.extend(_coco_detections_from_preds(img_id, boxes.cpu(), scores.cpu(), labels_t.cpu()))

    if not detections:
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    return _extract_metrics_payload(coco_eval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1, help="reserved (inference is one image at a time)")
    parser.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--threshold", type=float, default=0.001, help="box score threshold — low keeps all detections for mAP (like YOLO conf=0.001)")
    parser.add_argument("--text-threshold", type=float, default=0.25, dest="text_threshold",
                        help="token activation threshold for label decoding — use GDINO default 0.25, NOT 0.001")
    parser.add_argument("--nms-iou", type=float, default=0.6, dest="nms_iou")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)

    global RUN_NAME
    if "tiny" in args.model.lower():
        RUN_NAME = "grounding_dino_tiny"
    elif "base" in args.model.lower():
        RUN_NAME = "grounding_dino_base"

    print()
    print(f"  Grounding DINO  |  zero-shot  |  model={args.model}")
    print(f"  prompts: {CLASS_PROMPTS}")
    print(f"  {'=' * 46}")
    print(f"  {'Dataset':<16} {'mAP50':>9}  {'mAP50-95':>10}")
    print(f"  {'-' * 46}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    results: dict[str, dict] = {}

    for ds_name, yaml_name in TARGET_DOMAINS:
        print(f"  evaluating {ds_name}...", end=" ", flush=True)
        r = evaluate_domain(
            processor,
            model,
            device,
            ds_name,
            yaml_name,
            args.threshold,
            args.text_threshold,
            args.nms_iou,
        )
        results[ds_name] = r
        print(f"mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}")

    print(f"  {'-' * 46}")
    mean50 = sum(r["mAP50"] for r in results.values()) / len(results)
    mean5095 = sum(r["mAP50_95"] for r in results.values()) / len(results)
    print(f"  {'Mean':<16} {mean50:>9.4f}  {mean5095:>10.4f}")
    print(f"  {'=' * 46}")
    print()

    out_dir = config.RUNS_DIR / "results" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_name": RUN_NAME,
        "model": args.model,
        "class_prompts": CLASS_PROMPTS,
        "timestamp": datetime.now().isoformat(),
        "threshold": args.threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
    }
    for ds, r in results.items():
        payload[ds] = r

    out_path = out_dir / "baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
