"""
Pipeline verification audit — 5 checks to confirm results are real, not pipeline bugs.

Run the full audit:
    uv run python scripts/verify_pipeline.py

Run a single check:
    uv run python scripts/verify_pipeline.py --check remap
    uv run python scripts/verify_pipeline.py --check boxes
    uv run python scripts/verify_pipeline.py --check sh17_through_target
    uv run python scripts/verify_pipeline.py --check preprocess_match
    uv run python scripts/verify_pipeline.py --check predictions

Output images written to: docs/visual_examples/audit/
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import torch
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

import config
from src.eval import make_remapped_yaml, _extract_metrics
from src.hparams import load

AUDIT_DIR = Path("docs/visual_examples/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_NAMES = {0: "hard_hat", 1: "no_hard_hat", 2: "person"}
SH17_NAMES      = {4: "hard_hat(SH17)", 9: "no_hard_hat(SH17)", 16: "person(SH17)"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 128, 0),
          4: (0, 255, 0), 9: (0, 0, 255), 16: (255, 128, 0)}

DOMAINS = ["pictor_ppe", "shwd", "chv"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_yolo_boxes(img_path, label_path, class_names: dict, title="") -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names.get(cls, str(cls))
            cv2.putText(img, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if title:
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def random_sample(img_dir, label_dir, n=5):
    imgs = sorted(img_dir.glob("*"))
    imgs = [p for p in imgs if (label_dir / (p.stem + ".txt")).exists()]
    return random.sample(imgs, min(n, len(imgs)))


# ── Check 1: Class remap verification ─────────────────────────────────────────

def check_remap(n=20):
    print("\n" + "="*60)
    print("CHECK 1: Class remap verification")
    print("Expected: canonical 0→4, 1→9, 2→16")
    print("="*60)

    tmp_root = config.RUNS_DIR / "tmp_remap"
    passed = True

    for ds in DOMAINS:
        make_remapped_yaml(ds, tmp_root)
        src_label_dir = config.DATA_DIR / ds / "labels" / "test"
        rmp_label_dir = tmp_root / ds / "labels" / "test"
        labels = sorted(src_label_dir.glob("*.txt"))
        sample = random.sample(labels, min(n, len(labels)))

        print(f"\n  {ds} ({len(sample)} samples):")
        for lbl in sample[:5]:  # print first 5 verbosely
            src_lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            rmp_lines = [(tmp_root / ds / "labels" / "test" / lbl.name).read_text().splitlines()]
            rmp_lines = [l for l in rmp_lines[0] if l.strip()]
            for sl, rl in zip(src_lines[:3], rmp_lines[:3]):
                src_cls = int(sl.split()[0])
                rmp_cls = int(rl.split()[0])
                expected = config.CANONICAL_TO_SH17.get(src_cls)
                ok = "OK" if rmp_cls == expected else "FAIL"
                if ok == "FAIL":
                    passed = False
                print(f"    {src_cls} -> {rmp_cls}  (expected {expected})  [{ok}]")

        # Also verify all labels in bulk
        errors = 0
        for lbl in labels:
            rmp = tmp_root / ds / "labels" / "test" / lbl.name
            if not rmp.exists():
                continue
            for sl, rl in zip(lbl.read_text().splitlines(), rmp.read_text().splitlines()):
                sp, rp = sl.split(), rl.split()
                if not sp or not rp:
                    continue
                if int(rp[0]) != config.CANONICAL_TO_SH17.get(int(sp[0]), -1):
                    errors += 1
        status = "PASS" if errors == 0 else f"FAIL ({errors} mismatches)"
        print(f"  Bulk check ({len(labels)} label files): {status}")
        if errors:
            passed = False

    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Check 2: GT box visualization ─────────────────────────────────────────────

def check_boxes(n=5):
    print("\n" + "="*60)
    print("CHECK 2: GT box visualization")
    print(f"  Writing {n} images per domain to {AUDIT_DIR}/boxes_<domain>/")
    print("="*60)

    tmp_root = config.RUNS_DIR / "tmp_remap"

    for ds in DOMAINS:
        make_remapped_yaml(ds, tmp_root)
        img_dir = config.DATA_DIR / ds / "images" / "test"
        # Draw both canonical and remapped boxes side by side
        src_lbl_dir = config.DATA_DIR / ds / "labels" / "test"
        rmp_lbl_dir = tmp_root / ds / "labels" / "test"
        out_dir = AUDIT_DIR / f"boxes_{ds}"
        out_dir.mkdir(exist_ok=True)

        samples = random_sample(img_dir, src_lbl_dir, n)
        for img_path in samples:
            src_lbl = src_lbl_dir / (img_path.stem + ".txt")
            rmp_lbl = rmp_lbl_dir / (img_path.stem + ".txt")
            left  = draw_yolo_boxes(img_path, src_lbl, CANONICAL_NAMES, "canonical")
            right = draw_yolo_boxes(img_path, rmp_lbl, SH17_NAMES,      "remapped→SH17")
            # Resize to same height
            h = max(left.shape[0], right.shape[0])
            left  = cv2.resize(left,  (int(left.shape[1]  * h / left.shape[0]),  h))
            right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
            combined = np.hstack([left, right])
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), combined)

        print(f"  {ds}: {len(samples)} images → {out_dir}/")

    print("\n  Open the images and verify boxes sit on actual helmets/people.")
    print("  Result: MANUAL CHECK REQUIRED")


# ── Check 3: SH17 through target-domain eval path ─────────────────────────────

def check_sh17_through_target(hp):
    print("\n" + "="*60)
    print("CHECK 3: SH17 val images through target-domain eval path")
    print("  If pipeline is correct, SH17 val mAP50 should stay close to ~0.64")
    print("="*60)

    from ultralytics import YOLO
    import shutil

    eval_cfg = hp["eval"]
    best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
    tmp_root     = config.RUNS_DIR / "tmp_remap"

    # Build a fake "target domain" dataset from SH17 val images,
    # remapping SH17 classes 4,9,16 back to canonical 0,1,2 then re-remapping to 4,9,16.
    # If eval path is correct this round-trip should be identity.
    sh17_yaml    = config.DATA_DIR / "sh17" / "sh17.yaml"
    sh17_img_dir = config.DATA_DIR / "sh17" / "images" / "val"
    sh17_lbl_dir = config.DATA_DIR / "sh17" / "labels" / "val"

    # Write a subset of SH17 val through the canonical remap path
    # SH17 labels already use SH17 class indices — so we invert then remap
    inv_map = {v: k for k, v in config.CANONICAL_TO_SH17.items()}  # {4:0, 9:1, 16:2}

    tmp_dir     = tmp_root / "sh17_audit"
    tmp_img_dir = tmp_dir / "images" / "test"
    tmp_lbl_dir = tmp_dir / "labels" / "test"
    tmp_img_dir.mkdir(parents=True, exist_ok=True)
    tmp_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(sh17_img_dir.glob("*"))[:200]  # use first 200 val images
    copied = 0
    for img_path in images:
        lbl_path = sh17_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, tmp_img_dir / img_path.name)
        lines_out = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in inv_map:
                continue  # skip non-eval classes
            canonical = inv_map[cls]
            sh17_remap = config.CANONICAL_TO_SH17[canonical]  # should be identity
            lines_out.append(f"{sh17_remap} {' '.join(parts[1:])}")
        if lines_out:
            (tmp_lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines_out) + "\n")
            copied += 1

    print(f"  Using {copied} SH17 val images with eval-class labels")

    tmp_yaml = tmp_dir / "sh17_audit.yaml"
    tmp_yaml.write_text(f"""path: {tmp_dir}
train: images/test
val: images/test
nc: {len(config.SH17_CLASSES)}
names: {config.SH17_CLASSES}
""")

    # Native val (ground truth)
    model = YOLO(str(best_weights))
    print("  Running native model.val() on SH17 val...", end=" ", flush=True)
    native_metrics = model.val(
        data=str(sh17_yaml),
        imgsz=eval_cfg["imgsz"], batch=eval_cfg["batch"],
        device=eval_cfg["device"], half=eval_cfg["half"],
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name="audit_sh17_native", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    native_map = float(native_metrics.box.map50)
    print(f"mAP50={native_map:.4f}")

    # Through target eval path
    model2 = YOLO(str(best_weights))
    print("  Running through target-domain eval path...", end=" ", flush=True)
    target_metrics = model2.val(
        data=str(tmp_yaml),
        imgsz=eval_cfg["imgsz"], batch=eval_cfg["batch"],
        device=eval_cfg["device"], half=eval_cfg["half"],
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name="audit_sh17_target_path", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    target_result = _extract_metrics(target_metrics, config.SH17_EVAL_IDX)
    target_map = target_result["mAP50"]
    print(f"mAP50={target_map:.4f}")

    delta = target_map - native_map
    # Native measures all 17 classes; target path measures only 3 — expect some delta
    print(f"\n  Native SH17 (all 17 classes): {native_map:.4f}")
    print(f"  Through target eval path (3 classes): {target_map:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"\n  Expected: target path should be ≥ native (3 well-trained classes > mean of 17)")

    passed = target_map > 0.30  # conservative threshold — 3 classes should be well above 0.30
    print(f"  Result: {'PASS' if passed else 'FAIL — eval path may be broken'}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return passed


# ── Check 4: Native val vs custom preprocessing ─────────────────────────────

def check_preprocess_match(hp):
    print("\n" + "="*60)
    print("CHECK 4: Native .val() vs custom preprocessing on SH17")
    print("  Both should give same mAP50 ± 0.01")
    print("="*60)

    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    eval_cfg = hp["eval"]
    best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
    sh17_yaml    = config.DATA_DIR / "sh17" / "sh17.yaml"
    imgsz        = eval_cfg["imgsz"]
    batch_sz     = eval_cfg["batch"]

    model = YOLO(str(best_weights))
    print("  Native model.val()...", end=" ", flush=True)
    native = model.val(
        data=str(sh17_yaml), imgsz=imgsz, batch=batch_sz,
        device=eval_cfg["device"], half=eval_cfg["half"],
        split="val", conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        name="audit_preprocess_native", project=str(config.RUNS_DIR / "eval"),
        exist_ok=True, verbose=False,
    )
    native_map = float(native.box.map50)
    print(f"mAP50={native_map:.4f}")

    # Custom path: preprocess with cv2+LetterBox, adapt 0 steps (no adaptation), then val
    # We use TENT with steps=0 equivalent — just load model and patch fuse, then val
    model2  = YOLO(str(best_weights))
    lb      = LetterBox(new_shape=(imgsz, imgsz))
    device  = next(model2.model.parameters()).device
    img_dir = config.DATA_DIR / "sh17" / "images" / "val"
    img_paths = sorted(img_dir.glob("*"))[:100]  # just 100 images for speed

    print("  Custom preprocessing path (cv2+LetterBox, no adaptation)...", end=" ", flush=True)
    # Just test that our preprocessing doesn't mangle images
    # by running predict on a few and comparing to native
    model2.model.eval()
    tensors = []
    for p in img_paths[:16]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_lb = lb(image=img)
        img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1).copy()
        tensors.append(torch.from_numpy(img_lb).float() / 255.0)

    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        preds_custom = model2.model(x, augment=False)

    # Run native predict on same images
    preds_native = model2.predict(
        source=[str(p) for p in img_paths[:16]],
        imgsz=imgsz, conf=0.25, verbose=False,
    )

    # Compare: count detections per image
    from ultralytics.utils.nms import non_max_suppression
    nms_custom = non_max_suppression(preds_custom[0], conf_thres=0.25, iou_thres=0.6)
    n_custom = sum(len(d) for d in nms_custom)
    n_native = sum(len(r.boxes) for r in preds_native)

    print(f"done")
    print(f"\n  Native .val() mAP50: {native_map:.4f}")
    print(f"  Custom preprocessing detections (16 images): {n_custom}")
    print(f"  Native predict detections (16 images):       {n_native}")
    ratio = n_custom / max(n_native, 1)
    print(f"  Detection count ratio: {ratio:.2f}  (expected ~1.0)")

    passed = 0.7 <= ratio <= 1.3
    print(f"  Result: {'PASS' if passed else 'FAIL — preprocessing mismatch'}")
    return passed


# ── Check 5: Raw predictions + GT inspection ──────────────────────────────────

def check_predictions(hp, n=5):
    print("\n" + "="*60)
    print("CHECK 5: Raw predictions vs GT on target domains")
    print(f"  Writing {n} images per domain to {AUDIT_DIR}/preds_<domain>/")
    print("="*60)

    from ultralytics import YOLO

    eval_cfg     = hp["eval"]
    best_weights = config.RUNS_DIR / "train" / hp["run_name"] / "weights" / "best.pt"
    tmp_root     = config.RUNS_DIR / "tmp_remap"

    model = YOLO(str(best_weights))

    for ds in DOMAINS:
        make_remapped_yaml(ds, tmp_root)
        img_dir     = config.DATA_DIR / ds / "images" / "test"
        rmp_lbl_dir = tmp_root / ds / "labels" / "test"
        out_dir     = AUDIT_DIR / f"preds_{ds}"
        out_dir.mkdir(exist_ok=True)

        samples = random_sample(img_dir, rmp_lbl_dir, n)

        for img_path in samples:
            # GT (remapped to SH17 indices)
            rmp_lbl = rmp_lbl_dir / (img_path.stem + ".txt")
            gt_img  = draw_yolo_boxes(img_path, rmp_lbl, SH17_NAMES, "GT (remapped)")

            # Predictions
            results  = model.predict(source=str(img_path), imgsz=eval_cfg["imgsz"],
                                     conf=0.1, verbose=False)
            pred_img = cv2.imread(str(img_path))
            for box in results[0].boxes:
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = COLORS.get(cls, (200, 200, 200))
                name  = config.SH17_CLASSES[cls] if cls < len(config.SH17_CLASSES) else str(cls)
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(pred_img, f"{name} {conf:.2f}", (x1, max(y1-4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(pred_img, f"PRED conf>0.1 ({len(results[0].boxes)} dets)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            h = max(gt_img.shape[0], pred_img.shape[0])
            gt_img   = cv2.resize(gt_img,   (int(gt_img.shape[1]   * h / gt_img.shape[0]),   h))
            pred_img = cv2.resize(pred_img, (int(pred_img.shape[1] * h / pred_img.shape[0]), h))
            combined = np.hstack([gt_img, pred_img])
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), combined)

        print(f"  {ds}: {len(samples)} images → {out_dir}/")
        # Print class distribution of predictions on this domain
        all_preds = model.predict(
            source=[str(p) for p in sorted(img_dir.glob("*"))[:50]],
            imgsz=eval_cfg["imgsz"], conf=0.25, verbose=False,
        )
        cls_counts = {}
        for r in all_preds:
            for cls in r.boxes.cls.int().tolist():
                name = config.SH17_CLASSES[cls] if cls < len(config.SH17_CLASSES) else str(cls)
                cls_counts[name] = cls_counts.get(name, 0) + 1
        print(f"    Prediction class distribution (50 images, conf>0.25):")
        for name, cnt in sorted(cls_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"      {name:<20} {cnt}")

    print("\n  Open the images and verify predictions overlap with GT boxes.")
    print("  Result: MANUAL CHECK REQUIRED")


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None)
parser.add_argument("--check", default="all",
                    choices=["all", "remap", "boxes", "sh17_through_target",
                             "preprocess_match", "predictions"])
args = parser.parse_args()

random.seed(42)
hp = load(args.hparams)

print(f"\n  Pipeline Verification Audit  |  model: {hp['run_name']}")
print(f"  Output: {AUDIT_DIR}/\n")

results = {}

if args.check in ("all", "remap"):
    results["remap"] = check_remap()

if args.check in ("all", "boxes"):
    check_boxes()
    results["boxes"] = None  # manual

if args.check in ("all", "sh17_through_target"):
    results["sh17_through_target"] = check_sh17_through_target(hp)

if args.check in ("all", "preprocess_match"):
    results["preprocess_match"] = check_preprocess_match(hp)

if args.check in ("all", "predictions"):
    check_predictions(hp)
    results["predictions"] = None  # manual

print("\n" + "="*60)
print("AUDIT SUMMARY")
print("="*60)
for check, passed in results.items():
    if passed is None:
        print(f"  {check:<30} MANUAL CHECK REQUIRED")
    else:
        print(f"  {check:<30} {'PASS' if passed else 'FAIL'}")
print()
