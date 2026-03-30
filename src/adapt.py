"""
Run TENT test-time adaptation on Pictor-PPE and evaluate.

Flow:
  1. Load source-trained weights.
  2. Wrap model with TENT adapter.
  3. Process target images in batches: adapt → predict.
  4. Accumulate predictions and run mAP evaluation.
  5. Save results JSON for comparison with baseline.
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from tent import TENT

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Source-trained best.pt")
    p.add_argument("--data", default=str(ROOT / "data/pictor_ppe/pictor.yaml"))
    p.add_argument("--lr", type=float, default=0.001, help="TENT adaptation LR")
    p.add_argument("--steps", type=int, default=1, help="Gradient steps per batch")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", default="tent_pictor")
    p.add_argument("--project", default=str(ROOT / "results/tent"))
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------------------------
    # Phase 1: evaluate WITHOUT adaptation (baseline reference, same weights)
    # --------------------------------------------------------------------------
    print("\n=== Phase 1: Source-only zero-shot evaluation ===")
    base_model = YOLO(args.weights)
    base_metrics = base_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        conf=args.conf,
        iou=args.iou,
        name=args.name + "_baseline",
        project=args.project,
        exist_ok=True,
        verbose=False,
    )
    baseline_map50 = float(base_metrics.box.map50)
    print(f"  Baseline mAP50: {baseline_map50:.4f}")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --------------------------------------------------------------------------
    # Phase 2: TENT adapted evaluation
    # --------------------------------------------------------------------------
    print(f"\n=== Phase 2: TENT adaptation (lr={args.lr}, steps={args.steps}) ===")
    model = YOLO(args.weights)
    adapter = TENT(model, lr=args.lr, steps=args.steps)

    # Run TENT-adapted validation using ultralytics' built-in val loop
    # We hook the TENT step into each batch via a custom callback
    adapted_predictions = []

    def on_val_batch_start(validator):
        """
        Callback: adapt the model on each validation batch before inference.
        This runs the entropy minimization gradient step.
        """
        batch_imgs = validator.batch.get("im_file", [])
        if batch_imgs:
            adapter.step(list(batch_imgs))

    model.add_callback("on_val_batch_start", on_val_batch_start)

    tent_metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        conf=args.conf,
        iou=args.iou,
        name=args.name,
        project=args.project,
        exist_ok=True,
        verbose=True,
    )
    tent_map50 = float(tent_metrics.box.map50)
    recovery = tent_map50 - baseline_map50

    # --------------------------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------------------------
    results = {
        "weights": args.weights,
        "data": args.data,
        "tent_lr": args.lr,
        "tent_steps": args.steps,
        "baseline_mAP50": baseline_map50,
        "tent_mAP50": tent_map50,
        "recovery_mAP50": recovery,
        "baseline_mAP50_95": float(base_metrics.box.map) if False else "see baseline run",
        "tent_mAP50_95": float(tent_metrics.box.map),
        "per_class_AP50": {
            name: float(ap)
            for name, ap in zip(tent_metrics.names.values(), tent_metrics.box.ap50)
        },
    }

    out_path = Path(args.project) / args.name / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== TENT Results ===")
    print(f"Baseline mAP50 (source-only): {baseline_map50:.4f}")
    print(f"TENT mAP50 (adapted):         {tent_map50:.4f}")
    print(f"Recovery:                     {recovery:+.4f}")
    for cls, ap in results["per_class_AP50"].items():
        print(f"  {cls}: {ap:.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
