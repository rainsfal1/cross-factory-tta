"""
Zero-shot evaluation of a trained model on a target domain.

Produces per-class AP and mAP50 / mAP50-95, writes JSON to results/.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--data", default=str(ROOT / "data/pictor_ppe/pictor.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", default="val", help="val or test split")
    p.add_argument("--name", default="baseline_pictor")
    p.add_argument("--project", default=str(ROOT / "results/baseline"))
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        name=args.name,
        project=args.project,
        conf=args.conf,
        iou=args.iou,
        exist_ok=True,
        verbose=True,
    )

    results = {
        "weights": args.weights,
        "data": args.data,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_AP50": {
            name: float(ap)
            for name, ap in zip(metrics.names.values(), metrics.box.ap50)
        },
    }

    out_path = (Path(args.project) / args.name / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Zero-shot Evaluation Results ===")
    print(f"mAP50:    {results['mAP50']:.4f}")
    print(f"mAP50-95: {results['mAP50_95']:.4f}")
    for cls, ap in results["per_class_AP50"].items():
        print(f"  {cls}: {ap:.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
