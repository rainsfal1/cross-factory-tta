"""Train YOLOv8m on SH17 source domain."""

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/sh17/sh17.yaml"))
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--name", default="sh17_yolov8m")
    p.add_argument("--project", default=str(ROOT / "runs/train"))
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=args.lr0,
        name=args.name,
        project=args.project,
        exist_ok=True,
        verbose=True,
    )
    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
