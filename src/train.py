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
    p.add_argument("--device", default="mps")
    p.add_argument("--cfg", default=str(ROOT / "configs/train_sh17.yaml"))
    p.add_argument("--name", default="sh17_yolov8m")
    p.add_argument("--project", default=str(ROOT / "runs/train"))
    p.add_argument(
        "--compact-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one concise line per epoch (recommended for Kaggle).",
    )
    return p.parse_args()


def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _compact_epoch_line(trainer) -> str:
    # Ultralytics exposes loss items on the trainer, but the structure can vary by version.
    box = cls = dfl = None
    loss_items = getattr(trainer, "loss_items", None)
    if loss_items is not None and hasattr(loss_items, "__len__") and len(loss_items) >= 3:
        box, cls, dfl = (_maybe_float(loss_items[0]), _maybe_float(loss_items[1]), _maybe_float(loss_items[2]))

    metrics = {}
    trainer_metrics = getattr(trainer, "metrics", None)
    if isinstance(trainer_metrics, dict):
        metrics = trainer_metrics
    else:
        # Some versions store a Results-like object; best-effort extraction.
        results_dict = getattr(trainer_metrics, "results_dict", None)
        if isinstance(results_dict, dict):
            metrics = results_dict

    map50 = (
        metrics.get("metrics/mAP50(B)")
        or metrics.get("metrics/mAP50")
        or metrics.get("mAP50")
    )
    map5095 = (
        metrics.get("metrics/mAP50-95(B)")
        or metrics.get("metrics/mAP50-95")
        or metrics.get("mAP50-95")
    )
    map50 = _maybe_float(map50)
    map5095 = _maybe_float(map5095)

    epoch = int(getattr(trainer, "epoch", -1)) + 1
    epochs = int(getattr(trainer, "epochs", -1))

    parts = [f"epoch {epoch}/{epochs}"]
    if box is not None:
        parts.append(f"box={box:.4f}")
    if cls is not None:
        parts.append(f"cls={cls:.4f}")
    if dfl is not None:
        parts.append(f"dfl={dfl:.4f}")
    if map50 is not None:
        parts.append(f"mAP50={map50:.4f}")
    if map5095 is not None:
        parts.append(f"mAP50-95={map5095:.4f}")
    return " | ".join(parts)


def main():
    args = parse_args()
    model = YOLO(args.model)

    if args.compact_logs:
        print("epoch/epochs | box | cls | dfl | mAP50 | mAP50-95")

        def on_fit_epoch_end(trainer):
            # Runs after the epoch (and validation if enabled), so mAP is usually available.
            try:
                print(_compact_epoch_line(trainer))
            except Exception:
                # Never break training because of logging.
                pass

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # The cfg file overrides other individual parameters in Ultralytics
    results = model.train(
        data=args.data,
        cfg=args.cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        exist_ok=True,
        verbose=not args.compact_logs,
    )
    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
