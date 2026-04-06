import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

import config


def compact_epoch_callback(trainer):
    try:
        loss_items = getattr(trainer, "loss_items", None)
        box = cls = dfl = None
        if loss_items is not None and hasattr(loss_items, "__len__") and len(loss_items) >= 3:
            box, cls, dfl = float(loss_items[0]), float(loss_items[1]), float(loss_items[2])

        metrics = getattr(trainer, "metrics", {})
        if not isinstance(metrics, dict):
            metrics = getattr(metrics, "results_dict", {})

        map50   = metrics.get("metrics/mAP50(B)")    or metrics.get("metrics/mAP50")
        map5095 = metrics.get("metrics/mAP50-95(B)") or metrics.get("metrics/mAP50-95")

        epoch  = int(getattr(trainer, "epoch",  -1)) + 1
        epochs = int(getattr(trainer, "epochs", -1))

        parts = [f"epoch {epoch}/{epochs}"]
        if box     is not None: parts.append(f"box={box:.3f}")
        if cls     is not None: parts.append(f"cls={cls:.3f}")
        if dfl     is not None: parts.append(f"dfl={dfl:.3f}")
        if map50   is not None: parts.append(f"mAP50={float(map50):.3f}")
        if map5095 is not None: parts.append(f"mAP50-95={float(map5095):.3f}")

        print(" | ".join(parts))
    except Exception as e:
        print(f"[WARN] Callback error: {e}")


def run_training(hp: dict):
    run_name          = hp["run_name"]
    pretrained        = hp.get("pretrained_weights")
    train_cfg         = hp["train"]
    checkpoint_path   = config.RUNS_DIR / "train" / run_name / "weights" / "last.pt"
    best_weights_path = config.RUNS_DIR / "train" / run_name / "weights" / "best.pt"

    if pretrained is not None:
        src = Path(pretrained)
        if not src.exists():
            raise RuntimeError(f"pretrained_weights set but file not found: {src}")
        best_weights_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, best_weights_path)
        print(f"Loaded pre-trained weights from {src}, skipping training.")
        return best_weights_path

    if checkpoint_path.exists():
        print(f"Resuming interrupted training from {checkpoint_path}")
        model = YOLO(str(checkpoint_path))
        model.train(resume=True)
        print(f"Training complete. Best weights: {best_weights_path}")
        return best_weights_path

    def _internet_available() -> bool:
        try:
            import urllib.request
            urllib.request.urlopen("https://ultralytics.com", timeout=5)
            return True
        except Exception:
            return False

    if not Path("yolov8m.pt").exists() and not _internet_available():
        raise RuntimeError(
            "No internet and no local yolov8m.pt found. "
            "Either connect to the internet or place yolov8m.pt in the project root."
        )

    print("epoch/epochs | box | cls | dfl | mAP50 | mAP50-95")
    model = YOLO("yolov8m.pt")
    model.add_callback("on_fit_epoch_end", compact_epoch_callback)

    results = model.train(
        data=str(config.DATA_DIR / "sh17" / "sh17.yaml"),
        name=run_name,
        project=str(config.RUNS_DIR / "train"),
        **train_cfg,
    )

    print(f"\nTraining complete. Best weights: {best_weights_path}")
    if best_weights_path.exists():
        print(f"SH17 val mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"SH17 val mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return best_weights_path
