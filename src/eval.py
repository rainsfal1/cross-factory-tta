import shutil
from pathlib import Path

from ultralytics import YOLO

import config


def make_remapped_yaml(ds_name: str, tmp_root: Path) -> Path:
    """
    Remap canonical label indices [0,1,2] → SH17 indices [4,9,16] and write
    a temp yaml so .val() matches predictions at the correct class slots.

        0 (hard_hat)    → 4  (Hard Hat in SH17)
        1 (no_hard_hat) → 9  (No Hard Hat in SH17)
        2 (person)      → 16 (person in SH17)
    """
    tmp_dir    = tmp_root / ds_name
    tmp_labels = tmp_dir / "labels" / "test"
    tmp_images = tmp_dir / "images" / "test"
    tmp_labels.mkdir(parents=True, exist_ok=True)
    tmp_images.mkdir(parents=True, exist_ok=True)

    src_images = config.DATA_DIR / ds_name / "images" / "test"
    for img in src_images.iterdir():
        shutil.copy2(img, tmp_images / img.name)

    src_labels = config.DATA_DIR / ds_name / "labels" / "test"
    for lbl in src_labels.glob("*.txt"):
        lines_out = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            lines_out.append(f"{config.CANONICAL_TO_SH17[cls]} {' '.join(parts[1:])}")
        (tmp_labels / lbl.name).write_text(
            "\n".join(lines_out) + "\n" if lines_out else ""
        )

    tmp_yaml = tmp_dir / f"{ds_name}_remap.yaml"
    tmp_yaml.write_text(f"""path: {tmp_dir}
train: images/test
val: images/test

nc: {len(config.SH17_CLASSES)}
names: {config.SH17_CLASSES}
""")
    return tmp_yaml


def run_baseline_eval(best_weights_path: Path, hp: dict) -> dict:
    print("\n=== Baseline Zero-Shot Evaluation (All Target Domains) ===")

    eval_cfg  = hp["eval"]
    tmp_root  = config.RUNS_DIR / "tmp_remap"
    target_domains = [
        ("pictor_ppe", "pictor.yaml"),
        ("shwd",       "shwd.yaml"),
        ("chv",        "chv.yaml"),
    ]

    baseline_model = YOLO(str(best_weights_path))
    baseline_results = {}

    for ds_name, _ in target_domains:
        tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
        metrics = baseline_model.val(
            data=str(tmp_yaml),
            imgsz=eval_cfg["imgsz"],
            batch=eval_cfg["batch"],
            split="val",
            conf=eval_cfg["conf"],
            iou=eval_cfg["iou"],
            name=f"baseline_{ds_name}",
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True,
            verbose=False,
        )

        class_idx = (
            metrics.box.ap_class_index
            if hasattr(metrics.box, "ap_class_index")
            else range(len(metrics.box.ap50))
        )

        per_class_ap50, per_class_ap5095 = {}, {}
        for idx, ap50, ap5095 in zip(class_idx, metrics.box.ap50, metrics.box.ap):
            if idx in config.SH17_EVAL_IDX:
                name = config.SH17_EVAL_IDX[idx]
                per_class_ap50[name]   = float(ap50)
                per_class_ap5095[name] = float(ap5095)

        map50   = sum(per_class_ap50.values())   / len(per_class_ap50)   if per_class_ap50   else 0.0
        map5095 = sum(per_class_ap5095.values()) / len(per_class_ap5095) if per_class_ap5095 else 0.0

        baseline_results[ds_name] = {"mAP50": map50, "mAP50_95": map5095}
        print(f"  {ds_name:<14} mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

        shutil.rmtree(tmp_root / ds_name, ignore_errors=True)

    return baseline_results


def run_tent_eval(best_weights_path: Path, baseline_results: dict, hp: dict):
    from src.tent import TENT

    print("\n=== TENT Adaptation on Pictor ===")

    eval_cfg = hp["eval"]
    tent_cfg = hp["tent"]
    tmp_root = config.RUNS_DIR / "tmp_remap"
    tmp_yaml_pictor = make_remapped_yaml("pictor_ppe", tmp_root)

    tent_model = YOLO(str(best_weights_path))
    adapter = TENT(tent_model, lr=tent_cfg["lr"], steps=tent_cfg["steps"])

    def on_val_batch_start(validator):
        batch_imgs = validator.batch.get("im_file", [])
        if batch_imgs:
            adapter.step(list(batch_imgs))

    tent_model.add_callback("on_val_batch_start", on_val_batch_start)

    tent_metrics = tent_model.val(
        data=str(tmp_yaml_pictor),
        imgsz=eval_cfg["imgsz"],
        batch=eval_cfg["batch"],
        split="val",
        conf=eval_cfg["conf"],
        iou=eval_cfg["iou"],
        name="tent_pictor",
        project=str(config.RUNS_DIR / "eval"),
        exist_ok=True,
        verbose=False,
    )

    class_idx = (
        tent_metrics.box.ap_class_index
        if hasattr(tent_metrics.box, "ap_class_index")
        else range(len(tent_metrics.box.ap50))
    )

    tent_per_class_ap50, tent_per_class_ap5095 = {}, {}
    for idx, ap50, ap5095 in zip(class_idx, tent_metrics.box.ap50, tent_metrics.box.ap):
        if idx in config.SH17_EVAL_IDX:
            name = config.SH17_EVAL_IDX[idx]
            tent_per_class_ap50[name]   = float(ap50)
            tent_per_class_ap5095[name] = float(ap5095)

    tent_map50   = sum(tent_per_class_ap50.values())   / len(tent_per_class_ap50)   if tent_per_class_ap50   else 0.0
    tent_map5095 = sum(tent_per_class_ap5095.values()) / len(tent_per_class_ap5095) if tent_per_class_ap5095 else 0.0

    baseline_map50 = baseline_results.get("pictor_ppe", {}).get("mAP50", 0.0)
    recovery = tent_map50 - baseline_map50

    shutil.rmtree(tmp_root / "pictor_ppe", ignore_errors=True)

    print(f"TENT mAP50:    {tent_map50:.4f}")
    print(f"TENT mAP50-95: {tent_map5095:.4f}")
    print(f"Recovery:      {recovery:+.4f}")

    return {"mAP50": tent_map50, "mAP50_95": tent_map5095, "recovery": recovery}
