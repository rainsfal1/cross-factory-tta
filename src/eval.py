import logging
import shutil
from pathlib import Path

logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO

import config


def make_remapped_yaml(ds_name: str, tmp_root: Path) -> Path:
    """
    Remap canonical label indices [0,1,2] -> SH17 indices [4,9,16] and write
    a temp yaml so .val() matches predictions at the correct class slots.

        0 (hard_hat)    -> 4  (Hard Hat in SH17)
        1 (no_hard_hat) -> 9  (No Hard Hat in SH17)
        2 (person)      -> 16 (person in SH17)
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


def _extract_metrics(metrics, eval_idx: dict) -> dict:
    class_idx = (
        metrics.box.ap_class_index
        if hasattr(metrics.box, "ap_class_index")
        else range(len(metrics.box.ap50))
    )
    per_class_ap50, per_class_ap5095 = {}, {}
    for idx, ap50, ap5095 in zip(class_idx, metrics.box.ap50, metrics.box.ap):
        if idx in eval_idx:
            name = eval_idx[idx]
            per_class_ap50[name]   = float(ap50)
            per_class_ap5095[name] = float(ap5095)

    map50   = sum(per_class_ap50.values())   / len(per_class_ap50)   if per_class_ap50   else 0.0
    map5095 = sum(per_class_ap5095.values()) / len(per_class_ap5095) if per_class_ap5095 else 0.0
    return {"mAP50": map50, "mAP50_95": map5095}


def run_sh17_sanity(best_weights_path: Path, hp: dict) -> dict:
    """Eval on SH17 val (source domain). Expected mAP50 >= 0.70. Low values indicate a data or weight issue."""
    eval_cfg = hp["eval"]
    sh17_yaml = config.DATA_DIR / "sh17" / "sh17.yaml"

    model = YOLO(str(best_weights_path))
    print("  evaluating sh17 (val)...", end=" ", flush=True)
    metrics = model.val(
        data=str(sh17_yaml),
        imgsz=eval_cfg["imgsz"],
        batch=eval_cfg["batch"],
        device=eval_cfg["device"],
        half=eval_cfg["half"],
        split="val",
        conf=eval_cfg["conf"],
        iou=eval_cfg["iou"],
        name="source_domain_sh17",
        project=str(config.RUNS_DIR / "eval"),
        exist_ok=True,
        verbose=False,
    )
    print("done")
    return {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}


def run_baseline_eval(best_weights_path: Path, hp: dict) -> dict:
    eval_cfg  = hp["eval"]
    tmp_root  = config.RUNS_DIR / "tmp_remap"
    target_domains = [
        ("pictor_ppe", "pictor.yaml"),
        ("shwd",       "shwd.yaml"),
        ("chv",        "chv.yaml"),
    ]

    model = YOLO(str(best_weights_path))
    results = {}

    for ds_name, _ in target_domains:
        print(f"  evaluating {ds_name}...", end=" ", flush=True)
        tmp_yaml = make_remapped_yaml(ds_name, tmp_root)
        metrics = model.val(
            data=str(tmp_yaml),
            imgsz=eval_cfg["imgsz"],
            batch=eval_cfg["batch"],
            device=eval_cfg["device"],
            half=eval_cfg["half"],
            split="val",
            conf=eval_cfg["conf"],
            iou=eval_cfg["iou"],
            name=f"baseline_{ds_name}",
            project=str(config.RUNS_DIR / "eval"),
            exist_ok=True,
            verbose=False,
        )
        results[ds_name] = _extract_metrics(metrics, config.SH17_EVAL_IDX)
        print("done")
        shutil.rmtree(tmp_root / ds_name, ignore_errors=True)

    return results


def run_tent_eval(best_weights_path: Path, baseline_results: dict, hp: dict) -> dict:
    from src.tent import TENT

    eval_cfg = hp["eval"]
    tent_cfg = hp["tent"]
    tmp_root = config.RUNS_DIR / "tmp_remap"
    tmp_yaml_pictor = make_remapped_yaml("pictor_ppe", tmp_root)

    model = YOLO(str(best_weights_path))
    adapter = TENT(model, lr=tent_cfg["lr"], steps=tent_cfg["steps"])

    def on_val_batch_start(validator):
        batch_imgs = validator.batch.get("im_file", [])
        if batch_imgs:
            adapter.step(list(batch_imgs))

    model.add_callback("on_val_batch_start", on_val_batch_start)

    print("  evaluating pictor_ppe (TENT)...", end=" ", flush=True)
    tent_metrics = model.val(
        data=str(tmp_yaml_pictor),
        imgsz=eval_cfg["imgsz"],
        batch=eval_cfg["batch"],
        device=eval_cfg["device"],
        half=eval_cfg["half"],
        split="val",
        conf=eval_cfg["conf"],
        iou=eval_cfg["iou"],
        name="tent_pictor",
        project=str(config.RUNS_DIR / "eval"),
        exist_ok=True,
        verbose=False,
    )
    print("done")

    result = _extract_metrics(tent_metrics, config.SH17_EVAL_IDX)
    baseline_map50 = baseline_results.get("pictor_ppe", {}).get("mAP50", 0.0)
    result["recovery"] = result["mAP50"] - baseline_map50

    shutil.rmtree(tmp_root / "pictor_ppe", ignore_errors=True)
    return result
