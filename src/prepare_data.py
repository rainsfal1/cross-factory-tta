"""
Prepare datasets in the repository format expected by:
  - src/train.py (SH17 YOLO 17-class)
  - src/evaluate.py / src/adapt.py (Pictor-PPE YOLO test split)

This script is meant to be Kaggle-friendly:
  - it copies from /kaggle/input mounted dataset folders into the repo's `data/` folder
  - it avoids the hardcoded /tmp + kagglehub-cache assumptions in src/setup_data.py

Supported input formats
SH17:
  1) YOLO layout:
       <sh17-dir>/images/{train,val}/...
       <sh17-dir>/labels/{train,val}/...
  2) kagglehub-cache layout:
       <sh17-dir>/images/, <sh17-dir>/labels/,
       and <sh17-dir>/{train_files.txt,val_files.txt}

Pictor-PPE:
  1) YOLO layout:
       <pictor-dir>/images/test/... and <pictor-dir>/labels/test/...
  2) VOC-like layout as expected by src/setup_data.py conversion:
       <pictor-dir>/Images/...
       <pictor-dir>/Labels/pictor_ppe_crowdsourced_approach-01_test.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from setup_data import SH17_CLASSES, PICTOR_CLASSES, setup_pictor, setup_sh17, write_yamls


ROOT = Path(__file__).parent.parent
DEFAULT_OUT_DIR = ROOT / "data"


def _copytree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected path does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _ensure_sh17_from_yolo(sh17_src: Path, out_dir: Path) -> None:
    for split in ("train", "val"):
        _copytree_if_exists(sh17_src / "images" / split, out_dir / "sh17" / "images" / split)
        _copytree_if_exists(sh17_src / "labels" / split, out_dir / "sh17" / "labels" / split)


def _ensure_sh17_from_kagglehub_cache(sh17_cache: Path, out_dir: Path) -> None:
    setup_sh17(sh17_cache=sh17_cache, data_dir=out_dir)


def _ensure_pictor_from_yolo(pictor_src: Path, out_dir: Path) -> None:
    _copytree_if_exists(pictor_src / "images" / "test", out_dir / "pictor_ppe" / "images" / "test")
    _copytree_if_exists(pictor_src / "labels" / "test", out_dir / "pictor_ppe" / "labels" / "test")


def _ensure_pictor_from_voc(pictor_src: Path, out_dir: Path) -> None:
    setup_pictor(pictor_src=pictor_src, data_dir=out_dir)


def _write_simple_yaml(out_dir: Path) -> None:
    # We still call src/setup_data.py's writer to keep class lists consistent.
    write_yamls(data_dir=out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sh17-dir", type=str, default=None, help="Path to SH17 dataset folder")
    p.add_argument("--pictor-dir", type=str, default=None, help="Path to pictor_ppe dataset folder")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Where to write repo data/")
    p.add_argument("--force", action="store_true", help="Overwrite/copy even if data already exists")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    out_sh17_yaml = out_dir / "sh17" / "sh17.yaml"
    out_pictor_yaml = out_dir / "pictor_ppe" / "pictor.yaml"

    out_sh17_ready = (
        (out_dir / "sh17" / "images" / "train").exists()
        and out_sh17_yaml.exists()
    )
    out_pictor_ready = (
        (out_dir / "pictor_ppe" / "images" / "test").exists()
        and out_pictor_yaml.exists()
    )

    if not args.force and out_sh17_ready and out_pictor_ready:
        print("Data already exists; skipping preparation.")
        return

    if not args.force and out_sh17_ready:
        print("SH17 already present; skipping SH17 preparation.")
    else:
        if not args.sh17_dir:
            raise ValueError("Missing --sh17-dir (required when SH17 is not already prepared).")
        sh17_src = Path(args.sh17_dir)

        yolo_layout_ok = (
            (sh17_src / "images" / "train").exists()
            and (sh17_src / "labels" / "train").exists()
            and (sh17_src / "images" / "val").exists()
            and (sh17_src / "labels" / "val").exists()
        )
        cache_layout_ok = (
            (sh17_src / "images").exists()
            and (sh17_src / "labels").exists()
            and (sh17_src / "train_files.txt").exists()
            and (sh17_src / "val_files.txt").exists()
        )

        if yolo_layout_ok:
            print(f"Preparing SH17 from YOLO layout: {sh17_src}")
            _ensure_sh17_from_yolo(sh17_src, out_dir)
        elif cache_layout_ok:
            print(f"Preparing SH17 from kagglehub-cache layout: {sh17_src}")
            _ensure_sh17_from_kagglehub_cache(sh17_src, out_dir)
        else:
            raise ValueError(
                "Unsupported SH17 input layout. Expected either:\n"
                "  - images/{train,val}/ and labels/{train,val}/ (YOLO)\n"
                "  - images/, labels/, train_files.txt, val_files.txt (kagglehub cache layout)\n"
                f"Got: {sh17_src}"
            )

    if not args.force and out_pictor_ready:
        print("Pictor-PPE already present; skipping Pictor preparation.")
    else:
        if not args.pictor_dir:
            raise ValueError("Missing --pictor-dir (required when Pictor is not already prepared).")
        pictor_src = Path(args.pictor_dir)

        yolo_layout_ok = (
            (pictor_src / "images" / "test").exists()
            and (pictor_src / "labels" / "test").exists()
        )
        voc_layout_ok = (
            (pictor_src / "Images").exists()
            and (pictor_src / "Labels").exists()
            and (pictor_src / "Labels" / "pictor_ppe_crowdsourced_approach-01_test.txt").exists()
        )

        if yolo_layout_ok:
            print(f"Preparing Pictor-PPE from YOLO layout: {pictor_src}")
            _ensure_pictor_from_yolo(pictor_src, out_dir)
        elif voc_layout_ok:
            print(f"Preparing Pictor-PPE from VOC-like layout: {pictor_src}")
            _ensure_pictor_from_voc(pictor_src, out_dir)
        else:
            raise ValueError(
                "Unsupported Pictor-PPE input layout. Expected either:\n"
                "  - images/test/ and labels/test/ (YOLO)\n"
                "  - Images/, Labels/, pictor_ppe_crowdsourced_approach-01_test.txt (VOC-like)\n"
                f"Got: {pictor_src}"
            )

    _write_simple_yaml(out_dir)
    print("\nDone.")
    print(f"  SH17 yaml: {out_dir / 'sh17' / 'sh17.yaml'}")
    print(f"  Pictor yaml: {out_dir / 'pictor_ppe' / 'pictor.yaml'}")


if __name__ == "__main__":
    main()

