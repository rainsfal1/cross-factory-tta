"""
Data setup script: organizes SH17 and Pictor-PPE into the expected directory structure.

SH17 (YOLO format) from kagglehub cache → data/sh17/{images,labels}/{train,val}/
Pictor-PPE (VOC bbox format) from /tmp/pictor_extract → data/pictor_ppe/{images,labels}/test/
"""

import os
import shutil
from pathlib import Path
from PIL import Image

SH17_CACHE = Path.home() / ".cache/kagglehub/datasets/mugheesahmad/sh17-dataset-for-ppe-detection/versions/1"
PICTOR_SRC = Path("/tmp/pictor_extract/pictor-ppe")
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


# SH17: 17 classes
SH17_CLASSES = [
    "Coverall", "Face Shield", "Gloves", "Goggles", "Hard Hat",
    "No Coverall", "No Face Shield", "No Gloves", "No Goggles", "No Hard Hat",
    "No Safety Boot", "No Safety Vest", "No Vest", "Safety Boot", "Safety Vest",
    "Vest", "person",
]

# Pictor classes (from paper): 0=helmet, 1=head, 2=person
PICTOR_CLASSES = ["helmet", "head", "person"]


def setup_sh17():
    """Move SH17 images and labels into data/sh17/ using train/val split files."""
    print("\n=== Setting up SH17 ===")
    src_images = SH17_CACHE / "images"
    src_labels = SH17_CACHE / "labels"

    for split in ("train", "val"):
        out_images = DATA / "sh17" / "images" / split
        out_labels = DATA / "sh17" / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        split_file = SH17_CACHE / f"{split}_files.txt"
        filenames = split_file.read_text().splitlines()
        filenames = [f.strip() for f in filenames if f.strip()]
        print(f"  {split}: {len(filenames)} files")

        for fname in filenames:
            img_src = src_images / fname
            stem = Path(fname).stem
            label_src = src_labels / f"{stem}.txt"

            if img_src.exists():
                shutil.copy2(img_src, out_images / fname)
            else:
                print(f"  [WARN] Missing image: {fname}")

            if label_src.exists():
                shutil.copy2(label_src, out_labels / f"{stem}.txt")

    print(f"  SH17 train images: {len(list((DATA / 'sh17/images/train').glob('*')))}")
    print(f"  SH17 val   images: {len(list((DATA / 'sh17/images/val').glob('*')))}")


def pictor_voc_to_yolo(line: str, img_w: int, img_h: int) -> list[str]:
    """Convert one Pictor annotation line to YOLO txt lines."""
    parts = line.strip().split("\t")
    yolo_lines = []
    for ann in parts[1:]:
        ann = ann.strip()
        if not ann:
            continue
        vals = ann.split(",")
        if len(vals) != 5:
            continue
        x1, y1, x2, y2, cls = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])
        # YOLO: cx cy w h normalised
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def setup_pictor():
    """Convert Pictor annotations to YOLO format and place in data/pictor_ppe/."""
    print("\n=== Setting up Pictor-PPE ===")
    src_images = PICTOR_SRC / "Images"
    # Use approach-01 test split (most commonly referenced in paper)
    ann_file = PICTOR_SRC / "Labels" / "pictor_ppe_crowdsourced_approach-01_test.txt"

    out_images = DATA / "pictor_ppe" / "images" / "test"
    out_labels = DATA / "pictor_ppe" / "labels" / "test"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    lines = ann_file.read_text().splitlines()
    ok, skipped = 0, 0
    for line in lines:
        parts = line.strip().split("\t")
        if not parts:
            continue
        fname = parts[0].strip()
        img_src = src_images / fname
        if not img_src.exists():
            skipped += 1
            continue

        try:
            with Image.open(img_src) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"  [WARN] Cannot open {fname}: {e}")
            skipped += 1
            continue

        yolo_lines = pictor_voc_to_yolo(line, img_w, img_h)
        if not yolo_lines:
            skipped += 1
            continue

        shutil.copy2(img_src, out_images / fname)
        stem = Path(fname).stem
        (out_labels / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")
        ok += 1

    print(f"  Pictor test images placed: {ok}  (skipped: {skipped})")


def write_yamls():
    """Write dataset YAML config files."""
    print("\n=== Writing YAML configs ===")

    # SH17
    sh17_yaml = DATA / "sh17" / "sh17.yaml"
    sh17_yaml.write_text(f"""\
path: {DATA / 'sh17'}
train: images/train
val: images/val

nc: {len(SH17_CLASSES)}
names: {SH17_CLASSES}
""")

    # Pictor (only 3 classes; we will remap to hard-hat + person for cross-domain eval)
    pictor_yaml = DATA / "pictor_ppe" / "pictor.yaml"
    pictor_yaml.write_text(f"""\
path: {DATA / 'pictor_ppe'}
train: images/test   # no train split used
val: images/test

nc: {len(PICTOR_CLASSES)}
names: {PICTOR_CLASSES}
""")

    print("  Written:", sh17_yaml)
    print("  Written:", pictor_yaml)


if __name__ == "__main__":
    setup_sh17()
    setup_pictor()
    write_yamls()
    print("\nDone. Verify with:")
    print("  ls data/sh17/images/train | wc -l")
    print("  ls data/pictor_ppe/images/test | wc -l")
