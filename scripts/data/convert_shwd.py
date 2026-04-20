"""Convert SHWD (VOC2028) test split → YOLO format."""
import pathlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
import config

_ROOT      = config._ROOT
ANN_DIR    = _ROOT / "data/raw/shwd/VOC2028/Annotations"
SPLIT_FILE = _ROOT / "data/raw/shwd/VOC2028/ImageSets/Main/test.txt"
IMG_DIR    = _ROOT / "data/raw/shwd/VOC2028/JPEGImages"
OUT_LABELS = _ROOT / "data/shwd/labels/test"
OUT_IMAGES = _ROOT / "data/shwd/images/test"
CLASS_MAP  = {"hat": 0, "helmet": 0, "person": 2}

stems = SPLIT_FILE.read_text().split()
OUT_LABELS.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

counts = Counter()
skipped = 0
for stem in stems:
    tree = ET.parse(ANN_DIR / f"{stem}.xml")
    root = tree.getroot()
    size = root.find("size")
    W, H = int(size.find("width").text), int(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_MAP:
            skipped += 1
            continue
        cls = CLASS_MAP[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        w  = (xmax - xmin) / W
        h  = (ymax - ymin) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls] += 1
    if lines:
        (OUT_LABELS / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    src = IMG_DIR / f"{stem}.jpg"
    if src.exists():
        shutil.copy2(src, OUT_IMAGES / f"{stem}.jpg")

print(f"Images:  {sum(1 for _ in OUT_IMAGES.glob('*.jpg'))}")
print(f"Labels:  {sum(1 for _ in OUT_LABELS.glob('*.txt'))}")
print(f"Boxes:   {dict(counts)}  (0=hard_hat, 2=person)")
print(f"Skipped: {skipped} unknown-class boxes")
