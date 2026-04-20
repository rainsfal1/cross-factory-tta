"""
Convert Pictor-PPE test split from TSV pixel-bbox format → YOLO format.

Source format (tab-separated):
    filename    x1,y1,x2,y2,class    x1,y1,x2,y2,class ...

Classes: 0=helmet→0(hard_hat), 1=head→1(no_hard_hat), 2=person→2 (clean 1:1)
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

SRC        = config._ROOT / "data/raw/pictor_ppe_source/pictor-ppe"
LABEL_FILE = SRC / "Labels" / "pictor_ppe_crowdsourced_approach-01_test.txt"
IMG_DIR    = SRC / "Images"
OUT_IMAGES = config.PICTOR_DIR / "images" / "test"
OUT_LABELS = config.PICTOR_DIR / "labels" / "test"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

counts = Counter()
for line in LABEL_FILE.read_text().splitlines():
    parts = line.strip().split("\t")
    if not parts or not parts[0]:
        continue
    fname = parts[0]
    img_path = IMG_DIR / fname
    if not img_path.exists():
        print(f"  [WARN] image not found: {fname}")
        continue

    W, H = Image.open(img_path).size
    yolo_lines = []
    for bbox in parts[1:]:
        x1, y1, x2, y2, cls = bbox.split(",")
        x1, y1, x2, y2, cls = float(x1), float(y1), float(x2), float(y2), int(cls)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls] += 1

    stem = Path(fname).stem
    (OUT_LABELS / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")
    shutil.copy2(img_path, OUT_IMAGES / fname)

print(f"Images: {len(list(OUT_IMAGES.glob('*')))}")
print(f"Labels: {len(list(OUT_LABELS.glob('*.txt')))}")
print(f"Boxes:  {dict(counts)}  (0=hard_hat, 1=no_hard_hat, 2=person)")
