"""Convert CHV test split → YOLO format with canonical class remapping."""
import pathlib
import shutil
from collections import Counter

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
import config

_ROOT      = config._ROOT
CHV_DIR    = config.CHV_DIR
SPLIT_FILE = CHV_DIR / "data split/test.txt"
ANN_DIR    = CHV_DIR / "annotations"
IMG_DIR    = CHV_DIR / "images"
OUT_LABELS = CHV_DIR / "labels/test"
OUT_IMAGES = CHV_DIR / "images/test"
CLASS_MAP  = {0: 2, 2: 0, 3: 0, 4: 0, 5: 0}  # 1=vest → drop

# Split file lines are like "CHV_dataset/images/ppe_0193.jpg"
stems = [pathlib.Path(l).stem for l in SPLIT_FILE.read_text().splitlines() if l.strip()]
OUT_LABELS.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

counts = Counter()
vest_dropped = 0
for stem in stems:
    src_ann = ANN_DIR / f"{stem}.txt"
    lines_out = []
    for line in src_ann.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        raw_cls = int(parts[0])
        if raw_cls == 1:
            vest_dropped += 1
            continue
        canonical = CLASS_MAP[raw_cls]
        counts[canonical] += 1
        lines_out.append(f"{canonical} {' '.join(parts[1:])}")
    if lines_out:
        (OUT_LABELS / f"{stem}.txt").write_text("\n".join(lines_out) + "\n")
    shutil.copy2(IMG_DIR / f"{stem}.jpg", OUT_IMAGES / f"{stem}.jpg")

print(f"Images:       {sum(1 for _ in OUT_IMAGES.glob('*.jpg'))}")
print(f"Labels:       {sum(1 for _ in OUT_LABELS.glob('*.txt'))}")
print(f"Boxes:        {dict(counts)}  (0=hard_hat, 2=person)")
print(f"Vest dropped: {vest_dropped}")
