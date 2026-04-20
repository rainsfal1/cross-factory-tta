"""
Verify all datasets against audited counts and class distributions.
Run after conversion + prepare_data.py to confirm setup is correct.

Usage:
    uv run python scripts/verify_datasets.py
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

data = config.DATA_DIR  # data/prepared/

errors = []
warnings = []

CHECKS = {
    "shwd/images/test":       (1515, "images"),  # 2 jpgs missing from raw VOC2028
    "shwd/labels/test":       (1517, "labels"),
    "pictor_ppe/images/test": (152,  "images"),
    "pictor_ppe/labels/test": (152,  "labels"),
    "chv/images/test":        (133,  "images"),
    "chv/labels/test":        (133,  "labels"),
    "sh17/images/train":      (6479, "images"),
    "sh17/images/val":        (1620, "images"),
    "sh17/labels/train":      (6479, "labels"),
    "sh17/labels/val":        (1620, "labels"),
}

for rel, (expected, kind) in CHECKS.items():
    p = data / rel
    if not p.exists():
        errors.append(f"MISSING dir: {p}")
        continue
    ext = "*" if kind == "images" else "*.txt"
    n = len(list(p.glob(ext)))
    if n != expected:
        errors.append(f"{rel}: {n} {kind} (expected {expected})")
    else:
        print(f"[OK] {rel}: {n} {kind}")


def cls_counts(label_dir):
    counts = {}
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                c = int(line.split()[0])
                counts[c] = counts.get(c, 0) + 1
    return counts


print()
shwd_cls   = cls_counts(data / "shwd/labels/test")
pictor_cls = cls_counts(data / "pictor_ppe/labels/test")
chv_cls    = cls_counts(data / "chv/labels/test")

if set(shwd_cls.keys()) - {0, 2}:
    errors.append(f"SHWD unexpected classes: {shwd_cls}")
elif shwd_cls.get(0) != 1878 or shwd_cls.get(2) != 22558:
    errors.append(f"SHWD annotation counts wrong: {shwd_cls} (expected {{0:1878, 2:22558}})")
else:
    print(f"[OK] shwd classes: {shwd_cls}")

if set(pictor_cls.keys()) - {0, 1, 2}:
    errors.append(f"Pictor unexpected classes: {pictor_cls}")
elif pictor_cls.get(1, 0) != 7:
    warnings.append(f"Pictor class 1 count: {pictor_cls.get(1,0)} (expected 7)")
    print(f"[WARN] pictor classes: {pictor_cls}")
else:
    print(f"[OK] pictor classes: {pictor_cls}")

if set(chv_cls.keys()) - {0, 2}:
    errors.append(f"CHV unexpected classes: {chv_cls} (class 1/vest must be absent)")
else:
    print(f"[OK] chv classes: {chv_cls}")

YAML_CHECKS = [
    (data / "sh17/sh17.yaml",        17),
    (data / "shwd/shwd.yaml",         3),
    (data / "pictor_ppe/pictor.yaml", 3),
    (data / "chv/chv.yaml",           3),
]
print()
for yaml_path, expected_nc in YAML_CHECKS:
    if not yaml_path.exists():
        errors.append(f"MISSING yaml: {yaml_path}")
        continue
    cfg = yaml.safe_load(yaml_path.read_text())
    nc    = cfg.get("nc")
    names = cfg.get("names", [])
    if nc != expected_nc:
        errors.append(f"{yaml_path.name}: nc={nc} (expected {expected_nc})")
    elif yaml_path.name in ("pictor.yaml", "shwd.yaml", "chv.yaml"):
        if names != ["hard_hat", "no_hard_hat", "person"]:
            errors.append(f"{yaml_path.name}: names={names} (expected canonical)")
        else:
            print(f"[OK] {yaml_path.name}: nc={nc}, names={names}")
    else:
        print(f"[OK] {yaml_path.name}: nc={nc}")

print()
for ds in ("shwd", "pictor_ppe", "chv"):
    bad = []
    for f in (data / ds / "labels/test").glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                bad.append(f"{f.name}: wrong cols")
                break
            if any(not (0 <= float(v) <= 1) for v in parts[1:]):
                bad.append(f"{f.name}: coord out of range")
                break
    if bad:
        errors.append(f"{ds} malformed labels: {bad[:5]}")
    else:
        print(f"[OK] {ds} label format: all coords in [0,1], 5 cols")

print()
print("=" * 45)
if errors:
    print("ERRORS:")
    for e in errors:
        print(f"  x {e}")
if warnings:
    print("WARNINGS:")
    for w in warnings:
        print(f"  ! {w}")
if not errors:
    print("ALL CHECKS PASSED — setup is identical.")
print("=" * 45)
