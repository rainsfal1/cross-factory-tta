"""
Class index remapping between SH17 and Pictor-PPE.

SH17 classes (17 total):
 0  Coverall
 1  Face Shield
 2  Gloves
 3  Goggles
 4  Hard Hat
 5  No Coverall
 6  No Face Shield
 7  No Gloves
 8  No Goggles
 9  No Hard Hat
10  No Safety Boot
11  No Safety Vest
12  No Vest
13  Safety Boot
14  Safety Vest
15  Vest
16  person

Pictor-PPE classes (3 total, approach-01 annotation scheme):
 0  helmet      (≈ SH17 Hard Hat = 4)
 1  head        (≈ SH17 No Hard Hat = 9, i.e., bare head)
 2  person      (≈ SH17 person = 16)

For cross-domain evaluation we project both datasets onto a
shared label space of 2 classes that exist in both:
  0 → hard_hat   (SH17 cls 4 | Pictor cls 0)
  1 → person     (SH17 cls 16 | Pictor cls 2)

All other classes are ignored during evaluation.
"""

# SH17 → shared label mapping (None = ignore)
SH17_TO_SHARED: dict[int, int | None] = {
    0: None,   # Coverall
    1: None,   # Face Shield
    2: None,   # Gloves
    3: None,   # Goggles
    4: 0,      # Hard Hat  → shared 0
    5: None,
    6: None,
    7: None,
    8: None,
    9: None,
    10: None,
    11: None,
    12: None,
    13: None,
    14: None,
    15: None,
    16: 1,     # person    → shared 1
}

# Pictor → shared label mapping
PICTOR_TO_SHARED: dict[int, int | None] = {
    0: 0,      # helmet → hard_hat
    1: None,   # head (bare) — excluded; no direct SH17 equivalent at inference
    2: 1,      # person
}

SHARED_CLASSES = ["hard_hat", "person"]
NC_SHARED = len(SHARED_CLASSES)


def remap_yolo_line(line: str, remap: dict[int, int | None]) -> str | None:
    """
    Given a YOLO annotation line and a class remap dict,
    return the remapped line or None if the class should be ignored.
    """
    parts = line.strip().split()
    if not parts:
        return None
    cls_id = int(parts[0])
    new_cls = remap.get(cls_id)
    if new_cls is None:
        return None
    return f"{new_cls} " + " ".join(parts[1:])


def remap_label_file(src_path, dst_path, remap: dict[int, int | None]) -> int:
    """Remap a YOLO label file, writing only non-ignored lines. Returns count written."""
    import pathlib
    src = pathlib.Path(src_path)
    dst = pathlib.Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text().splitlines()
    out = []
    for line in lines:
        remapped = remap_yolo_line(line, remap)
        if remapped is not None:
            out.append(remapped)
    dst.write_text("\n".join(out) + ("\n" if out else ""))
    return len(out)
