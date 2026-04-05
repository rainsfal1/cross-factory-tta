"""
Notebook-style figure: one row, three panels (matplotlib subplots).

Default strip: two SH17 val images + one SHWD test image (sorted, first paths).

Run from repo root:
  python docs/visual_examples/make_sample_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg


ROOT = Path(__file__).resolve().parents[2]


def image_list(folder: Path, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    out: list[Path] = []
    for ext in exts:
        out.extend(sorted(folder.glob(f"*{ext}")))
    return sorted(set(out))


def main() -> None:
    sh17 = image_list(ROOT / "data/sh17/images/val")
    shwd = image_list(ROOT / "data/shwd/images/test")
    if len(sh17) < 2 or len(shwd) < 1:
        print("Need >=2 SH17 and >=1 SHWD images.", file=sys.stderr)
        sys.exit(1)

    a, b, c = sh17[0], sh17[1], shwd[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, path, label in zip(
        axes,
        (a, b, c),
        (
            f"SH17 (source)\n{a.name}",
            f"SH17 (source)\n{b.name}",
            f"SHWD (target)\n{c.name}",
        ),
    ):
        ax.imshow(mpimg.imread(path))
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Domain examples: SH17 val vs SHWD test (helmets mapped to class 4)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    out = ROOT / "docs/visual_examples/sh17_shwd_three_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
