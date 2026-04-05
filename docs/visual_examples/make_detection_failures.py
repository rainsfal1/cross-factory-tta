"""
Visualize YOLO predictions on SHWD: empty vs non-empty.

Default: raw Ultralytics `Results.plot()` only (no text overlay).

Optional caption box on top:
  python docs/visual_examples/make_detection_failures.py --caption

Run from repo root:
  python docs/visual_examples/make_detection_failures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "best.pt"
SHWD_TEST = ROOT / "data/shwd/images/test"
OUT_EMPTY = ROOT / "docs/visual_examples/shwd_prediction_empty.png"
OUT_NONEMPTY = ROOT / "docs/visual_examples/shwd_prediction_nonempty.png"

FONT_SIZE = 30
LINE_GAP = 8
TEXT_PAD_X = 16
TEXT_PAD_Y = 12
MIN_CAPTION_H = 84


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Word-wrap a paragraph to fit max_width (pixels)."""
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if not cur:
            cur = [w]
            continue
        if _line_width(draw, trial, font) <= max_width:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def compose_figure(
    bgr_content,
    title: str,
    subline: str,
):
    """
    Keep original image dimensions and add a compact top white caption area.
    """
    h, w = bgr_content.shape[:2]
    rgb = cv2.cvtColor(bgr_content, cv2.COLOR_BGR2RGB)
    max_text_w = w - 2 * TEXT_PAD_X

    font = _try_font(FONT_SIZE)
    title_lines = wrap_to_width(title, font, max_text_w)
    sub_lines = wrap_to_width(subline, font, max_text_w)

    caption_h = max(
        MIN_CAPTION_H,
        TEXT_PAD_Y * 2
        + len(title_lines) * (FONT_SIZE + LINE_GAP)
        + len(sub_lines) * (FONT_SIZE + LINE_GAP),
    )

    cap = Image.new("RGB", (w, caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(cap)
    y = TEXT_PAD_Y
    for line in title_lines:
        draw.text((TEXT_PAD_X, y), line, fill=(15, 15, 15), font=font)
        y += FONT_SIZE + LINE_GAP
    for line in sub_lines:
        draw.text((TEXT_PAD_X, y), line, fill=(45, 45, 45), font=font)
        y += FONT_SIZE + LINE_GAP

    out_h = caption_h + h
    full = Image.new("RGB", (w, out_h), (255, 255, 255))
    full.paste(cap, (0, 0))
    full.paste(Image.fromarray(rgb), (0, caption_h))
    return cv2.cvtColor(np.array(full), cv2.COLOR_RGB2BGR)


def find_empty_prediction(
    model: YOLO,
    paths: list[Path],
    confs: list[float],
) -> tuple[Path, float, Any]:
    for conf in confs:
        for p in paths:
            r = model.predict(source=str(p), conf=conf, verbose=False, imgsz=640)[0]
            if len(r.boxes) == 0:
                return p, conf, r
    raise RuntimeError("No empty prediction found; try more images or higher conf list.")


def find_nonempty_smallest(
    model: YOLO, paths: list[Path], conf: float, *, max_boxes: int = 8
) -> tuple[Path, Any] | None:
    best: tuple[Path, Any] | None = None
    best_n = 10**9
    for p in paths:
        r = model.predict(source=str(p), conf=conf, verbose=False, imgsz=640)[0]
        n = len(r.boxes)
        if n == 0:
            continue
        if n <= max_boxes:
            return p, r
        if n < best_n:
            best_n = n
            best = (p, r)
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--caption",
        action="store_true",
        help="Add a compact white caption strip above each image",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not WEIGHTS.exists():
        print(f"Missing weights: {WEIGHTS}", file=sys.stderr)
        sys.exit(1)
    paths = sorted(SHWD_TEST.glob("*.jpg")) + sorted(SHWD_TEST.glob("*.png"))
    if not paths:
        print(f"No images under {SHWD_TEST}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(WEIGHTS))
    confs = [0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]
    img_path, conf, res = find_empty_prediction(model, paths, confs)

    plotted = res.plot()
    title_e = f"SHWD test / {img_path.name} | best.pt | conf={conf} | 0 detections"
    sub_e = "No boxes above threshold (empty output)."

    conf_eval = 0.25
    hit = find_nonempty_smallest(model, paths, conf_eval, max_boxes=8)

    if hit is not None:
        p_ne, r_ne = hit
        n = len(r_ne.boxes)
        title_n = (
            f"SHWD test / {p_ne.name} | best.pt | conf={conf_eval} | {n} detections"
        )
        sub_n = "Non-empty contrast example."
    else:
        title_n, sub_n = "", ""

    if args.caption:
        empty_out = compose_figure(plotted, title_e, sub_e)
    else:
        empty_out = plotted

    OUT_EMPTY.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_EMPTY), empty_out)
    eh, ew = empty_out.shape[:2]
    print(f"Wrote {OUT_EMPTY} ({ew}x{eh})" + (" (caption)" if args.caption else " (no caption)"))
    print(f"  source image: {img_path}")

    if hit is not None:
        plot2 = r_ne.plot()
        if args.caption:
            nonempty_out = compose_figure(plot2, title_n, sub_n)
        else:
            nonempty_out = plot2
        cv2.imwrite(str(OUT_NONEMPTY), nonempty_out)
        nh, nw = nonempty_out.shape[:2]
        print(
            f"Wrote {OUT_NONEMPTY} ({nw}x{nh}) ({n} boxes)"
            + (" (caption)" if args.caption else " (no caption)")
        )
    else:
        print(f"No non-empty prediction at conf={conf_eval}; skipped {OUT_NONEMPTY.name}")


if __name__ == "__main__":
    main()
