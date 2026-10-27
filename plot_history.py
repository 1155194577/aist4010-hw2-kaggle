#!/usr/bin/env python3
"""Plot training vs eval loss for a run, from its history.csv.

    python plot_history.py                      # newest logs/*/history.csv
    python plot_history.py --run logs/cnn-*/    # a specific run directory
    python plot_history.py --out loss.png       # choose the output file

Dependency-free apart from Pillow (no matplotlib needed), so it runs in the
course container and in any plain Python install.

Writes a PNG next to the run's CSV (logs/<run>/loss.png) and prints the path.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

CANVAS = (1600, 900)
MARGIN = {"left": 130, "right": 70, "top": 110, "bottom": 110}
TRAIN_COLOR = (31, 100, 200)
EVAL_COLOR = (220, 110, 20)
GRID_COLOR = (222, 226, 232)
AXIS_COLOR = (90, 96, 104)
TEXT_COLOR = (30, 34, 40)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def latest_run(logs_dir: Path) -> Path:
    runs = [p for p in logs_dir.glob("*/history.csv")]
    if not runs:
        raise SystemExit(f"no */history.csv under {logs_dir}/ — train something first")
    return max(runs, key=lambda p: p.stat().st_mtime).parent


def read_history(csv_path: Path) -> Tuple[List[int], List[float], List[float]]:
    epochs, train, evals = [], [], []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if "epoch" not in row or "train_loss" not in row:
                raise SystemExit(f"{csv_path} does not look like a run history")
            epochs.append(int(float(row["epoch"])))
            train.append(float(row["train_loss"]))
            evals.append(float(row["eval_loss"]))
    if not epochs:
        raise SystemExit(f"{csv_path} has no rows")
    return epochs, train, evals


def read_best(run_dir: Path) -> str:
    """Best macro-F1 from summary.json, if the run got that far."""
    import json

    summary = run_dir / "summary.json"
    if not summary.exists():
        return ""
    try:
        data = json.loads(summary.read_text())
    except (OSError, ValueError):
        return ""
    best = data.get("best_macro_f1")
    return f"best macro-F1 {best:.4f}" if isinstance(best, (int, float)) else ""


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def nice_step(span: float, target: int = 6) -> float:
    """A 1/2/5 x 10^k step covering `span` in roughly `target` intervals."""
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * magnitude >= raw:
            return multiple * magnitude
    return 10 * magnitude


def ticks(lo: float, hi: float, integer: bool = False) -> List[float]:
    step = nice_step(hi - lo)
    if integer:
        step = max(1.0, round(step))
    start = math.ceil(lo / step) * step
    out, value = [], start
    while value <= hi + step * 1e-6:
        out.append(round(value, 10))
        value += step
    return out


def draw_vertical(image: Image.Image, center: Tuple[int, int], text: str,
                  fnt: ImageFont.FreeTypeFont, fill: Tuple[int, int, int]) -> None:
    """Paste `text` rotated 90 degrees counter-clockwise, centred on `center`."""
    box = fnt.getbbox(text)
    width, height = box[2] - box[0], box[3] - box[1]
    layer = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((4 - box[0], 4 - box[1]), text, font=fnt, fill=fill)
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated,
                (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
                rotated)


def draw_frame(image: Image.Image, draw: ImageDraw.ImageDraw,
               box: Tuple[int, int, int, int],
               xs: Sequence[float], ys: Sequence[Sequence[float]],
               colors: Sequence[Tuple[int, int, int]], names: Sequence[str],
               xlabel: str, ylabel: str, title: str, subtitle: str) -> None:
    left, top, right, bottom = box
    x_lo, x_hi = min(xs), max(xs)
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    flat = [v for series in ys for v in series]
    y_lo, y_hi = min(flat), max(flat)
    pad = (y_hi - y_lo) * 0.06 or 0.05
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def sx(x: float) -> float:
        return left + (x - x_lo) / (x_hi - x_lo) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - y_lo) / (y_hi - y_lo) * (bottom - top)

    small = font(26)
    for value in ticks(x_lo, x_hi, integer=True):
        x = sx(value)
        draw.line([(x, top), (x, bottom)], fill=GRID_COLOR, width=2)
        draw.text((x, bottom + 14), f"{value:g}", font=small, fill=AXIS_COLOR, anchor="ma")
    for value in ticks(y_lo, y_hi):
        y = sy(value)
        draw.line([(left, y), (right, y)], fill=GRID_COLOR, width=2)
        draw.text((left - 16, y), f"{value:.3f}", font=small, fill=AXIS_COLOR, anchor="rm")

    draw.line([(left, bottom), (right, bottom)], fill=AXIS_COLOR, width=3)
    draw.line([(left, top), (left, bottom)], fill=AXIS_COLOR, width=3)

    for series, color, name in zip(ys, colors, names):
        points = [(sx(x), sy(y)) for x, y in zip(xs, series)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color,
                         outline=(255, 255, 255), width=2)

    draw.text((left + (right - left) / 2, bottom + 60), xlabel,
              font=font(30), fill=TEXT_COLOR, anchor="ma")
    draw_vertical(image, (left - 120, top + (bottom - top) // 2), ylabel,
                  font(30), TEXT_COLOR)

    draw.text((left, 40), title, font=font(40, bold=True), fill=TEXT_COLOR, anchor="lm")
    if subtitle:
        draw.text((left, 82), subtitle, font=font(26), fill=AXIS_COLOR, anchor="lm")

    # Right-to-left so the labels read left-to-right in `names` order.
    lx = right - 30
    for name, color in reversed(list(zip(names, colors))):
        label = f"{name}"
        w = draw.textlength(label, font=font(28))
        lx -= w + 46
        draw.line([(lx, 46), (lx + 30, 46)], fill=color, width=6)
        draw.ellipse([lx + 9, 40, lx + 21, 52], fill=color)
        draw.text((lx + 38, 46), label, font=font(28), fill=TEXT_COLOR, anchor="lm")
        lx -= 20


def make_plot(run_dir: Path, out_path: Path) -> Path:
    epochs, train, evals = read_history(run_dir / "history.csv")
    image = Image.new("RGB", CANVAS, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    left, top = MARGIN["left"], MARGIN["top"]
    right, bottom = CANVAS[0] - MARGIN["right"], CANVAS[1] - MARGIN["bottom"]

    subtitle = read_best(run_dir)
    draw_frame(
        image, draw, (left, top, right, bottom),
        xs=epochs, ys=[train, evals],
        colors=[TRAIN_COLOR, EVAL_COLOR],
        names=["train_loss", "eval_loss"],
        xlabel="epoch", ylabel="loss",
        title=f"{run_dir.name}: training vs eval loss",
        subtitle=subtitle,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="run directory or its history.csv (default: newest)")
    ap.add_argument("--out", default=None, help="output PNG (default: <run>/loss.png)")
    ap.add_argument("--logs", default="logs", help="where run directories live")
    args = ap.parse_args()

    if args.run is None:
        run_dir = latest_run(Path(args.logs))
    else:
        given = Path(args.run)
        run_dir = given if given.is_dir() else given.parent
    if not (run_dir / "history.csv").exists():
        raise SystemExit(f"no history.csv in {run_dir}")

    out = Path(args.out) if args.out else run_dir / "loss.png"
    path = make_plot(run_dir, out)

    epochs, train, evals = read_history(run_dir / "history.csv")
    print(f"run    : {run_dir}")
    print(f"epochs : {len(epochs)}   train {train[0]:.4f} -> {train[-1]:.4f}   "
          f"eval {evals[0]:.4f} -> {evals[-1]:.4f}")
    print(f"plot   : {path}")


if __name__ == "__main__":
    main()
