"""Generate a font-only smoke dataset; its accuracy is not CCPD acceptance."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lpr.charset import CHARSET
from lpr.charnorm import normalize_char, unify_polarity
from lpr.io_utils import imwrite
from .augment import augment_char

DEFAULT_FONT = Path("C:/Windows/Fonts/simhei.ttf")
# RGB plate backgrounds and text; conversion to gray is explicit below.
PALETTES = {
    "blue": ((20, 65, 180), (245, 245, 245)),
    "yellow": ((235, 195, 25), (15, 15, 15)),
    "green": ((130, 220, 150), (15, 15, 15)),
    "white": ((235, 235, 235), (15, 15, 15)),
    "black": ((20, 20, 20), (240, 240, 240)),
}


def render_char(char: str, font: ImageFont.FreeTypeFont, color: str) -> np.ndarray:
    """Render tightly cropped glyph with a small plate-colored margin."""
    left, top, right, bottom = font.getbbox(char)
    background, foreground = PALETTES[color]
    image = Image.new("RGB", (right - left + 8, bottom - top + 8), background)
    ImageDraw.Draw(image).text((4 - left, 4 - top), char, font=font, fill=foreground)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return normalize_char(unify_polarity(gray, color))


def generate_dataset(out: Path, *, per_class: int = 200, seed: int = 42,
                     font_path: Path = DEFAULT_FONT, val_fraction: float = 0.2) -> list[dict]:
    """per_class is total train+val per label; refuse stale output datasets."""
    out = Path(out)
    if per_class < 2 or not 0 < val_fraction < 1:
        raise ValueError("per_class >= 2 and 0 < val_fraction < 1 required")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}")
    # Fail before producing a partial dataset if the requested font is unavailable.
    font = ImageFont.truetype(str(font_path), 48)
    out.mkdir(parents=True, exist_ok=True)
    val_count = max(1, min(per_class - 1, round(per_class * val_fraction)))
    split_seeds = np.random.SeedSequence(seed).spawn(2)
    rows = []
    for split, count, split_seed in zip(("train", "val"), (per_class - val_count, val_count), split_seeds):
        rng = np.random.default_rng(split_seed)
        for label, char in enumerate(CHARSET):
            for index in range(count):
                color = str(rng.choice(list(PALETTES)))
                image = augment_char(render_char(char, font, color), rng)
                relative = Path(split) / str(label) / f"{index:05d}.png"
                imwrite(out / relative, image)
                rows.append({"path": relative.as_posix(), "split": split, "label": label,
                             "char": char, "plate_color": color, "source": "font_synthetic",
                             "font": str(font_path), "seed": seed})
    with (out / "chars_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("label char train val total")
    for label, char in enumerate(CHARSET):
        print(f"{label:2d} {char} {per_class - val_count:5d} {val_count:3d} {per_class:5d}")
    print(f"TOTAL train={65 * (per_class - val_count)} val={65 * val_count} total={65 * per_class}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    generate_dataset(args.out, per_class=args.per_class, seed=args.seed,
                     font_path=args.font, val_fraction=args.val_fraction)


if __name__ == "__main__":
    main()
