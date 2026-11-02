"""Generate a font-only smoke dataset; its accuracy is not CCPD acceptance."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from lpr.charnorm import normalize_char, unify_polarity
from lpr.charset import CHAR_TO_INDEX, CHARSET, PROVINCES
from lpr.io_utils import imwrite
from PIL import Image, ImageDraw, ImageFont

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
                     font_path: Path = DEFAULT_FONT, val_fraction: float = 0.2,
                     provinces_only: bool = False, train_only: bool = False,
                     labels: list[int | str] | None = None,
                     class_counts: dict[int | str, int] | None = None) -> list[dict]:
    """Generate selected font samples; per_class counts train+val unless train_only."""
    out = Path(out)
    if per_class < (1 if train_only else 2) or (not train_only and not 0 < val_fraction < 1):
        raise ValueError("per_class must be >=1 for train-only or >=2 with 0 < val_fraction < 1")
    if provinces_only and labels is not None:
        raise ValueError("--provinces-only and explicit labels cannot be combined")
    selected = list(range(len(PROVINCES))) if provinces_only else (
        list(range(len(CHARSET))) if labels is None else [_resolve_label(label) for label in labels])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("labels must be a non-empty set without duplicates")
    counts_by_label = {_resolve_label(label): int(count) for label, count in (class_counts or {}).items()}
    if set(counts_by_label) - set(selected):
        raise ValueError("class_counts can only override selected labels")
    for label in selected:
        count = counts_by_label.get(label, per_class)
        if count < (1 if train_only else 2):
            raise ValueError(f"sample count for {CHARSET[label]} must be >=1 for train-only or >=2 otherwise")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}")
    # Fail before producing a partial dataset if the requested font is unavailable.
    font = ImageFont.truetype(str(font_path), 48)
    out.mkdir(parents=True, exist_ok=True)
    split_seeds = np.random.SeedSequence(seed).spawn(2)
    rows = []
    splits = ("train",) if train_only else ("train", "val")
    split_seed_by_name = dict(zip(("train", "val"), split_seeds, strict=False))
    for split in splits:
        split_seed = split_seed_by_name[split]
        rng = np.random.default_rng(split_seed)
        for label in selected:
            char = CHARSET[label]
            total = counts_by_label.get(label, per_class)
            val_count = 0 if train_only else max(1, min(total - 1, round(total * val_fraction)))
            count = total if train_only else (total - val_count if split == "train" else val_count)
            for index in range(count):
                color = str(rng.choice(list(PALETTES)))
                image = augment_char(render_char(char, font, color), rng)
                relative = Path(split) / str(label) / f"{index:05d}.png"
                imwrite(out / relative, image)
                rows.append({"path": relative.as_posix(), "split": split, "label": label,
                             "char": char, "plate_color": color, "source": "font_synthetic",
                             "font": str(font_path), "seed": seed})
    with (out / "chars_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "label", "char", "plate_color",
                                                    "source", "font", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    print("label char train val total")
    total_train = total_val = total_all = 0
    for label in selected:
        char = CHARSET[label]
        total = counts_by_label.get(label, per_class)
        val_count = 0 if train_only else max(1, min(total - 1, round(total * val_fraction)))
        train_count = total if train_only else total - val_count
        total_train += train_count
        total_val += val_count
        total_all += total
        print(f"{label:2d} {char} {train_count:5d} {val_count:3d} {total:5d}")
    print(f"TOTAL train={total_train} val={total_val} total={total_all}")
    return rows


def _resolve_label(value: int | str) -> int:
    if isinstance(value, int):
        label = value
    else:
        raw = str(value)
        label = int(raw) if raw.isdecimal() else CHAR_TO_INDEX.get(raw, -1)
    if not 0 <= label < len(CHARSET):
        raise ValueError(f"unknown character label: {value}")
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--provinces-only", action="store_true",
                        help="generate only the 31 province labels")
    parser.add_argument("--train-only", action="store_true",
                        help="write all samples under train; do not create validation samples")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="selected label characters or charset indexes (for example 陕 26)")
    parser.add_argument("--class-count", action="append", default=[], metavar="LABEL=COUNT",
                        help="override this label's total sample count; repeat as needed")
    args = parser.parse_args()
    class_counts = {}
    for item in args.class_count:
        if "=" not in item:
            parser.error(f"--class-count must be LABEL=COUNT: {item}")
        label, count = item.split("=", 1)
        try:
            class_counts[label] = int(count)
        except ValueError:
            parser.error(f"--class-count must have an integer count: {item}")
    generate_dataset(args.out, per_class=args.per_class, seed=args.seed,
                     font_path=args.font, val_fraction=args.val_fraction,
                     provinces_only=args.provinces_only, train_only=args.train_only,
                     labels=args.labels, class_counts=class_counts)


if __name__ == "__main__":
    main()
