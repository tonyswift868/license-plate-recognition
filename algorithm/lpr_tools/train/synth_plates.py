"""Generate a reproducible, font-only, train-only rare-province plate set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lpr.charset import DIGITS, LETTERS, PROVINCES
from lpr.plate_rules import match_layout

DEFAULT_FONT = Path("C:/Windows/Fonts/simhei.ttf")
FIELDS = ("file", "plate_no", "layout", "color", "source", "orig_path")
BLUE_BG, BLUE_FG = (32, 74, 174), (247, 248, 246)
GREEN_BG, GREEN_FG = (142, 205, 139), (17, 33, 25)
LAYOUT_CENTERS = {
    7: (0.075397, 0.204365, 0.380952, 0.507937, 0.636905, 0.763889, 0.894841),
    8: (0.076087, 0.193841, 0.378623, 0.489130, 0.599638, 0.710145, 0.818841, 0.927536),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _plate_number(province: str, rng: np.random.Generator, layout: int) -> str:
    prefix = province + str(rng.choice(LETTERS))
    if layout == 7:
        plate = prefix + "".join(rng.choice(np.asarray(list(DIGITS + LETTERS)), size=5).tolist())
    elif rng.random() < .5:
        plate = prefix + str(rng.choice(tuple("DF"))) + str(rng.choice(DIGITS + LETTERS))
        plate += "".join(rng.choice(np.asarray(list(DIGITS)), size=4).tolist())
    else:
        plate = prefix + "".join(rng.choice(np.asarray(list(DIGITS)), size=5).tolist())
        plate += str(rng.choice(tuple("DF")))
    if match_layout(plate) is None:
        raise AssertionError(f"generated invalid plate: {plate}")
    return plate


def _render(text: str, layout: int, font_paths: tuple[Path, ...], rng: np.random.Generator) -> np.ndarray:
    width = 252 if layout == 7 else 276
    background, foreground = (BLUE_BG, BLUE_FG) if layout == 7 else (GREEN_BG, GREEN_FG)
    canvas = Image.new("RGB", (width, 80), background)
    font_path = font_paths[int(rng.integers(len(font_paths)))]
    centers = np.asarray(LAYOUT_CENTERS[layout]) * width
    for index, (char, center) in enumerate(zip(text, centers)):
        glyph_h = int(rng.integers(49, 63))
        glyph_w = int(rng.integers(25, 31)) if index == 0 else int(rng.integers(20, 28))
        font = _font(str(font_path), glyph_h + 18)
        box = font.getbbox(char)
        mask = Image.new("L", (box[2] - box[0] + 8, box[3] - box[1] + 8), 0)
        ImageDraw.Draw(mask).text((4 - box[0], 4 - box[1]), char, font=font, fill=255)
        cropped = mask.crop(mask.getbbox()).resize((glyph_w, glyph_h), Image.Resampling.LANCZOS)
        x = round(center - glyph_w / 2 + rng.uniform(-1.25, 1.25))
        y = round(40 - glyph_h / 2 + rng.uniform(-2.0, 2.0))
        canvas.paste(foreground, (x, y), cropped)

    image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    # Mild print/capture variation, bounded to avoid changing character identity.
    angle = float(rng.uniform(-1.5, 1.5))
    matrix = cv2.getRotationMatrix2D((width / 2, 40), angle, float(rng.uniform(.97, 1.03)))
    matrix[0, 2] += float(rng.uniform(-2, 2))
    matrix[1, 2] += float(rng.uniform(-1.5, 1.5))
    image = cv2.warpAffine(image, matrix, (width, 80), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .3:
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, 79], [0, 79]])
        jitter = rng.uniform(-2.0, 2.0, (4, 2)).astype(np.float32)
        perspective = cv2.getPerspectiveTransform(source, source + jitter)
        image = cv2.warpPerspective(image, perspective, (width, 80), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .25:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(.2, .8)))
    image = np.clip(image.astype(np.float32) * rng.uniform(.88, 1.12) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
    if rng.random() < .35:
        noise = rng.normal(0, rng.uniform(1, 5), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < .08:
        x = int(rng.integers(8, width - 12)); y = int(rng.integers(8, 70))
        cv2.line(image, (x, y), (min(width - 1, x + int(rng.integers(3, 9))), y),
                 color=(112, 112, 112) if layout == 7 else (85, 110, 85), thickness=1)
    return image


def generate_dataset(out: Path, *, real_labels: Path, min_per_province: int = 1500,
                     seed: int = 42, font_paths: tuple[Path, ...] | None = None) -> dict:
    """Fill every province with <1000 real train rows to min_per_province total.

    The output contains only a ``train`` directory. Synthetic labels never enter val.
    """
    out, real_labels = Path(out), Path(real_labels)
    if min_per_province < 1000:
        raise ValueError("min_per_province must be at least 1000")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}")
    fonts = tuple(font_paths or (DEFAULT_FONT, Path("C:/Windows/Fonts/simkai.ttf"),
                                 Path("C:/Windows/Fonts/simfang.ttf")))
    if not fonts or any(not path.is_file() for path in fonts):
        raise FileNotFoundError("all configured font files must exist")
    # Preflight every font and every target glyph before creating output files.
    for path in fonts:
        font = _font(str(path), 32)
        if any(font.getbbox(char) is None for char in PROVINCES + list(DIGITS) + list(LETTERS)):
            raise ValueError(f"font lacks a required plate glyph: {path}")
    with real_labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"real labels schema must be {FIELDS}")
        counts = Counter()
        for row in reader:
            if row["source"] == "ground_truth" and row["file"].startswith("train/"):
                if row["plate_no"] and row["plate_no"][0] in PROVINCES:
                    counts[row["plate_no"][0]] += 1
    rare = {province: max(0, min_per_province - counts[province])
            for province in PROVINCES if counts[province] < 1000}
    if out.exists() and not out.is_dir():
        raise ValueError(f"output path is not a directory: {out}")
    (out / "train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int]] = []
    synthetic_counts = Counter()
    for province, needed in rare.items():
        for number in range(needed):
            layout = 8 if rng.random() < .22 else 7
            color = "green" if layout == 8 else "blue"
            plate_no = _plate_number(province, rng, layout)
            image = _render(plate_no, layout, fonts, rng)
            payload = cv2.imencode(".png", image)[1].tobytes()
            name = hashlib.sha256(f"{seed}\0{province}\0{number}\0{plate_no}".encode()).hexdigest() + ".png"
            relative = f"train/{name}"
            (out / relative).write_bytes(payload)
            rows.append({"file": relative, "plate_no": plate_no, "layout": layout,
                         "color": color, "source": "font_synthetic", "orig_path": ""})
            synthetic_counts[province] += 1
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "generator": "lpr_tools.train.synth_plates",
        "seed": seed,
        "train_only": True,
        "min_per_province_including_real_train": min_per_province,
        "real_labels_sha256": _sha256(real_labels),
        "fonts": [{"path": str(path), "sha256": _sha256(path)} for path in fonts],
        "rare_provinces_real_train_ground_truth": {p: counts[p] for p in rare},
        "synthetic_train_by_province": dict(synthetic_counts),
        "synthetic_train_by_layout_color": {
            "7/blue": sum(row["layout"] == 7 for row in rows),
            "8/green": sum(row["layout"] == 8 for row in rows),
        },
        "rows": len(rows),
        "labels_sha256": _sha256(out / "labels.csv"),
    }
    (out / "synthetic_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-labels", type=Path, required=True)
    parser.add_argument("--min-per-province", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.out, real_labels=args.real_labels,
                                      min_per_province=args.min_per_province, seed=args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
