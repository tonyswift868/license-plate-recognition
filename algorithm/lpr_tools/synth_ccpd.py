"""Generate a small, CCPD-shaped synthetic dataset for pipeline development.

Images and filename tokens follow the conventions consumed by
``lpr_tools.data.ccpd``. The pixels and labels are synthetic; this dataset is
only useful for exercising the data flow, not measuring recognition quality.
"""
from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lpr.charset import LETTERS, PROVINCES
from lpr.io_utils import imwrite
from lpr_tools.data.ccpd import ADS, ALPHABETS, PROVINCES_CCPD
from lpr_tools.synth_scenes import CENTERS7, CENTERS8, draw_plate

IMAGE_SIZE = (720, 1160)  # width, height: portrait CCPD frame
SUBSETS_2019 = ("ccpd_base", "ccpd_db", "ccpd_tilt", "ccpd_blur", "ccpd_np")
SPLITS_2020 = ("train", "val", "test")
FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")


def _plate_text(rng: random.Random, layout: int) -> str:
    province = rng.choice(PROVINCES)
    letter = rng.choice(LETTERS)
    if layout == 7:
        tail = "".join(rng.choice(LETTERS + list("0123456789")) for _ in range(5))
        return province + letter + tail
    # CCPD2020 small new-energy layout: PROV, LETTER, D/F, ALNUM, DIGIT x 4.
    return province + letter + rng.choice("DF") + rng.choice(LETTERS + list("0123456789")) + "".join(
        rng.choice("0123456789") for _ in range(4)
    )


def _ccpd_indices(text: str) -> str:
    tables = [PROVINCES_CCPD, ALPHABETS] + [ADS] * (len(text) - 2)
    return "_".join(str(table.index(ch)) for table, ch in zip(tables, text))


@lru_cache(maxsize=1)
def _province_font():
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"approved Chinese font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), 100)


def _draw_plate(text: str, green: bool) -> np.ndarray:
    """Render one compact plate; Pillow supplies the Chinese province glyph."""
    plate = draw_plate(text, "green" if green else "blue")
    foreground = (12, 12, 12) if green else (245, 245, 245)
    centers = CENTERS8 if len(text) == 8 else CENTERS7
    rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = _province_font()
    center = round(float(centers[0]))
    # Fit the province into its physical slot; font point size is not glyph width.
    mask = Image.new("L", (140, 140), 0)
    painter = ImageDraw.Draw(mask)
    bbox = painter.textbbox((0, 0), text[0], font=font)
    painter.text((-bbox[0], -bbox[1]), text[0], fill=255, font=font)
    mask = mask.crop(mask.getbbox()).resize((43, 88), Image.Resampling.LANCZOS)
    background = tuple(reversed(plate[70, 65].tolist()))
    draw.rectangle((center - 24, 20, center + 24, 120), fill=background)
    canvas.paste(tuple(reversed(foreground)), (center - 21, 26), mask)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _make_sample(text: str | None, subset: str, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (92, 98, 104), dtype=np.uint8)
    # A simple vehicle-like shape gives the crop context and preserves the portrait frame.
    cv2.rectangle(image, (105, 230), (615, 920), (61, 72, 82), -1)
    cv2.rectangle(image, (180, 300), (540, 470), (40, 47, 54), -1)
    cv2.circle(image, (165, 870), 58, (20, 20, 20), -1)
    cv2.circle(image, (555, 870), 58, (20, 20, 20), -1)
    if text is None:
        quad = [(0, 0)] * 4
        tilt = 0.0
    else:
        plate = _draw_plate(text, len(text) == 8)
        ph, pw = plate.shape[:2]
        center = np.array([360.0, 650.0])
        tilt = float(rng.uniform(-24, 24)) if subset == "ccpd_tilt" else float(rng.uniform(-5, 5))
        angle = np.deg2rad(tilt)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        half = np.array([[ -pw/2, -ph/2], [pw/2, -ph/2], [pw/2, ph/2], [-pw/2, ph/2]])
        perspective = (0.12, 0.20) if subset == "ccpd_tilt" else (0.0, 0.05)
        # A tapered top edge adds pronounced perspective to the tilt subset.
        half[0, 0] += pw * rng.uniform(*perspective)
        half[1, 0] -= pw * rng.uniform(*perspective)
        half[2, 0] -= pw * rng.uniform(*perspective)
        half[3, 0] += pw * rng.uniform(*perspective)
        half[0, 1] -= ph * rng.uniform(*perspective)
        half[1, 1] += ph * rng.uniform(*perspective)
        half[2, 1] += ph * rng.uniform(*perspective)
        half[3, 1] -= ph * rng.uniform(*perspective)
        points = np.rint(half @ rot.T + center).astype(np.float32)
        quad = [(int(x), int(y)) for x, y in points]
        src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
        matrix = cv2.getPerspectiveTransform(src, points)
        warped = cv2.warpPerspective(plate, matrix, IMAGE_SIZE)
        mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, IMAGE_SIZE)
        image[mask > 0] = warped[mask > 0]
        if subset == "ccpd_db":
            image = np.clip(image.astype(np.float32) * 0.38, 0, 255).astype(np.uint8)
        elif subset == "ccpd_blur":
            image = cv2.GaussianBlur(image, (13, 13), 3.0)
    noise = rng.normal(0, 2.5, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), quad, tilt


def _filename(text: str | None, quad: list[tuple[int, int]], image: np.ndarray,
              index: int, rng: np.random.Generator, tilt_deg: float) -> tuple[str, dict]:
    if text is None:
        return f"noplate_{index:05d}.jpg", {"area_ratio": 0.0, "tilt_degrees": 0.0,
                                              "brightness_method": "not_applicable", "blurriness_method": "not_applicable"}
    # Filename vertex order is RB, LB, LT, RT; parse_ccpd_name reorders to LT,RT,RB,LB.
    lt, rt, rb, lb = quad
    xs, ys = zip(*quad)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    area_ratio = cv2.contourArea(np.asarray(quad, dtype=np.float32)) / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
    area = max(1, round(area_ratio * 1000))  # custom synthetic area token: projected polygon fraction × 1000
    labels = _ccpd_indices(text)
    crop = image[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = int(round(float(np.mean(gray))))
    blurriness = min(255, int(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 100)))
    # Angles are measured from the integer quad; the 90-degree horizontal
    # offset keeps all tokens unsigned, as required by the existing decoder.
    tilt_h = round(90 + np.degrees(np.arctan2(rt[1]-lt[1], rt[0]-lt[0])))
    tilt_v = round(np.degrees(np.arctan2(lb[1]-lt[1], lb[0]-lt[0])))
    return (f"{area:03d}-{tilt_h}_{tilt_v}-"
            f"{x1}&{y1}_{x2}&{y2}-{rb[0]}&{rb[1]}_{lb[0]}&{lb[1]}_"
            f"{lt[0]}&{lt[1]}_{rt[0]}&{rt[1]}-{labels}-{brightness}-{blurriness}.jpg",
            {"area_ratio": float(area_ratio), "tilt_degrees": abs(float(tilt_deg)),
             "area_token": area, "bbox": [x1, y1, x2-x1, y2-y1],
             "tilt_h": tilt_h, "tilt_v": tilt_v, "brightness": brightness, "blurriness": blurriness,
             "area_token_formula": "round(projected quadrilateral area / image area * 1000)",
             "tilt_token_formula": "horizontal=90+top-edge angle, vertical=left-edge angle; degrees from integer quad; official orientation convention pending real-data verification",
             "brightness_method": "mean grayscale of axis-aligned plate bounding crop",
             "blurriness_method": "min(255, round(variance of Laplacian on bounding crop / 100))"})


def _clear_previous_output(out: Path) -> None:
    """Allow repeat runs only when the directory is wholly owned by this tool."""
    if not out.exists() or not any(out.iterdir()):
        return
    manifest = out / "truth.json"
    if not manifest.is_file():
        raise ValueError(f"refusing to overwrite nonempty output without truth.json: {out}")
    try:
        old = json.loads(manifest.read_text(encoding="utf-8"))
        owned = {Path(row["path"]).resolve() for row in old}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot validate existing generated output: {manifest}") from exc
    root = out.resolve()
    if any(path == root or root not in path.parents for path in owned):
        raise ValueError("truth.json contains paths outside its output directory")
    known_split_names = {"train.txt", "val.txt", "test.txt", *(f"{name}.txt" for name in SUBSETS_2019)}
    owned_files = set(owned)
    owned_files.update(p.resolve() for p in (out / "ccpd2019" / "splits").glob("*.txt")
                       if p.name in known_split_names)
    owned_files.add(manifest.resolve())
    if any(path == root or not path.is_relative_to(root) for path in owned_files):
        raise ValueError("managed output resolves outside its output directory")
    actual_files = {p.resolve() for p in out.rglob("*") if p.is_file()}
    if not actual_files.issubset(owned_files):
        extras = sorted(str(p) for p in actual_files - owned_files)
        raise ValueError(f"refusing to overwrite untracked files under output: {extras[:3]}")
    for path in owned_files & actual_files:
        path.unlink()


def generate(out: str | Path, per_subset: int = 50, seed: int = 42) -> list[dict]:
    """Write 5 CCPD2019-like subsets and 3 CCPD2020 splits.

    Returns independent truth rows with absolute image path, encoded filename,
    plate number, quad (LT,RT,RB,LB), dataset/subset/split, and plate flag.
    """
    if not isinstance(per_subset, int) or per_subset < 0 or not isinstance(seed, int) or seed < 0:
        raise ValueError("per_subset and seed must be nonnegative integers")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _clear_previous_output(out)
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)
    truth: list[dict] = []
    split_rows: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    subset_rows: dict[str, list[str]] = {name: [] for name in SUBSETS_2019}
    for subset in SUBSETS_2019:
        for i in range(per_subset):
            has_plate = subset != "ccpd_np"
            text = _plate_text(py_rng, 7) if has_plate else None
            image, quad, tilt = _make_sample(text, subset, rng)
            filename, measured = _filename(text, quad, image, i, rng, tilt)
            relative = Path(subset) / filename
            target = out / "ccpd2019" / relative
            imwrite(target, image)
            rel_posix = relative.as_posix()
            if subset == "ccpd_base":
                split = ("train", "val", "test")[i % 3]
                split_rows[split].append(rel_posix)
            else:
                split = "test"
                subset_rows[subset].append(rel_posix)
            truth.append({"path": str(target.resolve()), "file": filename, "plate_no": text or "",
                          "plate_text": text or "", "quad": [tuple(p) for p in quad], "dataset": "ccpd2019",
                          "subset": subset, "split": split, "has_plate": has_plate, "layout": len(text) if text else 0,
                          "metrics": measured})
    split_dir = out / "ccpd2019" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in split_rows.items():
        (split_dir / f"{name}.txt").write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    for subset, paths in subset_rows.items():
        (split_dir / f"{subset}.txt").write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")

    for split in SPLITS_2020:
        for i in range(per_subset):
            text = _plate_text(py_rng, 8)
            image, quad, tilt = _make_sample(text, "ccpd_base", rng)
            filename, measured = _filename(text, quad, image, i, rng, tilt)
            target = out / "ccpd2020" / "ccpd_green" / split / filename
            imwrite(target, image)
            truth.append({"path": str(target.resolve()), "file": filename, "plate_no": text,
                          "plate_text": text, "quad": [tuple(p) for p in quad], "dataset": "ccpd2020",
                          "subset": "ccpd_green", "split": split, "has_plate": True, "layout": 8,
                          "metrics": measured})

    truth_path = out / "truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="algorithm/data/raw_mock")
    parser.add_argument("--per-subset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        rows = generate(args.out, args.per_subset, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(rows)} CCPD-shaped samples under {Path(args.out)}; seed={args.seed}; per_subset={args.per_subset}")
    print(f"Truth metadata: {Path(args.out) / 'truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
