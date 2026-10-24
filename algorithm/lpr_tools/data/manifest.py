"""Versionable G2 manifest I/O; relative image paths resolve beside the CSV."""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

from lpr.plate_rules import match_layout

FIELDS = ("path", "plate_no", "plate_color", "category", "source", "quad", "note")
CATEGORIES = ("clear", "night", "tilt", "dirty", "local", "np")
COLORS = {"blue", "yellow", "green", "white", "black", "unknown"}


@dataclass(frozen=True)
class ManifestRow:
    path: str
    plate_no: str
    plate_color: str
    category: str
    source: str
    quad: tuple[tuple[float, float], ...] | None = None
    note: str = ""


def parse_quad(value: str) -> tuple[tuple[float, float], ...] | None:
    if not value.strip():
        return None
    numbers = [float(v) for v in value.split()]
    if len(numbers) != 8 or not all(math.isfinite(v) and v >= 0 for v in numbers):
        raise ValueError("quad must contain eight finite nonnegative coordinates (LT RT RB LB)")
    return tuple(zip(numbers[::2], numbers[1::2]))


def image_path(row: ManifestRow, manifest_path: str | Path) -> Path:
    p = Path(row.path)
    return p if p.is_absolute() else Path(manifest_path).resolve().parent / p


def validate_manifest(rows: Iterable[ManifestRow], *, base_dir: str | Path = ".",
                      check_files: bool = False) -> list[ManifestRow]:
    rows = list(rows)
    seen = set()
    for i, row in enumerate(rows, 2):
        prefix = f"manifest row {i}: "
        if not row.path.strip() or row.category not in CATEGORIES or row.plate_color not in COLORS:
            raise ValueError(prefix + "invalid path/category/color")
        if row.category == "np":
            if row.plate_no:
                raise ValueError(prefix + "np plate_no must be empty")
        elif match_layout(row.plate_no) is None:
            raise ValueError(prefix + "unsupported plate format")
        p = Path(row.path)
        p = (p if p.is_absolute() else Path(base_dir) / p).resolve()
        key = os.path.normcase(str(p))
        if key in seen:
            raise ValueError(prefix + "duplicate path")
        seen.add(key)
        if row.quad is not None:
            parse_quad(" ".join(str(v) for point in row.quad for v in point))
        if check_files and not p.is_file():
            raise ValueError(prefix + f"missing image: {p}")
    return rows


def read_manifest(path: str | Path, *, check_files: bool = False) -> list[ManifestRow]:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("invalid manifest columns")
        rows = []
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("invalid manifest row width")
            rows.append(ManifestRow(**{**raw, "quad": parse_quad(raw["quad"])}))
    return validate_manifest(rows, base_dir=path.resolve().parent, check_files=check_files)


def write_manifest(path: str | Path, rows: Iterable[ManifestRow], *, check_files: bool = False) -> None:
    path = Path(path)
    rows = validate_manifest(rows, base_dir=path.resolve().parent, check_files=check_files)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (" ".join(f"{v:g}" for point in row.quad for v in point)
                                     if row.quad else "") if field == "quad" else getattr(row, field)
                             for field in FIELDS})
