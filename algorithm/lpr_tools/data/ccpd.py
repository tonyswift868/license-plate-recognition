"""CCPD filename decoding and index generation, without dataset downloads.

Directory assumptions and the opaque area field await verification on delivery.
Paths written to the index are absolute, so read_index is cwd-independent.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from lpr.charset import PROVINCES
from lpr.io_utils import IMAGE_EXTS
from lpr.plate_rules import match_layout

PROVINCES_CCPD = PROVINCES + ["警", "学", "O"]
ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]
# Public aliases follow official CCPD nomenclature.
provinces, alphabets, ads = PROVINCES_CCPD, ALPHABETS, ADS
DATASET_2019 = "ccpd2019"
DATASET_2020 = "ccpd2020"
GREEN_SUBDIR = "ccpd_green"
SPLITS_SUBDIR = "splits"
SPLIT_NAMES = {"train": "train", "val": "val", "test": "test"}
DEFAULT_2019_SPLIT = "test"
AREA_MEANING = "raw filename token; scale/units pending verification"
INDEX_FIELDS = ("path", "dataset", "subset", "split", "has_plate", "plate_no", "layout",
                "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness", "blurriness",
                "area", "skip_reason")


@dataclass(frozen=True)
class CcpdRecord:
    plate_no: str
    layout: int
    quad: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    tilt_h: int
    tilt_v: int
    brightness: int
    blurriness: int
    area: str
    skip_reason: str = ""


def _numbers(token: str, separator: str, count: int) -> list[int]:
    pieces = token.split(separator)
    if len(pieces) != count or any(not p.isascii() or not p.isdigit() for p in pieces):
        raise ValueError(f"invalid numeric field: {token}")
    return [int(p) for p in pieces]


def parse_ccpd_name(name: str | Path) -> CcpdRecord:
    """Decode seven filename fields; unsupported labels carry a skip reason."""
    stem = Path(str(name).replace("\\", "/")).stem
    fields = stem.split("-")
    if len(fields) != 7 or not fields[0].isascii() or not fields[0].isdigit():
        raise ValueError(f"invalid CCPD filename: {name}")
    area, tilt, bounds, vertices, labels, brightness, blurriness = fields
    th, tv = _numbers(tilt, "_", 2)
    corners = [_numbers(v, "&", 2) for v in bounds.split("_")]
    if len(corners) != 2:
        raise ValueError("bbox must have two corners")
    (x1, y1), (x2, y2) = corners
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    quad = [tuple(_numbers(v, "&", 2)) for v in vertices.split("_")]
    if len(quad) != 4:
        raise ValueError("quad must have four vertices")
    label_parts = labels.split("_")
    if len(label_parts) not in (7, 8):
        raise ValueError("plate must have 7 or 8 indices")
    indices = _numbers(labels, "_", len(label_parts))
    tables = [PROVINCES_CCPD, ALPHABETS] + [ADS] * (len(indices) - 2)
    if any(i >= len(table) for i, table in zip(indices, tables)):
        raise ValueError("CCPD character index out of range")
    plate = "".join(table[i] for i, table in zip(indices, tables))
    reason = ("placeholder" if "O" in plate else "unsupported_province" if plate[0] in "警学"
              else "invalid_format" if match_layout(plate) is None else "")
    return CcpdRecord(plate, len(plate), [quad[i] for i in (2, 3, 0, 1)],
                      (x1, y1, x2-x1, y2-y1), th, tv,
                      _numbers(brightness, "_", 1)[0], _numbers(blurriness, "_", 1)[0], area, reason)


def _official_splits(root: Path, images: list[Path]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}
    for p in images:
        by_name.setdefault(p.name, []).append(p.relative_to(root).as_posix())
    for split_file in sorted((root / SPLITS_SUBDIR).glob("*.txt")):
        split = SPLIT_NAMES.get(split_file.stem)
        if split is None:
            if not split_file.stem.startswith("ccpd_"):
                continue
            split = "test"
        for line in split_file.read_text(encoding="utf-8-sig").splitlines():
            key = line.strip().replace("\\", "/").removeprefix("./")
            if not key:
                continue
            if key.startswith(DATASET_2019 + "/"):
                key = key[len(DATASET_2019)+1:]
            if "/" not in key:
                matches = by_name.get(key, [])
                if len(matches) > 1:
                    raise ValueError(f"ambiguous split filename: {key}")
                if matches:
                    key = matches[0]
            if key in assignments and assignments[key] != split:
                raise ValueError(f"conflicting official splits for {key}")
            assignments[key] = split
    return assignments


def build_index(raw_root: str | Path, out_csv: str | Path) -> list[dict[str, str]]:
    raw_root, out_csv = Path(raw_root).resolve(), Path(out_csv)
    if not raw_root.is_dir():
        raise ValueError(f"raw directory does not exist: {raw_root}")
    rows: list[dict[str, str]] = []
    for dataset in (DATASET_2019, DATASET_2020):
        root = raw_root / dataset
        scan = root if dataset == DATASET_2019 else root / GREEN_SUBDIR
        images = sorted(p for p in scan.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        splits = _official_splits(root, images) if dataset == DATASET_2019 else {}
        for image in images:
            relative = image.relative_to(root)
            subset = relative.parts[0]
            split = (splits.get(relative.as_posix(), DEFAULT_2019_SPLIT) if dataset == DATASET_2019
                     else relative.parts[1] if len(relative.parts) >= 3 else "")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unknown CCPD2020 split: {image}")
            row = dict.fromkeys(INDEX_FIELDS, "")
            row.update(path=image.as_posix(), dataset=dataset, subset=subset, split=split,
                       plate_color="green" if dataset == DATASET_2020 else "blue", has_plate="1")
            try:
                record = parse_ccpd_name(image.name)
            except ValueError:
                row.update(has_plate="0", skip_reason="no_plate" if subset == "ccpd_np" else "malformed_name")
            else:
                data = asdict(record)
                data["quad"] = " ".join(str(v) for point in record.quad for v in point)
                data["bbox"] = " ".join(map(str, record.bbox))
                row.update({k: str(v) for k, v in data.items()})
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"indexed: {len(rows)}")
    for title, counts in (
        ("subsets", Counter(f"{r['dataset']}/{r['subset']}" for r in rows)),
        ("splits", Counter(r["split"] for r in rows)),
        ("skip_reasons", Counter(r["skip_reason"] or "accepted" for r in rows))):
        print(f"{title}: {json.dumps(dict(sorted(counts.items())), ensure_ascii=False)}")
    return rows


def read_index(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(INDEX_FIELDS):
            raise ValueError("invalid CCPD index columns")
        return list(reader)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build CCPD index; area and directory assumptions await data verification")
    parser.add_argument("--raw", default="algorithm/data/raw", help="raw root containing ccpd2019/ and ccpd2020/")
    parser.add_argument("--out", default="algorithm/data/index/ccpd_index.csv", help="UTF-8 index CSV")
    args = parser.parse_args(argv)
    try:
        build_index(args.raw, args.out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
