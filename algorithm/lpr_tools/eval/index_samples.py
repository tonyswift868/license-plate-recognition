"""Validated index records and balanced, reproducible validation sampling."""
from __future__ import annotations

import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from lpr.plate_rules import match_layout
from lpr_tools.data.ccpd import read_index
from lpr_tools.data.manifest import ManifestRow, parse_quad, validate_manifest

DEFAULT_VALIDATION_N = 300
DEFAULT_SEED = 42
SUBSET_CATEGORIES = {
    "ccpd_base": "clear", "ccpd_green": "clear", "ccpd_fn": "clear",
    "ccpd_db": "night", "ccpd_tilt": "tilt", "ccpd_rotate": "tilt",
    "ccpd_blur": "dirty", "ccpd_weather": "dirty", "ccpd_challenge": "dirty",
    "ccpd_np": "np",
}


@dataclass(frozen=True)
class IndexSample:
    row: ManifestRow
    dataset: str
    subset: str
    split: str
    bbox: tuple[float, float, float, float] | None

    @property
    def stratum(self) -> str:
        return f"{self.dataset}/{self.subset}"


def path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _bbox(value: str) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in value.split())
    if (len(values) != 4 or not all(math.isfinite(v) for v in values)
            or min(values[:2]) < 0 or min(values[2:]) <= 0):
        raise ValueError("index bbox must contain finite x y w h with positive size")
    return values


def read_samples(index_path: str | Path) -> list[IndexSample]:
    """Skip unsupported labels; reject duplicate paths, including cross-split aliases.

    Relative image paths are relative to the index file, never the current cwd.
    No-plate records are valid only for ccpd_np with an empty/no_plate skip reason.
    """
    index_path = Path(index_path).resolve()
    seen = set()
    samples = []
    for raw in read_index(index_path):
        if None in raw or any(v is None for v in raw.values()) or not raw["path"].strip():
            raise ValueError("malformed CCPD index row")
        image = Path(raw["path"])
        image = (image if image.is_absolute() else index_path.parent / image).resolve()
        key = path_key(image)
        if key in seen:
            raise ValueError(f"duplicate or conflicting index path: {image}")
        seen.add(key)
        if raw["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid index split: {raw['split']}")
        np_sample = raw["subset"] == "ccpd_np" and raw["has_plate"] == "0"
        if np_sample:
            if raw["skip_reason"] not in {"", "no_plate"}:
                continue
        elif raw["has_plate"] != "1" or raw["skip_reason"]:
            continue
        category = SUBSET_CATEGORIES.get(raw["subset"])
        if category is None:
            raise ValueError(f"unsupported index subset category: {raw['subset']}")
        plate = "" if np_sample else raw["plate_no"]
        if not np_sample and (match_layout(plate) is None or str(len(plate)) != raw["layout"]):
            raise ValueError(f"invalid plate/layout in index: {image}")
        note = "validation grouping; image quality not manually verified"
        row = ManifestRow(image.as_posix(), plate, raw["plate_color"], category,
                          f"{raw['dataset']}/{raw['subset']}",
                          None if np_sample else parse_quad(raw["quad"]), note)
        validate_manifest([row])
        samples.append(IndexSample(row, raw["dataset"], raw["subset"], raw["split"],
                                   None if np_sample else _bbox(raw["bbox"])))
    return samples


def sample_validation(index_path: str | Path, *, n: int = DEFAULT_VALIDATION_N,
                      seed: int = DEFAULT_SEED) -> tuple[list[IndexSample], dict]:
    """Equal allocation by dataset/subset, redistributing exhausted strata.

    n is the total upper bound, without replacement; small pools use all records.
    A seeded stratum order avoids always favoring the same subset when n is tiny.
    """
    if n < 1:
        raise ValueError("--n must be positive")
    pools = defaultdict(list)
    for sample in read_samples(index_path):
        if sample.split == "val":
            pools[sample.stratum].append(sample)
    if not pools:
        raise ValueError("ccpd_index.csv contains no supported val samples")
    rng = random.Random(seed)
    order = sorted(pools)
    for key in order:
        pools[key].sort(key=lambda sample: path_key(sample.row.path))
        rng.shuffle(pools[key])
    rng.shuffle(order)
    sizes = {key: len(pools[key]) for key in sorted(pools)}
    selected = []
    offset = 0
    while len(selected) < min(n, sum(sizes.values())):
        for key in order:
            if offset < len(pools[key]):
                selected.append(pools[key][offset])
                if len(selected) == n:
                    break
        offset += 1
    return selected, {"split": "val", "seed": seed, "requested_n": n,
                      "selected_n": len(selected), "available_n": sum(sizes.values()),
                      "allocation": "balanced round-robin by dataset/subset; no replacement",
                      "available_by_subset": sizes,
                      "selected_by_subset": dict(sorted(Counter(s.stratum for s in selected).items())),
                      "paths": [s.row.path for s in selected]}
