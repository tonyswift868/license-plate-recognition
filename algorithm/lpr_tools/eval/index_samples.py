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
from lpr_tools.data.manifest import COLORS, ManifestRow, image_path, parse_quad, read_manifest

DEFAULT_VALIDATION_N = 300
DEFAULT_SEED = 42
SUBSET_CATEGORIES = {
    "ccpd_base": "clear", "ccpd_green": "clear", "ccpd_fn": "clear",
    "ccpd_db": "night", "ccpd_tilt": "tilt", "ccpd_rotate": "tilt",
    "ccpd_blur": "dirty", "ccpd_weather": "dirty", "ccpd_challenge": "dirty",
    "ccpd_np": "np",
}
HARD_SUBSETS = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
                "ccpd_fn", "ccpd_challenge", "ccpd_weather")


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
    candidate = Path(path)
    # read_samples and manifest image resolution already make paths absolute;
    # normalize those lexically to avoid repeated filesystem resolution over a
    # 366k-row index. Relative inputs still resolve against the current context.
    value = str(candidate) if candidate.is_absolute() else str(candidate.resolve())
    return os.path.normcase(os.path.normpath(value))


def exclusion_paths(exclude_manifest: str | Path | list[str | Path] | tuple[str | Path, ...] | None
                    ) -> list[Path]:
    """Normalize one or more exclusion manifest arguments to a path list."""
    if exclude_manifest is None:
        return []
    if isinstance(exclude_manifest, (str, Path)):
        return [Path(exclude_manifest)]
    return [Path(path) for path in exclude_manifest]


def excluded_paths(exclude_manifest: str | Path | list[str | Path] | tuple[str | Path, ...] | None
                   ) -> tuple[set[str], list[str]]:
    """Read exclusion manifests once, resolving aliases before deduplication."""
    manifests = exclusion_paths(exclude_manifest)
    keys = set()
    for manifest in manifests:
        for row in read_manifest(manifest):
            keys.add(path_key(image_path(row, manifest).resolve()))
    return keys, [str(manifest.resolve()) for manifest in manifests]


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
        if raw["plate_color"] not in COLORS:
            raise ValueError(f"invalid index plate color: {raw['plate_color']}")
        plate = "" if np_sample else raw["plate_no"]
        if not np_sample and (match_layout(plate) is None or str(len(plate)) != raw["layout"]):
            raise ValueError(f"invalid plate/layout in index: {image}")
        note = "validation grouping; image quality not manually verified"
        row = ManifestRow(image.as_posix(), plate, raw["plate_color"], category,
                          f"{raw['dataset']}/{raw['subset']}",
                          None if np_sample else parse_quad(raw["quad"]), note)
        samples.append(IndexSample(row, raw["dataset"], raw["subset"], raw["split"],
                                   None if np_sample else _bbox(raw["bbox"])))
    return samples


def sample_validation(index_path: str | Path, *, n: int = DEFAULT_VALIDATION_N,
                      seed: int = DEFAULT_SEED, exclude_manifest: str | Path | None = None,
                      subsets: set[str] | None = None) -> tuple[list[IndexSample], dict]:
    """Equal allocation by dataset/subset, redistributing exhausted strata.

    n is the total upper bound, without replacement; small pools use all records.
    A seeded stratum order avoids always favoring the same subset when n is tiny.
    """
    if n < 1:
        raise ValueError("--n must be positive")
    excluded, exclusion_sources = excluded_paths(exclude_manifest)
    pools = defaultdict(list)
    for sample in read_samples(index_path):
        if (sample.split == "val" and (subsets is None or sample.subset in subsets)
                and path_key(sample.row.path) not in excluded):
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
                      "excluded_n": len(excluded), "exclusion_sources": exclusion_sources,
                      "paths": [s.row.path for s in selected]}


def sample_dev_hard(index_path: str | Path, *, n: int = 300, seed: int = DEFAULT_SEED,
                    exclude_manifest: str | Path | None = None) -> tuple[list[IndexSample], dict]:
    """Sample n remaining images per hard subset, regardless of official index split."""
    if n < 1:
        raise ValueError("--n must be positive")
    excluded, exclusion_sources = excluded_paths(exclude_manifest)
    pools = defaultdict(list)
    index_splits = defaultdict(Counter)
    for sample in read_samples(index_path):
        if sample.subset in HARD_SUBSETS and path_key(sample.row.path) not in excluded:
            pools[sample.subset].append(sample)
            index_splits[sample.subset][sample.split] += 1
    missing = [subset for subset in HARD_SUBSETS if not pools[subset]]
    if missing:
        raise ValueError("no supported development samples for hard subsets: " + ", ".join(missing))
    rng = random.Random(seed)
    selected = []
    available = {}
    selected_by_subset = {}
    for subset in HARD_SUBSETS:
        pool = sorted(pools[subset], key=lambda item: path_key(item.row.path))
        available[subset] = len(pool)
        rng.shuffle(pool)
        chosen = pool[:n]
        selected.extend(chosen)
        selected_by_subset[subset] = len(chosen)
    return selected, {"split": "dev-hard/index-splits", "seed": seed, "requested_per_subset": n,
                      "selected_n": len(selected), "available_by_subset": available,
                      "selected_by_subset": selected_by_subset,
                      "index_splits_by_subset": {subset: dict(sorted(counts.items()))
                                                  for subset, counts in sorted(index_splits.items())},
                      "excluded_n": len(excluded), "exclusion_sources": exclusion_sources,
                      "allocation": "seeded sampling without replacement within each hard subset",
                      "paths": [sample.row.path for sample in selected]}
