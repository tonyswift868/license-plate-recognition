"""Validated index records and balanced, reproducible validation sampling."""
from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from lpr.config import ALGORITHM_DIR
from lpr.plate_rules import match_layout

from lpr_tools.data.ccpd import INDEX_FIELDS, read_index
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
FIXED_VAL_MANIFEST = ALGORITHM_DIR / "data/index/val1000_v1.csv"
FIXED_HARD_MANIFEST = ALGORITHM_DIR / "data/index/devhard_v1.csv"
FIXED_VAL_N = 1000
FIXED_HARD_N = 2100
FIXED_HARD_PER_SUBSET = 300
FROZEN_DAILY_MANIFESTS = (
    ALGORITHM_DIR / "data/index/testset_manifest_v1.csv",
    ALGORITHM_DIR / "tests/data/golden/golden_manifest.csv",
    ALGORITHM_DIR / "data/index/undecodable.csv",
)


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


def read_samples(index_path: str | Path, *, only_paths: set[str] | None = None) -> list[IndexSample]:
    """Skip unsupported labels; reject duplicate paths, including cross-split aliases.

    Relative image paths are relative to the index file, never the current cwd.
    No-plate records are valid only for ccpd_np with an empty/no_plate skip reason.
    """
    index_path = Path(index_path).resolve()
    seen = set()
    samples = []
    if only_paths is None:
        raw_rows = read_index(index_path)
    else:
        raw_rows = []
        with index_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(INDEX_FIELDS):
                raise ValueError("invalid CCPD index columns")
            for raw in reader:
                raw_path = Path(raw["path"])
                raw_path = raw_path if raw_path.is_absolute() else index_path.parent / raw_path
                if path_key(raw_path) in only_paths:
                    raw_rows.append(raw)
    for raw in raw_rows:
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


def fixed_manifest_samples(index_path: str | Path, manifest_path: str | Path, *,
                           purpose: str) -> tuple[list[IndexSample], dict]:
    """Load an ordered, versioned daily manifest and verify it against CCPD/index exclusions."""
    if purpose not in {"val", "dev-hard"}:
        raise ValueError("fixed manifest purpose must be val or dev-hard")
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixed {purpose} manifest missing: {manifest_path}")
    rows = read_manifest(manifest_path, check_files=True)
    expected = FIXED_VAL_N if purpose == "val" else FIXED_HARD_N
    if len(rows) != expected:
        raise ValueError(f"fixed {purpose} manifest must contain exactly {expected} rows; got {len(rows)}")
    manifest_keys = {path_key(image_path(row, manifest_path).resolve()) for row in rows}
    samples = read_samples(index_path, only_paths=manifest_keys)
    by_path = {path_key(s.row.path): s for s in samples}
    if set(by_path) != manifest_keys:
        raise ValueError(f"fixed {purpose} manifest has paths missing from CCPD index")
    selected = []
    seen = set()
    for row in rows:
        key = path_key(image_path(row, manifest_path).resolve())
        if key in seen:
            raise ValueError(f"duplicate path in fixed {purpose} manifest: {key}")
        seen.add(key)
        sample = by_path.get(key)
        if sample is None:
            raise ValueError(f"fixed {purpose} image is absent from supported CCPD index: {key}")
        if (row.plate_no, row.plate_color, row.category, row.source, row.quad) != (
                sample.row.plate_no, sample.row.plate_color, sample.row.category,
                sample.row.source, sample.row.quad):
            raise ValueError(f"fixed {purpose} label/metadata differs from CCPD index: {key}")
        if purpose == "val" and sample.split != "val":
            raise ValueError(f"fixed validation image is not in official val split: {key}")
        if purpose == "dev-hard" and sample.subset not in HARD_SUBSETS:
            raise ValueError(f"fixed dev-hard manifest contains an unsupported subset: {sample.subset}")
        selected.append(sample)
    if purpose == "dev-hard":
        counts = Counter(sample.subset for sample in selected)
        if any(counts[subset] != FIXED_HARD_PER_SUBSET for subset in HARD_SUBSETS):
            raise ValueError(f"fixed dev-hard manifest must contain exactly {FIXED_HARD_PER_SUBSET} rows per hard subset: {dict(counts)}")

    blocked_paths = set()
    blocked_hashes = set()
    manifests = [*FROZEN_DAILY_MANIFESTS, FIXED_HARD_MANIFEST if purpose == "val" else FIXED_VAL_MANIFEST]
    for frozen in manifests:
        if not frozen.is_file():
            raise FileNotFoundError(f"required daily exclusion manifest missing: {frozen}")
        if frozen.suffix.lower() == ".csv":
            with frozen.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "path" not in reader.fieldnames:
                    raise ValueError(f"daily exclusion manifest needs a path column: {frozen}")
                for raw in reader:
                    if raw.get("path", "").strip():
                        candidate = Path(raw["path"].strip())
                        candidate = candidate if candidate.is_absolute() else frozen.parent / candidate
                        candidate = candidate.resolve()
                        if not candidate.is_file():
                            raise FileNotFoundError(f"required daily exclusion image missing: {candidate}")
                        blocked_paths.add(path_key(candidate))
                        blob = candidate.read_bytes()
                        blocked_hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
        else:
            raise ValueError(f"unsupported daily exclusion manifest format: {frozen}")
    for selected_sample in selected:
        source = Path(selected_sample.row.path)
        blob = source.read_bytes()
        digest = (len(blob), hashlib.sha256(blob).hexdigest())
        if path_key(source) in blocked_paths or digest in blocked_hashes:
            raise ValueError(f"fixed {purpose} image overlaps a frozen exclusion by path or bytes: {source}")
    return selected, {"mode": "fixed_manifest", "purpose": purpose, "manifest": str(manifest_path),
                      "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "selected_n": len(selected), "paths_sha256": hashlib.sha256(
                          ("\n".join(path_key(sample.row.path) for sample in selected) + "\n").encode("utf-8")).hexdigest(),
                      "excluded_manifests": [str(path.resolve()) for path in manifests],
                      "selected_by_subset": dict(sorted(Counter(s.stratum for s in selected).items())),
                      "paths": [s.row.path for s in selected]}


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
