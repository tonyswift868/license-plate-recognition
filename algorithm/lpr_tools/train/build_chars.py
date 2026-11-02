"""Build normalized character crops from CCPD ground-truth plate corners."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from lpr.charnorm import normalize_char, unify_polarity
from lpr.charset import CHAR_TO_INDEX, CHARSET
from lpr.config import DEFAULT_CONFIG_PATH, load_config
from lpr.io_utils import decode_image, imread, imwrite
from lpr.locate import box_iou, locate
from lpr.pipeline import _layouts_to_try, _rectifications
from lpr.preprocess import preprocess
from lpr.segment import segment as segment_plate
from lpr.types import PlateCandidate, RectifiedPlate

from lpr_tools.data.ccpd import INDEX_FIELDS as CCPD_INDEX_FIELDS
from lpr_tools.data.manifest import image_path as manifest_image_path
from lpr_tools.data.manifest import read_manifest

INDEX_FIELDS = ("path", "source_path", "split", "label", "char", "plate_no", "position",
                "source_variant", "candidate_index", "rectification_index", "candidate_iou")
REJECTION_FIELDS = ("row", "split", "path", "reason")
PREFLIGHT_BATCH_SIZE = 256
PROGRESS_INTERVAL = 5000


def _canonical(path: Path) -> str:
    return str(path.resolve()).casefold()


def _iter_index(path: str | Path):
    """Yield one strictly shaped CCPD index row at a time."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CCPD_INDEX_FIELDS):
            raise ValueError("invalid CCPD index columns")
        for row_no, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"invalid CCPD index row width at row {row_no}")
            yield row


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(v) for v in value.split())
    except (AttributeError, ValueError) as exc:
        raise ValueError("invalid bbox: expected four numeric x y width height values") from exc
    if (len(values) != 4 or not all(math.isfinite(v) for v in values)
            or min(values[:2]) < 0 or min(values[2:]) <= 0):
        raise ValueError("invalid bbox: expected finite nonnegative origin and positive size")
    return values


def _parse_quad(value: str, image_shape: tuple[int, ...]) -> np.ndarray:
    try:
        parts = [float(v) for v in value.split()]
    except (AttributeError, ValueError) as exc:
        raise ValueError("invalid quad: expected eight numeric coordinates") from exc
    if len(parts) != 8 or not all(math.isfinite(v) and v >= 0 for v in parts):
        raise ValueError("invalid quad: expected eight finite nonnegative coordinates")
    points = np.asarray(parts, dtype=np.float32).reshape(4, 2)
    h, w = image_shape[:2]
    if np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("invalid quad: point outside source image")
    if abs(cv2.contourArea(points)) < 1:
        raise ValueError("invalid quad: degenerate area")
    return points


def _excluded_paths(manifest: str | Path | None, raw_root: Path) -> tuple[set[str], set[str]]:
    if manifest is None:
        return set(), set()
    manifest = Path(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("exclude manifest must contain a path column")
        result, content_hashes = set(), set()
        for row in reader:
            p = Path(row["path"])
            if not p.is_absolute():
                candidates = (manifest.resolve().parent / p, raw_root / p)
            else:
                candidates = (p,)
            for candidate in candidates:
                result.add(_canonical(candidate))
                # Keep the manifest's content identity as well as its paths so
                # renamed copies cannot bypass the train/val exclusion.
                if candidate.is_file():
                    content_hashes.add(_hash_file(candidate.resolve()))
        return result, content_hashes


def _excluded_golden(manifest: str | Path | None) -> tuple[set[str], set[str]]:
    """Resolve golden manifest paths by its contract and protect byte-identical copies."""
    if manifest is None:
        return set(), set()
    manifest_path = Path(manifest)
    rows = read_manifest(manifest_path, check_files=False)
    paths: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        source = manifest_image_path(row, manifest_path).resolve()
        paths.add(str(source).casefold())
        if source.is_file():
            hashes.add(_hash_file(source))
    return paths, hashes


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", cwd=repo_root)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _build_metadata(source: str, config, config_path: Path,
                    include_located_iou: float | None = None) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    algorithm_root = script_path.parents[2]
    repo_root = script_path.parents[3]
    segment_path = algorithm_root / "lpr" / "segment.py"
    charnorm_path = algorithm_root / "lpr" / "charnorm.py"
    dependency_paths = {
        "preprocess_code_sha256": algorithm_root / "lpr" / "preprocess.py",
        "locate_code_sha256": algorithm_root / "lpr" / "locate.py",
        "rectify_code_sha256": algorithm_root / "lpr" / "rectify.py",
        "pipeline_code_sha256": algorithm_root / "lpr" / "pipeline.py",
    }
    default_config = DEFAULT_CONFIG_PATH.resolve()
    def sha256(path: Path) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
    effective_json = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")
    result = {
        "source": source,
        "include_located_iou": include_located_iou,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "default_config_path": str(default_config),
        "default_config_sha256": sha256(default_config),
        "effective_config_sha256": hashlib.sha256(effective_json).hexdigest(),
        "build_chars_sha256": sha256(script_path),
        "segment_code_sha256": sha256(segment_path),
        "charnorm_code_sha256": sha256(charnorm_path),
        "segment_git_commit": _git_value(repo_root, "log", "-1", "--format=%H", "--", "algorithm/lpr/segment.py"),
        "segment_git_blob": _git_value(repo_root, "rev-parse", "HEAD:algorithm/lpr/segment.py"),
        "build_git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "build_git_branch": _git_value(repo_root, "branch", "--show-current"),
    }
    result.update({key: sha256(path) for key, path in dependency_paths.items()})
    return result


def _split_quotas(available: dict[str, int], cap: int) -> dict[str, int]:
    total = sum(available.values())
    budget = min(total, cap)
    if not total:
        return {"train": 0, "val": 0}
    if budget == 1:
        chosen = "train" if available.get("train", 0) else "val"
        return {"train": int(chosen == "train"), "val": int(chosen == "val")}
    quotas = {split: min(available.get(split, 0), budget * available.get(split, 0) // total)
              for split in ("train", "val")}
    if available.get("train", 0) and available.get("val", 0):
        for split in ("train", "val"):
            if quotas[split] == 0:
                other = "val" if split == "train" else "train"
                if quotas[other] > 1:
                    quotas[other] -= 1
                quotas[split] = 1
    remaining = budget - sum(quotas.values())
    fractions = {split: (budget * available.get(split, 0) / total) % 1 for split in ("train", "val")}
    while remaining:
        candidates = [s for s in ("train", "val") if quotas[s] < available.get(s, 0)]
        if not candidates:
            break
        chosen = max(candidates, key=lambda s: (fractions[s], available.get(s, 0) - quotas[s], s == "train"))
        quotas[chosen] += 1
        fractions[chosen] = -1
        remaining -= 1
    return quotas


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_file_sha256(revision: str, relative_path: str, repo_root: Path) -> str:
    result = subprocess.run(["git", "show", f"{revision}:{relative_path}"], check=True,
                            capture_output=True, cwd=repo_root)
    return hashlib.sha256(result.stdout).hexdigest()


def _locate_with_memory_retry(image: np.ndarray, config):
    """Retry only transient allocation failures in the per-image detector stage."""
    for attempt in range(3):
        try:
            pre = preprocess(image, config.preprocess, config.lane_roi(None))
            return pre, locate(pre, config.locate, config.runtime.max_candidates)
        except (MemoryError, SystemError) as exc:
            memory_cause = isinstance(exc, MemoryError) or isinstance(exc.__cause__, MemoryError)
            if not memory_cause or attempt == 2:
                raise
            gc.collect()
            time.sleep(float(attempt + 1))


def _load_resume_origin(path: str | Path, *, out: Path, index: Path,
                        exclude_manifest: str | Path | None, exclude_golden: str | Path | None,
                        source: str, include_located_iou: float | None, max_per_class: int,
                        skip_invalid: bool, workers: int, build_metadata: dict,
                        variant_caps: dict[str, int]) -> dict:
    """Validate the specific locate-before-write attestation and its closed CSV prefix."""
    # shortcut: only the attested failure before a source emits rows is resumable;
    # mid-write recovery requires a separate atomic per-source checkpoint contract.
    checkpoint_path = Path(path).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if (checkpoint.get("schema_version") != 1
            or checkpoint.get("attestation") != "locate_failure_before_current_source_write"
            or checkpoint.get("last_source_completed") is not True
            or checkpoint.get("prior_counters_unknown") is not True):
        raise ValueError("resume checkpoint lacks the supported locate-before-write attestation")
    if Path(checkpoint.get("partial_output_dir", "")).resolve() != out:
        raise ValueError("resume checkpoint output directory does not match --out")
    args = checkpoint.get("build_args", {})
    expected_args = {"source": source, "include_located_iou": include_located_iou,
                     "max_per_class": max_per_class, "skip_invalid": skip_invalid,
                     "workers": workers}
    if args != expected_args:
        raise ValueError("resume build arguments differ from origin checkpoint")
    if _hash_file(index.resolve()) != checkpoint.get("index_sha256"):
        raise ValueError("resume index SHA-256 differs from origin checkpoint")
    for argument, key in ((exclude_manifest, "exclude_manifest_sha256"),
                          (exclude_golden, "golden_manifest_sha256")):
        actual = _hash_file(Path(argument).resolve()) if argument is not None else None
        if actual != checkpoint.get(key):
            raise ValueError(f"resume {key} differs from origin checkpoint")
    origin_metadata = checkpoint.get("origin_build_metadata", {})
    if (checkpoint.get("origin_build_chars_sha256") != origin_metadata.get("build_chars_sha256")
            or checkpoint.get("origin_head") != origin_metadata.get("build_git_commit")):
        raise ValueError("resume origin build metadata is inconsistent")
    repo_root = Path(__file__).resolve().parents[3]
    try:
        committed_origin_sha = _git_file_sha256(
            checkpoint["origin_head"], "algorithm/lpr_tools/train/build_chars.py", repo_root)
    except (KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("resume origin commit or build source is unavailable") from exc
    if (committed_origin_sha != checkpoint["origin_build_chars_sha256"]
            or origin_metadata.get("build_git_branch") != build_metadata.get("build_git_branch")):
        raise ValueError("resume origin commit source or branch does not match its attestation")
    fingerprints = checkpoint.get("runtime_fingerprints", {})
    fingerprint_keys = {
        "config_sha256", "default_config_sha256", "effective_config_sha256",
        "segment_code_sha256", "charnorm_code_sha256", "segment_git_commit",
        "segment_git_blob", "preprocess_code_sha256", "locate_code_sha256",
        "rectify_code_sha256", "pipeline_code_sha256",
    }
    if (set(fingerprints) != fingerprint_keys
            or build_metadata.get("config_path") != origin_metadata.get("config_path")
            or build_metadata.get("default_config_path") != origin_metadata.get("default_config_path")):
        raise ValueError("resume checkpoint omits required runtime fingerprints or config identity")
    for key, expected in fingerprints.items():
        if build_metadata.get(key) != expected:
            raise ValueError(f"resume runtime fingerprint changed: {key}")
    if not out.is_dir():
        raise ValueError("resume output directory does not exist")
    chars_index = out / "chars_index.csv"
    if not chars_index.is_file():
        raise ValueError("resume chars_index.csv is missing")
    if chars_index.stat().st_size != checkpoint.get("chars_index_bytes"):
        raise ValueError("resume chars_index.csv byte length changed")
    if _hash_file(chars_index) != checkpoint.get("chars_index_sha256"):
        raise ValueError("resume chars_index.csv SHA-256 changed")
    if (out / "chars_rejected.csv").exists():
        raise ValueError("resume partial output unexpectedly contains chars_rejected.csv")

    counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    by_variant: dict[str, dict[str, Counter[int]]] = {
        variant: {"train": Counter(), "val": Counter()} for variant in variant_caps}
    seen_paths: set[str] = set()
    source_splits: dict[str, tuple[str, str]] = {}
    source_order: list[str] = []
    completed_groups: set[str] = set()
    current_group = None
    current_raw_source = None
    current_source_path = None
    current_source_key = None
    row_count = 0
    last_source = None
    with chars_index.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(INDEX_FIELDS):
            raise ValueError("resume chars_index.csv schema differs from current builder")
        for row in reader:
            row_count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError("resume chars_index.csv contains a malformed row width")
            split, char, plate = row["split"], row["char"], row["plate_no"]
            label, position = int(row["label"]), int(row["position"])
            variant = row["source_variant"]
            raw_source = row["source_path"]
            if raw_source != current_raw_source:
                current_source_path = Path(raw_source).resolve()
                current_source_key = str(current_source_path).casefold()
                current_raw_source = raw_source
            rel = Path(row["path"])
            if (split not in ("train", "val") or variant not in variant_caps
                    or len(plate) not in (7, 8) or not 0 <= position < len(plate)
                    or CHAR_TO_INDEX.get(char) != label or plate[position] != char):
                raise ValueError("resume chars_index.csv contains an invalid label or variant row")
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 3
                    or rel.parts[0] != split or rel.parts[1] != str(label)):
                raise ValueError("resume chars_index.csv contains an unsafe or inconsistent output path")
            rel_text = rel.as_posix()
            if rel_text in seen_paths:
                raise ValueError("resume chars_index.csv contains a duplicate output path")
            seen_paths.add(rel_text)
            crop_path = out / rel
            if not crop_path.is_file() or crop_path.stat().st_size == 0:
                raise ValueError(f"resume crop is missing or empty: {crop_path}")
            try:
                crop_path.resolve().relative_to(out)
            except ValueError as exc:
                raise ValueError(f"resume crop escapes the output directory: {crop_path}") from exc
            if variant == "gt":
                if (row["candidate_index"] != "-1" or row["rectification_index"] != "-1"
                        or row["candidate_iou"] != ""):
                    raise ValueError("resume GT row has candidate metadata")
            else:
                if (int(row["candidate_index"]) < 0 or int(row["rectification_index"]) < 0
                        or not float(include_located_iou) <= float(row["candidate_iou"]) <= 1.):
                    raise ValueError("resume located row has invalid candidate metadata")
            key = current_source_key
            if key != current_group:
                if key in completed_groups:
                    raise ValueError("resume source rows are not contiguous")
                if current_group is not None:
                    completed_groups.add(current_group)
                source_order.append(key)
                current_group = key
            old = source_splits.setdefault(key, (split, plate))
            if old != (split, plate):
                raise ValueError("resume source path has conflicting split or plate labels")
            counts[label] += 1
            split_counts[split] += 1
            by_variant[variant][split][label] += 1
            last_source = str(current_source_path)
    if row_count != checkpoint.get("chars_index_data_rows") or not row_count:
        raise ValueError("resume chars_index.csv row count differs from checkpoint")
    if last_source != checkpoint.get("last_source_path"):
        raise ValueError("resume last source path differs from checkpoint")
    output_files = [p for p in out.rglob("*") if p.is_file()]
    png_count = sum(1 for p in output_files if p.suffix.casefold() == ".png")
    if png_count != row_count:
        raise ValueError("resume output directory contains orphan PNGs or missing index rows")
    unexpected_files = [p for p in output_files
                        if p != chars_index and p.suffix.casefold() != ".png"]
    if unexpected_files:
        raise ValueError("resume output directory contains unexpected non-PNG files")
    allowed = {"chars_index.csv"}
    if {p.name for p in out.iterdir() if p.is_file()} != allowed:
        raise ValueError("resume output directory contains unexpected root-level files")
    return {"checkpoint": checkpoint, "counts": counts, "split_counts": split_counts,
            "variant_counts": by_variant, "source_splits": source_splits,
            "source_order": source_order, "row_count": row_count,
            "last_source": str(Path(last_source).resolve())}


def _preflight_one(item, config):
    row_no, row, source = item
    split = row["split"]
    if not source.is_file():
        if split == "test":
            return None, None, None
        return None, None, f"missing image: {source}"
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        return None, None, str(exc)
    digest = hashlib.sha256(source_bytes).hexdigest()
    if split == "test":
        return digest, None, None
    try:
        plate = row.get("plate_no", "")
        try:
            layout = int(row.get("layout", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid layout") from exc
        if layout not in (7, 8) or len(plate) != layout or any(ch not in CHAR_TO_INDEX for ch in plate):
            raise ValueError(f"unsupported plate label/layout: {plate!r}")
        if row.get("plate_color", "") not in {"blue", "yellow", "green", "white", "black"}:
            raise ValueError(f"invalid plate_color: {row.get('plate_color', '')!r}")
        image = decode_image(source_bytes)
        if image is None:
            raise ValueError(f"无法解码图像：{source}")
        quad = _parse_quad(row.get("quad", ""), image.shape)
        centers = config.segment.layout7_centers if layout == 7 else config.segment.layout8_centers
        char_w = config.segment.layout7_char_w if layout == 7 else config.segment.layout8_char_w
        if len(centers) != layout or not 0 < char_w <= 1 or any(not 0 < c < 1 for c in centers):
            raise ValueError(f"invalid slot configuration for layout {layout}")
        return digest, quad, None
    except (OSError, ValueError, cv2.error, TypeError) as exc:
        return digest, None, str(exc)


def build_chars(index: str | Path, raw: str | Path, out: str | Path, *,
                exclude_manifest: str | Path | None = None, max_per_class: int = 8000,
                skip_invalid: bool = False, workers: int = 1,
                source: str = "slots", exclude_golden: str | Path | None = None,
                include_located_iou: float | None = None,
                resume_origin: str | Path | None = None) -> dict[str, int]:
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    if workers not in (1, 2, 3, 4):
        raise ValueError("workers must be between 1 and 4")
    if source not in {"slots", "segment"}:
        raise ValueError("source must be 'slots' or 'segment'")
    if include_located_iou is not None and (source != "segment" or not 0 <= include_located_iou <= 1):
        raise ValueError("include_located_iou requires --source segment and a threshold in [0, 1]")
    if include_located_iou is not None and max_per_class < 3:
        raise ValueError("max_per_class must be at least 3 when located candidates are included")
    source_mode = source
    cv2.setNumThreads(1)
    raw_root, out = Path(raw).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()) and resume_origin is None:
        raise ValueError(f"output directory is not empty: {out}")
    excluded, excluded_hashes = _excluded_paths(exclude_manifest, raw_root)
    golden_paths, golden_hashes = _excluded_golden(exclude_golden)
    excluded.update(golden_paths)
    excluded_hashes.update(golden_hashes)
    config = load_config()
    config_path = Path(os.environ.get("LPR_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    build_metadata = _build_metadata(source_mode, config, config_path, include_located_iou)
    if include_located_iou is None:
        variant_caps = {"gt" if source_mode == "segment" else "slots": max_per_class}
    else:
        located_cap = max_per_class // 3
        variant_caps = {"gt": max_per_class - located_cap, "located": located_cap}
    resume_state = None
    resume_after_source_no = 0
    if resume_origin is not None:
        resume_state = _load_resume_origin(
            resume_origin, out=out, index=Path(index), exclude_manifest=exclude_manifest,
            exclude_golden=exclude_golden, source=source_mode,
            include_located_iou=include_located_iou, max_per_class=max_per_class,
            skip_invalid=skip_invalid, workers=workers, build_metadata=build_metadata,
            variant_caps=variant_caps)
        origin = resume_state["checkpoint"]
        build_metadata["resume_origin_head"] = origin["origin_head"]
        build_metadata["resume_origin_build_chars_sha256"] = origin["origin_build_chars_sha256"]
        build_metadata["resume_origin_chars_index_sha256"] = origin["chars_index_sha256"]
        build_metadata["resume_origin_checkpoint_sha256"] = _hash_file(Path(resume_origin).resolve())
        build_metadata["resume_build_commit"] = build_metadata["build_git_commit"]
        build_metadata["resume_build_chars_sha256"] = build_metadata["build_chars_sha256"]
    seen_paths: dict[str, tuple[str, int]] = {}
    seen_content: dict[str, str] = {}
    counts: Counter[int] = Counter()
    truncated: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    skipped_excluded = 0

    # Validate train/val sources fully. Test rows contribute path/content identities
    # when their files exist, but missing test files do not block character building.
    candidates = []
    preflight_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    for row_no, row in enumerate(_iter_index(index), 2):
        split = row.get("split")
        if row.get("has_plate") != "1" or split not in {"train", "val", "test"} or row.get("skip_reason", "").strip():
            continue
        source_text = row.get("path", "").strip()
        if not source_text:
            if split in {"train", "val"}:
                raise ValueError(f"row {row_no}: missing image path")
            continue
        source = Path(source_text)
        if not source.is_absolute():
            source = raw_root / source
        source = source.resolve()
        # source is resolved once here; retain _canonical's casefold semantics
        # without repeating filesystem resolution in the hot path.
        key = str(source).casefold()
        if split in {"train", "val"} and key in excluded:
            skipped_excluded += 1
            continue
        old_path = seen_paths.get(key)
        if old_path is not None:
            old_split, old_row = old_path
            if old_split != split:
                raise ValueError(f"image leakage across splits: {old_split} and {split}: {source}")
            raise ValueError(f"duplicate source path in index rows {old_row} and {row_no}: {source}")
        seen_paths[key] = (split, row_no)
        if split == "test":
            compact_row = {"split": "test"}
        else:
            compact_row = {key: row[key] for key in
                           ("split", "plate_no", "layout", "plate_color", "quad", "bbox")}
        candidates.append((row_no, compact_row, source))

    # Only submit bounded batches; executor.map preserves the source index order.
    valid = []
    next_preflight_progress = PROGRESS_INTERVAL
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(candidates), PREFLIGHT_BATCH_SIZE):
            batch = candidates[start:start + PREFLIGHT_BATCH_SIZE]
            results = list(pool.map(lambda item: _preflight_one(item, config), batch))
            for item, (digest, quad, error) in zip(batch, results, strict=False):
                row_no, row, source = item
                split = row["split"]
                if digest is not None:
                    if split != "test" and digest in excluded_hashes:
                        raise ValueError(f"image content matches exclude manifest: {source}")
                    old_split = seen_content.get(digest)
                    if old_split is not None and old_split != split:
                        raise ValueError(f"image leakage across splits: {old_split} and {split}: {source}")
                    seen_content[digest] = split
                if error:
                    if not skip_invalid:
                        raise ValueError(f"row {row_no} ({split}): {error}")
                    rejected.append({"row": str(row_no), "split": split,
                                     "path": str(source), "reason": error})
                    continue
                if digest is None:
                    continue  # Missing test images are not training inputs.
                if split in {"train", "val"}:
                    for char in row["plate_no"]:
                        preflight_counts[split][CHAR_TO_INDEX[char]] += 1
                    valid.append((row_no, row, source, quad))
            processed = min(start + len(batch), len(candidates))
            while processed >= next_preflight_progress:
                print(f"preflight: {next_preflight_progress}/{len(candidates)}", flush=True)
                next_preflight_progress += PROGRESS_INTERVAL

    # Leak checks are complete; keep only accepted train/val records for output.
    del candidates, seen_paths, seen_content
    quotas = {
        variant: {label: _split_quotas(
            {split: preflight_counts[split][label] for split in ("train", "val")}, cap)
                  for label in range(len(CHARSET))}
        for variant, cap in variant_caps.items()
    }
    variant_split_class_counts: dict[str, dict[str, Counter[int]]] = {
        variant: {"train": Counter(), "val": Counter()} for variant in variant_caps}
    if resume_state is not None:
        source_order = resume_state["source_order"]
        source_numbers = {str(source).casefold(): i for i, (_, _, source, _) in enumerate(valid, 1)}
        validated_sources = {str(source).casefold(): (row["split"], row["plate_no"])
                             for _, row, source, _ in valid}
        missing_sources = [source for source in source_order if source not in source_numbers]
        if missing_sources:
            raise ValueError("resume CSV contains sources absent from the validated input prefix")
        previous_number = 0
        for source in source_order:
            number = source_numbers[source]
            if number <= previous_number:
                raise ValueError("resume CSV source order does not match the input index")
            if resume_state["source_splits"][source] != validated_sources[source]:
                raise ValueError("resume CSV source labels differ from the validated input index")
            previous_number = number
        resume_after_source_no = source_numbers[str(Path(resume_state["last_source"]).resolve()).casefold()]
        for variant in variant_caps:
            for split in ("train", "val"):
                for label, n in resume_state["variant_counts"][variant][split].items():
                    if n > quotas[variant][label][split]:
                        raise ValueError("resume prefix exceeds its frozen per-variant quota")
                    variant_split_class_counts[variant][split][label] = n
        counts.update(resume_state["counts"])
        split_counts.update(resume_state["split_counts"])
    segment_failed_images = 0
    segment_length_mismatch_images = 0
    segment_invalid_images = 0
    segment_attempted_images = 0
    segment_success_images = 0
    segment_attempted_by_split: Counter[str] = Counter()
    segment_success_by_split: Counter[str] = Counter()
    variant_attempted: Counter[str] = Counter()
    variant_success: Counter[str] = Counter()
    located_candidates_seen = 0
    located_candidates_iou = 0

    out.mkdir(parents=True, exist_ok=True)
    index_mode = "a" if resume_state is not None else "w"
    with (out / "chars_index.csv").open(index_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if resume_state is None:
            writer.writeheader()
        next_output_progress = ((resume_after_source_no // PROGRESS_INTERVAL) + 1) * PROGRESS_INTERVAL
        for source_no, (row_no, row, source, quad) in enumerate(valid, 1):
            if source_no <= resume_after_source_no:
                continue
            plate = row["plate_no"]
            layout = int(row["layout"])
            color = row["plate_color"]
            split = row["split"]
            available = [CHAR_TO_INDEX[ch] for ch in plate
                         if any(variant_split_class_counts[v][split][CHAR_TO_INDEX[ch]]
                                < quotas[v][CHAR_TO_INDEX[ch]][split] for v in variant_caps)]
            if not available:
                truncated.update(CHAR_TO_INDEX[ch] for ch in plate)
                if source_no >= next_output_progress:
                    print(f"output: {source_no}/{len(valid)}", flush=True)
                    while source_no >= next_output_progress:
                        next_output_progress += PROGRESS_INTERVAL
                continue
            available_labels = {CHAR_TO_INDEX[ch] for ch in plate}
            variant_needs = {
                variant: any(variant_split_class_counts[variant][split][label]
                             < quotas[variant][label][split] for label in available_labels)
                for variant in variant_caps
            }
            image = imread(source)
            width, height = config.rectify.size_7 if layout == 7 else config.rectify.size_8
            variants = []
            if variant_needs.get("gt", False) or source_mode == "slots":
                destination = np.asarray([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(quad, destination)
                rectified = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
                if source_mode == "segment":
                    gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(quad)
                    gt_candidate = PlateCandidate([tuple(map(float, point)) for point in quad],
                                                  (gt_x, gt_y, gt_w, gt_h), color, 1.0, "oracle")
                    gt_rect = RectifiedPlate(rectified, layout, color, matrix, gt_candidate)
                    variants.append(("gt", -1, -1, None, gt_rect))
            if (source_mode == "segment" and include_located_iou is not None
                    and variant_needs.get("located", False)):
                pre, proposals = _locate_with_memory_retry(image, config)
                gt_box = _parse_bbox(row["bbox"])
                for candidate_index, proposal in enumerate(proposals):
                    located_candidates_seen += 1
                    overlap = box_iou(gt_box, proposal.bbox)
                    if overlap < include_located_iou:
                        continue
                    located_candidates_iou += 1
                    for rectification_index, proposal_rect in enumerate(
                            _rectifications(image, proposal, config.rectify)):
                        variants.append(("located", candidate_index, rectification_index,
                                         overlap, proposal_rect))
            if source_mode == "segment":
                variant_chars = []
                for variant, candidate_index, rectification_index, overlap, rect in variants:
                    # Run only the known GT layout when the real pipeline would
                    # consider it for this rectified candidate. This avoids
                    # selecting a view from recognizer confidence or labels.
                    if variant == "located" and layout not in _layouts_to_try(rect):
                        continue
                    segment_attempted_images += 1
                    segment_attempted_by_split[split] += 1
                    variant_attempted[variant] += 1
                    segmented = segment_plate(rect, layout, config.segment)
                    if not segmented.ok:
                        segment_failed_images += 1
                        continue
                    segment_success_images += 1
                    segment_success_by_split[split] += 1
                    variant_success[variant] += 1
                    if not isinstance(segmented.chars, np.ndarray) or segmented.chars.ndim == 0:
                        segment_invalid_images += 1
                        continue
                    if len(segmented.chars) != layout:
                        segment_length_mismatch_images += 1
                        continue
                    if (segmented.chars.shape != (layout, 32, 32)
                            or segmented.chars.dtype != np.uint8):
                        segment_invalid_images += 1
                        continue
                    variant_chars.append((variant, candidate_index, rectification_index,
                                          overlap, segmented.chars))
                # Validate every view before emitting any character from this source image.
                if not variant_chars:
                    continue
            else:
                variant_chars = [("slots", -1, -1, None, None)]
                gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
                gray = unify_polarity(gray, color)
                centers = config.segment.layout7_centers if layout == 7 else config.segment.layout8_centers
                char_w = config.segment.layout7_char_w if layout == 7 else config.segment.layout8_char_w
                if len(centers) != layout or not 0 < char_w <= 1 or any(not 0 < c < 1 for c in centers):
                    raise ValueError(f"row {row_no}: invalid slot configuration for layout {layout}")
            base = source.stem
            path_token = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()[:10]
            safe_base = f"{base}_{path_token}"
            for variant, candidate_index, rectification_index, overlap, char_images in variant_chars:
                for position, char in enumerate(plate):
                    label = CHAR_TO_INDEX[char]
                    if variant_split_class_counts[variant][split][label] >= quotas[variant][label][split]:
                        truncated[label] += 1
                        continue
                    if source_mode == "segment":
                        normalized = char_images[position]
                        if (not isinstance(normalized, np.ndarray) or normalized.shape != (32, 32)
                                or normalized.dtype != np.uint8):
                            raise ValueError(f"row {row_no}: segment returned invalid character crop at {position}")
                    else:
                        center = centers[position] * width
                        half = char_w * width / 2
                        left, right = max(0, math.floor(center-half)), min(width, math.ceil(center+half))
                        if right <= left:
                            raise ValueError(f"row {row_no}: empty character slot {position}")
                        normalized = normalize_char(gray[:, left:right])
                    variant_token = (f"{variant}_{candidate_index}_{rectification_index}_{overlap:.4f}"
                                     if overlap is not None else variant)
                    variant_hash = hashlib.sha256(variant_token.encode("ascii")).hexdigest()[:8]
                    rel = Path(split) / str(label) / f"{safe_base}_{variant_hash}_{position}.png"
                    target = out / rel
                    if target.exists():
                        raise ValueError(f"output path collision: {target}")
                    imwrite(target, normalized)
                    counts[label] += 1
                    variant_split_class_counts[variant][split][label] += 1
                    split_counts[split] += 1
                    writer.writerow({"path": rel.as_posix(), "source_path": str(source), "split": split,
                                     "label": str(label), "char": char, "plate_no": plate, "position": str(position),
                                     "source_variant": variant, "candidate_index": str(candidate_index),
                                     "rectification_index": str(rectification_index),
                                     "candidate_iou": "" if overlap is None else f"{overlap:.6f}"})
            if source_no >= next_output_progress:
                print(f"output: {source_no}/{len(valid)}", flush=True)
                while source_no >= next_output_progress:
                    next_output_progress += PROGRESS_INTERVAL
    with (out / "chars_rejected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REJECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rejected)
    total_chars = sum(counts.values())
    current_metadata = _build_metadata(source_mode, config, config_path, include_located_iou)
    frozen_keys = ("config_sha256", "default_config_sha256", "effective_config_sha256",
                   "build_chars_sha256", "segment_code_sha256", "charnorm_code_sha256",
                   "segment_git_commit", "segment_git_blob", "build_git_commit", "build_git_branch",
                   "preprocess_code_sha256", "locate_code_sha256", "rectify_code_sha256",
                   "pipeline_code_sha256")
    changed = [key for key in frozen_keys if current_metadata.get(key) != build_metadata.get(key)]
    if changed:
        raise ValueError(f"build inputs changed while generating characters: {', '.join(changed)}")
    resumed = resume_state is not None
    stats = {"total": total_chars, "skipped_excluded_images": skipped_excluded,
             "skip_invalid": skip_invalid, "rejected_count": len(rejected), "rejected": rejected,
             "build": build_metadata,
             "counter_scope": "resumed_phase_only" if resumed else "full_build",
             "prior_counters_unknown": resumed,
             "resume_prior_counters": ({"segment_attempted": None, "segment_success": None,
                                         "segment_failures": None, "segment_length_mismatches": None,
                                         "segment_invalid": None, "truncated_per_class": None}
                                        if resumed else None),
             "resume_origin": (str(Path(resume_origin).resolve()) if resumed else None),
             "resume_after_source_no": resume_after_source_no if resumed else None,
             "valid_source_images": len(valid),
             "segment_attempted_images": segment_attempted_images,
             "segment_success_images": segment_success_images,
             "segment_attempted_by_split": dict(sorted(segment_attempted_by_split.items())),
             "segment_success_by_split": dict(sorted(segment_success_by_split.items())),
             "segment_ok_false_images": segment_failed_images,
             "segment_wrong_count_images": segment_length_mismatch_images,
             "segment_invalid_images": segment_invalid_images,
             "segment_failed_images": segment_failed_images + segment_invalid_images,
             "segment_length_mismatch_images": segment_length_mismatch_images,
             "segment_dropped_images": segment_failed_images + segment_length_mismatch_images + segment_invalid_images,
             "include_located_iou": include_located_iou,
             "variant_max_per_class": variant_caps,
             "located_candidates_seen": located_candidates_seen,
             "located_candidates_iou": located_candidates_iou,
             "segment_attempted_by_variant": dict(sorted(variant_attempted.items())),
             "segment_success_by_variant": dict(sorted(variant_success.items())),
             "per_split": dict(sorted(split_counts.items())),
             "per_class": {str(i): counts[i] for i in range(len(CHARSET))},
             "truncated_per_class": {str(i): truncated[i] for i in range(len(CHARSET))}}
    (out / "chars_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("class char count")
    for i, char in enumerate(CHARSET):
        print(f"{i:2d} {char} {counts[i]}")
    print(f"TOTAL chars={total_chars} excluded_images={skipped_excluded}")
    print("split_counts " + json.dumps(stats["per_split"], ensure_ascii=False))
    print("truncated " + json.dumps({CHARSET[i]: n for i, n in truncated.items() if n}, ensure_ascii=False))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build normalized character samples from CCPD ground truth")
    parser.add_argument("--index", required=True, help="ccpd_index.csv")
    parser.add_argument("--raw", required=True, help="raw CCPD root for relative index paths")
    parser.add_argument("--out", required=True, help="output character directory")
    parser.add_argument("--exclude-manifest", help="test manifest; matching images are excluded")
    parser.add_argument("--exclude-golden", help="golden manifest; original paths and byte-identical copies are protected")
    parser.add_argument("--source", choices=("slots", "segment"), default="slots",
                        help="character crop source (default: fixed layout slots)")
    parser.add_argument("--include-located-iou", type=float,
                        help="also add detected/rectified candidates whose bbox IoU with GT bbox meets this threshold")
    parser.add_argument("--resume-origin",
                        help="resume only from a verified locate-before-source-write checkpoint")
    parser.add_argument("--max-per-class", type=int, default=8000)
    parser.add_argument("--skip-invalid", action="store_true", help="record and skip invalid train/val images")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="bounded preflight workers (1-4; default 1)")
    args = parser.parse_args(argv)
    try:
        build_chars(args.index, args.raw, args.out, exclude_manifest=args.exclude_manifest,
                    max_per_class=args.max_per_class, skip_invalid=args.skip_invalid, workers=args.workers,
                    source=args.source, exclude_golden=args.exclude_golden,
                    include_located_iou=args.include_located_iou,
                    resume_origin=args.resume_origin)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    except (MemoryError, SystemError) as exc:
        if isinstance(exc, MemoryError) or isinstance(exc.__cause__, MemoryError):
            parser.error("image processing ran out of memory; partial output was retained")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
