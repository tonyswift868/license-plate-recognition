"""Build normalized character crops from CCPD ground-truth plate corners."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lpr.charnorm import normalize_char, unify_polarity
from lpr.charset import CHAR_TO_INDEX, CHARSET
from lpr.config import load_config
from lpr.io_utils import decode_image, imread, imwrite
from lpr_tools.data.ccpd import read_index

INDEX_FIELDS = ("path", "source_path", "split", "label", "char", "plate_no", "position")
REJECTION_FIELDS = ("row", "split", "path", "reason")
PREFLIGHT_BATCH_SIZE = 256
PROGRESS_INTERVAL = 5000


def _canonical(path: Path) -> str:
    return str(path.resolve()).casefold()


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
                skip_invalid: bool = False, workers: int = 1) -> dict[str, int]:
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    if workers not in (1, 2, 3, 4):
        raise ValueError("workers must be between 1 and 4")
    cv2.setNumThreads(1)
    raw_root, out = Path(raw).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    rows = read_index(index)
    excluded, excluded_hashes = _excluded_paths(exclude_manifest, raw_root)
    config = load_config()
    seen_paths: dict[str, tuple[str, int]] = {}
    seen_content: dict[str, str] = {}
    counts: Counter[int] = Counter()
    truncated: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    out_rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    skipped_excluded = 0

    # Validate train/val sources fully. Test rows contribute path/content identities
    # when their files exist, but missing test files do not block character building.
    candidates = []
    preflight_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    for row_no, row in enumerate(rows, 2):
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
        candidates.append((row_no, row, source))

    # Only submit bounded batches; executor.map preserves the source index order.
    valid = []
    next_preflight_progress = PROGRESS_INTERVAL
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(candidates), PREFLIGHT_BATCH_SIZE):
            batch = candidates[start:start + PREFLIGHT_BATCH_SIZE]
            results = list(pool.map(lambda item: _preflight_one(item, config), batch))
            for item, (digest, quad, error) in zip(batch, results):
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
                    valid.append((row_no, row, source, digest, quad))
            processed = min(start + len(batch), len(candidates))
            while processed >= next_preflight_progress:
                print(f"preflight: {next_preflight_progress}/{len(candidates)}", flush=True)
                next_preflight_progress += PROGRESS_INTERVAL

    quotas = {label: _split_quotas({split: preflight_counts[split][label] for split in ("train", "val")}, max_per_class)
              for label in range(len(CHARSET))}
    split_class_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}

    next_output_progress = PROGRESS_INTERVAL
    for source_no, (row_no, row, source, digest, quad) in enumerate(valid, 1):
        plate = row.get("plate_no", "")
        layout = int(row["layout"])
        color = row.get("plate_color", "")
        split = row["split"]
        available = [CHAR_TO_INDEX[ch] for ch in plate
                     if split_class_counts[split][CHAR_TO_INDEX[ch]] < quotas[CHAR_TO_INDEX[ch]][split]]
        if not available:
            truncated.update(CHAR_TO_INDEX[ch] for ch in plate)
            if source_no >= next_output_progress:
                print(f"output: {source_no}/{len(valid)}", flush=True)
                while source_no >= next_output_progress:
                    next_output_progress += PROGRESS_INTERVAL
            continue
        image = imread(source)
        width, height = config.rectify.size_7 if layout == 7 else config.rectify.size_8
        destination = np.asarray([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, destination)
        rectified = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        gray = unify_polarity(gray, color)
        centers = config.segment.layout7_centers if layout == 7 else config.segment.layout8_centers
        char_w = config.segment.layout7_char_w if layout == 7 else config.segment.layout8_char_w
        if len(centers) != layout or not 0 < char_w <= 1 or any(not 0 < c < 1 for c in centers):
            raise ValueError(f"row {row_no}: invalid slot configuration for layout {layout}")
        base = source.stem
        path_token = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()[:10]
        safe_base = f"{base}_{path_token}"
        for position, char in enumerate(plate):
            label = CHAR_TO_INDEX[char]
            if split_class_counts[split][label] >= quotas[label][split]:
                truncated[label] += 1
                continue
            center = centers[position] * width
            half = char_w * width / 2
            left, right = max(0, math.floor(center-half)), min(width, math.ceil(center+half))
            if right <= left:
                raise ValueError(f"row {row_no}: empty character slot {position}")
            normalized = normalize_char(gray[:, left:right])
            rel = Path(split) / str(label) / f"{safe_base}_{position}.png"
            target = out / rel
            if target.exists():
                raise ValueError(f"output path collision: {target}")
            imwrite(target, normalized)
            counts[label] += 1
            split_class_counts[split][label] += 1
            split_counts[split] += 1
            out_rows.append({"path": rel.as_posix(), "source_path": str(source), "split": split,
                             "label": str(label), "char": char, "plate_no": plate, "position": str(position)})
        if source_no >= next_output_progress:
            print(f"output: {source_no}/{len(valid)}", flush=True)
            while source_no >= next_output_progress:
                next_output_progress += PROGRESS_INTERVAL

    out.mkdir(parents=True, exist_ok=True)
    with (out / "chars_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    with (out / "chars_rejected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REJECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rejected)
    stats = {"total": len(out_rows), "skipped_excluded_images": skipped_excluded,
             "skip_invalid": skip_invalid, "rejected_count": len(rejected), "rejected": rejected,
             "per_split": dict(sorted(split_counts.items())),
             "per_class": {str(i): counts[i] for i in range(len(CHARSET))},
             "truncated_per_class": {str(i): truncated[i] for i in range(len(CHARSET))}}
    (out / "chars_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("class char count")
    for i, char in enumerate(CHARSET):
        print(f"{i:2d} {char} {counts[i]}")
    print(f"TOTAL chars={len(out_rows)} excluded_images={skipped_excluded}")
    print("split_counts " + json.dumps(stats["per_split"], ensure_ascii=False))
    print("truncated " + json.dumps({CHARSET[i]: n for i, n in truncated.items() if n}, ensure_ascii=False))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build normalized character samples from CCPD ground truth")
    parser.add_argument("--index", required=True, help="ccpd_index.csv")
    parser.add_argument("--raw", required=True, help="raw CCPD root for relative index paths")
    parser.add_argument("--out", required=True, help="output character directory")
    parser.add_argument("--exclude-manifest", help="test manifest; matching images are excluded")
    parser.add_argument("--max-per-class", type=int, default=8000)
    parser.add_argument("--skip-invalid", action="store_true", help="record and skip invalid train/val images")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="bounded preflight workers (1-4; default 1)")
    args = parser.parse_args(argv)
    try:
        build_chars(args.index, args.raw, args.out, exclude_manifest=args.exclude_manifest,
                    max_per_class=args.max_per_class, skip_invalid=args.skip_invalid, workers=args.workers)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
