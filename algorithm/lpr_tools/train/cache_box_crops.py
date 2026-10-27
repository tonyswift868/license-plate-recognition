"""Build a temporary, resumable full-resolution JPEG cache for box_reader_v1.

The source image is decoded one at a time. This cache contains exactly the
GT quad's axis-aligned bbox expanded by 50% on each side, clipped to frame.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile

import cv2
import numpy as np

from lpr.boxcrop import crop_box
from lpr_tools.train.box_data import bbox_from_quad, iter_rows, sha256_file

CACHE_FIELDS = ("path", "label", "subset", "brightness", "sha256", "quad", "cache_file",
                "x0", "y0", "orig_w", "orig_h")
PACK_FIELDS = ("path", "label", "subset", "brightness", "sha256", "quad", "cache_pack",
               "pack_offset", "pack_length", "x0", "y0", "orig_w", "orig_h")
DEFAULT_CACHE = Path(r"D:\keshe2\algorithm\data\boxcrop_cache\box_reader_v1")
DEFAULT_LIMIT = 2_800_000_000
PACK_TAR_LIMIT = 2_750_000_000
PACK_TOTAL_LIMIT = 3_000_000_000


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable source image: {path}")
    return image


def _encode_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if hasattr(cv2, "IMWRITE_JPEG_OPTIMIZE"):
        params.extend((cv2.IMWRITE_JPEG_OPTIMIZE, 1))
    success, encoded = cv2.imencode(".jpg", image, params)
    if not success:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return encoded


def _encode_pool_row(row: dict[str, str], quality: int) -> tuple[dict[str, str], bytes, tuple[int, int, int, int]]:
    """Decode one original, make GT bbox+50% crop, and encode in memory."""
    bbox = bbox_from_quad(row["quad"])
    x, y, width, height = bbox
    source = _read_image(row["path"])
    ih, iw = source.shape[:2]
    x0 = max(0, math.floor(x - .5 * width))
    y0 = max(0, math.floor(y - .5 * height))
    crop = crop_box(source, bbox, .5, .5)
    if not crop.size:
        raise ValueError(f"empty GT bbox crop: {row['path']}")
    encoded = _encode_jpeg(crop, quality)
    result = (dict(row), encoded.tobytes(), (x0, y0, iw, ih))
    del source, crop, encoded
    return result


def _bounded_encode(rows, quality: int, *, workers: int = 3, pending_limit: int = 6):
    """Yield completed encodes in the input order with at most six buffers live."""
    pending = deque()
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < pending_limit and not exhausted:
                try:
                    row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.submit(_encode_pool_row, row, quality))
            if pending:
                yield pending.popleft().result()


def probe_packed(pool: str | Path, cache: str | Path = DEFAULT_CACHE, *,
                 quality: int = 65, seed: int = 42, per_group: int = 120,
                 workers: int = 3) -> dict[str, object]:
    """Estimate packed tar, index and total bytes from uniform group samples."""
    pool, cache = Path(pool).resolve(), Path(cache).resolve()
    counts: Counter[str] = Counter()
    reservoirs: dict[str, list[dict[str, str]]] = {}
    rng = random.Random(seed)
    for row in csv_rows(pool):
        group = row["subset"]
        counts[group] += 1
        bucket = reservoirs.setdefault(group, [])
        if len(bucket) < per_group:
            bucket.append(row)
        else:
            index = rng.randrange(counts[group])
            if index < per_group:
                bucket[index] = row
    if not counts or any(len(reservoirs[group]) != min(per_group, count)
                         for group, count in counts.items()):
        raise ValueError("could not build per-group packed-cache sample")

    sampled_counts: Counter[str] = Counter()
    jpeg_bytes: Counter[str] = Counter()
    tar_entry_bytes: Counter[str] = Counter()
    projected_index_rows: Counter[str] = Counter()
    encoded_samples = [row for group in sorted(reservoirs) for row in reservoirs[group]]
    encoded_started = time.perf_counter()
    pack_path = str(cache / "crops.tar")
    for row, encoded, shape in _bounded_encode(encoded_samples, quality, workers=workers):
        group = row["subset"]
        size = len(encoded)
        sampled_counts[group] += 1
        jpeg_bytes[group] += size
        tar_entry_bytes[group] += 512 + ((size + 511) // 512) * 512
        x0, y0, iw, ih = shape
        record = {key: row[key] for key in PACK_FIELDS[:6]}
        record.update(cache_pack=pack_path, pack_offset=2_750_000_000, pack_length=size,
                      x0=x0, y0=y0, orig_w=iw, orig_h=ih)
        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=PACK_FIELDS, lineterminator="\n").writerow(record)
        projected_index_rows[group] += len(buffer.getvalue().encode("utf-8"))
    encode_seconds = time.perf_counter() - encoded_started

    jpeg_projection = {group: round(jpeg_bytes[group] / sampled_counts[group] * count)
                       for group, count in counts.items()}
    tar_projection = {group: round(tar_entry_bytes[group] / sampled_counts[group] * count)
                      for group, count in counts.items()}
    index_projection = {group: round(projected_index_rows[group] / sampled_counts[group] * count)
                        for group, count in counts.items()}
    tar_bytes = sum(tar_projection.values()) + 1024
    index_bytes = len((",".join(PACK_FIELDS) + "\n").encode("utf-8")) + sum(index_projection.values())
    probe_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()) if cache.is_dir() else 0
    total_bytes = tar_bytes + index_bytes + probe_bytes
    return {
        "schema": "box-reader-packed-cache-probe-v1", "pool_sha256": sha256_file(pool),
        "seed": seed, "per_group": per_group, "workers": workers, "pending_limit": 6,
        "pool_rows_by_group": dict(counts), "sample_rows_by_group": dict(sampled_counts),
        "quality": quality, "jpeg_optimize": True,
        "sample_jpeg_bytes_by_group": dict(jpeg_bytes),
        "projected_jpeg_bytes_by_group": jpeg_projection,
        "projected_jpeg_bytes": sum(jpeg_projection.values()),
        "projected_tar_bytes_by_group_including_ustar_headers": tar_projection,
        "projected_tar_bytes": tar_bytes,
        "projected_index_row_bytes_by_group": index_projection,
        "projected_index_bytes": index_bytes,
        "existing_probe_jpeg_bytes": probe_bytes,
        "projected_total_cache_bytes": total_bytes,
        "tar_limit_bytes": PACK_TAR_LIMIT, "total_limit_bytes": PACK_TOTAL_LIMIT,
        "within_limits": tar_bytes <= PACK_TAR_LIMIT and total_bytes <= PACK_TOTAL_LIMIT,
        "sample_encode_seconds": encode_seconds,
        "estimated_full_cache_seconds": encode_seconds / sum(sampled_counts.values()) * sum(counts.values()),
        "note": "uniform seed-42 reservoir sample per group; image bytes existed only in bounded worker memory; USTAR overhead/index/probe included in projected disk",
    }


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def prepare_packed_cache(pool: str | Path, cache: str | Path = DEFAULT_CACHE, *,
                         quality: int = 65, seed: int = 42, workers: int = 3,
                         pending_limit: int = 6) -> dict[str, object]:
    """Write one USTAR cache and seek index, preserving all pool records."""
    pool, cache = Path(pool).resolve(), Path(cache).resolve()
    if not pool.is_file() or not 1 <= quality <= 100 or workers != 3 or pending_limit != 6:
        raise ValueError("packed cache requires a valid pool, workers=3, pending_limit=6")
    cache.mkdir(parents=True, exist_ok=True)
    tar_path, index_path, audit_path = cache / "crops.tar", cache / "index.csv", cache / "audit.json"
    if audit_path.is_file() and tar_path.is_file() and index_path.is_file():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if (old.get("pool_sha256") == sha256_file(pool) and old.get("quality") == quality
                and old.get("cache_index_sha256") == sha256_file(index_path)):
            return old
        raise FileExistsError("packed cache already exists for a different pool or quality")
    if any(path.exists() for path in (tar_path, index_path, audit_path,
                                      cache / "crops.tar.partial", cache / "index.partial.csv")):
        raise FileExistsError("partial or unaudited packed cache exists; preserve it for inspection")

    # Metadata-only pool stays in memory so its order is a reproducible global shuffle.
    rows = list(csv_rows(pool))
    random.Random(seed).shuffle(rows)
    pool_sha = sha256_file(pool)
    tar_partial, index_partial = cache / "crops.tar.partial", cache / "index.partial.csv"
    started = time.perf_counter()
    jpeg_bytes = 0
    with tarfile.open(tar_partial, mode="w", format=tarfile.USTAR_FORMAT) as archive, \
            index_partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACK_FIELDS, lineterminator="\n")
        writer.writeheader()
        for count, (row, encoded, shape) in enumerate(
                _bounded_encode(rows, quality, workers=workers, pending_limit=pending_limit), start=1):
            size = len(encoded)
            next_tar_size = archive.offset + 512 + ((size + 511) // 512) * 512
            if next_tar_size + 1024 > PACK_TAR_LIMIT:
                raise RuntimeError(f"packed tar would exceed hard limit {PACK_TAR_LIMIT} bytes")
            member = tarfile.TarInfo(name=f"{row['sha256']}.jpg")
            member.size = size
            member.mtime = 0
            member.mode = 0o644
            pack_offset = archive.offset + 512
            archive.addfile(member, io.BytesIO(encoded))
            jpeg_bytes += size
            x0, y0, iw, ih = shape
            record = {key: row[key] for key in PACK_FIELDS[:6]}
            record.update(cache_pack=str(tar_path), pack_offset=pack_offset, pack_length=size,
                          x0=x0, y0=y0, orig_w=iw, orig_h=ih)
            writer.writerow(record)
            if count % 5000 == 0:
                stream.flush()
                print(json.dumps({"packed_count": count, "jpeg_bytes": jpeg_bytes,
                                  "tar_bytes": archive.offset, "seconds": time.perf_counter() - started}), flush=True)
    tar_partial.replace(tar_path)
    index_partial.replace(index_path)
    tar_bytes, index_bytes = tar_path.stat().st_size, index_path.stat().st_size
    owned = {tar_path.resolve(), index_path.resolve(), audit_path.resolve(),
             tar_partial.resolve(), index_partial.resolve()}
    probe_bytes = sum(path.stat().st_size for path in cache.rglob("*")
                      if path.is_file() and path.resolve() not in owned)
    total_bytes = tar_bytes + index_bytes + probe_bytes
    if tar_bytes > PACK_TAR_LIMIT or total_bytes > PACK_TOTAL_LIMIT:
        raise RuntimeError(f"packed cache exceeded disk budget: tar={tar_bytes}, total={total_bytes}")
    report: dict[str, object] = {
        "schema": "box-reader-packed-cache-v1", "pool_sha256": pool_sha,
        "cache_index_sha256": sha256_file(index_path), "cache_tar_sha256": sha256_file(tar_path),
        "cache_path": str(cache), "cache_tar_path": str(tar_path),
        "pool_count": len(rows), "quality": quality, "workers": workers,
        "pending_limit": pending_limit, "jpeg_bytes": jpeg_bytes, "tar_bytes": tar_bytes,
        "index_bytes": index_bytes, "partial_probe_jpeg_bytes": probe_bytes,
        "total_cache_bytes_excluding_audit": total_bytes,
        "seconds": time.perf_counter() - started,
        "scale": "original resolution", "gt_bbox_pad": [0.5, 0.5],
        "format": "USTAR; one JPEG member per source SHA-256",
        "offset_semantics": "pack_offset points to JPEG payload start (member header offset + 512)",
        "order": "global random.Random(seed=42) shuffle of all pool CSV metadata rows",
        "seed": seed,
    }
    report["audit_bytes"] = 0
    report["total_cache_bytes"] = total_bytes
    if total_bytes > PACK_TOTAL_LIMIT:
        raise RuntimeError("packed cache plus audit exceeded the 3 GB limit")
    for _ in range(3):
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        actual_audit_bytes = audit_path.stat().st_size
        if report["audit_bytes"] == actual_audit_bytes:
            break
        report["audit_bytes"] = actual_audit_bytes
        report["total_cache_bytes"] = total_bytes + actual_audit_bytes
    if report["total_cache_bytes"] > PACK_TOTAL_LIMIT:
        raise RuntimeError("packed cache plus audit exceeded the 3 GB limit")
    return report


def prepare_cache(pool: str | Path, cache: str | Path = DEFAULT_CACHE, *,
                  quality: int = 70, limit: int = DEFAULT_LIMIT) -> dict[str, object]:
    """Create or validate a full pool cache while keeping only one image live."""
    pool, cache = Path(pool).resolve(), Path(cache).resolve()
    if not pool.is_file() or not 1 <= quality <= 100 or limit <= 0:
        raise ValueError("pool, JPEG quality, or cache byte limit is invalid")
    pool_sha = sha256_file(pool)
    cache.mkdir(parents=True, exist_ok=True)
    audit_path, index_path = cache / "audit.json", cache / "index.csv"
    if audit_path.is_file() and index_path.is_file():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if old.get("pool_sha256") == pool_sha and old.get("quality") == quality:
            if sha256_file(index_path) != old.get("cache_index_sha256"):
                raise ValueError("existing cache index does not match its audit")
            return old
        raise FileExistsError("cache exists for different pool/quality; do not mix cache runs")

    cv2.setNumThreads(1)
    used = sum(path.stat().st_size for path in cache.rglob("*.jpg"))
    if used > limit:
        raise RuntimeError(f"existing JPEGs exceed cache limit: {used} > {limit}")
    partial = cache / "index.partial.csv"
    if partial.exists():
        partial.unlink()
    count = 0
    started = time.perf_counter()
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CACHE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in iter_rows(pool):
            bbox = bbox_from_quad(row["quad"])
            x, y, width, height = bbox
            source = _read_image(row["path"])
            ih, iw = source.shape[:2]
            x0 = max(0, math.floor(x - .5 * width))
            y0 = max(0, math.floor(y - .5 * height))
            crop = crop_box(source, bbox, .5, .5)
            if not crop.size:
                raise ValueError(f"empty GT bbox crop: {row['path']}")
            target = cache / row["sha256"][:2] / (row["sha256"] + ".jpg")
            if not target.exists():
                encoded = _encode_jpeg(crop, quality)
                if used + encoded.nbytes > limit:
                    raise RuntimeError(f"JPEG cache would exceed {limit} bytes; encoded={used + encoded.nbytes}")
                target.parent.mkdir(parents=True, exist_ok=True)
                encoded.tofile(target)
                used += encoded.nbytes
            record = {key: row[key] for key in CACHE_FIELDS[:6]}
            record.update(cache_file=str(target.resolve()), x0=x0, y0=y0,
                          orig_w=iw, orig_h=ih)
            writer.writerow(record)
            count += 1
            del source, crop
            if count % 1000 == 0:
                stream.flush()
                print(json.dumps({"cache_count": count, "jpeg_bytes": used,
                                  "seconds": time.perf_counter() - started}), flush=True)
    partial.replace(index_path)
    total_bytes = used + index_path.stat().st_size
    if total_bytes > 3_000_000_000:
        raise RuntimeError(f"cache plus index exceeds 3 GB: {total_bytes}")
    report: dict[str, object] = {
        "schema": "box-reader-jpeg-cache-v1", "pool_sha256": pool_sha,
        "cache_index_sha256": sha256_file(index_path), "cache_path": str(cache),
        "samples": count, "jpeg_bytes": used, "index_bytes": index_path.stat().st_size,
        "total_bytes": total_bytes, "seconds": time.perf_counter() - started,
        "quality": quality, "jpeg_optimize": hasattr(cv2, "IMWRITE_JPEG_OPTIMIZE"),
        "scale": "original resolution", "gt_bbox_pad": [0.5, 0.5],
        "pool_rows_sha256": pool_sha,
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--packed", action="store_true", help="write all JPEGs into one USTAR file")
    parser.add_argument("--probe", action="store_true", help="estimate a seed-controlled sample in memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", type=Path, help="write the JSON result to this workspace path")
    args = parser.parse_args()
    if args.packed and args.probe:
        result = probe_packed(args.pool, args.cache, quality=args.quality, seed=args.seed,
                              workers=args.workers)
    elif args.packed:
        result = prepare_packed_cache(args.pool, args.cache, quality=args.quality,
                                      seed=args.seed, workers=args.workers)
    elif args.probe:
        parser.error("--probe requires --packed")
    else:
        result = prepare_cache(args.pool, args.cache, quality=args.quality, limit=args.limit)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
