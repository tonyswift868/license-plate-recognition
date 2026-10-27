"""Build a temporary, resumable full-resolution JPEG cache for box_reader_v1.

The source image is decoded one at a time. This cache contains exactly the
GT quad's axis-aligned bbox expanded by 50% on each side, clipped to frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from lpr.boxcrop import crop_box
from lpr_tools.train.box_data import bbox_from_quad, iter_rows, sha256_file

CACHE_FIELDS = ("path", "label", "subset", "brightness", "sha256", "quad", "cache_file",
                "x0", "y0", "orig_w", "orig_h")
DEFAULT_CACHE = Path(r"D:\keshe2\algorithm\data\boxcrop_cache\box_reader_v1")
DEFAULT_LIMIT = 2_800_000_000


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
    args = parser.parse_args()
    print(json.dumps(prepare_cache(args.pool, args.cache, quality=args.quality,
                                   limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
