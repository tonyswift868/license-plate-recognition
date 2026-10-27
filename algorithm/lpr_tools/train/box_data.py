"""Streaming metadata helpers for detector-box plate-reader training.

This module deliberately has no image, OpenCV, or torch imports.  It keeps
the data audit and measured geometric augmentation usable under tight RAM.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(r"D:\keshe2\algorithm\data")
V2 = DATA / "plates" / "v2"
DEFAULT_INDEX = DATA / "index" / "ccpd_index.csv"
DEFAULT_OUT = ROOT / "algorithm" / "out" / "data" / "batch10"
DEFAULT_GEOMETRY = ROOT / "algorithm" / "out" / "receipts" / "batch9" / "geometry" / "train_per_image.jsonl"
V2_LEDGER_SHA256 = "2a992d2b90ac2ab60c8eb6bdf4b5b89331cff784655bd4d8350c2bd73d329986"
HARD_SUBSETS = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
POOL_FIELDS = ("path", "label", "subset", "brightness", "sha256", "quad", "source")
TUNE_FIELDS = ("path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def _hash_ledger(path: Path) -> tuple[dict[str, str], str]:
    digest = sha256_file(path)
    if digest != V2_LEDGER_SHA256:
        raise ValueError(f"V2 source ledger changed: {digest}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for original, value in raw.items():
        sha = str(value.get("sha256", "")).lower()
        if len(sha) != 64:
            raise ValueError(f"malformed source ledger entry: {original}")
        result[_path_key(original)] = sha
    return result, digest


def _protected_manifests() -> tuple[set[str], set[str], list[dict[str, object]]]:
    """Load exact five protection lists and the byte-identical worker03 golden list."""
    manifests = (
        ("testset", DATA / "index" / "testset_manifest_v1.csv"),
        ("golden", ROOT / "algorithm" / "tests" / "data" / "golden" / "golden_manifest.csv"),
        ("devhard", DATA / "index" / "devhard_v1.csv"),
        ("val1000", DATA / "index" / "val1000_v1.csv"),
        ("undecodable", DATA / "index" / "undecodable.csv"),
        ("worker03_golden", Path(r"D:\keshe2\workers\03\algorithm\tests\data\golden\golden_manifest.csv")),
    )
    paths: set[str] = set()
    metadata: list[dict[str, object]] = []
    for name, manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"protected manifest missing: {manifest}")
        count = 0
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise ValueError(f"protected manifest has no path column: {manifest}")
            for row in reader:
                raw_path = row.get("path", "").strip()
                if raw_path:
                    paths.add(_path_key(raw_path))
                    count += 1
        metadata.append({"name": name, "path": str(manifest.resolve()), "rows": count,
                         "sha256": sha256_file(manifest)})
    hashes = set()
    for path_key in paths:
        path = Path(path_key)
        if not path.is_file():
            raise FileNotFoundError(f"protected image missing: {path}")
        hashes.add(sha256_file(path))
    return paths, hashes, metadata


def _quad_string(quad: str | Sequence[float]) -> str:
    return " ".join(f"{coordinate:.6f}" for point in parse_quad(quad) for coordinate in point)


def _brightness(path: str | Path) -> int | None:
    try:
        return int(Path(str(path)).stem.split("-")[-2])
    except (ValueError, IndexError):
        return None


def _label_geometry(path: str):
    # This parser only reads the CCPD filename; it does not open/decode image bytes.
    from lpr_tools.data.ccpd import parse_ccpd_name
    record = parse_ccpd_name(path)
    if record.skip_reason:
        raise ValueError(f"unsupported CCPD label {record.skip_reason}: {path}")
    return record.plate_no, _quad_string(record.quad)


def iter_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows without materializing the index in memory."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def parse_quad(value: str | Sequence[float]) -> tuple[tuple[float, float], ...]:
    """Parse eight whitespace/comma separated coordinates or four (x,y) pairs."""
    if isinstance(value, str):
        values = [float(part) for part in value.replace(",", " ").split()]
        if len(values) != 8:
            raise ValueError("quad must contain exactly eight coordinates")
        points = tuple((values[i], values[i + 1]) for i in range(0, 8, 2))
    else:
        if len(value) == 8 and not isinstance(value[0], (tuple, list)):
            points = tuple((float(value[i]), float(value[i + 1])) for i in range(0, 8, 2))
        elif len(value) == 4:
            points = tuple((float(point[0]), float(point[1])) for point in value)  # type: ignore[index]
        else:
            raise ValueError("quad must contain four points or eight coordinates")
    if not all(math.isfinite(c) for point in points for c in point):
        raise ValueError("quad coordinates must be finite")
    return points


def bbox_from_quad(quad: str | Sequence[float]) -> tuple[float, float, float, float]:
    """Return the axis-aligned enclosing (x, y, width, height) box."""
    points = parse_quad(quad)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("quad must enclose a positive-area box")
    return x0, y0, x1 - x0, y1 - y0


def box_delta(gt_quad: str | Sequence[float], candidate_quad: str | Sequence[float]
              ) -> tuple[float, float, float, float]:
    """Encode candidate bbox error as (dx/gt_w, dy/gt_h, log(w/gt_w), log(h/gt_h))."""
    gx, gy, gw, gh = bbox_from_quad(gt_quad)
    cx, cy, cw, ch = bbox_from_quad(candidate_quad)
    return ((cx - gx) / gw, (cy - gy) / gh,
            math.log(cw / gw), math.log(ch / gh))


def load_measured_jitter(jsonl_path: str | Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Read matched training rows from batch9 geometry output as bbox deltas.

    The measured proposal itself is used, not canonical corner deltas.  Each
    source contributes at most one correlated four-value sample, as recorded
    by batch9's one-match-per-source sampling contract.
    """
    samples: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subset = row.get("subset")
            if row.get("partition") != "train" or not row.get("matched"):
                continue
            if not isinstance(subset, str) or not subset:
                raise ValueError(f"missing subset at JSONL line {line_number}")
            gt = row.get("gt_quad_px")
            candidate = row.get("proposal_quad_px") or row.get("actual_quad_px")
            if gt is None or candidate is None:
                raise ValueError(f"matched row lacks quads at JSONL line {line_number}")
            delta = box_delta(gt, candidate)
            if not all(math.isfinite(value) for value in delta):
                raise ValueError(f"non-finite bbox delta at JSONL line {line_number}")
            samples[subset].append(delta)
    if not samples:
        raise ValueError("no matched train samples in crop-error JSONL")
    return dict(samples)


def read_jitter_samples(jsonl_path: str | Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Public loader name used by both training and offline evaluation."""
    return load_measured_jitter(jsonl_path)


def sample_jitter(rng: random.Random, subset: str,
                  samples: Mapping[str, Sequence[Sequence[float]]]
                  ) -> tuple[float, float, float, float]:
    """Draw one measured, correlated bbox error for subset (or all groups)."""
    choices = samples.get(subset)
    if not choices:
        choices = [sample for values in samples.values() for sample in values]
    if not choices:
        raise ValueError("jitter sampler has no measured samples")
    chosen = choices[rng.randrange(len(choices))]
    if len(chosen) != 4 or not all(math.isfinite(float(value)) for value in chosen):
        raise ValueError("jitter sample must contain four finite values")
    return tuple(float(value) for value in chosen)  # type: ignore[return-value]


def sample_box_delta(samples: Mapping[str, Sequence[Sequence[float]]], subset: str,
                     rng: random.Random) -> tuple[float, float, float, float]:
    """Compatibility spelling; prefer ``sample_jitter(rng, subset, samples)``."""
    return sample_jitter(rng, subset, samples)


def jitter_bbox(bbox: Sequence[float], delta: Sequence[float]
                ) -> tuple[float, float, float, float]:
    """Apply measured translation and independent width/height scale errors."""
    if len(bbox) != 4 or len(delta) != 4:
        raise ValueError("bbox and delta must each have four values")
    x, y, width, height = (float(value) for value in bbox)
    dx, dy, log_w, log_h = (float(value) for value in delta)
    if not all(math.isfinite(value) for value in (x, y, width, height, dx, dy, log_w, log_h)):
        raise ValueError("bbox and delta values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("bbox dimensions must be positive")
    result = (x + dx * width, y + dy * height,
              width * math.exp(log_w), height * math.exp(log_h))
    if not all(math.isfinite(value) for value in result) or result[2] <= 0 or result[3] <= 0:
        raise ValueError("jittered bbox is invalid")
    return result


def prepare_pool(index_csv: str | Path = DEFAULT_INDEX, out_dir: str | Path = DEFAULT_OUT,
                 *, v2_dir: str | Path = V2, geometry_jsonl: str | Path = DEFAULT_GEOMETRY
                 ) -> dict[str, object]:
    """Build safe one-row-per-origin metadata files, streaming source CSVs.

    Raw images stay in place. Hashes from the integrity-pinned V2 ledger are
    reused; only candidates absent from that ledger are hashed directly.
    Protected manifests are always hashed from their listed source files.
    """
    index_csv, out_dir, v2_dir = Path(index_csv), Path(out_dir), Path(v2_dir)
    ledger_path = v2_dir / "source_hashes.json"
    labels_path = v2_dir / "labels.csv"
    holdout_path = ROOT / "algorithm" / "out" / "data" / "batch8" / "holdout_origins.csv"
    audit_path = ROOT / "algorithm" / "out" / "data" / "batch8" / "audit.json"
    for path in (index_csv, ledger_path, labels_path, holdout_path, audit_path, geometry_jsonl):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations = {name: out_dir / name for name in
                    ("pool.csv", "tune.csv", "audit.json", "sampler.json")}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    ledger, ledger_sha = _hash_ledger(ledger_path)
    protected_paths, protected_hashes, protected_meta = _protected_manifests()
    holdout: dict[str, dict[str, str]] = {}
    with holdout_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"orig_path", "subset", "sha256", "brightness", "calibration_fold"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"batch8 holdout fields missing: {sorted(required)}")
        for row in reader:
            key = _path_key(row["orig_path"])
            if key in holdout:
                raise ValueError(f"duplicate batch8 holdout path: {row['orig_path']}")
            holdout[key] = row
    if len(holdout) != 2100:
        raise ValueError(f"expected 2100 batch8 holdout origins, found {len(holdout)}")
    holdout_hashes = {row["sha256"].lower() for row in holdout.values()}
    if len(holdout_hashes) != len(holdout):
        raise ValueError("batch8 holdout contains same-byte origins")

    counters: dict[str, object] = {
        "candidate_rows": defaultdict(int), "candidate_unique_paths": defaultdict(set),
        "excluded_protected_path": defaultdict(int), "excluded_protected_sha": defaultdict(int),
        "excluded_holdout_path": defaultdict(int), "excluded_holdout_sha": defaultdict(int),
        "excluded_duplicate_sha": defaultdict(int), "retained": defaultdict(int),
        "ledger_hits": defaultdict(int), "direct_hashes": defaultdict(int),
    }
    retained_hashes: set[str] = set()
    retained_by_subset: dict[str, int] = defaultdict(int)
    tune_by_subset_fold: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tune_hashes: set[str] = set()
    tune_hashes_by_fold: dict[str, set[str]] = {"fit": set(), "report": set()}
    cache_estimate_bytes = 0.0
    candidate_original_bytes = 0
    missing_ledger: list[str] = []

    def candidate_sha(path: str, group: str) -> str:
        key = _path_key(path)
        value = ledger.get(key)
        if value is not None:
            counters["ledger_hits"][group] += 1  # type: ignore[index]
            return value
        if not Path(path).is_file():
            raise FileNotFoundError(f"training source missing: {path}")
        counters["direct_hashes"][group] += 1  # type: ignore[index]
        missing_ledger.append(path)
        return sha256_file(path)

    def maybe_write(row: dict[str, str], group: str, writer) -> None:
        nonlocal candidate_original_bytes, cache_estimate_bytes
        counters["candidate_rows"][group] += 1  # type: ignore[index]
        key = _path_key(row["path"])
        counters["candidate_unique_paths"][group].add(key)  # type: ignore[index]
        digest = candidate_sha(row["path"], group)
        if key in protected_paths:
            counters["excluded_protected_path"][group] += 1  # type: ignore[index]
            return
        if digest in protected_hashes:
            counters["excluded_protected_sha"][group] += 1  # type: ignore[index]
            return
        if key in holdout:
            counters["excluded_holdout_path"][group] += 1  # type: ignore[index]
            return
        if digest in holdout_hashes:
            counters["excluded_holdout_sha"][group] += 1  # type: ignore[index]
            return
        if digest in retained_hashes:
            counters["excluded_duplicate_sha"][group] += 1  # type: ignore[index]
            return
        retained_hashes.add(digest)
        counters["retained"][group] += 1  # type: ignore[index]
        retained_by_subset[row["subset"]] += 1
        writer.writerow({**row, "sha256": digest})

        source = Path(row["path"])
        size = source.stat().st_size
        candidate_original_bytes += size
        # CCPD source frames are conventionally 1160x720. Estimate the JPEG
        # crop share from the filename quad and a 50% pad on each side; actual
        # cache size is recorded after encoding, never inferred from this value.
        x, y, width, height = bbox_from_quad(row["quad"])
        x0, y0 = max(0.0, x - width * 0.5), max(0.0, y - height * 0.5)
        x1, y1 = min(1160.0, x + width * 1.5), min(720.0, y + height * 1.5)
        share = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (1160.0 * 720.0)
        cache_estimate_bytes += size * min(1.0, share)

    pool_file = destinations["pool.csv"]
    with pool_file.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=POOL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in iter_rows(index_csv):
            if (row.get("split") != "train" or row.get("has_plate") != "1"
                    or row.get("skip_reason", "").strip()
                    or row.get("subset") not in {"ccpd_base", "ccpd_green"}):
                continue
            path = row["path"]
            subset = "green" if row["subset"] == "ccpd_green" else "base"
            maybe_write({"path": path, "label": row["plate_no"], "subset": subset,
                         "brightness": row.get("brightness", ""),
                         "quad": _quad_string(row["quad"]), "source": "ccpd_index_train"},
                        subset, writer)

    seen_hard_paths: set[str] = set()
    with pool_file.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=POOL_FIELDS, lineterminator="\n")
        with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                source = row.get("source", "")
                parts = source.split(":", 2)
                if (not row.get("file", "").startswith("train/") or len(parts) != 3
                        or parts[0] != "hard_train" or parts[1] not in HARD_SUBSETS):
                    continue
                path = row["orig_path"]
                key = _path_key(path)
                if key in seen_hard_paths:
                    continue
                seen_hard_paths.add(key)
                label, quad = _label_geometry(path)
                if row.get("plate_no") != label:
                    raise ValueError(f"hard label/filename mismatch: {path}")
                maybe_write({"path": path, "label": label, "subset": parts[1],
                             "brightness": str(_brightness(path) or ""), "quad": quad,
                             "source": source}, parts[1], writer)

    tune_file = destinations["tune.csv"]
    with tune_file.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TUNE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for key, item in sorted(holdout.items(), key=lambda pair: (pair[1]["subset"], pair[0])):
            path = item["orig_path"]
            digest = item["sha256"].lower()
            if key in protected_paths or digest in protected_hashes:
                raise ValueError(f"batch8 tune origin overlaps a protected list: {path}")
            actual = sha256_file(path)
            if actual != digest:
                raise ValueError(f"batch8 holdout hash changed for {path}")
            if digest in retained_hashes:
                raise ValueError(f"tune/train byte overlap for {path}")
            label, quad = _label_geometry(path)
            if item["subset"] not in HARD_SUBSETS:
                raise ValueError(f"unexpected tune subset: {item['subset']}")
            fold = item["calibration_fold"]
            if fold not in {"fit", "report"}:
                raise ValueError(f"unexpected tune calibration fold: {fold}")
            tune_hashes.add(digest)
            tune_hashes_by_fold[fold].add(digest)
            tune_by_subset_fold[item["subset"]][fold] += 1
            writer.writerow({"path": path, "label": label, "subset": item["subset"],
                             "brightness": item.get("brightness", ""), "quad": quad,
                             "sha256": digest, "calibration_fold": fold})
    if len(tune_hashes) != 2100:
        raise AssertionError("tune set must contain 2100 unique SHA-256 values")
    fold_counts = {fold: len(values) for fold, values in tune_hashes_by_fold.items()}
    if fold_counts != {"fit": 1050, "report": 1050}:
        raise AssertionError(f"batch8 tune fit/report counts changed: {fold_counts}")
    if tune_hashes_by_fold["fit"] & tune_hashes_by_fold["report"]:
        raise AssertionError("tune calibration fit/report SHA overlap")
    if retained_hashes & (protected_hashes | holdout_hashes):
        raise AssertionError("training pool contains protected or tune content")

    measured = read_jitter_samples(geometry_jsonl)
    geometry_sha = sha256_file(geometry_jsonl)
    sampler_payload = {
        "schema": "bbox-jitter-v1",
        "source": str(Path(geometry_jsonl).resolve()),
        "source_sha256": geometry_sha,
        "fit_partition": "train",
        "delta_fields": ["dx_over_gt_width", "dy_over_gt_height", "log_width_ratio", "log_height_ratio"],
        "geometry_note": "proposal_quad_px outer rectangle approximates candidate bbox; batch9 did not persist an exact bbox field",
        "sampling": "uniform measured row within requested subset; all groups fallback only if no matching sample exists",
        "samples_by_subset": {subset: measured.get(subset, []) for subset in HARD_SUBSETS},
        "sample_count_by_subset": {subset: len(measured.get(subset, [])) for subset in HARD_SUBSETS},
    }
    destinations["sampler.json"].write_text(
        json.dumps(sampler_payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    audit = {
        "schema": "box-reader-pool-v1",
        "source_files": {
            str(index_csv.resolve()): sha256_file(index_csv),
            str(labels_path.resolve()): sha256_file(labels_path),
            str(ledger_path.resolve()): ledger_sha,
            str(holdout_path.resolve()): sha256_file(holdout_path),
            str(audit_path.resolve()): sha256_file(audit_path),
            str(Path(geometry_jsonl).resolve()): geometry_sha,
        },
        "ledger_policy": "sha256 values reused only after source_hashes.json full-file SHA matches the pinned V2 ledger; absent candidates directly hashed",
        "protected_manifests": protected_meta,
        "protected_unique_path_count": len(protected_paths),
        "protected_unique_sha256_count": len(protected_hashes),
        "protected_policy": "all five requested lists plus worker03 golden alias excluded by normalized path and direct source-byte SHA-256",
        "batch8_tune": {"rows": len(holdout), "unique_sha256": len(tune_hashes),
                         "fold_unique_sha256": fold_counts,
                         "fit_report_sha_overlap": len(tune_hashes_by_fold["fit"] & tune_hashes_by_fold["report"]),
                         "path_unique": len(holdout),
                         "by_subset_fold": {s: dict(tune_by_subset_fold[s]) for s in HARD_SUBSETS}},
        "candidate_rows": {k: v for k, v in counters["candidate_rows"].items()},
        "candidate_unique_paths": {k: len(v) for k, v in counters["candidate_unique_paths"].items()},
        "excluded": {name: dict(counters[name]) for name in
                     ("excluded_protected_path", "excluded_protected_sha", "excluded_holdout_path",
                      "excluded_holdout_sha", "excluded_duplicate_sha")},
        "retained_by_group": dict(counters["retained"]),
        "retained_by_subset": dict(retained_by_subset),
        "ledger_hits_by_group": dict(counters["ledger_hits"]),
        "direct_hashes_by_group": dict(counters["direct_hashes"]),
        "direct_hash_paths": missing_ledger,
        "content_checks": {
            "train_unique_sha256": len(retained_hashes), "tune_unique_sha256": len(tune_hashes),
            "train_tune_sha_overlap": len(retained_hashes & tune_hashes),
            "train_protected_sha_overlap": len(retained_hashes & protected_hashes),
            "tune_protected_sha_overlap": len(tune_hashes & protected_hashes),
        },
        "cache_estimate": {
            "source_jpeg_bytes": candidate_original_bytes,
            "estimated_gt_bbox_pad50_jpeg_bytes": round(cache_estimate_bytes),
            "method": "source JPEG byte length multiplied by the expanded GT bbox area fraction; assumes 1160x720 CCPD frame and 50% pad clipped to frame; rough planning estimate only, not measured encoded crop size",
            "cap_bytes": 3_000_000_000,
            "generated": False,
        },
        "output": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)} for name, path in destinations.items()
                   if name != "audit.json"},
        "output_hashes": {name: sha256_file(path) for name, path in destinations.items()
                          if name != "audit.json"},
        "total_output_bytes_excluding_self_audit": sum(path.stat().st_size for name, path in destinations.items()
                                                       if name != "audit.json"),
        "notes": [
            "One train row per unique source image byte sequence; raw images are not copied or decoded.",
            "GT quad is in TL,TR,BR,BL order and bbox is derived from its axis-aligned outer rectangle.",
            "Batch9 records proposal quad but no exact candidate bbox; its outer rectangle is a documented approximation for bbox jitter.",
            "Train pool uses ccpd_index split=train base/green plus hard_train origins in V2 train labels; holdout paths and same-byte aliases are removed.",
        ],
    }
    destinations["audit.json"].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--v2-data", type=Path, default=V2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    args = parser.parse_args()
    print(json.dumps(prepare_pool(args.index, args.out, v2_dir=args.v2_data,
                                  geometry_jsonl=args.geometry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
