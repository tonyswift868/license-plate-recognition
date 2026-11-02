"""Stream prediction-only geometry and pixel features for a three-route trace.

This tool reruns only the configured locator (CornerNet + classical proposals),
never the segment, rectified-reader, or box-reader recognition routes. It does
not read labels or ground-truth quadrilaterals.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_rows(path: Path):
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    metadata = None
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if line_no == 1 and isinstance(row, dict) and "_meta" in row:
                metadata = row["_meta"]
                continue
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError(f"trace row {line_no} has no path")
            yield row
    if metadata is None:
        raise ValueError("trace is missing its first-line _meta identity")


def _read_trace_metadata(path: Path) -> dict:
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        first = stream.readline()
    try:
        row = json.loads(first)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("trace first line is not valid metadata JSON") from exc
    metadata = row.get("_meta") if isinstance(row, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("trace is missing first-line _meta identity")
    return metadata


def _validate_identity(trace_meta: dict, detector_identity: dict,
                       effective_config: dict) -> None:
    """Ensure the replay uses the exact detector and locator configuration."""
    if trace_meta.get("schema") != "three-route-live-v1":
        raise ValueError("trace metadata schema must be three-route-live-v1")
    expected_detector = trace_meta.get("models", {}).get("detector", {}).get("sha256")
    if not expected_detector or detector_identity.get("sha256") != expected_detector:
        raise ValueError("detector model SHA differs from source trace")
    expected_config = trace_meta.get("config", {}).get("effective_config")
    actual_config = effective_config.get("effective_config")
    if not isinstance(expected_config, dict) or not isinstance(actual_config, dict):
        raise ValueError("trace or replay is missing effective configuration")
    for key in ("preprocess", "locate", "lanes"):
        if actual_config.get(key) != expected_config.get(key):
            raise ValueError(f"locator config differs from trace in {key}")
    if (actual_config.get("runtime", {}).get("max_candidates") !=
            expected_config.get("runtime", {}).get("max_candidates")):
        raise ValueError("locator config differs from trace in runtime.max_candidates")


def _candidate_agrees(candidate, traced: dict, index: int) -> None:
    """Fail closed if the locator replay changes candidate identity or order."""
    expected = traced.get(str(index))
    if expected is None:
        raise ValueError(f"trace missing candidate index {index}")
    if int(expected.get("candidate_index", -1)) != index:
        raise ValueError(f"trace candidate index mismatch at {index}")
    if expected.get("source") != candidate.source:
        raise ValueError(f"candidate source mismatch at index {index}")
    box = [int(value) for value in candidate.bbox]
    if expected.get("bbox") != box:
        raise ValueError(f"candidate bbox mismatch at index {index}")
    if not math.isclose(float(expected.get("score")), float(candidate.score),
                        rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"candidate score mismatch at index {index}")


def _tilt_angle_deg(corners) -> float:
    """Mean absolute top/bottom horizontal-edge angle, in degrees.

    Corners must be ordered LT, RT, RB, LB in original-image coordinates.
    """
    points = [(float(x), float(y)) for x, y in corners]
    if len(points) != 4 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("candidate requires four finite ordered corners")
    top = math.degrees(math.atan2(points[1][1] - points[0][1],
                                 points[1][0] - points[0][0]))
    bottom = math.degrees(math.atan2(points[2][1] - points[3][1],
                                    points[2][0] - points[3][0]))
    return (abs(top) + abs(bottom)) / 2.0


def _layout_lengths(row: dict, candidate_index: int) -> list[int]:
    lengths = set()
    for rect in row.get("rects", []):
        if rect.get("candidate_index") != candidate_index:
            continue
        for route in ("segment", "reader"):
            attempt = rect.get(route)
            if attempt and attempt.get("valid") and isinstance(attempt.get("plate"), str):
                length = len(attempt["plate"])
                if length in (7, 8):
                    lengths.add(length)
    return sorted(lengths)


def _box_gray_mean(gray, bbox):
    x, y, width, height = (int(v) for v in bbox)
    h, w = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + width), min(h, y + height)
    crop = gray[y0:y1, x0:x1]
    return float(crop.mean()) if crop.size else None


def enrich_trace(*, trace_path, config_path, output_path, detector_model=None,
                 detector_loader=None):
    """Write gzip JSONL prediction-only features; refuses every output overwrite."""
    import cv2
    import numpy as np
    import torch
    from lpr.config import load_config
    from lpr.recognize import load_corner_detector

    from lpr_tools.eval.box_direct import _actual_candidates

    source = Path(trace_path).resolve()
    target = Path(output_path).resolve()
    config_file = Path(config_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite feature trace: {target}")
    trace_meta = _read_trace_metadata(source)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg = load_config(config_file, runtime__torch_threads=2)
    model_path = (Path(detector_model).resolve() if detector_model else
                  cfg.resolve_path(cfg.runtime.corner_model_path)
                  if cfg.runtime.corner_model_path else None)
    if model_path is None:
        raise ValueError("a configured or explicit CornerNet detector model is required")
    from lpr_tools.eval.three_route import _model_manifest_identity
    detector_identity = _model_manifest_identity(model_path)
    from lpr_tools.eval.common import effective_config_info

    effective_info = effective_config_info(cfg, config_file)
    _validate_identity(trace_meta, detector_identity, effective_info)
    if detector_loader:
        from lpr_tools.eval.three_route import _load_custom_factory
        detector = _load_custom_factory(detector_loader, model_path, "detect")
    else:
        detector = load_corner_detector(model_path)
    effective_payload = effective_info.get("effective_config", effective_info)
    effective_sha = hashlib.sha256(json.dumps(
        effective_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    meta = {
        "schema": "three-route-prediction-features-v1",
        "source_trace": str(source), "source_trace_sha256": sha256_file(source),
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
        "detector": {**detector_identity,
                     "loader": detector_loader or "lpr.recognize.load_corner_detector"},
        "config": {"path": str(config_file), "sha256": sha256_file(config_file),
                   "effective_config_sha256": effective_sha, "effective": effective_info},
        "runtime_threads": {"torch": 2, "opencv": 1, "blas": 1},
        "candidate_replay": "box_direct._actual_candidates: configured corner detector + classical locator, merged top 3",
        "feature_definitions": {
            "tilt_angle_deg": "mean(abs(top-edge atan2 angle), abs(bottom-edge atan2 angle)); corners LT,RT,RB,LB",
            "layout_lengths": "sorted unique lengths 7/8 from valid segment/reader attempts in the same candidate's trace rects",
            "brightness_filename": "existing trace brightness field only; CCPD filename annotation, not pixel measurement",
            "image_gray_mean": "mean of OpenCV BGR-to-gray image pixels, scale 0..255",
            "box_gray_mean": "mean gray over candidate bbox clipped to image; no padding or rectification",
            "color": "candidate locator's predicted color",
        },
        "scope": "prediction/pixel features only; no truth, GT quad, model recognition, or image output",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if target.suffix.lower() == ".gz" else open
    count = 0
    seen_paths = set()
    try:
        with opener(target, "wt", encoding="utf-8", newline="\n") as out:
            out.write(json.dumps({"_meta": meta}, ensure_ascii=False,
                                 separators=(",", ":")) + "\n")
            for row in _trace_rows(source):
                image_path = Path(row["path"])
                image_bytes = image_path.read_bytes()
                image_sha = hashlib.sha256(image_bytes).hexdigest()
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                del image_bytes
                if image is None:
                    raise ValueError(f"cannot decode trace image: {image_path}")
                path_key = str(image_path.resolve()).casefold()
                if path_key in seen_paths:
                    raise ValueError(f"duplicate image path in trace: {image_path}")
                seen_paths.add(path_key)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gray_mean = float(gray.mean())
                candidates = _actual_candidates(image, cfg, detector)
                traced_boxes = row.get("boxes", {})
                if int(row.get("candidate_count", -1)) != len(candidates):
                    raise ValueError(f"candidate count changed for {image_path}")
                if set(traced_boxes) != {str(i) for i in range(len(candidates))}:
                    raise ValueError(f"trace candidate keys differ for {image_path}")
                features = []
                for ci, candidate in enumerate(candidates):
                    _candidate_agrees(candidate, traced_boxes, ci)
                    box_gray_mean = _box_gray_mean(gray, candidate.bbox)
                    features.append({
                        "candidate_index": ci,
                        "source": candidate.source,
                        "score": float(candidate.score),
                        "bbox": [int(v) for v in candidate.bbox],
                        "corners": [[float(x), float(y)] for x, y in candidate.corners],
                        "tilt_angle_deg": _tilt_angle_deg(candidate.corners),
                        "color": str(candidate.color),
                        "layout_lengths": _layout_lengths(row, ci),
                        "box_gray_mean": box_gray_mean,
                    })
                out.write(json.dumps({
                    "path": row["path"], "candidate_count": len(candidates),
                    "image_sha256": image_sha,
                    "brightness_filename": row.get("brightness"),
                    "image_gray_mean": image_gray_mean, "candidates": features,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                if count % 100 == 0:
                    out.flush()
                    print(f"route_features={count}/3100", flush=True)
    except Exception:
        # A failed partial trace must not look like a completed artifact.
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    if count != 3100:
        target.unlink(missing_ok=True)
        raise ValueError(f"source trace must contain exactly 3100 records; got {count}")
    if target.stat().st_size > 500_000_000:
        target.unlink(missing_ok=True)
        raise ValueError("compressed feature trace exceeds 500 MiB output budget")
    return {"status": "complete", "records": count, "output": str(target),
            "source_trace_sha256": meta["source_trace_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="three-route JSONL or JSONL.gz")
    parser.add_argument("--config", required=True)
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-loader", help="optional module:factory implementing detect(image,max_candidates)")
    parser.add_argument("--out", required=True, help="new .jsonl.gz output (must not exist)")
    args = parser.parse_args(argv)
    result = enrich_trace(trace_path=args.trace, config_path=args.config,
                          detector_model=args.detector_model,
                          detector_loader=args.detector_loader, output_path=args.out)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
