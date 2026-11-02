"""Evaluate a standalone plate-corner refiner on ordered production candidates.

The reader path mirrors lpr_tools.eval.ceiling's ordered rectifications and
first-OK CRNN rule. Ground truth is used only for paired geometry metrics and
subset grouping, never to select an input candidate for recognition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpr.config import ALGORITHM_DIR, DEFAULT_CONFIG_PATH, load_config
from lpr.corner_net import CornerNet, CornerRefiner
from lpr.locate import box_iou, locate
from lpr.pipeline import _decode_text, _in_roi, _merge_candidates, _rectifications
from lpr.preprocess import preprocess
from lpr.recognize.crnn import CrnnReader
from lpr.types import PlateCandidate
from lpr_tools.data.manifest import FIELDS, image_path, read_manifest
from lpr_tools.eval.common import effective_config_info
from lpr_tools.eval.index_samples import IndexSample, path_key, read_samples

IOU_THRESHOLD = 0.5
CORNER_THRESHOLD_PX = 8.0
STATUS_RANK = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}
SOURCE_PATHS = (
    Path("algorithm/lpr/corner_net.py"), Path("algorithm/lpr/preprocess.py"),
    Path("algorithm/lpr/locate.py"), Path("algorithm/lpr/pipeline.py"),
    Path("algorithm/lpr/rectify.py"), Path("algorithm/lpr/recognize/crnn.py"),
    Path(__file__),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {str(path): _sha256(path) for path in SOURCE_PATHS}


def _load_samples(manifest_path: Path, index_path: Path) -> tuple[list[IndexSample], dict[str, str]]:
    """Read an ordered path manifest and reconcile every row to the shared index."""
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest_reader = csv.DictReader(stream)
        fields = manifest_reader.fieldnames or []
        raw_manifest_rows = list(manifest_reader)
    canonical_rows = fields == list(FIELDS)
    if canonical_rows:
        manifest_rows = read_manifest(manifest_path, check_files=True)
        manifest_paths = [image_path(row, manifest_path).resolve() for row in manifest_rows]
    elif "path" in fields and all("path" in row and row["path"] for row in raw_manifest_rows):
        manifest_rows = None
        manifest_paths = []
        for raw in raw_manifest_rows:
            path = Path(raw["path"])
            manifest_paths.append((path if path.is_absolute() else manifest_path.parent / path).resolve())
    else:
        raise ValueError("manifest must use the canonical plate manifest schema or contain a path column")
    ordered_keys: list[str] = []
    seen: set[str] = set()
    for path in manifest_paths:
        key = path_key(path)
        if key in seen:
            raise ValueError(f"duplicate path in manifest: {key}")
        seen.add(key)
        ordered_keys.append(key)
    if not ordered_keys:
        raise ValueError("manifest is empty")

    selected = read_samples(index_path, only_paths=set(ordered_keys))
    by_path = {path_key(sample.row.path): sample for sample in selected}
    if set(by_path) != set(ordered_keys):
        raise ValueError(f"manifest paths missing from index: {len(set(ordered_keys) - set(by_path))}")
    samples = [by_path[key] for key in ordered_keys]

    # The manifest parser already checks labels and image paths. Where labels
    # are present, verify them against canonical index metadata as the ceiling
    # evaluator does for its fixed manifests.
    if manifest_rows is not None:
        for entry, sample in zip(manifest_rows, samples, strict=False):
            if entry.plate_no != sample.row.plate_no or entry.plate_color != sample.row.plate_color:
                raise ValueError(f"manifest truth/color differs from shared index: {sample.row.path}")
            if entry.quad is not None and sample.row.quad is not None:
                if not np.allclose(np.asarray(entry.quad), np.asarray(sample.row.quad), atol=0.0, rtol=0.0):
                    raise ValueError(f"manifest quad differs from shared index: {sample.row.path}")

    brightness: dict[str, str] = {}
    wanted = set(ordered_keys)
    with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            p = Path(raw["path"])
            if not p.is_absolute():
                p = Path(index_path).resolve().parent / p
            key = path_key(p)
            if key in wanted:
                brightness[key] = raw.get("brightness", "")
    if set(brightness) != wanted:
        raise ValueError("brightness metadata missing for one or more manifest images")
    return samples, brightness


def _corner_error(candidate: PlateCandidate, truth: Iterable[tuple[float, float]]) -> float:
    predicted = np.asarray(candidate.corners, dtype=np.float64).reshape(4, 2)
    expected = np.asarray(list(truth), dtype=np.float64).reshape(4, 2)
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("corner coordinates must be finite")
    return float(np.linalg.norm(predicted - expected, axis=1).mean())


def _ordered_reader_read(rects, sample: IndexSample, reader: CrnnReader,
                         min_confidence: float) -> tuple[str, dict[str, Any]]:
    """Match ceiling.py: first valid OK wins, otherwise highest status/confidence."""
    best = None
    for rect in rects:
        text, confidences = reader.read(rect.image)
        decoded = _decode_text(text, confidences, rect.color, min_confidence)
        if decoded is None:
            continue
        rank = (STATUS_RANK[decoded.status.value], decoded.confidence)
        if best is None or rank > best[0]:
            best = (rank, decoded.plate_no, decoded.status.value)
        if decoded.status.value == "OK":
            return decoded.plate_no, {"status": "OK", "prediction": decoded.plate_no}
    if best is None:
        return "", {"status": "NO_READ", "prediction": ""}
    return best[1], {"status": best[2], "prediction": best[1]}


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    baseline_correct = sum(r["baseline_correct"] for r in records)
    refined_correct = sum(r["refined_correct"] for r in records)
    out: dict[str, Any] = {
        "images": n,
        "baseline_crnn_exact_correct": baseline_correct,
        "baseline_crnn_accuracy": baseline_correct / n if n else None,
        "refined_crnn_exact_correct": refined_correct,
        "refined_crnn_accuracy": refined_correct / n if n else None,
        "accuracy_delta_count": refined_correct - baseline_correct,
    }
    def geometry_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        before = np.asarray([c["before_corner_error_px"] for c in pairs], dtype=np.float64)
        after = np.asarray([c["after_corner_error_px"] for c in pairs], dtype=np.float64)
        return {
            "bbox_hit_pairs": len(pairs),
            "before_mean_corner_error_px": float(before.mean()) if len(before) else None,
            "after_mean_corner_error_px": float(after.mean()) if len(after) else None,
            "before_le_8px_ratio": float(np.mean(before <= CORNER_THRESHOLD_PX)) if len(before) else None,
            "after_le_8px_ratio": float(np.mean(after <= CORNER_THRESHOLD_PX)) if len(after) else None,
            "mean_error_reduction_ratio": float((before.mean() - after.mean()) / before.mean())
            if len(before) and before.mean() > 0 else None,
        }
    pairs = [(candidate["before_corner_error_px"], candidate["after_corner_error_px"])
             for record in records for candidate in record["candidates"]
             if candidate["bbox_hit"]]
    before = np.asarray([p[0] for p in pairs], dtype=np.float64)
    after = np.asarray([p[1] for p in pairs], dtype=np.float64)
    out["corner_geometry"] = {
        "bbox_hit_candidates": int(len(pairs)),
        "images_with_bbox_hit_candidate": len({r["path"] for r in records
                                                 if any(c["bbox_hit"] for c in r["candidates"])}),
        "before_mean_corner_error_px": float(before.mean()) if len(before) else None,
        "after_mean_corner_error_px": float(after.mean()) if len(after) else None,
        "before_le_8px_ratio": float(np.mean(before <= CORNER_THRESHOLD_PX)) if len(before) else None,
        "after_le_8px_ratio": float(np.mean(after <= CORNER_THRESHOLD_PX)) if len(after) else None,
        "mean_error_reduction_ratio": float((before.mean() - after.mean()) / before.mean())
        if len(before) and before.mean() > 0 else None,
    }
    matched_candidates = [c for record in records for c in record["candidates"] if c["bbox_hit"]]
    out["corner_geometry_by_source"] = {
        source: geometry_metrics([c for c in matched_candidates if c["source"] == source])
        for source in sorted({c["source"] for c in matched_candidates})
    }
    best_hit_per_image = []
    best_iou_buckets: dict[str, list[dict[str, Any]]] = {"bbox_iou_0.5_to_0.7": [], "bbox_iou_ge_0.7": []}
    best_miss_count = 0
    for record in records:
        best = max(record["candidates"], key=lambda c: c["bbox_iou"], default=None)
        if best is None or not best["bbox_hit"]:
            best_miss_count += 1
            continue
        best_hit_per_image.append(best)
        bucket = "bbox_iou_ge_0.7" if best["bbox_iou"] >= .7 else "bbox_iou_0.5_to_0.7"
        best_iou_buckets[bucket].append(best)
    out["best_iou_candidate_per_image"] = {
        "images": n,
        "images_without_bbox_hit": best_miss_count,
        **geometry_metrics(best_hit_per_image),
        "iou_buckets": {name: geometry_metrics(bucket) for name, bucket in best_iou_buckets.items()},
    }
    return out


def evaluate(manifest_path: Path, index_path: Path, reader_path: Path, output_path: Path, *,
             refiner_path: Path | None = None, detector_path: Path | None = None,
             config_path: Path | None = None, candidate_trace_path: Path | None = None,
             threads: int = 2, purpose: str = "holdout",
             frozen: bool = False) -> dict[str, Any]:
    if threads != 2:
        raise ValueError("corner-refiner timing/acceptance requires exactly two Torch threads")
    if purpose in {"val", "devhard"} and not frozen:
        raise ValueError("val/devhard are report-only and require --frozen after checkpoint selection")
    if purpose not in {"holdout", "val", "devhard"}:
        raise ValueError("purpose must be holdout, val, or devhard")
    if Path(output_path).exists() or Path(output_path).with_suffix(".jsonl").exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_path}")

    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(2)
    cfg = load_config(config_path)
    resolved_config = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH.resolve()
    config_info = effective_config_info(cfg, config_path)
    current_source_hashes = _source_hashes()
    detector_path = detector_path or Path("algorithm/models/corner_net_v1.pt")
    detector = CornerNet.from_file(detector_path, version="corner_net_v1")
    refiner = CornerRefiner.from_file(refiner_path) if refiner_path is not None else None
    reader = CrnnReader.from_file(reader_path, version="crnn_v2")
    samples, brightness_by_path = _load_samples(manifest_path, index_path)
    cached_candidates: dict[str, list[PlateCandidate]] | None = None
    if candidate_trace_path is not None:
        trace_report_path = candidate_trace_path.with_suffix(".json")
        if not trace_report_path.is_file():
            raise FileNotFoundError(f"candidate trace provenance report missing: {trace_report_path}")
        trace_report = json.loads(trace_report_path.read_text(encoding="utf-8"))
        expected_reader_hash = _sha256(reader_path)
        expected_manifest_hash = _sha256(manifest_path)
        if trace_report.get("purpose") != "holdout" or trace_report.get("refiner") is not None:
            raise ValueError("candidate cache must come from a holdout baseline without refinement")
        if trace_report.get("manifest", {}).get("sha256") != expected_manifest_hash:
            raise ValueError("candidate cache was generated for a different manifest")
        if trace_report.get("detector", {}).get("sha256") != _sha256(detector_path):
            raise ValueError("candidate cache detector weights differ")
        if trace_report.get("reader", {}).get("sha256") != expected_reader_hash:
            raise ValueError("candidate cache CRNN reader weights differ")
        if trace_report.get("source_hashes") != current_source_hashes:
            raise ValueError("candidate cache source hashes differ; regenerate the baseline trace")
        if trace_report.get("input_hashes", {}).get(str(resolved_config)) != _sha256(resolved_config):
            raise ValueError("candidate cache config file contents differ")
        if trace_report.get("effective_config", {}).get("effective_config_sha256") != \
                config_info.get("effective_config_sha256"):
            raise ValueError("candidate cache runtime configuration differs")
        cached_candidates = {}
        with candidate_trace_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                item = json.loads(line)
                key = path_key(item["path"])
                if key in cached_candidates:
                    raise ValueError(f"duplicate path in candidate trace line {line_number}: {key}")
                if item.get("refiner_applied"):
                    raise ValueError("candidate trace must contain pre-refinement detector candidates")
                cached_candidates[key] = [PlateCandidate(
                    corners=[tuple(point) for point in raw["corners"]],
                    bbox=tuple(raw["bbox"]), color=raw["color"],
                    score=float(raw["score"]), source=raw["source"])
                    for raw in item.get("candidates", [])]
        if set(cached_candidates) != {path_key(sample.row.path) for sample in samples}:
            raise ValueError("candidate trace paths must exactly match evaluation manifest")

    image_records: list[dict[str, Any]] = []
    refine_latencies: list[float] = []
    failed_refinements = 0
    refinement_fallbacks = 0
    refinement_errors: list[dict[str, str]] = []
    for sample in samples:
        image_path_value = Path(sample.row.path)
        encoded = np.fromfile(image_path_value, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode manifest image: {image_path_value}")

        if cached_candidates is None:
            pre = preprocess(image, cfg.preprocess)
            classic = locate(pre, cfg.locate, cfg.runtime.max_candidates)
            network = [candidate for candidate in detector.detect(image, cfg.runtime.max_candidates)
                       if _in_roi(candidate, cfg.lane_roi(None))]
            candidates = _merge_candidates(network, classic, cfg.runtime.max_candidates)
        else:
            candidates = cached_candidates[path_key(sample.row.path)]
        truth_bbox = sample.bbox
        if truth_bbox is None or sample.row.quad is None:
            raise ValueError(f"positive plate sample needs bbox and quad: {sample.row.path}")
        truth_corners = sample.row.quad
        key = path_key(sample.row.path)
        brightness_text = brightness_by_path[key].strip()
        brightness = int(brightness_text) if brightness_text.isdigit() else None
        groups = ["all", sample.subset]
        if sample.subset == "ccpd_db" and brightness is not None and brightness <= 19:
            groups.append("ccpd_db_brightness_le_19")

        _refined_candidates: list[PlateCandidate] = []
        candidate_records: list[dict[str, Any]] = []
        baseline_rects = []
        refined_rects = []
        for rank, candidate in enumerate(candidates[:cfg.runtime.max_candidates]):
            iou = box_iou(candidate.bbox, truth_bbox)
            before_error = _corner_error(candidate, truth_corners)
            if refiner is None:
                refined = candidate
                refine_ms = 0.0
            else:
                began = time.perf_counter()
                try:
                    refined = refiner.refine(image, candidate)
                    if not isinstance(refined, PlateCandidate):
                        raise TypeError("refiner.refine must return a PlateCandidate")
                    if refined.color != candidate.color or refined.score != candidate.score \
                            or refined.source != candidate.source:
                        raise ValueError("refiner must preserve candidate color/score/source and order")
                    _corner_error(refined, truth_corners)  # validates finite ordered output
                except Exception as exc:
                    failed_refinements += 1
                    refinement_errors.append({"path": sample.row.path, "candidate_rank": str(rank),
                                              "error": f"{type(exc).__name__}: {exc}"})
                    refined = candidate
                if refined is candidate:
                    refinement_fallbacks += 1
                refine_ms = (time.perf_counter() - began) * 1000.0
                refine_latencies.append(refine_ms)
            after_error = _corner_error(refined, truth_corners)
            matched = iou >= IOU_THRESHOLD
            rects_before = _rectifications(image, candidate, cfg.rectify)
            rects_after = _rectifications(image, refined, cfg.rectify)
            baseline_rects.extend(rects_before)
            refined_rects.extend(rects_after)
            candidate_records.append({
                "rank": rank,
                "source": candidate.source,
                "bbox": list(candidate.bbox),
                "corners": [list(point) for point in candidate.corners],
                "color": candidate.color,
                "score": candidate.score,
                "refined_bbox": list(refined.bbox),
                "bbox_iou": float(iou),
                "bbox_hit": matched,
                "before_corner_error_px": before_error,
                "after_corner_error_px": after_error,
                "refiner_call_ms": refine_ms,
                "baseline_rectifications": len(rects_before),
                "refined_rectifications": len(rects_after),
            })

        baseline_prediction, baseline_info = _ordered_reader_read(
            baseline_rects, sample, reader, cfg.postprocess.min_confidence)
        if refiner is None:
            refined_prediction, refined_info = baseline_prediction, baseline_info
        else:
            refined_prediction, refined_info = _ordered_reader_read(
                refined_rects, sample, reader, cfg.postprocess.min_confidence)
        truth = sample.row.plate_no
        image_records.append({
            "path": sample.row.path,
            "subset": sample.subset,
            "brightness": brightness,
            "truth": truth,
            "groups": groups,
            "candidate_count": len(candidates),
            "top3_hit": any(c["bbox_hit"] for c in candidate_records),
            "candidates": candidate_records,
            "baseline_prediction": baseline_prediction,
            "baseline_status": baseline_info["status"],
            "baseline_correct": baseline_prediction == truth,
            "refined_prediction": refined_prediction,
            "refined_status": refined_info["status"],
            "refined_correct": refined_prediction == truth,
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        for group in record["groups"]:
            grouped[group].append(record)
    summaries = {group: _summarize(rows)
                 for group, rows in sorted(grouped.items())}
    report = {
        "method": "same ordered production fused candidates; per-candidate optional corner refinement; ceiling.py first-OK exact CRNN reader",
        "purpose": purpose,
        "frozen_report_only": bool(frozen),
        "manifest": {"path": str(Path(manifest_path).resolve()),
                     "sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                     "images": len(samples)},
        "index": {"path": str(Path(index_path).resolve()),
                  "sha256": hashlib.sha256(Path(index_path).read_bytes()).hexdigest()},
        "detector": {"path": str(Path(detector_path).resolve()),
                     "sha256": hashlib.sha256(Path(detector_path).read_bytes()).hexdigest()},
        "refiner": None if refiner_path is None else {
            "path": str(Path(refiner_path).resolve()),
            "sha256": hashlib.sha256(Path(refiner_path).read_bytes()).hexdigest()},
        "candidate_trace": None if candidate_trace_path is None else {
            "path": str(Path(candidate_trace_path).resolve()),
            "sha256": hashlib.sha256(Path(candidate_trace_path).read_bytes()).hexdigest()},
        "reader": {"path": str(Path(reader_path).resolve()),
                   "sha256": _sha256(reader_path),
                   "version": reader.version},
        "protocol": {
            "candidate_sources": "network first, then classic fill-in via production _merge_candidates; original max_candidates and candidate order preserved",
            "geometry_population": "each production Top-3 candidate with its original bbox IoU >= 0.5; same candidate pair before/after refiner",
            "reader_population": "all manifest images, all production candidates in original order, no bbox-hit/ground-truth gate",
            "reader_selection": "ceiling.py _reader_read first valid OK across ordered candidate rectifications; otherwise best status then confidence",
            "candidate_trace_reuse": "optional; exact path coverage is required and cached pre-refinement candidates retain their original order",
            "corner_error": "mean of four corresponding LT/RT/RB/LB Euclidean distances in original image pixels",
            "bbox_hit_threshold": IOU_THRESHOLD,
            "corner_threshold_px": CORNER_THRESHOLD_PX,
            "cpu_threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads()},
        },
        "refiner_timing_ms": {
            "calls": len(refine_latencies),
            "mean": float(np.mean(refine_latencies)) if refine_latencies else 0.0,
            "p50": float(np.percentile(refine_latencies, 50)) if refine_latencies else 0.0,
            "p95": float(np.percentile(refine_latencies, 95)) if refine_latencies else 0.0,
            "max": float(np.max(refine_latencies)) if refine_latencies else 0.0,
            "failed_calls": failed_refinements,
            "fallback_results": refinement_fallbacks,
            "call_errors": refinement_errors,
            "scope": "single refiner.refine(image,candidate) wall time on CPU; excludes image read, detection and CRNN",
        },
        "config_path": str(resolved_config),
        "effective_config": config_info,
        "source_hashes": current_source_hashes,
        "input_hashes": {
            str(Path(manifest_path).resolve()): _sha256(manifest_path),
            str(Path(index_path).resolve()): _sha256(index_path),
            str(Path(detector_path).resolve()): _sha256(detector_path),
            str(Path(reader_path).resolve()): _sha256(reader_path),
            str(resolved_config): _sha256(resolved_config),
            **({str(Path(refiner_path).resolve()): _sha256(refiner_path)} if refiner_path else {}),
            **({str(Path(candidate_trace_path).resolve()): _sha256(candidate_trace_path)}
               if candidate_trace_path else {}),
        },
        "decode_errors": 0,
        "groups": summaries,
        "records": image_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace_path = output_path.with_suffix(".jsonl")
    with trace_path.open("w", encoding="utf-8") as stream:
        for record in image_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ALGORITHM_DIR / "data/index/ccpd_index.csv")
    parser.add_argument("--refiner", type=Path, help="omit for pre-refinement baseline")
    parser.add_argument("--candidate-trace", type=Path, help="reuse an exact manifest-matched pre-refinement trace")
    parser.add_argument("--detector", type=Path, default=Path("algorithm/models/corner_net_v1.pt"))
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("holdout", "val", "devhard"), default="holdout")
    parser.add_argument("--frozen", action="store_true", help="allow val/devhard report-only evaluation after freeze")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    report = evaluate(args.manifest, args.index, args.reader, args.output,
                      refiner_path=args.refiner, detector_path=args.detector,
                      config_path=args.config, candidate_trace_path=args.candidate_trace,
                      threads=args.threads,
                      purpose=args.purpose, frozen=args.frozen)
    print(json.dumps({"output": str(args.output), "groups": report["groups"],
                      "refiner_timing_ms": report["refiner_timing_ms"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
