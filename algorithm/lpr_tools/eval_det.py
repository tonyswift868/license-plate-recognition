"""Evaluation helpers and CPU runner for plate_det_v1.

The module imports detector/reader implementations only inside evaluate(), so
metric tests and report tooling do not allocate model memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from lpr.config import ALGORITHM_DIR

IOU_THRESHOLDS = (0.5, 0.7)
HARD_SUBSETS = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
                "ccpd_fn", "ccpd_challenge", "ccpd_weather")
STATUS_RANK = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}
CAMERA_SCALES = (0.4, 0.6, 0.8, 1.0, 1.3, 1.6)
THRESHOLD_GRID = tuple(round(i / 100, 2) for i in range(5, 100, 5))


def camera_canvas(image, bbox: Sequence[float], quad: Sequence[Sequence[float]],
                  scale: float, canvas: tuple[int, int] = (1280, 720)):
    """Resize a plate image to the camera canvas and transform its GT geometry.

    Scale is relative to the benchmark letterbox at 1.0. If requested content
    exceeds the canvas, crop around the ground-truth bbox center; remaining
    margins are zero-padded. Using GT for this synthetic crop is a measurement
    protocol, not a claim about camera-time crop robustness.
    """
    import cv2
    import numpy as np

    ch, cw = image.shape[:2]
    transform = camera_canvas_metadata(cw, ch, bbox, quad, scale, canvas)
    w, h = canvas
    target_w, target_h = transform["scaled_content_wh"]
    # Resize the original exactly once. This preserves detail at large scale
    # and uses the same 1.0 resize/interpolation as the benchmark letterbox.
    effective_scale = min(target_w / cw, target_h / ch)
    resized = cv2.resize(image, (target_w, target_h),
                         interpolation=cv2.INTER_AREA if effective_scale < 1 else cv2.INTER_LINEAR)
    crop_x, crop_y = transform["crop_xy"]
    pad_x, pad_y = transform["pad_xy"]
    cut = resized[crop_y:min(crop_y + h, target_h), crop_x:min(crop_x + w, target_w)]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[pad_y:pad_y + cut.shape[0], pad_x:pad_x + cut.shape[1]] = cut
    transformed_bbox = tuple(transform["bbox_xywh"])
    transformed_quad = transform["quad_xy"]
    return out, transformed_bbox, transformed_quad, transform


def camera_canvas_metadata(width: int, height: int, bbox: Sequence[float],
                          quad: Sequence[Sequence[float]], scale: float,
                          canvas: tuple[int, int] = (1280, 720)) -> dict[str, Any]:
    """Preflight transformed GT dimensions without decoding an image."""
    if (width <= 0 or height <= 0 or not math.isfinite(float(scale)) or scale <= 0
            or len(canvas) != 2 or min(canvas) <= 0):
        raise ValueError("source dimensions and scale must be positive")
    x, y, bw, bh = map(float, bbox)
    points = [tuple(map(float, p)) for p in quad]
    if (len(points) != 4 or not all(math.isfinite(v) for v in (x, y, bw, bh))
            or min(bw, bh) <= 0 or not all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in points)):
        raise ValueError("bbox and quad must be finite and nondegenerate")
    w, h = canvas
    base = min(w / width, h / height)
    base_w, base_h = min(w, max(1, round(width * base))), min(h, max(1, round(height * base)))
    target_w, target_h = max(1, round(base_w * scale)), max(1, round(base_h * scale))
    sx, sy = target_w / width, target_h / height
    gx, gy, gbw, gbh = x * sx, y * sy, bw * sx, bh * sy
    scaled_quad = [(px * sx, py * sy) for px, py in points]
    qx0, qy0 = min(p[0] for p in scaled_quad), min(p[1] for p in scaled_quad)
    qx1, qy1 = max(p[0] for p in scaled_quad), max(p[1] for p in scaled_quad)
    union_x0, union_y0 = min(gx, qx0), min(gy, qy0)
    union_x1, union_y1 = max(gx + gbw, qx1), max(gy + gbh, qy1)
    if union_x1 - union_x0 > w or union_y1 - union_y0 > h:
        raise ValueError("scaled GT bbox+quad union exceeds camera canvas")
    center_x, center_y = gx + gbw / 2, gy + gbh / 2
    max_x, max_y = max(0, target_w - w), max(0, target_h - h)
    lo_x, hi_x = max(0, math.ceil(union_x1 - w)), min(max_x, math.floor(union_x0))
    lo_y, hi_y = max(0, math.ceil(union_y1 - h)), min(max_y, math.floor(union_y0))
    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("scaled GT bbox+quad union cannot be retained on camera canvas")
    crop_x = min(max(round(center_x - w / 2), lo_x), hi_x)
    crop_y = min(max(round(center_y - h / 2), lo_y), hi_y)
    pad_x, pad_y = (w - target_w) // 2 if target_w < w else 0, (h - target_h) // 2 if target_h < h else 0
    dx, dy = pad_x - crop_x, pad_y - crop_y
    transformed_quad = [[px + dx, py + dy] for px, py in scaled_quad]
    tx, ty = gx + dx, gy + dy
    visible = (tx >= 0 and ty >= 0 and tx + gbw <= w and ty + gbh <= h and
               all(0 <= px <= w and 0 <= py <= h for px, py in transformed_quad))
    if not visible:
        raise ValueError("scaled GT bbox+quad is not fully visible")
    return {"bbox_xywh": [tx, ty, gbw, gbh], "quad_xy": transformed_quad,
            "plate_width_canvas_px": gbw, "plate_width_model256_px": gbw * (256 / w),
            "base_content_wh": [base_w, base_h], "scaled_content_wh": [target_w, target_h],
            "crop_xy": [crop_x, crop_y], "pad_xy": [pad_x, pad_y],
            "gt_bbox_fully_visible": visible, "gt_quad_fully_visible": visible}


def indexed_dimensions(index_path: Path, paths: set[str]) -> dict[str, tuple[int, int]]:
    """Read indexed dimensions or JPEG SOF metadata for selected paths, streaming CSV once."""
    import csv

    from lpr_tools.eval.index_samples import path_key
    from lpr_tools.train_det import jpeg_size
    found = {}
    with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("canonical index must contain a path column")
        has_dimensions = {"width", "height"}.issubset(reader.fieldnames)
        for raw in reader:
            key = path_key(raw["path"])
            if key in paths:
                if has_dimensions and raw["width"] and raw["height"]:
                    found[key] = (int(raw["width"]), int(raw["height"]))
                else:
                    found[key] = jpeg_size(raw["path"])
                if len(found) == len(paths):
                    break
    if len(found) != len(paths):
        raise ValueError(f"canonical index is missing dimensions for {len(paths) - len(found)} selected paths")
    return found


def threshold_curve(positive_records: Sequence[Sequence[float]],
                    negative_records: Sequence[Sequence[float]],
                    thresholds: Sequence[float] = THRESHOLD_GRID) -> list[dict[str, Any]]:
    """Score thresholds to positive Top-1 IoU.7 recall and NP image FP rate."""
    for threshold in thresholds:
        if not math.isfinite(float(threshold)) or not 0 < threshold < 1:
            raise ValueError("thresholds must be finite and between zero and one")
    out = []
    for threshold in thresholds:
        # Filtering preserves detector score order; Top-1 is the first
        # surviving proposal, even when a lower-ranked proposal matches GT.
        pos_hits = sum(bool((first := next((iou for score, iou in row
                                           if score >= threshold), None)) is not None and first >= .7)
                       for row in positive_records)
        neg_hits = sum(any(score >= threshold for score, _ in row) for row in negative_records)
        out.append({"threshold": float(threshold), "positive_top1_iou_0_7_recall":
                    pos_hits / len(positive_records) if positive_records else None,
                    "positive_hits": pos_hits, "positive_images": len(positive_records),
                    "np_false_positive_image_rate": neg_hits / len(negative_records) if negative_records else None,
                    "np_false_positive_images": neg_hits, "np_images": len(negative_records)})
    return out


def ranked_box_metrics(records: Sequence[Sequence[dict[str, Any]]], *,
                      max_candidates: int = 3) -> dict[str, Any]:
    """Aggregate ranked Top-1/3 IoU.7 hits and candidate count from one-pass rows."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    n = len(records)
    topk = {}
    for k in (1, 3):
        hits = sum(any(float(item["iou"]) >= .7 for item in row[:min(k, max_candidates)])
                   for row in records)
        topk[str(k)] = {"hits": hits, "n": n, "recall": hits / n if n else None}
    total_candidates = sum(min(len(row), max_candidates) for row in records)
    return {"images": n, "topk_iou_0_7": topk,
            "mean_candidates": total_candidates / n if n else None}


def width_distribution(widths: Sequence[float]) -> dict[str, Any]:
    """Return deterministic linear-interpolated p5/median/p95 width summaries."""
    values = sorted(float(v) for v in widths)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("plate widths must be finite and positive")
    def quantile(q: float):
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "p5": quantile(.05), "median": quantile(.5), "p95": quantile(.95)}


def evaluate_scale_suite(manifest_path: Path, index_path: Path, detector_path: Path,
                         output_path: Path, *, config_path: Path | None = None,
                         scales: Sequence[float] = CAMERA_SCALES,
                         threads: int = 2) -> dict[str, Any]:
    """Preflight GT against all requested scales, then evaluate one image at a time."""
    if threads != 2:
        raise ValueError("scale study requires exactly two Torch threads")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite scale output: {output_path}")
    scales = tuple(float(s) for s in scales)
    if not scales or any(s not in CAMERA_SCALES for s in scales):
        raise ValueError(f"scales must be drawn from {CAMERA_SCALES}")
    from lpr_tools.train_det import limit_cpu_working_set, process_memory
    if not limit_cpu_working_set(1_000_000_000):
        raise RuntimeError("could not enforce the 1 GB CPU working-set limit")
    import cv2
    import numpy as np
    import torch
    from lpr.config import load_config

    from lpr_tools.eval.index_samples import path_key
    from lpr_tools.eval_refine import _load_samples

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    samples, _ = _load_samples(manifest_path, index_path)
    dimensions = indexed_dimensions(index_path, {path_key(s.row.path) for s in samples})
    # Validate every GT bbox against all transforms before loading weights or
    # starting inference; do not silently drop geometrically impossible rows.
    for sample in samples:
        if sample.bbox is None or sample.row.quad is None:
            raise ValueError(f"scale study needs GT bbox and quad: {sample.row.path}")
        width, height = dimensions[path_key(sample.row.path)]
        for factor in scales:
            try:
                camera_canvas_metadata(width, height, sample.bbox, sample.row.quad, factor)
            except ValueError as exc:
                raise ValueError(f"scale preflight failed for {sample.row.path} at {factor}: {exc}") from exc
    cfg = load_config(config_path)
    detector = _load_candidates(detector_path, "plate_det_v1")
    by_scale: dict[str, list[dict[str, Any]]] = {str(s): [] for s in scales}
    widths_canvas = {str(s): [] for s in scales}
    widths_model = {str(s): [] for s in scales}
    latencies = {str(s): [] for s in scales}
    memory_samples = []
    started = time.monotonic()
    for number, sample in enumerate(samples, 1):
        path = Path(sample.row.path)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode scale study image: {path}")
        for factor in scales:
            canvas, truth, _, transform = camera_canvas(image, sample.bbox, sample.row.quad, factor)
            widths_canvas[str(factor)].append(truth[2])
            widths_model[str(factor)].append(truth[2] * (256 / 1280))
            began = time.perf_counter()
            candidates = detector.detect(canvas, cfg.runtime.max_candidates)
            latencies[str(factor)].append((time.perf_counter() - began) * 1000)
            by_scale[str(factor)].append([{"iou": bbox_iou(c.bbox, truth), "score": float(c.score)}
                                          for c in candidates[:cfg.runtime.max_candidates]])
        if number % 100 == 0:
            memory = process_memory()
            memory_samples.append({"images_completed": number, **memory})
            if memory["working_set"] > 1_000_000_000:
                raise MemoryError(f"scale evaluation exceeded 1 GB working set at image {number}")
            print(json.dumps({"study": "scale", "completed": number, "total": len(samples),
                              "elapsed_s": time.monotonic() - started, "memory": memory}), flush=True)
        del image
    reports = {}
    for factor in scales:
        key = str(factor)
        summaries = ranked_box_metrics(by_scale[key], max_candidates=cfg.runtime.max_candidates)
        reports[key] = {"top1_iou_0_7": summaries["topk_iou_0_7"]["1"],
                        "top3_iou_0_7": summaries["topk_iou_0_7"]["3"],
                        "plate_width_canvas_px": width_distribution(widths_canvas[key]),
                        "plate_width_model256_px": width_distribution(widths_model[key]),
                        "detector_latency_ms_mean": sum(latencies[key]) / len(latencies[key]) if latencies[key] else None}
    report = {"study": "scale_suite", "manifest": str(manifest_path.resolve()),
              "manifest_sha256": _sha256(manifest_path), "index_sha256": _sha256(index_path),
              "detector": str(detector_path.resolve()), "detector_sha256": _sha256(detector_path),
              "scales": reports,
              "protocol": {"canvas": [1280, 720], "model_input": [256, 256],
                           "baseline_fit": "same rounded dimensions, interpolation, and centered zero padding as eval.common.letterbox",
                           "scale_anchor": "scale is relative to baseline fitted content dimensions",
                           "oversized_crop": "GT bbox center anchors a crop clamped to content bounds and constrained to retain the union of GT bbox and four GT corners; residual margins are centered zero padding",
                           "ground_truth_use": "GT bbox is used only to anchor synthetic oversized crops; results do not establish camera-time robustness without GT",
                           "preflight": {"samples": len(samples), "scales": list(scales), "all_gt_boxes_fully_retained": True,
                                         "bbox_and_quad_union_retained": True},
                           "width_mapping": "256-pixel square model letterbox scales canvas x coordinates by 256/1280"},
              "memory": {"working_set_limit_bytes": 1_000_000_000,
                         "samples_every_100_images": memory_samples,
                         "final": process_memory()},
              "elapsed_seconds": time.monotonic() - started}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU for xywh boxes; malformed/nonfinite/degenerate boxes score zero."""
    if len(a) != 4 or len(b) != 4:
        return 0.0
    values = tuple(float(v) for v in (*a, *b))
    if not all(math.isfinite(v) for v in values):
        return 0.0
    ax, ay, aw, ah, bx, by, bw, bh = values
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def corner_error_px(predicted: Iterable[Sequence[float]], truth: Iterable[Sequence[float]]) -> float:
    """Mean corresponding LT/RT/RB/LB Euclidean error in original pixels."""
    import numpy as np
    p = np.asarray(list(predicted), dtype=np.float64).reshape(4, 2)
    t = np.asarray(list(truth), dtype=np.float64).reshape(4, 2)
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("corner coordinates must be finite")
    return float(np.linalg.norm(p - t, axis=1).mean())


def topk_hits(candidates: Sequence[Any], truth_box: Sequence[float], *, k: int,
              threshold: float) -> bool:
    """Whether any of the first k already-ranked candidates reaches IoU."""
    return any(bbox_iou(c.bbox, truth_box) >= threshold for c in candidates[:k])


def summarize_boxes(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ranked IoU recall and matched-candidate corner error."""
    out: dict[str, Any] = {"images": len(records), "topk": {}}
    for k in (1, 3):
        out["topk"][str(k)] = {}
        for threshold in IOU_THRESHOLDS:
            n = len(records)
            hits = sum(any(c["iou"] >= threshold for c in r["candidates"][:k]) for r in records)
            out["topk"][str(k)][str(threshold)] = {
                "hits": hits, "n": n, "recall": hits / n if n else None}
    errors = [c["corner_error_px"] for r in records for c in r["candidates"]
              if c["iou"] >= 0.5]
    out["bbox_hit_candidates"] = len(errors)
    out["mean_corner_error_px_on_iou_ge_0_5"] = sum(errors) / len(errors) if errors else None
    return out


def first_ok_read(ordered_reads: Iterable[tuple[str, Sequence[float], str]],
                  decode: Callable[[str, Sequence[float], str], Any]) -> tuple[str, str]:
    """Use first operational OK result; otherwise best status/confidence.

    Each input is (text, per-character confidence, color). No truth value is
    accepted by this function, preventing ground-truth candidate routing.
    """
    best: tuple[tuple[int, float], str, str] | None = None
    for text, confidences, color in ordered_reads:
        decoded = decode(text, confidences, color)
        if decoded is None:
            continue
        status = decoded.status.value if hasattr(decoded.status, "value") else str(decoded.status)
        plate = str(decoded.plate_no)
        confidence = float(decoded.confidence)
        rank = (STATUS_RANK.get(status, 0), confidence)
        if best is None or rank > best[0]:
            best = (rank, plate, status)
        if status == "OK":
            return plate, status
    return (best[1], best[2]) if best else ("", "NO_READ")


def summarize_np(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Image and proposal false-positive counts for a no-plate set."""
    positives = [r for r in records if r.get("candidate_count", 0) > 0]
    return {"images": len(records), "images_with_false_positive": len(positives),
            "false_positive_image_rate": len(positives) / len(records) if records else None,
            "false_positive_candidates": sum(int(r.get("candidate_count", 0)) for r in records)}


def validate_purpose(purpose: str, frozen: bool) -> None:
    if purpose not in {"holdout", "val", "devhard"}:
        raise ValueError("purpose must be holdout, val, or devhard")
    if purpose in {"val", "devhard"} and not frozen:
        raise ValueError("val/devhard are report-only and require --frozen after checkpoint selection")


def _load_candidates(path: Path, version: str):
    """Resolve checkpoints through the repository's architecture dispatcher."""
    from lpr.corner_net import CornerNet
    return CornerNet.from_file(path, version=version)


class BoxCrnnAdapter:
    """Run the existing CRNN preprocessing/model/decode path on arbitrary box crops.

    CrnnReader.read() has a legacy canonical-rectification shape guard. This
    adapter bypasses only that guard: plate_tensor still performs the reader's
    one shared gray/resize/normalization step, followed by the same model,
    temperature scaling, finiteness check, and CTC greedy decode.
    """

    def __init__(self, reader: Any) -> None:
        self.reader = reader
        self.name = getattr(reader, "name", "crnn")
        self.version = reader.version

    def read(self, plate_image):
        import numpy as np
        import torch
        from lpr.charset import NUM_CLASSES
        from lpr.recognize.crnn import _greedy_decode, plate_tensor

        if (not isinstance(plate_image, np.ndarray) or plate_image.dtype != np.uint8
                or plate_image.ndim != 3 or plate_image.shape[2] != 3
                or min(plate_image.shape[:2]) <= 0):
            return "", []
        tensor = plate_tensor(plate_image)
        with torch.inference_mode():
            logits = self.reader.model(tensor[None])[..., :NUM_CLASSES + 1]
            if self.reader.temperature != 1.0:
                logits = logits / self.reader.temperature
        if not torch.isfinite(logits).all():
            return "", []
        return _greedy_decode(logits[:, 0])


def evaluate(manifest_path: Path, index_path: Path, detector_path: Path,
             reader_path: Path | None, output_path: Path, *, v1_path: Path,
             np_manifest: Path | None = None, purpose: str = "holdout",
             frozen: bool = False, threads: int = 2, config_path: Path | None = None,
             boxes_only: bool = False, method: str = "new") -> dict[str, Any]:
    """Run paired evaluation or a report-only geometry selection pass."""
    validate_purpose(purpose, frozen)
    if boxes_only and purpose in {"val", "devhard"} and not frozen:
        raise ValueError("--boxes-only on val/devhard requires --frozen")
    if boxes_only and method not in {"new", "v1-fused"}:
        raise ValueError("boxes-only method must be new or v1-fused")
    if not boxes_only and reader_path is None:
        raise ValueError("full evaluation requires a CRNN reader checkpoint")
    if boxes_only and np_manifest is not None:
        raise ValueError("NP false-positive evaluation belongs to the final full evaluation")
    if threads != 2:
        raise ValueError("detector evaluation requires exactly two Torch threads")
    if output_path.exists() or output_path.with_suffix(".jsonl").exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_path}")

    # Heavy dependencies are deliberately deferred until an authorized run.
    import cv2
    import numpy as np
    import torch
    from lpr.config import load_config
    from lpr.locate import locate
    from lpr.pipeline import _decode_text, _in_roi, _merge_candidates
    from lpr.preprocess import preprocess

    from lpr_tools.eval.index_samples import path_key
    from lpr_tools.eval_refine import _load_samples

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    cfg = load_config(config_path)
    detector = _load_candidates(detector_path, "plate_det_v1") if not boxes_only or method == "new" else None
    v1 = _load_candidates(v1_path, "corner_net_v1") if not boxes_only or method == "v1-fused" else None
    if boxes_only:
        reader = None
    else:
        from lpr.recognize.crnn import CrnnReader
        reader = BoxCrnnAdapter(CrnnReader.from_file(reader_path, version="crnn_v2"))
    samples, brightness_by_path = _load_samples(manifest_path, index_path)
    expected_n = {"holdout": 2100, "val": 1000, "devhard": 2100}[purpose]
    if len(samples) != expected_n:
        raise ValueError(f"{purpose} manifest must resolve to exactly {expected_n} indexed rows; got {len(samples)}")
    if purpose == "val" and any(sample.split != "val" for sample in samples):
        raise ValueError("val report manifest contains an image outside the official val split")
    if purpose == "devhard":
        by_subset = Counter(sample.subset for sample in samples)
        if any(by_subset[subset] != 300 for subset in HARD_SUBSETS):
            raise ValueError(f"devhard manifest must contain exactly 300 images per hard subset: {dict(by_subset)}")
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = defaultdict(list)

    def detect_one(model, image, name):
        start = time.perf_counter()
        values = model.detect(image, cfg.runtime.max_candidates)
        latencies[name].append((time.perf_counter() - start) * 1000.0)
        return values[:cfg.runtime.max_candidates]

    evaluation_started = time.monotonic()
    for sample_number, sample in enumerate(samples, 1):
        path = Path(sample.row.path)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode manifest image: {path}")
        if sample.bbox is None or sample.row.quad is None:
            raise ValueError(f"positive sample needs bbox and corners: {path}")
        if boxes_only and method == "new":
            candidates_by_method = {"plate_det_v1": detect_one(detector, image, "plate_det_v1")}
        elif boxes_only:
            from lpr.locate import locate
            from lpr.pipeline import _in_roi, _merge_candidates
            from lpr.preprocess import preprocess
            old = detect_one(v1, image, "corner_net_v1")
            pre = preprocess(image, cfg.preprocess)
            classic = locate(pre, cfg.locate, cfg.runtime.max_candidates)
            old_roi = [c for c in old if _in_roi(c, cfg.lane_roi(None))]
            candidates_by_method = {"classic_plus_v1": _merge_candidates(
                old_roi, classic, cfg.runtime.max_candidates)}
        else:
            from lpr.locate import locate
            from lpr.pipeline import _in_roi, _merge_candidates
            from lpr.preprocess import preprocess
            new = detect_one(detector, image, "plate_det_v1")
            old = detect_one(v1, image, "corner_net_v1")
            pre = preprocess(image, cfg.preprocess)
            classic = locate(pre, cfg.locate, cfg.runtime.max_candidates)
            old_roi = [c for c in old if _in_roi(c, cfg.lane_roi(None))]
            fused = _merge_candidates(old_roi, classic, cfg.runtime.max_candidates)
            candidates_by_method = {"plate_det_v1": new, "corner_net_v1": old,
                                   "classic": classic, "classic_plus_v1": fused}
        brightness_text = brightness_by_path.get(path_key(path), "")
        record: dict[str, Any] = {"path": str(path), "subset": sample.subset,
                                  "brightness": int(brightness_text) if brightness_text.isdigit() else None,
                                  "truth": sample.row.plate_no, "methods": {}}
        for method_name, candidates in candidates_by_method.items():
            candidate_rows = []
            for rank, c in enumerate(candidates[:3]):
                candidate_rows.append({"rank": rank + 1, "bbox": list(c.bbox),
                                       "iou": bbox_iou(c.bbox, sample.bbox),
                                       "corner_error_px": corner_error_px(c.corners, sample.row.quad),
                                       "score": float(c.score), "source": c.source})
            result = {"candidate_count": len(candidates), "candidates": candidate_rows}
            if not boxes_only:
                from lpr.boxcrop import crop_box
                from lpr.pipeline import _decode_text
                prediction, status = _read_box_route(image, candidates, reader,
                                                      crop_box, _decode_text,
                                                      cfg.postprocess.min_confidence)
                result.update({"prediction": prediction, "status": status,
                               "exact": prediction == sample.row.plate_no})
            record["methods"][method_name] = result
        rows.append(record)
        if sample_number % 100 == 0:
            print(json.dumps({"purpose": purpose, "images_completed": sample_number,
                              "images_total": len(samples),
                              "elapsed_seconds": time.monotonic()-evaluation_started}), flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subset in HARD_SUBSETS:
        groups[subset]  # Keep empty required subsets visible in the report.
    groups["dark_db_le_19"]
    for row in rows:
        groups["all"].append(row)
        groups[row["subset"]].append(row)
        if row["subset"] == "ccpd_db" and row["brightness"] is not None and row["brightness"] <= 19:
            groups["dark_db_le_19"].append(row)
    report_groups = {}
    for group, group_rows in groups.items():
        report_groups[group] = {}
        if boxes_only:
            evaluated_methods = ("plate_det_v1",) if method == "new" else ("classic_plus_v1",)
        else:
            evaluated_methods = ("plate_det_v1", "corner_net_v1", "classic", "classic_plus_v1")
        for method_name in evaluated_methods:
            projected = [{"candidates": r["methods"][method_name]["candidates"]} for r in group_rows]
            box_metrics = summarize_boxes(projected)
            if not boxes_only:
                exact = sum(r["methods"][method_name]["exact"] for r in group_rows)
                box_metrics["crnn_box_exact"] = {"correct": exact, "n": len(group_rows),
                                                   "accuracy": exact / len(group_rows) if group_rows else None}
            report_groups[group][method_name] = box_metrics

    np_metrics = None
    if np_manifest is not None:
        np_metrics = _evaluate_np(np_manifest, detector, cfg, cv2, np)
    report = {"method": (f"{'frozen box report' if purpose in {'val', 'devhard'} else 'holdout box selection'}: {method}" if boxes_only else
                          "ranked candidate geometry plus crop_box(.15,.25) and offline CRNN first-OK; no truth-based routing"),
              "purpose": purpose, "frozen_report_only": frozen,
              "selection_mode": {"boxes_only": boxes_only, "method": method if boxes_only else "full"},
              "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path), "images": len(samples)},
              "index": {"path": str(index_path.resolve()), "sha256": _sha256(index_path)},
              "detector": None if detector is None else {"path": str(detector_path.resolve()), "sha256": _sha256(detector_path)},
              "v1": None if v1 is None else {"path": str(v1_path.resolve()), "sha256": _sha256(v1_path)},
              "reader": None if reader is None else {"path": str(reader_path.resolve()), "sha256": _sha256(reader_path), "version": reader.version},
              "groups": report_groups,
              "np_false_positives": np_metrics, "detector_latency_ms": {
                  name: {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
                         "p95": float(np.percentile(vals, 95)) if vals else None}
                  for name, vals in latencies.items()},
              "protocol": {"topk": [1, 3], "iou_thresholds": list(IOU_THRESHOLDS),
                           "corner_error_population": "all returned top-3 candidates, paired by rank; also aggregated only where bbox IoU >= .5",
                           "box_crop_pad": [0.15, 0.25], "threads": threads,
                           "reader_route": None if boxes_only else "all crop_box outputs in detector order; first valid OK wins, otherwise best status then confidence",
                           "crnn_box_adapter": None if boxes_only else "plate_tensor preprocessing, existing CrnnReader model/temperature, finite-logit guard, and _greedy_decode; bypasses only the legacy canonical-shape guard"}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _read_box_route(image, candidates, reader, crop_box_fn, decode_fn, min_confidence):
    """Read each expanded crop in detector order; stop at first valid OK."""
    status_rank = STATUS_RANK
    best = None
    for candidate in candidates:
        crop = crop_box_fn(image, candidate.bbox, 0.15, 0.25)
        if crop.size == 0:
            continue
        text, confs = reader.read(crop)
        decoded = decode_fn(text, confs, candidate.color, min_confidence)
        if decoded is None:
            continue
        status = decoded.status.value
        rank = (status_rank.get(status, 0), decoded.confidence)
        if best is None or rank > best[0]:
            best = (rank, decoded.plate_no, status)
        if status == "OK":
            return decoded.plate_no, status
    return (best[1], best[2]) if best else ("", "NO_READ")


def _evaluate_np(manifest_path, detector, cfg, cv2, np):
    """Count false positives from a path-only no-plate CSV manifest."""
    import csv
    paths = []
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("NP manifest must contain a path column")
        for raw in reader:
            path = Path(raw["path"].strip())
            paths.append(path if path.is_absolute() else Path(manifest_path).resolve().parent / path)
    if len(paths) != 601:
        raise ValueError(f"NP manifest must contain exactly 601 rows; got {len(paths)}")
    if len(set(map(str, paths))) != len(paths):
        raise ValueError("NP manifest contains duplicate paths")
    counts = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode NP image: {path}")
        counts.append({"candidate_count": len(detector.detect(image, cfg.runtime.max_candidates))})
    return summarize_np(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ALGORITHM_DIR / "data/index/ccpd_index.csv")
    parser.add_argument("--detector", type=Path, default=Path("algorithm/models/plate_det_v1.pt"))
    parser.add_argument("--v1", type=Path, default=Path("algorithm/models/corner_net_v1.pt"))
    parser.add_argument("--reader", type=Path)
    parser.add_argument("--np-manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("holdout", "val", "devhard"), default="holdout")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--boxes-only", action="store_true",
                        help="geometry report; val/devhard require --frozen and are report-only")
    parser.add_argument("--method", choices=("new", "v1-fused"), default="new",
                        help="method for --boxes-only")
    parser.add_argument("--scale-suite", action="store_true",
                        help="run all camera scales; uses only manifest/index/detector/output/config/threads")
    parser.add_argument("--scales", type=float, nargs="+", default=list(CAMERA_SCALES))
    args = parser.parse_args(argv)
    if args.scale_suite:
        report = evaluate_scale_suite(args.manifest, args.index, args.detector, args.output,
                                      config_path=args.config, scales=args.scales,
                                      threads=args.threads)
        print(json.dumps({"output": str(args.output), "scales": report["scales"],
                          "memory": report["memory"]}, ensure_ascii=False))
        return 0
    report = evaluate(args.manifest, args.index, args.detector, args.reader, args.output,
                      v1_path=args.v1, np_manifest=args.np_manifest, purpose=args.purpose,
                      frozen=args.frozen, threads=args.threads, config_path=args.config,
                      boxes_only=args.boxes_only, method=args.method)
    print(json.dumps({"output": str(args.output), "groups": report["groups"],
                      "np_false_positives": report["np_false_positives"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
