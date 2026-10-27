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
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

IOU_THRESHOLDS = (0.5, 0.7)
HARD_SUBSETS = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
                "ccpd_fn", "ccpd_challenge", "ccpd_weather")
STATUS_RANK = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}


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
    """Run final paired evaluation or low-load holdout-only box selection."""
    validate_purpose(purpose, frozen)
    if boxes_only and purpose != "holdout":
        raise ValueError("--boxes-only is restricted to the independent holdout")
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
    from lpr.pipeline import _decode_text, _in_roi, _merge_candidates
    from lpr.preprocess import preprocess
    from lpr.locate import locate
    from lpr_tools.eval_refine import _load_samples
    from lpr_tools.eval.index_samples import path_key

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
            from lpr.pipeline import _in_roi, _merge_candidates
            from lpr.preprocess import preprocess
            from lpr.locate import locate
            old = detect_one(v1, image, "corner_net_v1")
            pre = preprocess(image, cfg.preprocess)
            classic = locate(pre, cfg.locate, cfg.runtime.max_candidates)
            old_roi = [c for c in old if _in_roi(c, cfg.lane_roi(None))]
            candidates_by_method = {"classic_plus_v1": _merge_candidates(
                old_roi, classic, cfg.runtime.max_candidates)}
        else:
            from lpr.pipeline import _in_roi, _merge_candidates
            from lpr.preprocess import preprocess
            from lpr.locate import locate
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
                from lpr.pipeline import _decode_text
                from lpr.boxcrop import crop_box
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
    report = {"method": (f"holdout-only box selection: {method}" if boxes_only else
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
    parser.add_argument("--index", type=Path, default=Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv"))
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
                        help="holdout-only geometry selection; skips CRNN and other baselines")
    parser.add_argument("--method", choices=("new", "v1-fused"), default="new",
                        help="method for --boxes-only")
    args = parser.parse_args(argv)
    report = evaluate(args.manifest, args.index, args.detector, args.reader, args.output,
                      v1_path=args.v1, np_manifest=args.np_manifest, purpose=args.purpose,
                      frozen=args.frozen, threads=args.threads, config_path=args.config,
                      boxes_only=args.boxes_only, method=args.method)
    print(json.dumps({"output": str(args.output), "groups": report["groups"],
                      "np_false_positives": report["np_false_positives"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
