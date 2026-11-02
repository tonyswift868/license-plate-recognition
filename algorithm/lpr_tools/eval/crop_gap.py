"""Diagnose crop and candidate selection gaps on fixed development hard cases.

The primary buckets describe observed per-image rescue opportunities, not
causal attribution. Independent rescue flags retain overlapping hypotheses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from lpr.config import ALGORITHM_DIR, load_config
from lpr.locate import locate
from lpr.pipeline import (
    _decode_text,
    _in_roi,
    _layouts_to_try,
    _merge_candidates,
    _rank,
    _rectifications,
)
from lpr.postprocess import decode
from lpr.preprocess import preprocess
from lpr.segment import segment
from lpr.types import PlateCandidate, RecogStatus, RectifiedPlate

from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path

from .common import effective_config_info, model_identity
from .index_samples import FIXED_HARD_MANIFEST, fixed_manifest_samples, path_key

DEFAULT_INDEX = ALGORITHM_DIR / "data/index/ccpd_index.csv"
TARGET_SUBSETS = ("ccpd_db", "ccpd_tilt", "ccpd_rotate")
IOU_THRESHOLD = 0.5
BUCKET_PRIORITY = ("decode_error", "unsupported", "alternate_candidate", "same_candidate_route_selection",
                   "crop_expansion", "gt_geometry_rescue", "reader_or_segment_residual")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_index(index_path, selected_paths):
    lookup = {}
    with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "subset", "brightness", "bbox", "quad", "plate_no"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CCPD index missing crop-gap fields")
        for row in reader:
            p = Path(row["path"])
            if not p.is_absolute():
                p = Path(index_path).resolve().parent / p
            key = path_key(p)
            if key in selected_paths:
                if key in lookup:
                    raise ValueError(f"duplicate selected CCPD index path: {p}")
                lookup[key] = row
    if set(lookup) != selected_paths:
        raise ValueError(f"selected paths absent from CCPD index: {len(selected_paths-set(lookup))}")
    return lookup


def _iou(box_a, box_b):
    x = max(0., min(box_a[0]+box_a[2], box_b[0]+box_b[2])-max(box_a[0], box_b[0]))
    y = max(0., min(box_a[1]+box_a[3], box_b[1]+box_b[3])-max(box_a[1], box_b[1]))
    intersection = x*y
    union = box_a[2]*box_a[3]+box_b[2]*box_b[3]-intersection
    return intersection/union if union > 0 else 0.


def _gt_rect(image, row, cfg):
    if row.quad is None:
        raise ValueError(f"GT quad missing: {row.path}")
    layout = len(row.plate_no)
    width, height = cfg.rectify.size_7 if layout == 7 else cfg.rectify.size_8
    source = np.asarray(row.quad, np.float32)
    if abs(cv2.contourArea(source)) < 1:
        raise ValueError(f"GT quad is degenerate: {row.path}")
    target = np.asarray([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)], np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    x, y, w, h = cv2.boundingRect(source)
    candidate = PlateCandidate(list(row.quad), (x, y, w, h), row.plate_color, 1., "ground_truth")
    return RectifiedPlate(warped, layout, row.plate_color, matrix, candidate)


def _actual_candidates(image, cfg, recognizer):
    pre = preprocess(image, cfg.preprocess, cfg.lane_roi(None))
    classical = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    net = []
    if recognizer.corner_detector is not None:
        net = [candidate for candidate in recognizer.corner_detector.detect(image, cfg.runtime.max_candidates)
               if _in_roi(candidate, cfg.lane_roi(None))]
    merged = _merge_candidates(net, classical, cfg.runtime.max_candidates)
    return merged[:3]


def _attempt_payload(decision, truth):
    if decision is None:
        return {"status": None, "plate_no": "", "confidence": None, "correct": False}
    return {"status": decision.status.value, "plate_no": decision.plate_no,
            "confidence": float(decision.confidence), "correct": decision.plate_no == truth}


def _segment_route(rect, truth, recognizer):
    best = None
    attempts = []
    for layout in _layouts_to_try(rect):
        segmented = segment(rect, layout, recognizer.config.segment)
        if not segmented.ok:
            attempts.append({"layout": layout, "status": "SEG_FAILED", "plate_no": "",
                             "confidence": None, "correct": False})
            continue
        decision = decode(recognizer.classifier.predict_proba(segmented.chars), segmented.layout,
                          rect.color, recognizer.config.postprocess)
        attempts.append({"layout": layout, **_attempt_payload(decision, truth)})
        if best is None or _rank(decision) > _rank(best):
            best = decision
        if decision.status == RecogStatus.OK:
            return {"selected": _attempt_payload(decision, truth), "attempts": attempts}
    return {"selected": _attempt_payload(best, truth), "attempts": attempts}


def _reader_route(rect, truth, recognizer):
    text, confidences = recognizer.plate_reader.read(rect.image)
    decision = _decode_text(text, confidences, rect.color, recognizer.config.postprocess.min_confidence)
    return _attempt_payload(decision, truth)


def _routes_for_rect(rect, truth, recognizer):
    return {"segment_cnn": _segment_route(rect, truth, recognizer),
            "crnn": _reader_route(rect, truth, recognizer)}


def _bbox_crop(image, bbox, expand, cfg, color, truth_layout):
    x, y, width, height = bbox
    dx, dy = width*expand, height*expand
    left = max(0, int(np.floor(x-dx)))
    top = max(0, int(np.floor(y-dy)))
    right = min(image.shape[1], int(np.ceil(x+width+dx)))
    bottom = min(image.shape[0], int(np.ceil(y+height+dy)))
    if right <= left or bottom <= top:
        raise ValueError("expanded candidate bbox produced an empty crop")
    crop = image[top:bottom, left:right]
    target_w, target_h = cfg.rectify.size_7 if truth_layout == 7 else cfg.rectify.size_8
    scaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    points = [(float(left), float(top)), (float(right-1), float(top)),
              (float(right-1), float(bottom-1)), (float(left), float(bottom-1))]
    candidate = PlateCandidate(points, (left, top, right-left, bottom-top), color, 1., f"bbox_expand_{expand:.2f}")
    return RectifiedPlate(scaled, truth_layout, color, np.eye(3, dtype=np.float64), candidate)


def _crop_preview(image, max_side=400):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros((40, 160, 3), np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(1., max_side/max(h, w))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                           interpolation=cv2.INTER_AREA)
    return image


def _make_thumbnail(sample, output_dir, final_crop, gt_crop, examples):
    pieces = [("FINAL", final_crop), ("GT", gt_crop)]
    if examples.get("best_alternative_crop") is not None:
        pieces.append(("ALT", examples["best_alternative_crop"]))
    panel_width = 400 // len(pieces) if len(pieces) == 3 else 195
    previews = []
    for label, crop in pieces:
        previews.append((label, _crop_preview(crop, panel_width)))
    height = max(image.shape[0] for _, image in previews) + 16
    canvas = np.zeros((height, panel_width*len(previews), 3), np.uint8)
    for index, (label, image) in enumerate(previews):
        left = index * panel_width
        cv2.putText(canvas, label, (left+4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (0, 0, 255), 1, cv2.LINE_AA)
        canvas[16:16+image.shape[0], left:left+image.shape[1]] = image
    return canvas


def classify_primary_bucket(*, decode_error=False, unsupported=False, other_candidate=False,
                            same_candidate_route=False, expanded=False, gt_correct=False):
    """Mutually exclusive priority; independent rescue flags are reported separately."""
    conditions = (decode_error, unsupported, other_candidate, same_candidate_route,
                  expanded, gt_correct)
    for name, matches in zip(BUCKET_PRIORITY, conditions, strict=False):
        if matches:
            return name
    return "reader_or_segment_residual"


def _ceiling_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("ceiling report has no records list")
    return data, {path_key(row["path"]): row for row in records}


def _load_ceiling_cache(path, manifest_identity, samples, cfg):
    if path is None:
        return None, {}
    data, records = _ceiling_records(path)
    if data.get("manifests", {}).get("devhard", {}).get("sha256") != manifest_identity["manifest_sha256"]:
        raise ValueError("ceiling cache devhard SHA differs; omit --ceiling-report to compute current Top-3")
    archived = data.get("config_and_models", {}).get("effective_config")
    if not isinstance(archived, dict):
        raise ValueError("ceiling cache has no effective config; omit --ceiling-report to compute current Top-3")
    current_locator = _locator_config(cfg.model_dump(mode="json"))
    cached_locator = _locator_config(archived)
    if current_locator != cached_locator:
        raise ValueError("ceiling cache locator/config differs; omit --ceiling-report to compute current Top-3")
    corner_sha = model_identity(cfg, "corner_model_path")["sha256"]
    cached_corner_sha = data.get("config_and_models", {}).get("models", {}).get("corner", {}).get("sha256")
    if cached_corner_sha != corner_sha:
        raise ValueError("ceiling cache corner model differs; omit --ceiling-report to compute current Top-3")
    for sample in samples:
        prior = records.get(path_key(sample.row.path))
        if prior is None or prior.get("truth") != sample.row.plate_no or prior.get("subset") != sample.subset:
            raise ValueError(f"ceiling cache path/truth/subset mismatch: {sample.row.path}")
    return data, records


def _locator_config(config):
    runtime = config.get("runtime", {})
    return {"preprocess": config.get("preprocess"), "locate": config.get("locate"),
            "max_candidates": runtime.get("max_candidates")}


def _summarize(rows):
    result = {"n_final_wrong_top3_hit": len(rows), "buckets": {}, "potential_rescues": {}}
    details = [row.get("diagnostic") or row for row in rows]
    bucket_counts = Counter(item.get("primary_bucket", "decode_error") for item in details)
    result["buckets"] = {name: {"n": bucket_counts.get(name, 0),
                                      "share": bucket_counts.get(name, 0)/len(rows) if rows else None}
                         for name in BUCKET_PRIORITY}
    for rescue in ("alternate_candidate", "same_candidate_route_selection", "crop_expansion", "gt_geometry"):
        rescued = [row for row, item in zip(rows, details, strict=False)
                   if (item.get("potential_rescues") or {}).get(rescue)]
        result["potential_rescues"][rescue] = {"n": len(rescued),
                                                "examples": [{"subset": row["subset"], "path": row["path"]}
                                                             for row in rescued[:5]]}
    return result


def run(*, manifest, index, config, output, ceiling_report=None, cnn_model=None,
        corner_model=None, crnn_model=None, torch_threads=1, overwrite=False,
        max_thumbnails=8):
    if torch_threads not in (1, 2):
        raise ValueError("torch_threads must be 1 or 2")
    if not 0 <= max_thumbnails <= 8:
        raise ValueError("max_thumbnails must be between 0 and 8")
    out = Path(output)
    trace_path = out / "crop_gap.jsonl"
    if not overwrite and (out.exists() and any(out.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty crop-gap output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    overrides = {f"runtime__{field}": value for field, value in (
        ("model_path", cnn_model), ("corner_model_path", corner_model),
        ("plate_reader_model_path", crnn_model)) if value is not None}
    overrides["runtime__torch_threads"] = torch_threads
    overrides["postprocess__arbitration"] = "first_ok"
    cfg = load_config(config, **overrides)
    if not cfg.runtime.plate_reader_model_path:
        raise ValueError("crop-gap requires an enabled CRNN reader")

    samples, manifest_identity = fixed_manifest_samples(index, manifest, purpose="dev-hard")
    samples = [sample for sample in samples if sample.subset in TARGET_SUBSETS]
    counts = Counter(sample.subset for sample in samples)
    if counts != Counter({"ccpd_db": 300, "ccpd_tilt": 300, "ccpd_rotate": 300}):
        raise ValueError(f"required fixed hard groups must each have 300 rows; got {dict(counts)}")
    selected_paths = {path_key(sample.row.path) for sample in samples}
    index_rows = _read_index(index, selected_paths)

    ceiling_data, ceiling_by_path = _load_ceiling_cache(ceiling_report, manifest_identity, samples, cfg)

    import torch
    recognizer = __import__("lpr.pipeline", fromlist=["PlateRecognizer"]).PlateRecognizer.from_config(
        config, **overrides)
    torch.set_num_threads(torch_threads)
    cv2.setNumThreads(1)
    active_models = {"cnn": model_identity(cfg, "model_path"),
                     "corner": model_identity(cfg, "corner_model_path"),
                     "crnn": model_identity(cfg, "plate_reader_model_path")}
    source_digest = sha256_file(Path(__file__))
    metadata = {
        "status": "running", "config": str(Path(config).resolve()),
        "config_and_models": effective_config_info(cfg, config), "models": active_models,
        "tool_source_sha256": source_digest,
        "runtime_threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads()},
        "fixed_manifest": manifest_identity,
        "index": {"path": str(Path(index).resolve()), "sha256": sha256_file(index)},
        "top3_source": "validated_ceiling_cache" if ceiling_report is not None else "computed_from_current_locator",
        "architecture_ceiling": ({"path": str(Path(ceiling_report).resolve()),
                                  "sha256": sha256_file(ceiling_report),
                                  "effective_config_sha256": ceiling_data.get("config_and_models", {}).get("effective_config_sha256")}
                                 if ceiling_report is not None else None),
        "subset_counts": dict(counts), "top3_iou_threshold": IOU_THRESHOLD,
        "primary_bucket_priority": list(BUCKET_PRIORITY), "records": 0,
    }
    meta_path = out / "run_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path.write_text("", encoding="utf-8")
    wrong_rows = []
    decode_errors = []
    thumbnail_counts = Counter()
    thumbnail_index = []
    for i, sample in enumerate(samples, 1):
        row = sample.row
        key = path_key(row.path)
        arch = ceiling_by_path.get(key) if ceiling_by_path else None
        source_path = image_path(row, manifest)
        image, error = decode_image(source_path)
        if error:
            image, error = decode_image(source_path)
        raw_index = index_rows[key]
        candidates = _actual_candidates(image, cfg, recognizer) if image is not None and arch is None else None
        truth_bbox = tuple(float(v) for v in raw_index["bbox"].split())
        if arch is None:
            best_iou = max((_iou(candidate.bbox, truth_bbox) for candidate in candidates), default=0.) if candidates is not None else 0.
            top3_hit = best_iou >= IOU_THRESHOLD
        else:
            best_iou = arch.get("top3_best_iou")
            top3_hit = bool(arch["top3_hit"])
        final_result = recognizer.recognize(image) if image is not None else None
        final_wrong = (final_result is None or final_result.plate_no != row.plate_no)
        record = {"row": i, "path": row.path, "truth": row.plate_no, "subset": sample.subset,
                  "brightness": int(raw_index["brightness"]) if raw_index["brightness"].isdigit() else None,
                  "top3_hit": top3_hit, "top3_best_iou": best_iou,
                  "final_status": final_result.status.value if final_result is not None else None,
                  "final_prediction": final_result.plate_no if final_result is not None else "",
                  "final_confidence": float(final_result.confidence) if final_result is not None else None,
                  "final_bbox": list(final_result.bbox) if final_result is not None and final_result.bbox else None,
                  "decode_error": error, "diagnostic": None}
        if error:
            decode_errors.append({"path": row.path, "error": error})
        if final_wrong and top3_hit:
            if image is None:
                record["diagnostic"] = {"primary_bucket": "decode_error", "potential_rescues": {},
                                         "candidates": [], "expanded_crops": [],
                                         "final_crop": {"prediction": "", "correct": False},
                                         "ground_truth_crop": {"prediction": "", "correct": False}}
                wrong_rows.append(record)
            else:
                diagnosis = _diagnose_one(image, row, final_result, cfg, recognizer,
                                          truth_bbox=tuple(float(v) for v in raw_index["bbox"].split()),
                                          output_dir=out / "crops",
                                          sample_prefix=hashlib.sha256(row.path.encode("utf-8")).hexdigest()[:12],
                                          expected_best_iou=best_iou, candidates=candidates)
                record["diagnostic"] = diagnosis
                wrong_rows.append(record)
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace.flush()
            os.fsync(trace.fileno())
        metadata["records"] = i
        if i % 25 == 0:
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Output no more than the requested number of image examples per exclusive bucket.
    for record in wrong_rows:
        diagnostic = record.get("diagnostic") or {}
        bucket = diagnostic.get("primary_bucket", "decode_error")
        if thumbnail_counts[bucket] >= max_thumbnails or record.get("decode_error"):
            continue
        image, err = decode_image(image_path(record_to_manifest_row(record, samples), manifest))
        if err or image is None:
            continue
        final_crop = _load_preview_crop(diagnostic.get("final_crop_path"))
        gt_crop = _load_preview_crop(diagnostic.get("ground_truth_crop_path"))
        alt_crop = _load_preview_crop(diagnostic.get("best_alternative_crop_path"))
        thumb = _make_thumbnail(record, out, final_crop, gt_crop,
                                {"best_alternative_crop": alt_crop})
        name = f"bucket-{bucket}_{thumbnail_counts[bucket]+1:02d}.jpg"
        cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out / name))
        thumbnail_index.append({"bucket": bucket, "path": name, "source": record["path"],
                                "truth": record["truth"], "prediction": record["final_prediction"]})
        thumbnail_counts[bucket] += 1

    summaries = {"all": _summarize(wrong_rows)}
    for subset in (*TARGET_SUBSETS, "ccpd_db_brightness_le_19", "ccpd_tilt_plus_rotate"):
        selected = [record for record in wrong_rows
                    if (record["subset"] == subset
                        or (subset == "ccpd_db_brightness_le_19" and record["subset"] == "ccpd_db"
                            and record["brightness"] is not None and record["brightness"] <= 19)
                        or (subset == "ccpd_tilt_plus_rotate" and record["subset"] in {"ccpd_tilt", "ccpd_rotate"}))]
        summaries[subset] = _summarize(selected)
    metadata["status"] = "complete" if not decode_errors else "decode_errors"
    metadata["finished_records"] = len(samples)
    report = {"status": metadata["status"], "method": metadata,
              "final_wrong_top3_hit": len(wrong_rows), "decode_error_count": len(decode_errors),
              "decode_errors": decode_errors, "summaries": summaries,
              "thumbnails": thumbnail_index, "thumbnail_counts": dict(thumbnail_counts),
              "trace": str(trace_path.resolve()), "priority": list(BUCKET_PRIORITY),
              "bucket_semantics": {
                  "alternate_candidate": "some non-final top3 candidate/any rectification/read route emits exact truth",
                  "crop_expansion": "no alternate-candidate rescue, but bbox expanded 0/10/20% without perspective emits exact truth",
                  "same_candidate_route_selection": "same final crop via a different route or another rectification of the selected candidate emits exact truth",
                  "gt_geometry_rescue": "no candidate or expanded rescue, but GT quad crop emits exact truth",
                  "reader_or_segment_residual": "none of the tested crops emits exact truth; this does not prove source pixels are unreadable"}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_preview_crop(path):
    if not path:
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def record_to_manifest_row(record, samples,):
    key = path_key(record["path"])
    return next(sample.row for sample in samples if path_key(sample.row.path) == key)


def _diagnose_one(image, row, final_result, cfg, recognizer, *, truth_bbox, output_dir,
                  sample_prefix, expected_best_iou=None, candidates=None):
    if candidates is None:
        candidates = _actual_candidates(image, cfg, recognizer)
    observed_best_iou = max((_iou(candidate.bbox, truth_bbox) for candidate in candidates), default=0.)
    if observed_best_iou < IOU_THRESHOLD:
        raise ValueError(f"recomputed Top-3 no longer hits the ceiling report label for {row.path}")
    if (expected_best_iou is not None
            and abs(observed_best_iou-float(expected_best_iou)) > 1e-9):
        raise ValueError(f"recomputed Top-3 IoU differs from ceiling report for {row.path}")
    candidate_evidence = []
    used_rects = []
    for candidate_index, candidate in enumerate(candidates[:3], 1):
        iou = _iou(candidate.bbox, truth_bbox)
        rects = _rectifications(image, candidate, cfg.rectify)
        rect_evidence = []
        for rect_index, rect in enumerate(rects, 1):
            routes = _routes_for_rect(rect, row.plate_no, recognizer)
            crop_path = None
            if _route_correct(routes):
                crop_path = _save_crop(rect.image, output_dir,
                                       f"{sample_prefix}-candidate-{candidate_index}-rect-{rect_index}")
            selected_pipeline_crop = (final_result.plate_image is not None
                                      and np.array_equal(rect.image, final_result.plate_image))
            item = {"rectification": rect_index, "selected_pipeline_crop": selected_pipeline_crop,
                    "routes": routes,
                    "crop_path": str(crop_path) if crop_path else None}
            rect_evidence.append(item)
            used_rects.append((candidate_index, rect_index, rect, routes, crop_path))
        candidate_evidence.append({"index": candidate_index, "bbox": list(candidate.bbox),
                                   "source": candidate.source, "score": float(candidate.score),
                                   "bbox_iou": iou, "hit": iou >= IOU_THRESHOLD,
                                   "rectifications": rect_evidence})

    final_bbox = tuple(final_result.bbox) if final_result and final_result.bbox else None
    final_candidate_index = next((index for index, candidate in enumerate(candidates[:3], 1)
                                  if tuple(candidate.bbox) == final_bbox), None)
    final_rect_crop = final_result.plate_image if final_result is not None else None
    final_rect = RectifiedPlate(final_rect_crop, final_result.layout or len(row.plate_no),
                                final_result.plate_color, np.eye(3, dtype=np.float64),
                                PlateCandidate(final_result.corners or [], final_result.bbox or (0,0,0,0),
                                               final_result.plate_color, 1., "pipeline_final")) if final_rect_crop is not None else None
    final_routes = _routes_for_rect(final_rect, row.plate_no, recognizer) if final_rect is not None else None
    gt_rect = _gt_rect(image, row, cfg)
    gt_routes = _routes_for_rect(gt_rect, row.plate_no, recognizer)
    matched_candidates = [(candidate_index, candidate)
                          for candidate_index, candidate in enumerate(candidates[:3], 1)
                          if _iou(candidate.bbox, truth_bbox) >= IOU_THRESHOLD]
    expanded_evidence = []
    for candidate_index, candidate in matched_candidates:
        rects = _rectifications(image, candidate, cfg.rectify)
        layout = next((rect.layout for rect in rects if rect.layout in (7, 8)), len(row.plate_no))
        for fraction in (0., .10, .20):
            crop = _bbox_crop(image, candidate.bbox, fraction, cfg, candidate.color, layout)
            routes = _routes_for_rect(crop, row.plate_no, recognizer)
            path = _save_crop(crop.image, output_dir,
                              f"{sample_prefix}-matched-{candidate_index}-bbox-expand-{int(fraction*100)}")
            expanded_evidence.append({"candidate_index": candidate_index, "expand": fraction,
                                      "routes": routes, "crop_path": str(path)})

    other_candidate_correct = any(
        candidate["index"] != final_candidate_index
        and any(rect["routes"][route]["selected"]["correct"] if route == "segment_cnn"
                else rect["routes"][route]["correct"]
                for rect in candidate["rectifications"] for route in ("segment_cnn", "crnn"))
        for candidate in candidate_evidence)
    same_candidate_rect_correct = _same_candidate_alt_rect_rescue(candidate_evidence, final_candidate_index)
    expanded_correct = any(_route_correct(item["routes"]) for item in expanded_evidence)
    gt_correct = _route_correct(gt_routes)
    same_candidate_route = _route_correct(final_routes) or same_candidate_rect_correct
    unsupported = final_result.status == RecogStatus.UNSUPPORTED
    primary = classify_primary_bucket(unsupported=unsupported, other_candidate=other_candidate_correct,
                                      same_candidate_route=same_candidate_route, expanded=expanded_correct,
                                      gt_correct=gt_correct)
    best_alt_path = next((rect["crop_path"] for candidate in candidate_evidence
                          for rect in candidate["rectifications"]
                          if candidate["index"] != final_candidate_index and rect["crop_path"]), None)
    return {"primary_bucket": primary,
            "final_crop": {"status": final_result.status.value, "prediction": final_result.plate_no or "",
                           "confidence": float(final_result.confidence), "bbox": list(final_result.bbox or ())},
            "final_crop_routes": final_routes,
            "final_crop_path": str(_save_crop(final_rect_crop, output_dir, f"{sample_prefix}-final"))
                               if final_rect_crop is not None else None,
            "ground_truth_crop": gt_routes,
            "ground_truth_crop_path": str(_save_crop(gt_rect.image, output_dir, f"{sample_prefix}-gt")),
            "candidates": candidate_evidence, "expanded_crops": expanded_evidence,
            "potential_rescues": {"alternate_candidate": other_candidate_correct,
                                  "crop_expansion": expanded_correct,
                                  "same_candidate_route_selection": same_candidate_route,
                                  "gt_geometry": gt_correct},
            "unsupported": unsupported,
            "best_alternative_crop_path": best_alt_path}


def _route_correct(routes):
    if routes is None:
        return False
    return any(routes[route]["selected"]["correct"] if route == "segment_cnn"
               else routes[route]["correct"] for route in ("segment_cnn", "crnn"))


def _same_candidate_alt_rect_rescue(candidate_evidence, final_candidate_index):
    if final_candidate_index is None:
        return False
    return any(candidate["index"] == final_candidate_index
               and not rect.get("selected_pipeline_crop", False)
               and _route_correct(rect["routes"])
               for candidate in candidate_evidence for rect in candidate["rectifications"])


def _save_crop(image, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"failed to save diagnostic crop: {path}")
    encoded.tofile(str(path))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(FIXED_HARD_MANIFEST))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--ceiling-report", help="optional compatible Top-3 cache; omitted computes candidates with the current locator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cnn-model")
    parser.add_argument("--corner-model")
    parser.add_argument("--crnn-model")
    parser.add_argument("--threads", choices=(1, 2), type=int, default=1)
    parser.add_argument("--out", default="algorithm/out/receipts/第十批/crop-gap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-thumbnails", type=int, default=8)
    args = parser.parse_args(argv)
    report = run(manifest=args.manifest, index=args.index, ceiling_report=args.ceiling_report,
                 config=args.config, cnn_model=args.cnn_model, corner_model=args.corner_model,
                 crnn_model=args.crnn_model, torch_threads=args.threads, output=args.out,
                 overwrite=args.overwrite, max_thumbnails=args.max_thumbnails)
    print(f"report={Path(args.out) / 'report.json'}")
    return 2 if report["decode_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
