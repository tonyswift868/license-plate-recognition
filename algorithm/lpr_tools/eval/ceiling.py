"""Paired recognition ceilings for ground-truth and operational localization.

Each cell isolates one recognition route on one rectification source. It is a
diagnostic upper bound, not a counterfactual causal decomposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from lpr.config import load_config
from lpr.locate import locate
from lpr.pipeline import _decode_text, _in_roi, _layouts_to_try, _merge_candidates, _rectifications
from lpr.postprocess import decode
from lpr.preprocess import preprocess
from lpr.rectify import RectifiedPlate
from lpr.segment import segment
from lpr.types import PlateCandidate
from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path
from .common import effective_config_info
from .index_samples import (FIXED_HARD_MANIFEST, FIXED_VAL_MANIFEST, FIXED_HARD_N,
                            FIXED_VAL_N, fixed_manifest_samples, path_key)

INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
IOU_MIN = 0.5


def _iou(a, b):
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = x*y
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.


def _gt_rect(image, row, cfg):
    if row.quad is None:
        raise ValueError(f"missing GT quad: {row.path}")
    layout = len(row.plate_no)
    w, h = cfg.rectify.size_7 if layout == 7 else cfg.rectify.size_8
    source = np.asarray(row.quad, np.float32)
    if abs(cv2.contourArea(source)) < 1:
        raise ValueError(f"degenerate GT quad: {row.path}")
    target = np.asarray([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)], np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    plate = cv2.warpPerspective(image, matrix, (w, h))
    x, y, bw, bh = cv2.boundingRect(source)
    candidate = PlateCandidate(list(row.quad), (x, y, bw, bh), row.plate_color, 1., "ground_truth")
    return RectifiedPlate(plate, layout, row.plate_color, matrix, candidate)


def _actual_rects(image, cfg, recognizer):
    pre = preprocess(image, cfg.preprocess)
    candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    if recognizer.corner_detector is not None:
        net = [c for c in recognizer.corner_detector.detect(image, cfg.runtime.max_candidates)
               if _in_roi(c, cfg.lane_roi(None))]
        candidates = _merge_candidates(net, candidates, cfg.runtime.max_candidates)
    return [rect for candidate in candidates[:cfg.runtime.max_candidates]
            for rect in _rectifications(image, candidate, cfg.rectify)], candidates


def _segment_read(rects, row, recognizer):
    best = None
    ranks = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}
    for rect in rects:
        for layout in _layouts_to_try(rect):
            seg = segment(rect, layout, recognizer.config.segment)
            if not seg.ok:
                continue
            dec = decode(recognizer.classifier.predict_proba(seg.chars), seg.layout,
                         rect.color, recognizer.config.postprocess)
            if best is None or (ranks[dec.status.value], dec.confidence) > (ranks[best.status.value], best.confidence):
                best = dec
            if dec.status.value == "OK":
                return dec.plate_no, dec.plate_no == row.plate_no
    return (best.plate_no, best.plate_no == row.plate_no) if best is not None else ("", False)


def _reader_read(rects, row, recognizer):
    if recognizer.plate_reader is None:
        raise ValueError("CRNN is not enabled by config")
    best = None
    ranks = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}
    for rect in rects:
        text, confs = recognizer.plate_reader.read(rect.image)
        if len(text) in (7, 8) and len(confs) == len(text):
            dec = _decode_text(text, confs, rect.color, recognizer.config.postprocess.min_confidence)
            if dec is not None and (best is None or (ranks[dec.status.value], dec.confidence) >
                                    (ranks[best.status.value], best.confidence)):
                best = dec
            if dec is not None and dec.status.value == "OK":
                return text, text == row.plate_no
    return (best.plate_no, best.plate_no == row.plate_no) if best is not None else ("", False)


def _load_index_fields(index_path, needed):
    lookup = {}
    with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"path", "brightness", "bbox", "quad"}.issubset(reader.fieldnames):
            raise ValueError("CCPD index missing required metadata columns")
        for raw in reader:
            p = Path(raw["path"])
            if not p.is_absolute():
                p = Path(index_path).resolve().parent / p
            key = path_key(p)
            if key in needed:
                if key in lookup:
                    raise ValueError(f"duplicate index path: {p}")
                lookup[key] = raw
    if set(lookup) != needed:
        raise ValueError(f"index rows missing for manifest: {len(needed-set(lookup))}")
    return lookup


def _summarize(records):
    cells = defaultdict(list)
    groups = defaultdict(list)
    for item in records:
        for localization in ("ground_truth", "actual"):
            for route in ("segment_cnn", "crnn"):
                cells[f"{localization}/{route}"].append(item["cells"][localization][route]["correct"])
        keys = [item["subset"], "all"]
        if (item["subset"] == "ccpd_db" and item["brightness"] is not None
                and item["brightness"] <= 19):
            keys.append("ccpd_db_brightness_le_19")
        if item["subset"] in {"ccpd_tilt", "ccpd_rotate"}:
            keys.append("ccpd_tilt_plus_rotate")
        for key in keys:
            groups[key].append(item)
    def pack(rows):
        out = {}
        for loc in ("ground_truth", "actual"):
            for route in ("segment_cnn", "crnn"):
                values = [r["cells"][loc][route]["correct"] for r in rows]
                out[f"{loc}/{route}"] = {"correct": sum(values), "n": len(values),
                                          "accuracy": sum(values)/len(values) if values else None}
        actual = rows
        out["actual_top3"] = {"hits": sum(r["top3_hit"] is True for r in actual), "n": len(actual),
                              "recall": sum(r["top3_hit"] is True for r in actual)/len(actual) if actual else None}
        # For each route, report observed operational misses split by whether
        # the fused Top-3 covered the truth bbox. The GT/actual accuracy gap
        # remains a paired diagnostic and is not a causal additive estimate.
        out["actual_error_buckets"] = {}
        for route in ("segment_cnn", "crnn"):
            errors = [r for r in actual if not r["cells"]["actual"][route]["correct"]]
            out["actual_error_buckets"][route] = {
                "top3_localization_miss": sum(r["top3_hit"] is False for r in errors),
                "top3_hit_but_route_wrong": sum(r["top3_hit"] is True for r in errors),
                "top3_undefined": sum(r["top3_hit"] is None for r in errors),
                "total_wrong": len(errors)}
        out["paired_gt_actual_net_gain"] = {}
        for route in ("segment_cnn", "crnn"):
            rescued = sum(r["cells"]["ground_truth"][route]["correct"] and
                          not r["cells"]["actual"][route]["correct"] for r in actual)
            gt_still_wrong = sum(not r["cells"]["ground_truth"][route]["correct"] for r in actual)
            out["paired_gt_actual_net_gain"][route] = {
                "gt_correct_actual_wrong": rescued,
                "gt_wrong_actual_correct": sum(not r["cells"]["ground_truth"][route]["correct"] and
                                                r["cells"]["actual"][route]["correct"] for r in actual),
                "gt_also_wrong": gt_still_wrong,
                "net_accuracy_delta_count": rescued - sum(
                    not r["cells"]["ground_truth"][route]["correct"] and
                    r["cells"]["actual"][route]["correct"] for r in actual)}
        return out
    return {k: pack(v) for k, v in sorted(groups.items())}


def _ensure_output_available(out_path, overwrite=False):
    out = Path(out_path)
    trace = out.with_suffix(".jsonl")
    if not overwrite and (out.exists() or trace.exists()):
        raise FileExistsError(f"refusing to overwrite existing ceiling output: {out}; choose a new --out")


def _runtime_overrides(overrides, threads):
    return {**overrides, "runtime__torch_threads": threads}


def run(manifests, index_path, config_path, out_path, *, cnn_model=None, corner_model=None,
        crnn_model=None, overwrite=False, threads=1):
    if threads < 1:
        raise ValueError("--threads must be positive")
    output = Path(out_path)
    jsonl = output.with_suffix(".jsonl")
    _ensure_output_available(output, overwrite)
    overrides = {f"runtime__{field}": value for field, value in (
        ("model_path", cnn_model), ("corner_model_path", corner_model),
        ("plate_reader_model_path", crnn_model)) if value is not None}
    effective_overrides = _runtime_overrides(overrides, threads)
    cfg = load_config(config_path, **effective_overrides)
    if not cfg.runtime.plate_reader_model_path:
        raise ValueError("config must enable runtime.plate_reader_model_path")
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    rows_by_key, rows_src = {}, {}
    for name, manifest in manifests.items():
        path = Path(manifest).resolve()
        expected = FIXED_VAL_N if name == "val" else FIXED_HARD_N
        samples, identity = fixed_manifest_samples(index_path, path,
                                                  purpose="val" if name == "val" else "dev-hard")
        rows = [sample.row for sample in samples]
        if len(rows) != expected:
            raise ValueError(f"{name} must have {expected} fixed rows; got {len(rows)}")
        subset_by_key = {}
        for sample in samples:
            row = sample.row
            key = path_key(image_path(row, path).resolve())
            if key in rows_by_key:
                raise ValueError(f"overlapping fixed manifest path: {key}")
            rows_by_key[key] = (name, path, row)
            subset_by_key[key] = sample.subset
        rows_src[name] = (path, rows, subset_by_key, identity)
    index = _load_index_fields(index_path, set(rows_by_key))
    recognizer = __import__("lpr.pipeline", fromlist=["PlateRecognizer"]).PlateRecognizer.from_config(
        config_path, **effective_overrides)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("", encoding="utf-8")
    records = []
    for key, (split, manifest, row) in rows_by_key.items():
        raw = index[key]
        image, error = decode_image(image_path(row, manifest))
        if error:
            image, error = decode_image(image_path(row, manifest))
        item = {"split": split, "path": row.path, "truth": row.plate_no,
                "subset": rows_src[split][2][key],
                "brightness": int(raw["brightness"]) if raw["brightness"].isdigit() else None,
                "decode_error": error, "top3_hit": None, "cells": {}}
        if error:
            item["cells"] = {loc: {route: {"prediction": "", "correct": False}
                                   for route in ("segment_cnn", "crnn")}
                              for loc in ("ground_truth", "actual")}
            records.append(item)
            with jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            continue
        gt = [_gt_rect(image, row, cfg)]
        actual, candidates = _actual_rects(image, cfg, recognizer)
        best_iou = max((_iou(c.bbox, tuple(float(v) for v in raw["bbox"].split())) for c in candidates), default=0.)
        item["top3_hit"] = best_iou >= IOU_MIN
        item["top3_best_iou"] = best_iou
        for label, rects in (("ground_truth", gt), ("actual", actual)):
            seg_prediction, seg_correct = _segment_read(rects, row, recognizer)
            crnn_prediction, crnn_correct = _reader_read(rects, row, recognizer)
            item["cells"][label] = {
                "segment_cnn": {"prediction": seg_prediction, "correct": seg_correct},
                "crnn": {"prediction": crnn_prediction, "correct": crnn_correct}}
        records.append(item)
        with jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"method": "2x2 paired operational ceiling; one GT quad or ordered actual fused candidates; per-route first-OK exact plate text; no per-sample best-route oracle",
              "ground_truth_scope": "Oracle uses annotated quad, color and true plate layout; it is an upper bound and the GT-versus-actual difference is not a causal localization-only effect.",
              "error_bucket_scope": "Top-3 miss indicates candidate coverage failure (IoU < 0.5). Top-3 hit with wrong text means coverage exists but may still include candidate ranking, crop, rectification, segmentation or recognition errors; it is not a pure recognizer error.",
              "iou_threshold": IOU_MIN, "manifests": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(rows), "validated_identity": identity} for name,(path,rows,_subs,identity) in rows_src.items()},
              "index": {"path": str(Path(index_path).resolve()), "sha256": hashlib.sha256(Path(index_path).read_bytes()).hexdigest()},
              "config_and_models": effective_config_info(cfg, config_path),
              "runtime_threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads(),
                                  "requested_torch": threads},
              "decode_error_count": sum(bool(r.get("decode_error")) for r in records),
              "decode_errors": [{"split": r["split"], "path": r["path"], "error": r["decode_error"]}
                                for r in records if r.get("decode_error")],
              "records_jsonl": str(jsonl), "groups": {}, "records": records}
    for split in manifests:
        split_records = [r for r in records if r["split"] == split]
        report["groups"][split] = _summarize(split_records)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default=str(FIXED_VAL_MANIFEST))
    parser.add_argument("--dev-hard", default=str(FIXED_HARD_MANIFEST))
    parser.add_argument("--index", default=str(INDEX))
    parser.add_argument("--config", required=True, help="temporary TOML selecting CNN, corner and CRNN model paths")
    parser.add_argument("--cnn-model", help="override runtime.model_path")
    parser.add_argument("--corner-model", help="override runtime.corner_model_path")
    parser.add_argument("--crnn-model", help="override runtime.plate_reader_model_path")
    parser.add_argument("--out", required=True, help="new report path; refuses existing report unless --overwrite")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output (including JSONL trace)")
    parser.add_argument("--threads", type=int, default=1, help="Torch intra-op threads (default: 1)")
    args = parser.parse_args(argv)
    report = run({"val": args.val, "devhard": args.dev_hard}, args.index, args.config, args.out,
        cnn_model=args.cnn_model, corner_model=args.corner_model, crnn_model=args.crnn_model,
        overwrite=args.overwrite, threads=args.threads)
    print(f"report={Path(args.out)}")
    return 2 if report["decode_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
