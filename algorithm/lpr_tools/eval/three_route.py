"""Evaluate the experimental S/R/B routes without changing production runtime.

Entry points
============
* **Join + exact scan:** join the paired E1 S/R trace and ``box_direct`` B
  trace, then exhaust the finite family of observed confidence boundaries.
* **Fresh:** run the fixed val/devhard manifests through all three routes,
  write a fresh trace, and automatically scan the finite family of observed
  confidence boundaries to completion (requires the selected box pad).
* **Live verify:** independently rerun all three models against a joined
  trace, compare route outputs and the final policy decision per image.

Commands below are PowerShell examples from ``D:\\keshe2\\workers\\03``;
set ``$env:PYTHONPATH='D:\\keshe2\\workers\\03\\algorithm'`` first and
use ``.\\.venv\\Scripts\\python.exe`` throughout. The E1 and
box-direct traces/reports must already exist for join+scan.

Tests::

    .\\.venv\\Scripts\\python.exe -m pytest algorithm/tests/test_three_route.py --basetemp=algorithm/out/receipts/第十一批/pytest-three-route

Join existing paired + box traces and scan::

    .\\.venv\\Scripts\\python.exe -m lpr_tools.eval.three_route --paired-trace algorithm/out/receipts/第十批/e1/trace.jsonl --paired-identity algorithm/out/receipts/第十批/e1/trace_identity.json --box-trace algorithm/out/receipts/第十一批/box-direct/trace.jsonl.gz --box-report algorithm/out/receipts/第十一批/box-direct/report.json --joined-trace algorithm/out/receipts/第十一批/three-route/trace.jsonl.gz --out algorithm/out/receipts/第十一批/three-route/scan.json

Live verify a chosen scan policy and compare final per-image decisions::

    .\\.venv\\Scripts\\python.exe -m lpr_tools.eval.three_route --live-verify algorithm/out/receipts/第十一批/three-route/trace.jsonl.gz --policy-report algorithm/out/receipts/第十一批/three-route/scan.json --config algorithm/config/default.toml --live-output algorithm/out/receipts/第十一批/three-route/live-verify.jsonl.gz --out algorithm/out/receipts/第十一批/three-route/live-verify-command.json

Fresh inference + trace + exact scan (pad coordinates come from the selected
``box_direct`` report)::

    $boxReport = Get-Content algorithm/out/receipts/第十一批/box-direct/report.json | ConvertFrom-Json
    $pad = $boxReport.best_pad_from_actual_selection
    .\\.venv\\Scripts\\python.exe -m lpr_tools.eval.three_route --fresh --config algorithm/config/default.toml --box-pad-x $pad.pad_x --box-pad-y $pad.pad_y --live-output algorithm/out/receipts/第十一批/three-route/fresh.jsonl.gz --out algorithm/out/receipts/第十一批/three-route/fresh-command.json

Optional custom adapters use ``module:factory``. A stable factory is exactly
``factory(model_path: pathlib.Path) -> adapter``; detector adapters implement
``detect(image, max_candidates)`` and box-reader adapters implement
``read(native_crop)``. The crop passed to ``read`` retains the original
``crop_box`` dimensions. Each weight must have a sibling ``manifest.json``
entry whose SHA-256 matches the model file. Future model architectures need
their official factory/decoder before they can be loaded; a model path alone
does not imply architecture support. After an official factory is delivered,
pass it with ``--detector-loader module:factory`` or ``--box-loader
module:factory`` to ``--fresh`` for a complete new inference and scan.

This experimental three-route family uses same-candidate, same-rect two-route
agreement followed by one B solo event per candidate; it is not installed in
the production runtime.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import importlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROUTE_PAIRS = (("segment", "reader"), ("segment", "box"), ("reader", "box"))
FALLBACKS = ("agreement_then_reader", "majority", "box", "crnn")
PRECISION_DENOMINATOR = 1000
PRECISION_NUMERATOR = 995


def _valid(attempt):
    if not (attempt and attempt.get("valid", attempt.get("status") == "OK")
            and attempt.get("status", "OK") not in {"INVALID_FORMAT", "BAD_INPUT"}
            and attempt.get("plate") is not None):
        return False
    try:
        confidence = float(attempt["confidence"])
        return math.isfinite(confidence) and 0.0 <= confidence <= 1.0
    except (KeyError, TypeError, ValueError):
        return False


def _confidence(attempt):
    return float(attempt["confidence"])


def _candidate_index(attempt):
    if "candidate_index" in attempt:
        return int(attempt["candidate_index"])
    # box_direct numbers detections from one; the paired E1 trace numbers them
    # from zero. Convert only this documented schema here, at the join edge.
    if "index" in attempt:
        return int(attempt["index"]) - 1
    raise ValueError("B attempt lacks candidate_index/index")


def _normalize_box_attempt(attempt):
    item = dict(attempt)
    item.setdefault("plate", item.get("prediction"))
    item.setdefault("confidence", item.get("min_confidence"))
    item.setdefault("valid", bool(item.get("legal", True)) and
                    item.get("status", "OK") not in {"INVALID_FORMAT", "BAD_INPUT"})
    return item


def select_box_pad(row, pad_order):
    """Normalize one nine_ceiling box_direct row to a single B pad.

    This preserves its candidate order and converts its 1-based indices to
    the 0-based candidate indices in the paired E1 trace.
    """
    pads = row.get("pads")
    if not isinstance(pads, list):
        raise ValueError("box trace row has no pads list")
    selected = [p for p in pads if int(p.get("pad_order", -1)) == int(pad_order)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one pad_order={pad_order}, found {len(selected)}")
    pad = selected[0]
    normalized_candidates = []
    for attempt in pad.get("candidates", []):
        item = _normalize_box_attempt(attempt)
        # box_direct stores candidate_index as 1-based (matching its report);
        # paired trace candidate_index is 0-based.
        raw_index = attempt.get("candidate_index", attempt.get("index"))
        if raw_index is None:
            raise ValueError("box_direct candidate lacks candidate_index/index")
        item["source_candidate_index_1based"] = int(raw_index)
        item["candidate_index"] = int(raw_index) - 1
        normalized_candidates.append(item)
    return {**row, "candidates": normalized_candidates,
            "box_pad": {"pad_order": pad.get("pad_order"), "pad_x": pad.get("pad_x"),
                        "pad_y": pad.get("pad_y")}}


def join_routes(paired_rows, box_rows):
    """Join two traces by normalized path, preserving paired-trace order.

    Call ``select_box_pad`` first so each box row contains one pad's per-candidate
    attempts. Ambiguous multi-pad data is refused.
    """
    boxes = {}
    for row in box_rows:
        key = str(Path(row["path"]).as_posix()).casefold()
        if key in boxes:
            raise ValueError(f"duplicate box trace path: {row['path']}")
        if "box_pad" not in row:
            raise ValueError("box rows must be normalized with select_box_pad first")
        boxes[key] = row
    joined = []
    for row in paired_rows:
        key = str(Path(row["path"]).as_posix()).casefold()
        b_row = boxes.pop(key, None)
        if b_row is None:
            raise ValueError(f"missing box trace row for {row['path']}")
        if str(b_row.get("truth")) != str(row.get("truth")):
            raise ValueError(f"box/paired truth mismatch for {row['path']}")
        if bool(b_row.get("decode_error")) != bool(row.get("input_error")):
            raise ValueError(f"paired/box decode-error disagreement for {row['path']}")
        paired_split = "devhard" if row.get("split") in {"test", "dev-hard", "devhard"} else row.get("split")
        box_split = "devhard" if b_row.get("split") in {"test", "dev-hard", "devhard"} else b_row.get("split")
        if box_split != paired_split:
            raise ValueError(f"box/paired split mismatch for {row['path']}: {box_split} vs {paired_split}")
        by_candidate = {}
        for attempt in b_row.get("candidates", []):
            ci = _candidate_index(attempt)
            if ci in by_candidate:
                raise ValueError(f"duplicate B attempt candidate={ci}, path={row['path']}")
            by_candidate[ci] = attempt
        candidates = list(range(int(row.get("candidate_count", 0))))
        if len(by_candidate) != len(candidates):
            raise ValueError(f"candidate count mismatch for {row['path']}: paired={len(candidates)} box={len(by_candidate)}")
        source_by_index = {}
        for rect in row.get("rects", []):
            ci = int(rect["candidate_index"])
            if rect.get("candidate_source") is not None:
                source_by_index.setdefault(ci, rect.get("candidate_source"))
        unverified_source_indices = []
        for ci in candidates:
            attempt = by_candidate.get(ci)
            if attempt is None:
                raise ValueError(f"box candidate index mismatch: path={row['path']} index={ci}")
            box_source = attempt.get("source", attempt.get("candidate_source"))
            paired_source = source_by_index.get(ci)
            if box_source is not None and paired_source is not None and paired_source != box_source:
                raise ValueError(f"candidate source mismatch: path={row['path']} index={ci} "
                                 f"paired={paired_source!r} box={box_source!r}")
            if box_source is not None and paired_source is None:
                unverified_source_indices.append(ci)
        row_out = dict(row)
        row_out["split"] = paired_split
        row_out["index_split"] = b_row.get("index_split")
        row_out["brightness"] = b_row.get("brightness")
        row_out["plate_color"] = b_row.get("plate_color")
        row_out["box_decode_error"] = b_row.get("decode_error")
        row_out["locator_error"] = b_row.get("locator_error")
        row_out["candidate_source_unverified_indices"] = unverified_source_indices
        row_out["boxes"] = {str(ci): by_candidate.get(ci) for ci in candidates}
        joined.append(row_out)
    if boxes:
        raise ValueError(f"unmatched box trace rows: {len(boxes)}")
    validate_records(joined)
    return joined


def validate_records(records):
    """Fail closed on malformed split/order or non-finite route confidences."""
    paths = set()
    for row_number, record in enumerate(records, 1):
        path = str(record.get("path", ""))
        key = path.replace("\\", "/").casefold()
        if not key or key in paths:
            raise ValueError(f"missing or duplicate trace path at row {row_number}: {path!r}")
        paths.add(key)
        if record.get("split") not in {"val", "devhard"}:
            raise ValueError(f"invalid split at row {row_number}: {record.get('split')!r}")
        count = int(record.get("candidate_count", -1))
        if count < 0:
            raise ValueError(f"invalid candidate_count at row {row_number}")
        seen = set()
        previous = (-1, -1)
        per_candidate = defaultdict(list)
        for rect in record.get("rects", []):
            ci, ri = int(rect["candidate_index"]), int(rect["rect_index"])
            if not 0 <= ci < count or ri < 0 or (ci, ri) in seen or (ci, ri) < previous:
                raise ValueError(f"duplicate/out-of-order candidate/rect index at row {row_number}")
            seen.add((ci, ri))
            previous = (ci, ri)
            per_candidate[ci].append(ri)
            for route in ("segment", "reader"):
                _validate_attempt(rect.get(route), f"row {row_number} {route}")
        for ci, indices in per_candidate.items():
            if indices != list(range(len(indices))):
                raise ValueError(f"non-contiguous rect index at row {row_number}, candidate {ci}")
        boxes = record.get("boxes", {})
        if set(boxes) != {str(ci) for ci in range(count)}:
            raise ValueError(f"box candidate indexes mismatch at row {row_number}")
        for attempt in boxes.values():
            _validate_attempt(attempt, f"row {row_number} box")
    return True


def _validate_attempt(attempt, label):
    if attempt is None or not attempt.get("valid", False):
        return
    if not isinstance(attempt.get("plate"), str) or not attempt["plate"]:
        raise ValueError(f"valid route has empty/non-text plate: {label}")
    try:
        confidence = float(attempt["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"valid route has missing/non-numeric confidence: {label}") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"valid route confidence outside [0,1]: {label}")


def _attempt_for(rect, boxes, route):
    if route == "segment":
        return rect.get("segment")
    if route == "reader":
        return rect.get("reader")
    return boxes.get(str(int(rect["candidate_index"])))


def _ordered_events(record):
    """Yield same-rect agreements then one B-solo event per candidate.

    S/R/B pair votes require the same rectification. Candidate order is outer,
    rect order is inner; the candidate's B solo follows all of its rect pairs.
    A B-only candidate with no rectification still has a valid solo event.
    """
    if (record.get("input_error") or record.get("decode_error") or record.get("box_decode_error")
            or record.get("live_error")):
        return
    boxes = record.get("boxes", {})
    rects_by_candidate = defaultdict(list)
    for rect in record.get("rects", []):
        rects_by_candidate[int(rect["candidate_index"])].append(rect)
    candidates = range(int(record.get("candidate_count", 0)))
    if not record.get("candidate_count"):
        candidates = sorted({int(v) for v in boxes})
    for ci in candidates:
        rects = sorted(rects_by_candidate.get(ci, []), key=lambda item: int(item["rect_index"]))
        for rect in rects:
            routes = {name: _attempt_for(rect, boxes, name) for name in ("segment", "reader", "box")}
            for left, right in ROUTE_PAIRS:
                a, b = routes[left], routes[right]
                if _valid(a) and _valid(b) and a["plate"] == b["plate"]:
                    yield {"kind": "agree", "plate": a["plate"], "confidence": min(_confidence(a), _confidence(b)),
               "output_confidence": _confidence(a if left != "box" else b),
                           "selected_route": left if left != "box" else right,
                           "truth_correct": a["plate"] == record["truth"],
                           "candidate_index": ci, "rect_index": int(rect["rect_index"])}
        b = boxes.get(str(ci))
        if _valid(b):
            yield {"kind": "box", "plate": b["plate"], "confidence": _confidence(b),
               "output_confidence": _confidence(b),
                   "selected_route": "box", "truth_correct": b["plate"] == record["truth"],
                   "candidate_index": ci, "rect_index": None}


def replay_auto(record, tau_agree, tau_box):
    """Return the first automatically accepted event, or None."""
    if record.get("input_error") or record.get("decode_error"):
        return None
    # A candidate can have several rectifications. Its B result is reused but
    # tested only at that candidate's rects, never across candidates.
    for event in _ordered_events(record):
        threshold = tau_agree if event["kind"] == "agree" else tau_box
        if event["confidence"] >= threshold:
            return {**event, "status": "OK", "confidence": event["output_confidence"]}
    return None


def reference_decision(record, tau_agree, tau_box, fallback="agreement_then_reader"):
    """Independent straightforward policy executor for per-image audit output."""
    if (record.get("input_error") or record.get("decode_error") or record.get("box_decode_error")
            or record.get("live_error")):
        return {"status": "BAD_INPUT", "plate": None, "confidence": 0.0,
                "route": None, "candidate_index": None, "rect_index": None}
    boxes = record.get("boxes", {})
    by_candidate = defaultdict(list)
    for rect in record.get("rects", []):
        by_candidate[int(rect["candidate_index"])].append(rect)
    n_candidates = int(record.get("candidate_count", len(boxes)))
    for ci in range(n_candidates):
        for rect in sorted(by_candidate.get(ci, []), key=lambda r: int(r["rect_index"])):
            current = {name: _attempt_for(rect, boxes, name)
                       for name in ("segment", "reader", "box")}
            for left, right in ROUTE_PAIRS:
                a, b = current[left], current[right]
                if (_valid(a) and _valid(b) and a["plate"] == b["plate"] and
                        min(float(a["confidence"]), float(b["confidence"])) >= tau_agree):
                    chosen = a if left != "box" else b
                    return {"status": "OK", "plate": chosen["plate"],
                            "confidence": float(chosen["confidence"]), "route": left,
                            "candidate_index": ci, "rect_index": int(rect["rect_index"])}
        b = boxes.get(str(ci))
        if _valid(b) and float(b["confidence"]) >= tau_box:
            return {"status": "OK", "plate": b["plate"], "confidence": float(b["confidence"]),
                    "route": "box", "candidate_index": ci, "rect_index": None}
    chosen = _reference_fallback(record, fallback)
    if chosen is None:
        return {"status": "NO_READ", "plate": None, "confidence": 0.0,
                "route": None, "candidate_index": None, "rect_index": None}
    attempt, route, ci, ri, confidence = chosen
    return {"status": "LOW_CONFIDENCE", "plate": attempt.get("plate"),
            "confidence": float(confidence or 0.0), "route": route,
            "candidate_index": ci, "rect_index": ri}


def _reference_fallback(record, policy):
    if policy not in FALLBACKS:
        raise ValueError(f"unknown fallback {policy!r}")
    if policy in {"box", "crnn"}:
        route = "box" if policy == "box" else "reader"
        values = []
        if route == "box":
            for key, attempt in record.get("boxes", {}).items():
                if _valid(attempt):
                    values.append((attempt, route, int(key), None, float(attempt["confidence"])))
        else:
            for rect in record.get("rects", []):
                attempt = rect.get(route)
                if _valid(attempt):
                    values.append((attempt, route, int(rect["candidate_index"]),
                                   int(rect["rect_index"]), float(attempt["confidence"])))
        return max(values, key=lambda item: float(item[0]["confidence"])) if values else None

    rects = record.get("rects", [])
    boxes = record.get("boxes", {})
    if policy == "agreement_then_reader":
        best_agreement = None
        best_reader = None
        for rect in rects:
            s, r = rect.get("segment"), rect.get("reader")
            ci, ri = int(rect["candidate_index"]), int(rect["rect_index"])
            if _valid(r) and (best_reader is None or
                              float(r["confidence"]) > float(best_reader[0]["confidence"])):
                best_reader = (r, "reader", ci, ri, float(r["confidence"]))
            if _valid(s) and _valid(r) and s["plate"] == r["plate"]:
                score = min(float(s["confidence"]), float(r["confidence"]))
                if best_agreement is None or score > best_agreement[0]:
                    best_agreement = (score, (r, "reader", ci, ri, score))
        return best_agreement[1] if best_agreement else best_reader

    # Majority fallback returns the first same-rect two-of-three agreement;
    # absent one, choose highest B, then highest rectified CRNN.
    for rect in rects:
        current = {name: _attempt_for(rect, boxes, name)
                   for name in ("segment", "reader", "box")}
        for left, right in ROUTE_PAIRS:
            a, b = current[left], current[right]
            if _valid(a) and _valid(b) and a["plate"] == b["plate"]:
                chosen = a if left != "box" else b
                route = left if left != "box" else right
                return (chosen, route, int(rect["candidate_index"]), int(rect["rect_index"]),
                        min(float(a["confidence"]), float(b["confidence"])))
    for route in ("box", "reader"):
        values = []
        if route == "box":
            values = [(a, route, int(ci), None, float(a["confidence"]))
                      for ci, a in boxes.items() if _valid(a)]
        else:
            values = [(r["reader"], route, int(r["candidate_index"]), int(r["rect_index"]),
                       float(r["reader"]["confidence"]))
                      for r in rects if _valid(r.get("reader"))]
        if values:
            return max(values, key=lambda item: float(item[0]["confidence"]))
    return None


def _valid_attempts(record, route):
    if route == "box":
        return [v for v in record.get("boxes", {}).values() if _valid(v)]
    return [r.get(route) for r in record.get("rects", []) if _valid(r.get(route))]


def _fallback(record, policy):
    """Return a deterministic non-auto fallback prediction (never auto-pass)."""
    if policy not in FALLBACKS:
        raise ValueError(f"unknown fallback {policy!r}")
    if (record.get("input_error") or record.get("decode_error") or record.get("box_decode_error")
            or record.get("live_error")):
        return None
    rects = record.get("rects", [])
    if policy == "agreement_then_reader":
        pairs = []
        top_reader = None
        for rect in rects:
            s, r = rect.get("segment"), rect.get("reader")
            if _valid(s) and _valid(r) and s["plate"] == r["plate"]:
                pairs.append((min(_confidence(s), _confidence(r)), r,
                              int(rect["candidate_index"]), int(rect["rect_index"])))
            if _valid(r) and (top_reader is None or _confidence(r) > top_reader[0]):
                top_reader = (_confidence(r), r, int(rect["candidate_index"]), int(rect["rect_index"]))
        if pairs:
            score, attempt, ci, ri = max(pairs, key=lambda item: item[0])
            return {**attempt, "confidence": score, "status": "LOW_CONFIDENCE",
                    "route": "reader", "candidate_index": ci, "rect_index": ri}
        if top_reader:
            _conf, attempt, ci, ri = top_reader
            return {**attempt, "status": "LOW_CONFIDENCE", "route": "reader",
                    "candidate_index": ci, "rect_index": ri}
        return None
    if policy == "box":
        values = [(attempt, int(ci)) for ci, attempt in record.get("boxes", {}).items() if _valid(attempt)]
        if not values:
            return None
        attempt, ci = max(values, key=lambda item: _confidence(item[0]))
        return {**attempt, "status": "LOW_CONFIDENCE", "route": "box",
                "candidate_index": ci, "rect_index": None}
    if policy == "crnn":
        values = [(rect["reader"], int(rect["candidate_index"]), int(rect["rect_index"]))
                  for rect in rects if _valid(rect.get("reader"))]
        if not values:
            return None
        attempt, ci, ri = max(values, key=lambda item: _confidence(item[0]))
        return {**attempt, "status": "LOW_CONFIDENCE", "route": "reader",
                "candidate_index": ci, "rect_index": ri}
    # First same-rect two-of-three agreement, respecting candidate/rect order.
    # If none exists, the declared fallback chain is max-confidence B then R.
    boxes = record.get("boxes", {})
    for rect in rects:
        routes = {name: _attempt_for(rect, boxes, name) for name in ("segment", "reader", "box")}
        for left, right in ROUTE_PAIRS:
            a, b = routes[left], routes[right]
            if _valid(a) and _valid(b) and a["plate"] == b["plate"]:
                selected = a if left != "box" else b
                route = left if left != "box" else right
                return {**selected, "confidence": min(_confidence(a), _confidence(b)),
                        "status": "LOW_CONFIDENCE", "route": route,
                        "candidate_index": int(rect["candidate_index"]),
                        "rect_index": int(rect["rect_index"])}
    boxes_valid = [(attempt, int(ci)) for ci, attempt in boxes.items() if _valid(attempt)]
    if boxes_valid:
        attempt, ci = max(boxes_valid, key=lambda item: _confidence(item[0]))
        return {**attempt, "status": "LOW_CONFIDENCE", "route": "box",
                "candidate_index": ci, "rect_index": None}
    readers_valid = [(rect["reader"], int(rect["candidate_index"]), int(rect["rect_index"]))
                     for rect in rects if _valid(rect.get("reader"))]
    if readers_valid:
        attempt, ci, ri = max(readers_valid, key=lambda item: _confidence(item[0]))
        return {**attempt, "status": "LOW_CONFIDENCE", "route": "reader",
                "candidate_index": ci, "rect_index": ri}
    return None


def evaluate(records, tau_agree, tau_box, fallback="agreement_then_reader"):
    """Metrics per split and subset, including exact whole-plate accuracy."""
    out = defaultdict(lambda: {"n": 0, "auto_pass": 0, "auto_pass_correct": 0,
                               "whole_plate_correct": 0})
    for record in records:
        split = "val" if record["split"] == "val" else "devhard"
        auto = replay_auto(record, tau_agree, tau_box)
        pred = auto["plate"] if auto else None
        if pred is None:
            fb = _fallback(record, fallback)
            pred = fb.get("plate") if fb else None
        groups = (split, f"{split}/{record.get('subset', 'unknown')}",
                  f"{split}/{record.get('category', 'unknown')}")
        brightness = record.get("brightness")
        if brightness is None:
            brightness = record.get("darkness")
        if record.get("subset") == "ccpd_db" and brightness is not None and float(brightness) <= 19:
            groups += (f"{split}/ccpd_db_brightness_le_19",)
        if record.get("subset") in {"ccpd_tilt", "ccpd_rotate"}:
            groups += (f"{split}/ccpd_tilt_plus_rotate",)
        if str(record.get("plate_color", record.get("color", record.get("lead_color", "")))).lower() == "green":
            groups += (f"{split}/green",)
        for group in set(groups):
            row = out[group]
            row["n"] += 1
            row["whole_plate_correct"] += pred == record["truth"]
            if auto:
                row["auto_pass"] += 1
                row["auto_pass_correct"] += auto["plate"] == record["truth"]
    for row in out.values():
        row["whole_plate_accuracy"] = row["whole_plate_correct"] / row["n"] if row["n"] else None
        row["auto_pass_precision"] = (row["auto_pass_correct"] / row["auto_pass"]
                                       if row["auto_pass"] else None)
    return dict(out)


def _boundary_values(records, kind):
    values = {0.0, math.nextafter(1.0, math.inf)}
    for record in records:
        for event in _ordered_events(record):
            if event["kind"] == kind:
                values.add(float(event["confidence"]))
    return sorted(values)


class _ThreeRouteBits:
    """Small per-action bitset index used by the exact two-axis search."""

    def __init__(self, records):
        self.records = list(records)
        self.n = len(self.records)
        self.all = (1 << self.n) - 1
        self.groups = {name: _pack([("val" if r["split"] == "val" else "devhard") == name
                                    for r in self.records]) for name in ("val", "devhard")}
        per_row = [list(_ordered_events(r)) for r in self.records]
        self.slots = max((len(row) for row in per_row), default=0)
        self.agree_conf, self.box_conf, self.agree_correct, self.box_correct = [], [], [], []
        for j in range(self.slots):
            ac, bc, at, bt = [], [], [], []
            for row, events in zip(self.records, per_row):
                event = events[j] if j < len(events) else None
                if event and event["kind"] == "agree":
                    ac.append(event["confidence"]); bc.append(float("-inf"))
                    at.append(event["truth_correct"]); bt.append(False)
                elif event:
                    ac.append(float("-inf")); bc.append(event["confidence"])
                    at.append(False); bt.append(event["truth_correct"])
                else:
                    ac.append(float("-inf")); bc.append(float("-inf"))
                    at.append(False); bt.append(False)
            self.agree_conf.append(np.asarray(ac, dtype=np.float64))
            self.box_conf.append(np.asarray(bc, dtype=np.float64))
            self.agree_correct.append(_pack(at)); self.box_correct.append(_pack(bt))
        self._cache_a, self._cache_b = {}, {}

    def _masks(self, axis, threshold):
        cache = self._cache_a if axis == "agree" else self._cache_b
        threshold = float(threshold)
        if threshold not in cache:
            confs = self.agree_conf if axis == "agree" else self.box_conf
            cache[threshold] = tuple(_pack(conf >= threshold) for conf in confs)
        return cache[threshold]

    def _counts_from_masks(self, am, bm):
        seen = correct = 0
        for j in range(self.slots):
            ok = am[j] | bm[j]
            first = ok & ~seen & self.all
            correct |= first & (self.agree_correct[j] | self.box_correct[j])
            seen |= ok
        return {name: {"accepted": (seen & mask).bit_count(),
                       "correct": (correct & mask).bit_count()}
                for name, mask in self.groups.items()}

    def counts(self, tau_agree, tau_box):
        return self._counts_from_masks(self._masks("agree", tau_agree),
                                       self._masks("box", tau_box))

    def bounds(self, low_a, high_a, low_b, high_b):
        la, ha = self._masks("agree", low_a), self._masks("agree", high_a)
        lb, hb = self._masks("box", low_b), self._masks("box", high_b)
        possible = guaranteed = possible_correct = 0
        for j in range(self.slots):
            possible |= la[j] | lb[j]
            guaranteed |= ha[j] | hb[j]
            possible_correct |= ((la[j] & self.agree_correct[j]) |
                                 (lb[j] & self.box_correct[j]))
        return {name: {"possible_accept": (possible & mask).bit_count(),
                       "guaranteed_accept": (guaranteed & mask).bit_count(),
                       "possible_correct": (possible_correct & mask).bit_count()}
                for name, mask in self.groups.items()}


def _pack(values):
    return int.from_bytes(np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little").tobytes(),
                          "little")


def exact_search(records, progress=None):
    """Exhaustively optimize the two observed confidence-boundary axes.

    The objective is accepted images over val+devhard, subject to nonempty
    auto-pass and >=99.5% precision in each split. A two-dimensional branch
    and bound uses safe bounds: low-corner acceptance bounds coverage above;
    high-corner acceptance is unavoidable; any potentially correct accepted
    event bounds correct count above. It returns complete only after the
    frontier is empty.
    """
    axes = (_boundary_values(records, "agree"), _boundary_values(records, "box"))
    n_axis = tuple(len(axis) for axis in axes)
    if not all(n_axis):
        raise ValueError("missing threshold boundaries")

    bits = _ThreeRouteBits(records)

    # A small high-confidence seed pass improves pruning only; it is not the
    # search domain or an approximation. The proof still explores every box
    # not eliminated by the safe bounds below.
    seed_values = (0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999,
                   math.nextafter(1.0, math.inf))
    seed_indices = []
    for axis in axes:
        indices = {0, len(axis) - 1}
        indices.update(round((len(axis) - 1) * fraction)
                       for fraction in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999))
        indices.update(min(len(axis) - 1, bisect.bisect_left(axis, value)) for value in seed_values)
        seed_indices.append(sorted(indices))
    best = None
    for ai in seed_indices[0]:
        for bi in seed_indices[1]:
            counts = bits.counts(axes[0][ai], axes[1][bi])
            if all(counts[s]["accepted"] > 0 and
                   counts[s]["correct"] * PRECISION_DENOMINATOR >=
                   PRECISION_NUMERATOR * counts[s]["accepted"] for s in ("val", "devhard")):
                total = counts["val"]["accepted"] + counts["devhard"]["accepted"]
                if best is None or total > best["accepted"]:
                    best = {"accepted": total, "tau_agree": axes[0][ai], "tau_box": axes[1][bi],
                            "threshold_counts": counts, "nodes": 0, "seed": True}

    # Integer-index boxes are inclusive. Cache corner scores and event upper
    # bounds by index box to keep repeated boundary work bounded.
    stack = [(0, n_axis[0]-1, 0, n_axis[1]-1)]
    nodes = 0
    pruned = 0
    started = time.perf_counter()
    while stack:
        a0, a1, b0, b1 = stack.pop()
        nodes += 1
        low = bits.counts(axes[0][a0], axes[1][b0])
        high = bits.counts(axes[0][a1], axes[1][b1])
        bound = bits.bounds(axes[0][a0], axes[0][a1], axes[1][b0], axes[1][b1])
        feasible_possible = True
        for split in ("val", "devhard"):
            lo = high[split]
            c = bound[split]["possible_correct"]
            # Every point accepts at least the high-corner population. If even
            # that cannot fit under the optimistic correct upper bound, none
            # in this box can satisfy the precision constraint.
            if c * PRECISION_DENOMINATOR < lo["accepted"] * PRECISION_NUMERATOR:
                feasible_possible = False
                break
        upper_total = low["val"]["accepted"] + low["devhard"]["accepted"]
        if (not feasible_possible) or (best is not None and upper_total <= best["accepted"]):
            pruned += 1
            continue
        if a0 == a1 and b0 == b1:
            if all(high[s]["accepted"] > 0 and
                   high[s]["correct"] * PRECISION_DENOMINATOR >=
                   PRECISION_NUMERATOR * high[s]["accepted"] for s in ("val", "devhard")):
                total = high["val"]["accepted"] + high["devhard"]["accepted"]
                if best is None or total > best["accepted"]:
                    best = {"accepted": total, "tau_agree": axes[0][a0], "tau_box": axes[1][b0],
                            "threshold_counts": high, "nodes": nodes, "seed": False}
            continue
        if a1-a0 >= b1-b0:
            mid = (a0+a1)//2
            stack.extend(((mid+1, a1, b0, b1), (a0, mid, b0, b1)))
        else:
            mid = (b0+b1)//2
            stack.extend(((a0, a1, mid+1, b1), (a0, a1, b0, mid)))
        if progress and nodes % 10000 == 0:
            progress({"nodes": nodes, "pruned": pruned, "frontier": len(stack),
                      "best_accepted": best["accepted"] if best else None,
                      "elapsed_seconds": time.perf_counter() - started})
    return {"complete": True, "nodes": nodes, "pruned": pruned,
            "frontier_remaining": len(stack), "optimality_gap_accepted": 0,
            "boundary_counts": {"agree": n_axis[0], "box": n_axis[1]}, "best": best,
            "seed_points": len(seed_indices[0]) * len(seed_indices[1]),
            "search_family": "observed_confidence_boundaries_exact"}


def read_jsonl_gz(path):
    metadata = None
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "_meta" in row and metadata is None:
                metadata = row["_meta"]
            else:
                rows.append(row)
    return rows, metadata


def write_jsonl_gz(path, rows, metadata=None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
        if metadata is not None:
            stream.write(json.dumps({"_meta": metadata}, ensure_ascii=False, separators=(",", ":")) + "\n")
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_manifest_identity(model_path):
    """Fail closed unless the weight has a matching adjacent manifest entry."""
    model_path = Path(model_path).resolve()
    manifest_path = model_path.parent / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"model or manifest missing: {model_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get(model_path.name)
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError(f"model lacks SHA-verified manifest entry: {model_path.name}")
    actual = sha256(model_path)
    if entry["sha256"] != actual:
        raise ValueError(f"model manifest SHA mismatch: {model_path.name}")
    return {"path": str(model_path), "sha256": actual,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path), "version": entry.get("version"),
            "manifest_entry": entry}


def _load_custom_factory(spec, model_path, method):
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("custom loader must use module:factory form")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    # Stable plugin contract: factory(model_path: Path) -> route adapter.
    instance = factory(Path(model_path))
    if not callable(getattr(instance, method, None)):
        raise TypeError(f"{spec} factory result must implement {method}()")
    return instance


def _as_route_attempt(decision):
    if decision is None:
        return None
    return {"plate": decision.plate_no, "confidence": float(decision.confidence),
            "status": decision.status.value,
            "valid": decision.status.value != "INVALID_FORMAT"}


def _read_box_result(reader, crop, color, min_confidence, is_crnn, torch):
    from lpr.pipeline import _decode_text
    from lpr.recognize.crnn import CrnnReader
    from lpr_tools.eval.box_direct import _read_crnn_native

    if crop is None or getattr(crop, "size", 0) == 0:
        return {"plate": "", "confidence": None, "status": "BAD_CROP", "valid": False}
    if is_crnn:
        if not isinstance(reader, CrnnReader):
            raise TypeError("the default direct-crop path requires CrnnReader")
        text, confidences = _read_crnn_native(reader, crop, torch)
    else:
        result = reader.read(crop)
        if isinstance(result, dict):
            # Custom BoxReader adapters may return a normalized decision.
            if "plate" in result and "confidence" in result:
                return {"plate": result["plate"], "confidence": float(result["confidence"]),
                        "status": result.get("status", "OK"),
                        "valid": bool(result.get("valid", result.get("legal", True)))}
            text, confidences = result.get("text", ""), result.get("confidences", [])
        else:
            text, confidences = result
    decision = _decode_text(text, confidences, color, min_confidence)
    if decision is None:
        return {"plate": text or "", "confidence": min(confidences) if confidences else None,
                "status": "INVALID_OUTPUT", "valid": False}
    return _as_route_attempt(decision)


def _live_record(image, sample, cfg, classifier, detector, reader, box_reader,
                 box_is_crnn, pad_x, pad_y, torch, fixed_split):
    """Independently run candidate, rectified S/R, and native-box B routes."""
    from lpr import pipeline
    from lpr.boxcrop import crop_box
    from lpr.pipeline import _rank
    from lpr.segment import segment
    from lpr.types import RecogStatus as Status
    from lpr_tools.eval.box_direct import _actual_candidates

    row = sample.row
    truth = row.plate_no
    try:
        candidates = _actual_candidates(image, cfg, detector)
    except Exception as exc:
        return {"path": str(row.path), "truth": truth, "subset": sample.subset,
                "split": fixed_split, "index_split": sample.split,
                "category": row.category, "plate_color": row.plate_color,
                "input_error": None, "candidate_count": 0, "rects": [], "boxes": {},
                "locator_error": f"{type(exc).__name__}:{exc}"}
    rect_records = []
    boxes = {}
    route_errors = []
    for ci, candidate in enumerate(candidates):
        try:
            rects = pipeline._rectifications(image, candidate, cfg.rectify)
        except Exception as exc:
            route_errors.append({"candidate_index": ci, "route": "rectify",
                                 "error": f"{type(exc).__name__}:{exc}"})
            rects = []
        for ri, rect in enumerate(rects):
            segment_best = None
            for layout in pipeline._layouts_to_try(rect):
                try:
                    segmented = segment(rect, layout, cfg.segment)
                    if not segmented.ok:
                        continue
                    probabilities = classifier.predict_proba(segmented.chars)
                    decision = pipeline.decode(probabilities, segmented.layout, rect.color, cfg.postprocess)
                    if segment_best is None or _rank(decision) > segment_best[0]:
                        segment_best = (_rank(decision), _as_route_attempt(decision))
                    if decision.status == Status.OK:
                        break
                except Exception as exc:
                    route_errors.append({"candidate_index": ci, "rect_index": ri,
                                         "route": "segment", "error": f"{type(exc).__name__}:{exc}"})
                    break
            s_attempt = segment_best[1] if segment_best else None
            r_attempt = None
            if reader is not None:
                try:
                    text, confidences = reader.read(rect.image)
                    decision = pipeline._decode_text(text, confidences, rect.color,
                                                     cfg.postprocess.min_confidence)
                    r_attempt = _as_route_attempt(decision)
                except Exception as exc:
                    route_errors.append({"candidate_index": ci, "rect_index": ri,
                                         "route": "reader", "error": f"{type(exc).__name__}:{exc}"})
            rect_records.append({"candidate_index": ci, "rect_index": ri,
                                 "candidate_source": candidate.source, "color": rect.color,
                                 "segment": s_attempt, "reader": r_attempt})
        crop = crop_box(image, candidate.bbox, pad_x, pad_y)
        try:
            box_attempt = _read_box_result(box_reader, crop, candidate.color,
                                           cfg.postprocess.min_confidence, box_is_crnn, torch)
        except Exception as exc:
            route_errors.append({"candidate_index": ci, "route": "box",
                                 "error": f"{type(exc).__name__}:{exc}"})
            box_attempt = {"plate": "", "confidence": None, "status": "READ_ERROR", "valid": False}
        boxes[str(ci)] = {**box_attempt, "candidate_index": ci, "source": candidate.source,
                          "score": float(candidate.score), "bbox": [int(v) for v in candidate.bbox]}
    return {"path": str(row.path), "truth": truth, "subset": sample.subset,
            "split": fixed_split, "index_split": sample.split,
            "category": row.category, "plate_color": row.plate_color,
            "lead_color": candidates[0].color if candidates else "unknown",
            "input_error": None, "candidate_count": len(candidates),
            "rects": rect_records, "boxes": boxes, "route_errors": route_errors}


def _same_attempt(left, right, tolerance=1e-7):
    if left is None or right is None:
        return left is right
    if (left.get("plate") != right.get("plate") or left.get("status") != right.get("status")
            or bool(left.get("valid")) != bool(right.get("valid"))):
        return False
    lc, rc = left.get("confidence"), right.get("confidence")
    if lc is None or rc is None:
        return lc is rc
    return abs(float(lc) - float(rc)) <= tolerance


def _compare_live_record(expected, actual):
    diffs = []
    for field in ("truth", "split", "index_split", "candidate_count"):
        if expected.get(field) != actual.get(field):
            diffs.append({"field": field, "expected": expected.get(field), "actual": actual.get(field)})
    if expected.get("input_error") != actual.get("input_error"):
        diffs.append({"field": "input_error", "expected": expected.get("input_error"),
                      "actual": actual.get("input_error")})
    expected_rects, actual_rects = expected.get("rects", []), actual.get("rects", [])
    if len(expected_rects) != len(actual_rects):
        diffs.append({"field": "rect_count", "expected": len(expected_rects), "actual": len(actual_rects)})
    for index, (left, right) in enumerate(zip(expected_rects, actual_rects)):
        for field in ("candidate_index", "rect_index", "candidate_source", "color"):
            if left.get(field) != right.get(field):
                diffs.append({"field": f"rects[{index}].{field}",
                              "expected": left.get(field), "actual": right.get(field)})
        for route in ("segment", "reader"):
            if not _same_attempt(left.get(route), right.get(route)):
                diffs.append({"field": f"rects[{index}].{route}",
                              "expected": left.get(route), "actual": right.get(route)})
    expected_boxes, actual_boxes = expected.get("boxes", {}), actual.get("boxes", {})
    if set(expected_boxes) != set(actual_boxes):
        diffs.append({"field": "box_candidate_indices", "expected": sorted(expected_boxes),
                      "actual": sorted(actual_boxes)})
    for key in set(expected_boxes) & set(actual_boxes):
        left, right = expected_boxes[key], actual_boxes[key]
        for field in ("candidate_index", "source", "bbox", "score"):
            if left.get(field) != right.get(field):
                diffs.append({"field": f"boxes[{key}].{field}",
                              "expected": left.get(field), "actual": right.get(field)})
        if not _same_attempt(left, right):
            diffs.append({"field": f"boxes[{key}].attempt", "expected": left, "actual": right})
    return diffs


def _normalize_verification_meta(meta):
    """Normalize joined and live trace identities for strict live verification."""
    meta = meta or {}
    if meta.get("schema") == "three-route-live-v1":
        models = meta.get("models", {})
        config_info = meta.get("config", {})
        return {
            "models": {"model_path": models.get("cnn"),
                       "corner_model_path": models.get("detector"),
                       "plate_reader_model_path": models.get("reader")},
            # Preserve B identity independently: a future BoxReader need not
            # share weights or architecture with the rectified S/R reader.
            "box_models": {"box_reader": models.get("box")} if models.get("box") else {},
            "box_best_pad": meta.get("box_pad"),
            "effective_config": config_info.get("effective_config", {}),
        }
    return {
        "models": meta.get("models", {}),
        "box_models": meta.get("box_models", {}),
        "box_best_pad": meta.get("box_best_pad"),
        "effective_config": meta.get("effective_config", {}),
    }


def run_live(*, config, index, val_manifest, devhard_manifest, pad_x, pad_y,
             output, detector_model=None, box_model=None, detector_loader=None,
             box_loader=None, verify_trace=None, policy_report=None,
             policy_fallback="agreement_then_reader", search=False):
    """Stream a full fresh S/R/B run and optionally compare a saved joined trace."""
    import cv2
    import torch
    from lpr.config import load_config
    from lpr.recognize import load_corner_detector, load_plate_reader
    from lpr.recognize import load_classifier
    from lpr_tools.data.image_health import decode_image
    from lpr_tools.data.manifest import image_path
    from lpr_tools.eval.common import effective_config_info
    from lpr_tools.eval.index_samples import fixed_manifest_samples, path_key
    from lpr_tools.eval.box_direct import RSS_LIMIT_BYTES, _read_brightness, _rss_bytes

    cfg = load_config(config, runtime__torch_threads=2)
    if cfg.runtime.classifier != "cnn":
        raise ValueError("live three-route executor requires the configured CNN classifier")
    if abs(float(cfg.postprocess.min_confidence) - 0.8) > 1e-12:
        raise ValueError("live three-route executor requires min_confidence=0.8")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cnn_path = cfg.resolve_path(cfg.runtime.model_path)
    detector_path = (Path(detector_model).resolve() if detector_model else
                     cfg.resolve_path(cfg.runtime.corner_model_path) if cfg.runtime.corner_model_path else None)
    reader_path = (cfg.resolve_path(cfg.runtime.plate_reader_model_path)
                   if cfg.runtime.plate_reader_model_path else None)
    if cnn_path is None or detector_path is None or reader_path is None:
        raise ValueError("live run requires configured CNN, detector, and rectified reader model paths")
    box_path = Path(box_model).resolve() if box_model else reader_path
    cnn_identity = _model_manifest_identity(cnn_path)
    detector_identity = _model_manifest_identity(detector_path)
    reader_identity = _model_manifest_identity(reader_path)
    box_identity = _model_manifest_identity(box_path)
    classifier = load_classifier("cnn", cnn_path)
    detector = (_load_custom_factory(detector_loader, detector_path, "detect") if detector_loader
                else load_corner_detector(detector_path))
    reader = load_plate_reader(reader_path)
    if box_loader:
        box_reader = _load_custom_factory(box_loader, box_path, "read")
        box_is_crnn = False
    elif box_path == reader_path:
        box_reader, box_is_crnn = reader, True
    else:
        box_reader, box_is_crnn = load_plate_reader(box_path), True
    initial_rss = _rss_bytes()
    if initial_rss is not None and initial_rss >= RSS_LIMIT_BYTES:
        raise MemoryError(f"three-route live process RSS {initial_rss} exceeds {RSS_LIMIT_BYTES}")

    samples = []
    manifest_ids = {}
    for split, manifest, purpose in (("val", val_manifest, "val"),
                                     ("devhard", devhard_manifest, "dev-hard")):
        selected, identity = fixed_manifest_samples(index, manifest, purpose=purpose)
        samples.extend((sample, split, Path(manifest).resolve()) for sample in selected)
        manifest_ids[split] = identity
    if len(samples) != 3100:
        raise ValueError(f"live executor requires 3100 fixed rows; got {len(samples)}")
    brightness_by_path = _read_brightness(index, {path_key(sample.row.path) for sample, _, _ in samples})

    expected = {}
    expected_meta = None
    policy_params = None
    policy_decision_differences = 0
    first_policy_differences = []
    if policy_report:
        policy_data = json.loads(Path(policy_report).read_text(encoding="utf-8"))
        best = policy_data.get("search", {}).get("best")
        if not isinstance(best, dict) or "tau_agree" not in best or "tau_box" not in best:
            raise ValueError("policy report lacks exact search.best thresholds")
        policy_params = {"tau_agree": float(best["tau_agree"]), "tau_box": float(best["tau_box"])}
    if verify_trace:
        expected_records, expected_meta = read_jsonl_gz(verify_trace)
        expected_meta = _normalize_verification_meta(expected_meta)
        validate_records(expected_records)
        expected = {str(Path(row["path"]).as_posix()).casefold(): row for row in expected_records}
        if len(expected) != 3100:
            raise ValueError(f"verification trace has {len(expected)} rows; expected 3100")
        expected_order = [str(Path(row["path"]).as_posix()).casefold() for row in expected_records]
        fixed_order = [str(Path(sample.row.path).as_posix()).casefold() for sample, _, _ in samples]
        if expected_order != fixed_order:
            raise ValueError("verification trace path order differs from the fixed val/devhard manifests")
        if not policy_params:
            raise ValueError("live verification requires a selected --policy-report")
        if expected_meta and expected_meta.get("box_best_pad"):
            selected_pad = expected_meta["box_best_pad"]
            if pad_x is None:
                pad_x = selected_pad["pad_x"]
            if pad_y is None:
                pad_y = selected_pad["pad_y"]
            if (abs(float(selected_pad["pad_x"]) - float(pad_x)) > 1e-12 or
                    abs(float(selected_pad["pad_y"]) - float(pad_y)) > 1e-12):
                raise ValueError("live box pad differs from verification trace selected pad")
        expected_models = expected_meta.get("models", {}) if expected_meta else {}
        for label, current, expected_key in (("cnn", cnn_identity, "model_path"),
                                             ("detector", detector_identity, "corner_model_path"),
                                             ("reader", reader_identity, "plate_reader_model_path")):
            if current["sha256"] != expected_models.get(expected_key, {}).get("sha256"):
                raise ValueError(f"live verification {label} model SHA differs from joined trace")
        expected_box_models = expected_meta.get("box_models", {}) if expected_meta else {}
        expected_box = (expected_box_models.get("box_reader", {}).get("sha256") or
                        expected_box_models.get("crnn", {}).get("sha256"))
        if expected_box and box_identity["sha256"] != expected_box:
            raise ValueError("live verification B model SHA differs from joined trace")
        expected_config = expected_meta.get("effective_config", {}) if expected_meta else {}
        actual_config = cfg.model_dump(mode="json")
        for key in ("preprocess", "locate", "lanes"):
            if actual_config.get(key) != expected_config.get(key):
                raise ValueError(f"live verification config differs in {key}")
        if actual_config.get("runtime", {}).get("max_candidates") != \
                expected_config.get("runtime", {}).get("max_candidates"):
            raise ValueError("live verification config differs in runtime.max_candidates")
    if pad_x is None or pad_y is None or not (0 <= float(pad_x) <= 1 and 0 <= float(pad_y) <= 1):
        raise ValueError("live fresh run requires box pad values in [0,1]")

    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite live trace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    decision_path = Path(output).with_suffix(".decisions.jsonl.gz") if verify_trace else None
    report_path = Path(output).with_suffix(".report.json")
    if decision_path is not None and decision_path.exists():
        raise FileExistsError(f"refusing to overwrite policy decision trace: {decision_path}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite live report: {report_path}")
    meta = {"schema": "three-route-live-v1", "config": effective_config_info(cfg, config),
            "models": {"cnn": cnn_identity, "detector": detector_identity,
                       "reader": reader_identity, "box": box_identity},
            "box_pad": {"pad_x": float(pad_x), "pad_y": float(pad_y)},
            "manifests": manifest_ids, "runtime_threads": {"torch": torch.get_num_threads(),
                                                               "opencv": cv2.getNumThreads(),
                                                               "blas": 1},
            "loader_factories": {"detector": detector_loader or "lpr.recognize.load_corner_detector",
                                 "box": box_loader or "CrnnReader native-crop adapter"},
            "candidate_order": "corner detector first then classical locate, pipeline._merge_candidates",
            "route_order": "within candidate, all ordered same-rect pairs, then one B solo",
            "trace_source_sha256": sha256(Path(__file__).resolve())}
    differences = 0
    first_differences = []
    live_errors = []
    records = []
    decision_rows = []
    memory_abort = False
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"_meta": meta}, ensure_ascii=False, separators=(",", ":")) + "\n")
        for number, (sample, split, manifest) in enumerate(samples, 1):
            image, error = decode_image(image_path(sample.row, manifest))
            if error:
                actual = {"path": str(sample.row.path), "truth": sample.row.plate_no,
                          "subset": sample.subset, "split": split, "index_split": sample.split,
                          "category": sample.row.category,
                          "plate_color": sample.row.plate_color, "input_error": error,
                          "box_decode_error": error, "candidate_count": 0, "rects": [], "boxes": {}}
                live_errors.append({"path": str(sample.row.path), "error": error})
            else:
                try:
                    actual = _live_record(image, sample, cfg, classifier, detector, reader, box_reader,
                                          box_is_crnn, pad_x, pad_y, torch, split)
                except Exception as exc:
                    actual = {"path": str(sample.row.path), "truth": sample.row.plate_no,
                              "subset": sample.subset, "split": split, "index_split": sample.split,
                              "category": sample.row.category,
                              "plate_color": sample.row.plate_color, "input_error": None,
                              "live_error": f"{type(exc).__name__}:{exc}",
                              "candidate_count": 0, "rects": [], "boxes": {}}
                    live_errors.append({"path": str(sample.row.path), "error": actual["live_error"]})
            if actual.get("locator_error"):
                live_errors.append({"path": actual["path"], "error": actual["locator_error"]})
            actual["brightness"] = brightness_by_path[path_key(sample.row.path)]
            for route_error in actual.get("route_errors", []):
                live_errors.append({"path": actual["path"], **route_error})
            if verify_trace:
                key = str(Path(actual["path"]).as_posix()).casefold()
                reference = expected.get(key)
                if reference is None:
                    diffs = [{"field": "path", "error": "missing expected trace row"}]
                else:
                    diffs = _compare_live_record(reference, actual)
                if diffs:
                    differences += len(diffs)
                    if len(first_differences) < 30:
                        first_differences.append({"path": actual["path"], "differences": diffs[:8]})
                if reference is not None:
                    expected_decision = reference_decision(reference, **policy_params,
                                                           fallback=policy_fallback)
                    actual_decision = reference_decision(actual, **policy_params,
                                                         fallback=policy_fallback)
                    decision_diff = any(expected_decision.get(k) != actual_decision.get(k)
                                        for k in ("status", "plate", "route", "candidate_index", "rect_index"))
                    decision_diff = decision_diff or abs(float(expected_decision["confidence"]) -
                                                          float(actual_decision["confidence"])) > 1e-7
                    if decision_diff:
                        policy_decision_differences += 1
                        if len(first_policy_differences) < 30:
                            first_policy_differences.append({"path": actual["path"],
                                "expected": expected_decision, "actual": actual_decision})
                    decision_rows.append({"path": actual["path"], "expected": expected_decision,
                                          "actual": actual_decision, "matches": not decision_diff})
            records.append(actual)
            stream.write(json.dumps(actual, ensure_ascii=False, separators=(",", ":")) + "\n")
            if number % 100 == 0 or number == len(samples):
                stream.flush()
                rss = _rss_bytes()
                if rss is not None and rss >= RSS_LIMIT_BYTES:
                    memory_abort = True
                    live_errors.append({"path": actual["path"],
                                        "error": f"memory_limit:{rss}>={RSS_LIMIT_BYTES}"})
                    break
            if number % 100 == 0:
                print(f"live_three_route={number}/3100", flush=True)
            del image
    completed = len(records) == len(samples)
    if verify_trace:
        write_jsonl_gz(decision_path, decision_rows,
                       {"schema": "three-route-policy-decisions-v1",
                        "policy": {**(policy_params or {}), "fallback": policy_fallback},
                        "records": len(decision_rows), "difference_count": policy_decision_differences})
    report = {"status": ("aborted_memory_limit" if memory_abort else
                          "complete" if completed and not live_errors and not differences and
                          not policy_decision_differences else
                          "completed_with_errors" if completed else "incomplete"),
              "rows": len(records), "live_error_count": len(live_errors), "live_errors": live_errors,
              "verification_trace": str(Path(verify_trace).resolve()) if verify_trace else None,
              "verification_difference_count": differences,
              "verification_first_differences": first_differences,
              "policy_report": str(Path(policy_report).resolve()) if policy_report else None,
              "policy": {**(policy_params or {}), "fallback": policy_fallback},
              "final_decision_difference_count": policy_decision_differences,
              "final_decision_first_differences": first_policy_differences,
              "final_decision_rows": str(decision_path.resolve()) if decision_path else None,
              "final_decision_rows_sha256": sha256(decision_path) if decision_path else None,
              "fresh_trace": str(target), "fresh_trace_sha256": sha256(target), "metadata": meta}
    if verify_trace:
        report["verification_pass"] = (differences == 0 and policy_decision_differences == 0
                                        and not live_errors)
    if search and completed and not memory_abort:
        validate_records(records)
        scan = exact_search(records, progress=lambda event: print(
            json.dumps({"search_progress": event}, separators=(",", ":")), flush=True))
        report["search"] = scan
        if scan["best"]:
            report["fallback_metrics"] = {
                fallback: evaluate(records, scan["best"]["tau_agree"], scan["best"]["tau_box"], fallback)
                for fallback in FALLBACKS}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", help="existing normalized joined gzip JSONL trace")
    parser.add_argument("--paired-trace", help="fresh E1 ordered S/R JSONL trace")
    parser.add_argument("--paired-identity", help="E1 trace_identity.json")
    parser.add_argument("--box-trace", help="fresh box_direct trace.jsonl.gz")
    parser.add_argument("--box-report", help="box_direct report.json")
    parser.add_argument("--joined-trace", help="write normalized joined gzip JSONL here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fresh", action="store_true", help="run all fixed rows and create a fresh live trace")
    parser.add_argument("--live-verify", help="independently rerun models against this joined trace")
    parser.add_argument("--policy-report", help="exact scan report with selected thresholds")
    parser.add_argument("--policy-fallback", choices=FALLBACKS, default="agreement_then_reader")
    parser.add_argument("--live-output", help="fresh trace path for --fresh or --live-verify")
    parser.add_argument("--config")
    parser.add_argument("--index", default=r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
    parser.add_argument("--val-manifest", default=r"D:\keshe2\algorithm\data\index\val1000_v1.csv")
    parser.add_argument("--devhard-manifest", default=r"D:\keshe2\algorithm\data\index\devhard_v1.csv")
    parser.add_argument("--detector-model")
    parser.add_argument("--box-model")
    parser.add_argument("--detector-loader", help="optional module:factory implementing detect(image,max_candidates)")
    parser.add_argument("--box-loader", help="optional module:factory implementing read(native_crop)")
    parser.add_argument("--box-pad-x", type=float)
    parser.add_argument("--box-pad-y", type=float)
    parser.add_argument("--search", action="store_true", help="run the exact observed-boundary scan after fresh inference")
    parser.add_argument("--fallback", choices=FALLBACKS, action="append",
                        help="fallback to report; repeat as needed (default: all four)")
    args = parser.parse_args(argv)
    if args.fresh or args.live_verify:
        if not args.live_output:
            parser.error("--fresh/--live-verify requires --live-output")
        if not args.config:
            parser.error("--fresh/--live-verify requires --config with the intended model/config identity")
        if args.fresh and not args.live_verify and (args.box_pad_x is None or args.box_pad_y is None):
            parser.error("fresh inference requires --box-pad-x and --box-pad-y from the selected pad report")
        if args.live_verify and not args.policy_report:
            parser.error("--live-verify requires --policy-report to compare final per-image decisions")
        cli_report_path = Path(args.out).resolve()
        live_trace_path = Path(args.live_output).resolve()
        live_report_path = live_trace_path.with_suffix(".report.json")
        decision_trace_path = live_trace_path.with_suffix(".decisions.jsonl.gz")
        planned_outputs = [cli_report_path, live_trace_path, live_report_path]
        if args.live_verify:
            planned_outputs.append(decision_trace_path)
        if len(set(planned_outputs)) != len(planned_outputs):
            raise ValueError("--out and live output paths must be distinct")
        for target in planned_outputs:
            if target.exists():
                raise FileExistsError(f"refusing to overwrite live output: {target}")
        result = run_live(config=args.config, index=args.index, val_manifest=args.val_manifest,
                          devhard_manifest=args.devhard_manifest, pad_x=args.box_pad_x,
                          pad_y=args.box_pad_y, output=args.live_output,
                          detector_model=args.detector_model, box_model=args.box_model,
                          detector_loader=args.detector_loader, box_loader=args.box_loader,
                          verify_trace=args.live_verify, policy_report=args.policy_report,
                          policy_fallback=args.policy_fallback, search=args.search or args.fresh)
        cli_report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "rows": result["rows"],
                          "verification_pass": result.get("verification_pass"),
                          "fresh_trace": result["fresh_trace"],
                          "report": str(Path(args.live_output).with_suffix(".report.json"))},
                         ensure_ascii=False), flush=True)
        scan_ok = (not (args.search or args.fresh) or
                   (result.get("search", {}).get("complete") is True and
                    result.get("search", {}).get("best") is not None))
        return 0 if result["status"] == "complete" and result.get("verification_pass", True) and scan_ok else 2
    if any((args.paired_trace, args.paired_identity, args.box_trace, args.box_report, args.joined_trace)):
        if not all((args.paired_trace, args.paired_identity, args.box_trace,
                    args.box_report, args.joined_trace)):
            parser.error("fresh join requires paired trace+identity, box trace+report, and joined trace output")
        paired_path, identity_path = Path(args.paired_trace), Path(args.paired_identity)
        box_path, report_path = Path(args.box_trace), Path(args.box_report)
        from lpr_tools.data.manifest import image_path, read_manifest
        from lpr_tools.eval.index_samples import FIXED_HARD_MANIFEST, FIXED_VAL_MANIFEST
        with paired_path.open(encoding="utf-8") as stream:
            paired = [json.loads(line) for line in stream if line.strip()]
        box_rows, _ = read_jsonl_gz(box_path)
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        manifest_hashes = {str(path.resolve()): sha256(path)
                           for path in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST)}
        if identity.get("manifests") != manifest_hashes:
            raise ValueError("E1 trace identity fixed-manifest SHA mismatch")
        expected_paths = []
        for manifest in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST):
            expected_paths.extend(str(image_path(row, manifest).resolve()).replace("\\", "/").casefold()
                                  for row in read_manifest(manifest, check_files=True))
        paired_paths = [str(Path(row["path"]).resolve()).replace("\\", "/").casefold() for row in paired]
        if paired_paths != expected_paths:
            raise ValueError("paired trace paths/order do not match frozen val/devhard manifests")
        box_report = json.loads(report_path.read_text(encoding="utf-8"))
        if box_report.get("record_count") != 3100:
            raise ValueError(f"box report has unexpected record_count={box_report.get('record_count')}")
        if Path(box_report.get("trace", "")).resolve() != box_path.resolve():
            raise ValueError("box report does not reference the supplied box trace")
        if box_report.get("status") != "complete":
            raise ValueError(f"box report is not clean/complete: status={box_report.get('status')} "
                             f"decode={box_report.get('decode_error_count')} "
                             f"locator={box_report.get('locator_error_count')} "
                             f"read={box_report.get('read_error_count')}")
        if any(box_report.get(key, 0) for key in ("decode_error_count", "locator_error_count", "read_error_count")):
            raise ValueError("box report has fixed-row input/locator/read errors; refuse threshold scan")
        if identity.get("input_errors"):
            raise ValueError("paired trace contains input errors; refuse threshold scan")
        box_models = box_report.get("method", {}).get("models", {})
        paired_models = identity.get("models", {})
        for box_key, paired_key in (("cnn", "model_path"), ("corner", "corner_model_path")):
            if box_models.get(box_key, {}).get("sha256") != paired_models.get(paired_key, {}).get("sha256"):
                raise ValueError(f"paired/box model SHA mismatch for {box_key}")
        if "box_reader" in box_models:
            if not box_models["box_reader"].get("sha256"):
                raise ValueError("box_reader model identity lacks SHA256")
        elif box_models.get("crnn", {}).get("sha256") != \
                paired_models.get("plate_reader_model_path", {}).get("sha256"):
            raise ValueError("CRNNv2 proxy SHA must match the rectified reader model")
        paired_cfg = identity.get("effective_config", {})
        box_cfg = box_report.get("method", {}).get("config", {}).get("effective_config", {})
        if not box_cfg:
            raise ValueError("box report lacks effective config identity")
        runtime = paired_cfg.get("runtime", {})
        postprocess = paired_cfg.get("postprocess", {})
        if (runtime.get("classifier") != "cnn" or
                Path(runtime.get("model_path", "")).name != "char_cnn_v2.pt" or
                Path(runtime.get("corner_model_path", "")).name != "corner_net_v1.pt" or
                Path(runtime.get("plate_reader_model_path", "")).name != "crnn_v2.pt" or
                identity.get("expected_reader_model") != "crnn_v2.pt" or
                postprocess.get("arbitration") != "agree" or
                float(postprocess.get("min_confidence", -1)) != 0.8):
            raise ValueError("paired trace is not CNNv2+corner_v1+CRNNv2 with agree/min_confidence=.8")
        locator_keys = ("preprocess", "locate", "lanes")
        if any(paired_cfg.get(key) != box_cfg.get(key) for key in locator_keys) or \
                paired_cfg.get("runtime", {}).get("max_candidates") != \
                box_cfg.get("runtime", {}).get("max_candidates"):
            raise ValueError("paired/box candidate locator configuration differs")
        pad = box_report.get("best_pad_from_actual_selection")
        if not isinstance(pad, dict) or "pad_order" not in pad:
            raise ValueError("box report lacks best_pad_from_actual_selection.pad_order")
        records = join_routes(paired, [select_box_pad(row, pad["pad_order"]) for row in box_rows])
        counts = {split: sum(row.get("split") == split for row in records)
                  for split in ("val", "devhard")}
        if len(records) != 3100 or counts != {"val": 1000, "devhard": 2100}:
            raise ValueError(f"joined trace fixed-set mismatch: rows={len(records)} splits={counts}")
        trace_meta = {"schema": "three-route-v1", "records": len(records), "split_counts": counts,
                      "paired_trace": str(paired_path.resolve()), "paired_trace_sha256": sha256(paired_path),
                      "paired_identity": str(identity_path.resolve()),
                      "paired_identity_sha256": sha256(identity_path),
                      "paired_identity_verified": identity.get("identity_migration", {}).get("verified") is True,
                      "box_trace": str(box_path.resolve()), "box_trace_sha256": sha256(box_path),
                      "box_report": str(report_path.resolve()), "box_report_sha256": sha256(report_path),
                      "box_run_status": box_report.get("status"),
                      "box_error_counts": {key: box_report.get(key) for key in
                                           ("decode_error_count", "locator_error_count", "read_error_count")},
                      "box_best_pad": pad, "models": identity.get("models"),
                      "box_models": box_models,
                      "effective_config": identity.get("effective_config"),
                      "manifests": identity.get("manifests"), "pipeline_sha256": identity.get("pipeline_sha256"),
                      "rule_family": "same-candidate same-rect two-of-three, then one B solo per candidate",
                      "candidate_order": "E1 0-based order; box_direct 1-based index normalized explicitly",
                      "rect_order": "E1 rect_index order",
                      "limitations": ["E1 trace stores the chosen segment attempt per rect, not all layout attempts or candidate boxes.",
                                      "Join/scan uses box_direct's recorded crnn_v2 proxy; --fresh/--live-verify execute inference independently."]}
        paired_counts = defaultdict(int)
        for row in paired:
            paired_counts[row.get("split")] += 1
        paired_path_sha = hashlib.sha256(("\n".join(row["path"] for row in paired) + "\n").encode("utf-8")).hexdigest()
        if len(paired) != 3100 or dict(paired_counts) != {"val": 1000, "test": 2100}:
            raise ValueError(f"paired trace fixed splits mismatch: rows={len(paired)} counts={dict(paired_counts)}")
        if identity.get("trace_rows") != 3100 or identity.get("path_list_sha256") != paired_path_sha:
            raise ValueError("paired trace row count/path-list identity mismatch")
        if identity.get("trace_sha256") != trace_meta["paired_trace_sha256"]:
            raise ValueError("paired identity trace SHA does not match paired trace bytes")
        if not trace_meta["paired_identity_verified"]:
            raise ValueError("paired identity is not verified")
        if Path(args.joined_trace).exists():
            raise FileExistsError(f"refusing to overwrite joined trace: {args.joined_trace}")
        write_jsonl_gz(args.joined_trace, records, trace_meta)
        trace_path = Path(args.joined_trace)
    elif args.trace:
        trace_path = Path(args.trace)
        records, trace_meta = read_jsonl_gz(trace_path)
    else:
        parser.error("provide --trace or all fresh trace join inputs")
    validate_records(records)
    scan = exact_search(records, progress=lambda event: print(
        json.dumps({"search_progress": event}, separators=(",", ":")), flush=True))
    fallbacks = args.fallback or list(FALLBACKS)
    result = {"trace": str(trace_path.resolve()), "trace_sha256": sha256(trace_path),
              "trace_metadata": trace_meta, "fallbacks": fallbacks, "search": scan}
    if scan["best"]:
        best = scan["best"]
        result["fallback_metrics"] = {
            fallback: evaluate(records, best["tau_agree"], best["tau_box"], fallback)
            for fallback in fallbacks}
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite scan report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if scan["complete"] and scan["best"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
