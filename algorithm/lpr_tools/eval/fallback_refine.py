"""Fit-only selection of a frozen, small S/R/B fallback-rule family.

Automatic decisions are replayed unchanged. Fit chooses one fallback rule;
only then are hold metrics calculated for that rule and the A/M/L baselines.
The rule predicates consume prediction-only features and never labels/groups.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from lpr_tools.eval.route_split import assert_content_separation, path_key, split_assignments
from lpr_tools.eval.three_route import (
    _fallback,
    _reference_fallback,
    _valid,
    read_jsonl_gz,
    sha256,
)
from lpr_tools.eval.three_route_auto import _policy, reference_decision

PREREG_PATH = Path(__file__).with_name("fallback_rules_v1.json")
SPLIT_AMENDMENT_PATH = Path(__file__).with_name("fallback_split_v1.json")
SELECTION_PROTOCOL_PATH = Path(__file__).with_name("fallback_selection_v1.json")
OUTPUT_SCHEMA = "fallback-refine-v1"
BASELINES = ("A", "M", "L")
SPLITS = ("val", "devhard")


def _rules():
    rules = [{"id": key, "complexity": 0, "when_true": key}
             for key in BASELINES]
    rules.extend([
        {"id": "G8_A", "signal": "g8", "complexity": 1, "when_true": "A", "when_false": "M"},
        {"id": "G8_M", "signal": "g8", "complexity": 1, "when_true": "M", "when_false": "A"},
        {"id": "G8_L", "signal": "g8", "complexity": 1, "when_true": "L", "when_false": "M"},
    ])
    for prefix, signal, thresholds in (
            ("T", "tilt", (5, 10, 15, 20)),
            ("Y", "gray", (40, 60, 80, 100)),
            ("D", "delta", (0, .05, .1, .2))):
        for threshold in thresholds:
            token = (f"{int(threshold):02d}" if prefix == "T" else
                     "05" if threshold == .05 else
                     str(int(threshold)) if float(threshold).is_integer() else str(int(threshold * 100)))
            for target, other in (("A", "M"), ("M", "A")):
                rules.append({"id": f"{prefix}{token}_{target}", "signal": signal,
                              "threshold": threshold, "complexity": 1,
                              "when_true": target, "when_false": other})
    rules.extend([
        {"id": "G8Y60_A", "signal": "g8_gray60", "complexity": 2,
         "when_true": "A", "when_false": "M"},
        {"id": "G8Y60_M", "signal": "g8_gray60", "complexity": 2,
         "when_true": "M", "when_false": "A"},
        {"id": "G8Y60_L", "signal": "g8_gray60", "complexity": 2,
         "when_true": "L", "when_false": "M"},
    ])
    return rules


RULES = _rules()


def _read_features(path):
    meta, rows = None, []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if line_no == 1 and isinstance(row, dict) and "_meta" in row:
                meta = row["_meta"]
                continue
            rows.append(row)
    if not meta or meta.get("schema") != "three-route-prediction-features-v1":
        raise ValueError("features must have three-route-prediction-features-v1 metadata")
    return meta, rows


def _load_policy(path, trace_path):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = policy.get("selected_policy")
    parsed = _policy(selected)
    if policy.get("trace_rows") != 3100:
        raise ValueError("auto policy report must cover exactly 3100 rows")
    if policy.get("trace") and Path(policy["trace"]).resolve() != Path(trace_path).resolve():
        raise ValueError("auto policy report references a different source trace")
    if policy.get("trace_sha256") and policy["trace_sha256"] != sha256(trace_path):
        raise ValueError("auto policy report trace SHA differs from supplied trace")
    if policy.get("eligible") is not None and selected not in policy["eligible"]:
        raise ValueError("selected auto policy is not in the report's eligible set")
    thresholds = policy.get("thresholds", {})
    tau_agree, tau_box = thresholds.get("tau_agree"), thresholds.get("tau_box")
    if (tau_agree is None or tau_box is None or
            not math.isfinite(float(tau_agree)) or not math.isfinite(float(tau_box)) or
            float(tau_agree) < 0 or float(tau_box) < 0):
        raise ValueError("auto policy report lacks finite base tau_agree/tau_box thresholds")
    legacy_thresholds = ("legacy_sr", "legacy_solo_segment", "legacy_solo_reader")
    if any(key not in thresholds or not math.isfinite(float(thresholds[key])) or
           not 0 <= float(thresholds[key]) <= 1 for key in legacy_thresholds):
        raise ValueError("auto policy report lacks valid legacy S/R thresholds")
    switches = {key: parsed[key] for key in ("legacy_sr_agree", "legacy_solo_segment",
                "legacy_solo_reader", "province_support")}
    return policy, {"policy_id": selected, "safe_augment": parsed["safe_augment"],
                    "base_thresholds": {"tau_agree": float(tau_agree),
                                        "tau_box": float(tau_box)}, **switches}


def _legacy_sr(record, thresholds):
    """Replay old first-OK, then production's no-pass A or best valid S/R fallback."""
    legacy_sr = float(thresholds["legacy_sr"])
    solo_segment = float(thresholds["legacy_solo_segment"])
    solo_reader = float(thresholds["legacy_solo_reader"])

    if any(record.get(key) for key in ("input_error", "decode_error", "box_decode_error", "live_error")):
        return None
    _boxes = record.get("boxes", {})
    grouped = defaultdict(list)
    for rect in record.get("rects", []):
        grouped[int(rect["candidate_index"])].append(rect)
    candidate_count = int(record.get("candidate_count", 0))
    candidate_count = max(candidate_count, max(grouped, default=-1) + 1)
    for ci in range(candidate_count):
        rects = sorted(grouped.get(ci, []), key=lambda item: int(item["rect_index"]))
        for rect in rects:
            segment, reader = rect.get("segment"), rect.get("reader")
            if (_valid(segment) and _valid(reader) and segment.get("plate") == reader.get("plate")
                    and min(float(segment["confidence"]), float(reader["confidence"])) >= legacy_sr):
                return {**segment, "confidence": max(float(segment["confidence"]),
                                                      float(reader["confidence"])),
                        "status": "OK", "valid": True, "route": "segment",
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "legacy_event": "sr_agree"}
            if _valid(segment) and float(segment["confidence"]) >= solo_segment:
                return {**segment, "status": "OK", "route": "segment",
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "legacy_event": "solo_segment"}
            if _valid(reader) and float(reader["confidence"]) >= solo_reader:
                return {**reader, "status": "OK", "route": "reader",
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "legacy_event": "solo_reader"}
    ordered_rects = sorted(record.get("rects", []),
                           key=lambda item: (int(item["candidate_index"]),
                                             int(item["rect_index"])))
    first_rect = ordered_rects[0] if ordered_rects else None
    if first_rect and str(first_rect.get("color", "")).lower() in {"white", "black"}:
        return None  # legacy runtime returns UNSUPPORTED before text fallback
    fallback = _complete_fallback(record, "agreement_then_reader")
    if fallback is not None:
        return {**fallback, "status": "LOW_CONFIDENCE", "legacy_event": "production_fallback"}
    return _manual_best_sr(record)


def _manual_best_sr(record):
    """Mirror pipeline's low-confidence best attempt when A has no legal reader text."""
    best = None
    ordered_rects = sorted(record.get("rects", []),
                           key=lambda item: (int(item["candidate_index"]),
                                             int(item["rect_index"])))
    for rect in ordered_rects:
        segment, reader = rect.get("segment"), rect.get("reader")
        if _valid(segment):
            attempt, route = segment, "segment"
        elif _valid(reader):
            attempt, route = reader, "reader"
        else:
            continue
        candidate = {**attempt, "status": "LOW_CONFIDENCE", "valid": True,
                     "route": route, "candidate_index": int(rect["candidate_index"]),
                     "rect_index": int(rect["rect_index"]),
                     "legacy_event": "production_fallback_best_sr"}
        if best is None or float(candidate["confidence"]) > float(best["confidence"]):
            best = candidate
    return best


def _complete_fallback(record, policy):
    """Attach deterministic route/index evidence to the lightweight fallback result."""
    simple = _fallback(record, policy)
    selected = _reference_fallback(record, policy)
    if simple is None or selected is None:
        if simple is not selected:
            raise ValueError(f"fallback reference availability differs for {policy}")
        return None
    attempt, route, candidate_index, rect_index, confidence = selected
    if (simple.get("plate") != attempt.get("plate") or
            abs(float(simple["confidence"]) - float(confidence)) > 1e-12):
        raise ValueError(f"fallback reference text/confidence differs for {policy}")
    return {**simple, "confidence": float(confidence), "route": route,
            "candidate_index": candidate_index, "rect_index": rect_index}


def _clean_attempt(attempt):
    if not attempt:
        return None
    return {key: attempt.get(key) for key in
            ("status", "plate", "confidence", "route", "candidate_index", "rect_index")}


def _fallbacks(record, legacy_thresholds):
    annotated_a = _complete_fallback(record, "agreement_then_reader")
    legacy = _legacy_sr(record, legacy_thresholds)
    return {"A": annotated_a, "M": _complete_fallback(record, "majority"), "L": legacy}


def _fallback_feature_view(feature_row, fallback_rows, record):
    """Expose only permitted prediction/pixel features to rule evaluation."""
    candidates = []
    for candidate in feature_row.get("candidates", []):
        candidates.append({key: candidate.get(key) for key in
                           ("candidate_index", "color", "tilt_angle_deg", "layout_lengths")})
    box_attempts = [row for row in record.get("boxes", {}).values() if _valid(row)]
    box_confidence = max((float(row["confidence"]) for row in box_attempts), default=None)
    fallback_predictions = [row for row in fallback_rows.values() if row]
    reader_lengths = [len(rect["reader"]["plate"]) for rect in record.get("rects", [])
                      if _valid(rect.get("reader")) and isinstance(rect["reader"].get("plate"), str)]
    return {"image_gray_mean": feature_row.get("image_gray_mean"),
            "candidates": candidates,
            "box_confidence": box_confidence,
            "rect_colors": [str(rect.get("color", "")).lower()
                            for rect in record.get("rects", [])],
            "reader_lengths": reader_lengths,
            "valid_lengths": [len(row["plate"]) for row in fallback_predictions
                              if isinstance(row.get("plate"), str)]}


def _signal(rule, view, fallbacks):
    signal = rule.get("signal")
    if signal == "g8":
        green = ("green" in view["rect_colors"] or
                 any(str(candidate.get("color", "")).lower() == "green"
                     for candidate in view["candidates"]))
        length8 = 8 in view["reader_lengths"] or 8 in view["valid_lengths"] or any(
            8 in (candidate.get("layout_lengths") or []) for candidate in view["candidates"])
        return green or length8
    if signal == "tilt":
        first = next((candidate for candidate in view["candidates"]
                      if candidate.get("candidate_index") == 0), None)
        angle = first.get("tilt_angle_deg") if first else None
        return angle is not None and math.isfinite(float(angle)) and float(angle) >= rule["threshold"]
    if signal == "gray":
        gray = view.get("image_gray_mean")
        return gray is not None and math.isfinite(float(gray)) and float(gray) <= rule["threshold"]
    if signal == "delta":
        a = fallbacks.get("A")
        box_confidence = view.get("box_confidence")
        if not a or a.get("confidence") is None or box_confidence is None:
            return False
        return float(box_confidence) - float(a["confidence"]) >= rule["threshold"]
    if signal == "g8_gray60":
        return _signal({"signal": "g8"}, view, fallbacks) and _signal(
            {"signal": "gray", "threshold": 60}, view, fallbacks)
    raise ValueError(f"unknown rule signal {signal!r}")


def _rule_target(rule, view, fallbacks):
    if not rule.get("signal"):
        return rule["when_true"]
    return rule["when_true"] if _signal(rule, view, fallbacks) else rule["when_false"]


def _resolve_fallback(target, fallbacks):
    order = (target, *(key for key in BASELINES if key != target))
    for key in order:
        if fallbacks.get(key) is not None:
            return key, fallbacks[key]
    return target, None


def _decision(record, auto, fallbacks, view, rule):
    if auto["status"] == "OK":
        return {**auto, "fallback_id": None, "auto_accepted": True}
    if auto["status"] == "BAD_INPUT":
        return {**auto, "fallback_id": None, "auto_accepted": False}
    target = _rule_target(rule, view, fallbacks)
    selected, attempt = _resolve_fallback(target, fallbacks)
    if attempt is None:
        return {**auto, "status": "NO_READ" if auto["status"] == "NO_AUTO" else auto["status"],
                "fallback_id": selected, "auto_accepted": False}
    return {"status": "LOW_CONFIDENCE", "plate": attempt.get("plate"),
            "confidence": float(attempt.get("confidence") or 0.0),
            "route": attempt.get("route"),
            "candidate_index": attempt.get("candidate_index"),
            "rect_index": attempt.get("rect_index"), "event": None,
            "fallback_id": selected, "auto_accepted": False}


def _groups(record):
    split = record["split"]
    groups = {split, f"{split}/{record.get('subset', 'unknown')}",
              f"{split}/{record.get('category', 'unknown')}"}
    brightness = record.get("brightness", record.get("darkness"))
    if record.get("subset") == "ccpd_db" and brightness is not None and float(brightness) <= 19:
        groups.add(f"{split}/ccpd_db_brightness_le_19")
    if record.get("subset") in {"ccpd_tilt", "ccpd_rotate"}:
        groups.add(f"{split}/ccpd_tilt_plus_rotate")
    if record.get("subset") == "ccpd_green":
        groups.add(f"{split}/green")
    return groups


def _score_rows(rows, decisions):
    stats = defaultdict(lambda: {"n": 0, "correct": 0, "auto_accepted": 0,
                                 "auto_correct": 0, "whole_plate_correct": 0})
    for record, decision in zip(rows, decisions, strict=False):
        correct = decision.get("plate") == record["truth"]
        accepted = decision.get("status") == "OK"
        for group in _groups(record):
            item = stats[group]
            item["n"] += 1
            item["correct"] += int(correct)
            item["whole_plate_correct"] += int(correct)
            item["auto_accepted"] += int(accepted)
            item["auto_correct"] += int(accepted and correct)
    for item in stats.values():
        item["whole_plate_accuracy"] = item["whole_plate_correct"] / item["n"] if item["n"] else None
        item["auto_pass_precision"] = (item["auto_correct"] / item["auto_accepted"]
                                       if item["auto_accepted"] else None)
    return dict(stats)


def _sha_json(value):
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _green_gate_minimum(group_size, production_default_correct, guard_mode):
    if guard_mode == "relative_default":
        return production_default_correct
    if guard_mode == "strict_absolute":
        return max(math.ceil(.926 * group_size), production_default_correct)
    raise ValueError(f"unknown fallback green guard mode: {guard_mode}")


def _load_frozen_rule_id(path, *, trace_sha, features_sha, split_sha, prereg_sha,
                         split_amendment_sha, selection_protocol_sha, guard_mode):
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    if frozen.get("selection_failed") or not frozen.get("selected_rule"):
        raise ValueError("frozen fallback report has no selected rule")
    if frozen.get("auto_policy", {}).get("policy_id") != "000":
        raise ValueError("fallback must be selected once against the architecture baseline policy 000")
    expected = {"trace_sha256": trace_sha, "features_sha256": features_sha,
                "split_assignment_sha256": split_sha, "preregistration_sha256": prereg_sha,
                "split_amendment_sha256": split_amendment_sha,
                "selection_protocol_sha256": selection_protocol_sha,
                "guard_mode": guard_mode}
    for field, value in expected.items():
        if frozen.get(field) != value:
            raise ValueError(f"frozen fallback report {field} mismatch")
    baseline_counts = frozen.get("auto_precision_check", {})
    accepted = sum(int(baseline_counts.get(split, {}).get("accepted", 0))
                   for split in SPLITS)
    baseline_report_path = Path(frozen.get("auto_policy_report", "")).resolve()
    baseline_report_sha = frozen.get("auto_policy_report_sha256")
    if not baseline_report_path.is_file() or sha256(baseline_report_path) != baseline_report_sha:
        raise ValueError("frozen baseline auto policy report is missing or changed")
    baseline_policy = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_policy.get("selected_policy") != "000":
        raise ValueError("frozen rule selection was not based on auto policy 000")
    baseline_metrics = baseline_policy.get("policies", {}).get("000", {})
    for split in SPLITS:
        frozen_metrics = frozen.get("auto_precision_check", {}).get(split, {})
        report_metrics = baseline_metrics.get(split, {})
        if (not frozen_metrics or not report_metrics or
                int(frozen_metrics.get("accepted", -1)) != int(report_metrics.get("accepted", -2)) or
                int(frozen_metrics.get("correct", -1)) != int(report_metrics.get("accepted_correct", -2))):
            raise ValueError(f"frozen baseline automatic counts differ from its policy report on {split}")
    expected_total = baseline_policy.get("baseline_expected_accepted")
    accepted_total = sum(int(baseline_metrics[split]["accepted"]) for split in SPLITS)
    if accepted != accepted_total:
        raise ValueError("frozen fallback accepted count differs from its bound baseline report")
    if expected_total is not None and accepted_total != int(expected_total):
        raise ValueError("baseline policy 000 accepted count differs from its bound base report")
    return frozen, frozen["selected_rule"]


def _load_shared_split(path, expected_assignments, trace_path, auto_policy,
                       split_amendment_sha):
    if not Path(path).exists():
        return None
    raw_sha = sha256(path)
    expected_auto_sha = auto_policy.get("split_assignment_sha256")
    if expected_auto_sha and expected_auto_sha != raw_sha:
        raise ValueError("auto policy and fallback use different split JSON bytes")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = payload.get("assignments", payload.get("path_to_half", payload))
    if not isinstance(assignments, dict):
        raise ValueError("shared split JSON must contain a path-to-half mapping")
    normalized = {path_key(key): value for key, value in assignments.items()}
    if normalized != expected_assignments:
        raise ValueError("shared split assignments differ from route_split(seed=42)")
    source_trace_sha = payload.get("trace_sha256") if isinstance(payload, dict) else None
    if source_trace_sha and source_trace_sha != sha256(trace_path):
        raise ValueError("shared split file is bound to different trace bytes")
    if payload.get("split_amendment_sha256") != split_amendment_sha:
        raise ValueError("shared split file uses a different content-split amendment")
    assignment_sha = payload.get("assignment_sha256") if isinstance(payload, dict) else None
    if assignment_sha and assignment_sha != _sha_json(expected_assignments):
        raise ValueError("shared split assignment SHA mismatch")
    return raw_sha


def _write_gzip_jsonl(path, header, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"_meta": header}, ensure_ascii=False, separators=(",", ":")) + "\n")
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_refine(trace_path, features_path, auto_policy_path, output_dir=None,
               report_path=None, split_path=None, frozen_rule_report=None,
               guard_mode="relative_default"):
    trace_path, features_path = Path(trace_path).resolve(), Path(features_path).resolve()
    auto_policy_path = Path(auto_policy_path).resolve()
    if report_path is None:
        if output_dir is None:
            raise ValueError("provide output_dir or report_path")
        output_dir = Path(output_dir).resolve()
        report_path = output_dir / "fallback-refine.json"
    else:
        report_path = Path(report_path).resolve()
        output_dir = Path(output_dir).resolve() if output_dir else report_path.parent
    split_path = Path(split_path).resolve() if split_path else output_dir / "split.json"
    decisions_path = report_path.with_suffix(".decisions.jsonl.gz")
    if len({report_path, split_path, decisions_path}) != 3:
        raise ValueError("report, split, and decisions outputs must be distinct")
    for target in (report_path, decisions_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    records, trace_meta = read_jsonl_gz(trace_path)
    feature_meta, feature_rows = _read_features(features_path)
    if len(records) != 3100 or len(feature_rows) != 3100:
        raise ValueError(f"expected 3100 trace/features rows, got {len(records)}/{len(feature_rows)}")
    if feature_meta.get("source_trace_sha256") != sha256(trace_path):
        raise ValueError("feature file is not bound to these trace bytes")
    auto_policy, selected_auto = _load_policy(auto_policy_path, trace_path)
    if trace_meta.get("schema") not in {"three-route-live-v1", "three-route-v1"}:
        raise ValueError("unsupported S/R/B trace schema")
    meta_sha = hashlib.sha256(json.dumps(trace_meta, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    if auto_policy.get("trace_metadata_sha256") and auto_policy["trace_metadata_sha256"] != meta_sha:
        raise ValueError("auto policy trace metadata identity differs from supplied trace")

    feature_by_path = {}
    image_sha_by_path = {}
    for row in feature_rows:
        key = path_key(row["path"])
        if key in feature_by_path:
            raise ValueError(f"duplicate feature path {row['path']}")
        if not row.get("image_sha256"):
            raise ValueError(f"feature row lacks image_sha256: {row['path']}")
        if any(field in row for field in ("truth", "plate_color", "subset", "category", "ground_truth")):
            raise ValueError("feature row contains label/group fields")
        feature_by_path[key] = row
        image_sha_by_path[key] = row["image_sha256"]
    trace_keys = [path_key(row["path"]) for row in records]
    feature_keys = [path_key(row["path"]) for row in feature_rows]
    if len(set(trace_keys)) != len(trace_keys) or set(trace_keys) != set(feature_by_path):
        raise ValueError("trace/features path sets differ or contain duplicate normalized paths")
    if trace_keys != feature_keys:
        raise ValueError("feature row order differs from source trace")
    for record in records:
        feature = feature_by_path[path_key(record["path"])]
        if int(feature.get("candidate_count", -1)) != int(record.get("candidate_count", -2)):
            raise ValueError(f"candidate count mismatch: {record['path']}")
        candidates = feature.get("candidates", [])
        if len(candidates) != int(record.get("candidate_count", 0)):
            raise ValueError(f"candidate feature rows incomplete: {record['path']}")
        for index, candidate in enumerate(candidates):
            if int(candidate.get("candidate_index", -1)) != index:
                raise ValueError(f"candidate feature order mismatch: {record['path']} index={index}")
            if any(key in candidate for key in ("truth", "plate_color", "subset", "category", "ground_truth")):
                raise ValueError("candidate features contain label/group fields")
            traced = record.get("boxes", {}).get(str(index))
            expected_source = traced.get("source") if traced else next(
                (rect.get("candidate_source") for rect in record.get("rects", [])
                 if int(rect["candidate_index"]) == index), None)
            if expected_source is not None and candidate.get("source") != expected_source:
                raise ValueError(f"feature candidate source mismatch: {record['path']} index={index}")
            if traced and traced.get("bbox") is not None and candidate.get("bbox") != traced.get("bbox"):
                raise ValueError(f"feature candidate bbox mismatch: {record['path']} index={index}")
            if traced and traced.get("score") is not None and candidate.get("score") is not None and \
                    abs(float(candidate["score"]) - float(traced["score"])) > 1e-7:
                raise ValueError(f"feature candidate score mismatch: {record['path']} index={index}")
    split_amendment_sha = sha256(SPLIT_AMENDMENT_PATH)
    selection_protocol_sha = sha256(SELECTION_PROTOCOL_PATH)
    if guard_mode not in {"relative_default", "strict_absolute"}:
        raise ValueError(f"unknown fallback green guard mode: {guard_mode}")
    assignments = split_assignments(records, seed=42, image_sha_by_path=image_sha_by_path)
    assert_content_separation(assignments, image_sha_by_path)
    shared_split_sha = _load_shared_split(split_path, assignments, trace_path, auto_policy,
                                          split_amendment_sha)
    if shared_split_sha is None and (auto_policy.get("split_assignment_sha256") or
                                     auto_policy.get("split_assignment_method") not in
                                     (None, "generated_route_split_seed42")):
        raise ValueError("auto policy used a shared split file that was not supplied to fallback refinement")
    split_counts = defaultdict(int)
    records_by_key = {path_key(row["path"]): row for row in records}
    for key, half in assignments.items():
        record = records_by_key[key]
        split_counts[f"{record['split']}_{half}"] += 1
        split_counts[f"{record['split']}/{record['subset']}_{half}"] += 1
    if dict(split_counts) != {"val_fit": 500, "val_hold": 500,
                             "devhard_fit": 1050, "devhard_hold": 1050,
                             "val/ccpd_base_fit": 250, "val/ccpd_base_hold": 250,
                             "val/ccpd_green_fit": 250, "val/ccpd_green_hold": 250,
                             **{f"devhard/{subset}_{half}": 150
                                for subset in ("ccpd_db", "ccpd_blur", "ccpd_tilt",
                                               "ccpd_rotate", "ccpd_fn", "ccpd_challenge",
                                               "ccpd_weather")
                                for half in ("fit", "hold")}}:
        raise ValueError(f"unexpected fixed split counts: {dict(split_counts)}")

    preregistration = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    rules = _rules()
    registered_ids = [item["id"] for item in preregistration["rules"]]
    if [rule["id"] for rule in rules] != registered_ids:
        raise ValueError("implemented fallback rule IDs differ from frozen pre-registration")
    rules_by_id = {rule["id"]: rule for rule in rules}
    frozen_report = None
    frozen_rule_id = None
    if frozen_rule_report:
        frozen_report, frozen_rule_id = _load_frozen_rule_id(
            frozen_rule_report, trace_sha=sha256(trace_path), features_sha=sha256(features_path),
            split_sha=_sha_json(assignments), prereg_sha=sha256(PREREG_PATH),
            split_amendment_sha=split_amendment_sha,
            selection_protocol_sha=selection_protocol_sha, guard_mode=guard_mode)
        if frozen_rule_id not in rules_by_id:
            raise ValueError(f"frozen rule id is not in registered family: {frozen_rule_id}")
    fallback_rows, auto_rows, feature_views = [], [], []
    production_default_rows = []
    for record in records:
        fallbacks = _fallbacks(record, auto_policy["thresholds"])
        production_default_rows.append(fallbacks["L"] or {"status": "NO_READ", "plate": None})
        auto = reference_decision(record, selected_auto)
        fallback_rows.append(fallbacks)
        auto_rows.append(auto)
        feature_views.append(_fallback_feature_view(
            feature_by_path[path_key(record["path"])], fallbacks, record))

    fit_indices = [i for i, row in enumerate(records)
                   if assignments[path_key(row["path"])] == "fit"]
    hold_indices = [i for i, row in enumerate(records)
                    if assignments[path_key(row["path"])] == "hold"]
    fit_records = [records[i] for i in fit_indices]
    production_default_fit = _score_rows(
        fit_records, [production_default_rows[i] for i in fit_indices])
    fit_scores = {}
    scored_rules = rules if frozen_rule_id is None else [rules_by_id[frozen_rule_id],
                                                         *(rules_by_id[key] for key in BASELINES)]
    for rule in scored_rules:
        decisions = [_decision(records[i], auto_rows[i], fallback_rows[i], feature_views[i], rule)
                     for i in fit_indices]
        fit_scores[rule["id"]] = _score_rows(fit_records, decisions)

    eligible = []
    eligibility_rules = rules if frozen_rule_id is None else [rules_by_id[frozen_rule_id]]
    for rule in eligibility_rules:
        group = fit_scores[rule["id"]].get("val/green", {"n": 0, "correct": 0})
        legacy_green = production_default_fit.get("val/green", {"n": 0, "correct": 0})
        minimum = _green_gate_minimum(group["n"], legacy_green["correct"], guard_mode)
        if group["n"] and group["correct"] >= minimum:
            eligible.append(rule)

    def selection_key(rule):
        score = fit_scores[rule["id"]]
        return (-score.get("devhard", {}).get("correct", 0),
                -score.get("devhard/ccpd_db_brightness_le_19", {}).get("correct", 0),
                -score.get("devhard/ccpd_tilt_plus_rotate", {}).get("correct", 0),
                -score.get("val", {}).get("correct", 0),
                int(rule["complexity"]), rule["id"])

    selected_rule = (rules_by_id[frozen_rule_id] if frozen_rule_id is not None else
                     min(eligible, key=selection_key) if eligible else None)
    # Audit the already-frozen automatic policy only after fallback fit selection.
    # These full counts cannot affect either rule choice or eligibility.
    auto_counts = defaultdict(lambda: {"n": 0, "accepted": 0, "correct": 0})
    for record, auto in zip(records, auto_rows, strict=False):
        row = auto_counts[record["split"]]
        row["n"] += 1
        if auto["status"] == "OK":
            row["accepted"] += 1
            row["correct"] += int(auto.get("plate") == record["truth"])
    for split in ("val", "devhard"):
        row = auto_counts[split]
        if (row["accepted"] <= 0 or row["correct"] * 1000 < row["accepted"] * 995) and \
                auto_policy.get("selection_scope") != "fit_only":
            raise ValueError(f"frozen auto policy fails 99.5% precision on {split}: {row}")
        reported = auto_policy.get("policies", {}).get(selected_auto["policy_id"], {}).get(split)
        if not reported or (row["accepted"] != int(reported.get("accepted", -1)) or
                            row["correct"] != int(reported.get("accepted_correct", -1))):
            raise ValueError(f"selected auto replay differs from policy report on {split}")
    if selected_rule:
        selected_id = selected_rule["id"]
        hold_result = {}
        for rule_id in (selected_id, *BASELINES):
            hold_result[rule_id] = _score_rows(
                [records[i] for i in hold_indices],
                [_decision(records[i], auto_rows[i], fallback_rows[i], feature_views[i],
                           rules_by_id[rule_id]) for i in hold_indices])
        full_result = {}
        full_decisions = {}
        for rule_id in (selected_id, *BASELINES):
            decisions = [_decision(record, auto, fallback, view, rules_by_id[rule_id])
                         for record, auto, fallback, view in
                         zip(records, auto_rows, fallback_rows, feature_views, strict=False)]
            full_decisions[rule_id] = decisions
            full_result[rule_id] = _score_rows(records, decisions)
        production_default_hold = _score_rows(
            [records[i] for i in hold_indices],
            [production_default_rows[i] for i in hold_indices])
        production_default_full = _score_rows(records, production_default_rows)
    else:
        selected_id, hold_result, full_result, full_decisions = None, {}, {}, {}
        production_default_hold, production_default_full = {}, {}

    split_payload = {"schema": "fallback-refine-split-v1", "seed": 42,
                     "strategy": "stratified fixed_split+subset; seed42 path split, exact-byte groups assigned to canonical path half, then singleton rebalance",
                     "split_amendment": str(SPLIT_AMENDMENT_PATH.resolve()),
                     "split_amendment_sha256": split_amendment_sha,
                     "trace_sha256": sha256(trace_path), "assignments": assignments,
                     "assignment_sha256": _sha_json(assignments), "counts": dict(split_counts)}
    if shared_split_sha is not None:
        split_file_sha = shared_split_sha
    else:
        split_file_sha = None
    decisions_out = []
    if selected_id:
        for index, record in enumerate(records):
            decisions_out.append({"path": record["path"], "partition": assignments[path_key(record["path"])],
                                  "auto": _clean_attempt(auto_rows[index]),
                                  "selected": _clean_attempt(full_decisions[selected_id][index]),
                                  "A": _clean_attempt(full_decisions["A"][index]),
                                  "M": _clean_attempt(full_decisions["M"][index]),
                                  "L": _clean_attempt(full_decisions["L"][index])})
            if auto_rows[index]["status"] == "OK":
                for rule_id in (selected_id, *BASELINES):
                    if any(full_decisions[rule_id][index].get(key) != auto_rows[index].get(key)
                           for key in ("status", "plate", "confidence", "route", "candidate_index", "rect_index")):
                        raise AssertionError(f"fallback modified accepted auto result: {record['path']}")

    report = {"schema": OUTPUT_SCHEMA, "trace": str(trace_path), "trace_sha256": sha256(trace_path),
              "features": str(features_path), "features_sha256": sha256(features_path),
              "auto_policy_report": str(auto_policy_path), "auto_policy_report_sha256": sha256(auto_policy_path),
              "auto_policy": selected_auto, "auto_precision_check": dict(auto_counts),
              "preregistration": str(PREREG_PATH), "preregistration_sha256": sha256(PREREG_PATH),
              "split_amendment": str(SPLIT_AMENDMENT_PATH.resolve()),
              "split_amendment_sha256": split_amendment_sha,
              "selection_protocol": str(SELECTION_PROTOCOL_PATH.resolve()),
              "selection_protocol_sha256": selection_protocol_sha,
              "guard_mode": guard_mode,
              "split_file": str(split_path), "split_assignment_sha256": _sha_json(assignments),
              "split_sha256": split_file_sha,
              "split_counts": dict(split_counts), "rules": rules,
              "selection_eligible": [item["id"] for item in eligible],
              "selected_rule": selected_id,
              "frozen_rule_report": str(Path(frozen_rule_report).resolve()) if frozen_rule_report else None,
              "frozen_rule_report_sha256": sha256(frozen_rule_report) if frozen_rule_report else None,
              "selection_reused_without_refit": frozen_rule_id is not None,
              "fit_scores": fit_scores, "hold_scores": hold_result, "full_scores": full_result,
              "production_default_scores": {"fit": production_default_fit,
                                            "hold": production_default_hold,
                                            "full": production_default_full},
              "selection_failed": selected_id is None,
              "status_contract": "fallback predictions with a legal route use LOW_CONFIDENCE; no legal fallback uses NO_READ; input failures retain BAD_INPUT. These analysis labels are not claimed as a production runtime mapping (architect integration maps no-read to SEG_FAILED and preserves existing special-color/error handling).",
              "decisions_file": str(decisions_path) if selected_id else None}

    for target in (report_path, decisions_path if selected_id else None):
        if target is not None and target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_id:
        _write_gzip_jsonl(decisions_path,
                          {"schema": "fallback-refine-decisions-v1", "selected_rule": selected_id,
                           "auto_policy": selected_auto, "trace_sha256": sha256(trace_path)}, decisions_out)
    if shared_split_sha is None:
        split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report["split_sha256"] = sha256(split_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--auto-policy", required=True, help="three_route_auto report JSON")
    parser.add_argument("--out", required=True, help="fallback refinement report JSON")
    parser.add_argument("--split-json", help="fixed split output (default: sibling split.json)")
    parser.add_argument("--frozen-fallback-report", help="reuse selected_rule without re-selection")
    parser.add_argument("--guard-mode", choices=("relative_default", "strict_absolute"),
                        default="relative_default")
    args = parser.parse_args(argv)
    report = run_refine(args.trace, args.features, args.auto_policy,
                        report_path=args.out, split_path=args.split_json,
                        frozen_rule_report=args.frozen_fallback_report,
                        guard_mode=args.guard_mode)
    print(json.dumps({"selected_rule": report["selected_rule"],
                      "eligible": report["selection_eligible"],
                      "selection_failed": report["selection_failed"],
                      "report": str(Path(args.out).resolve())}, ensure_ascii=False), flush=True)
    return 0 if not report["selection_failed"] else 2


def apply_frozen(trace_path, features_path, auto_policy_path, frozen_fallback_report,
                 output_dir):
    """Apply the baseline-frozen fallback ID to another frozen auto policy."""
    output_dir = Path(output_dir).resolve()
    frozen = json.loads(Path(frozen_fallback_report).read_text(encoding="utf-8"))
    split_path = Path(frozen["split_file"]).resolve()
    guard_mode = frozen.get("guard_mode")
    if guard_mode not in {"relative_default", "strict_absolute"}:
        raise ValueError("frozen fallback report lacks a supported guard mode")
    return run_refine(trace_path, features_path, auto_policy_path, output_dir=output_dir,
                      report_path=output_dir / "frozen-application.json",
                      split_path=split_path, frozen_rule_report=frozen_fallback_report,
                      guard_mode=guard_mode)


if __name__ == "__main__":
    raise SystemExit(main())
