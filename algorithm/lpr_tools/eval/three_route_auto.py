"""Replay a preregistered, finite family of three-route automatic rules.

This evaluator is deliberately read-only with respect to production runtime.
It does not tune thresholds: the archived three-route thresholds and three
legacy switches are fixed before scoring.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from lpr_tools.eval.three_route import ROUTE_PAIRS, _fallback, _valid, validate_records
from lpr_tools.eval.route_split import path_key, split_assignments

TAU_AGREE = 0.8839523792266846
TAU_BOX = 0.9605637192726135
LEGACY_SR = 0.8225060701370239
LEGACY_SOLO_SEGMENT = 0.938401460647583
LEGACY_SOLO_READER = 0.9818063974380493
SPLITS = ("val", "devhard")


def _policy(policy):
    if isinstance(policy, str):
        safe = policy.startswith("safe-")
        policy_id = policy[5:] if safe else policy
        if len(policy_id) != 3 or any(bit not in "01" for bit in policy_id):
            raise ValueError("policy ID must be 000..111 or safe-000..safe-111")
        bits = tuple(bit == "1" for bit in policy_id)
        return {"policy_id": policy, "safe_augment": safe,
                "base_thresholds": {"tau_agree": TAU_AGREE, "tau_box": TAU_BOX},
                "legacy_sr_agree": bits[0], "legacy_solo_segment": bits[1],
                "legacy_solo_reader": bits[2]}
    if not isinstance(policy, dict):
        raise ValueError("policy must be a three-bit ID or a policy dictionary")
    switches = {key: bool(policy.get(key, False)) for key in
                ("legacy_sr_agree", "legacy_solo_segment", "legacy_solo_reader")}
    short_id = "".join("1" if switches[key] else "0" for key in switches)
    safe = bool(policy.get("safe_augment", False))
    policy_id = f"safe-{short_id}" if safe else short_id
    if policy.get("policy_id", policy_id) != policy_id:
        raise ValueError("policy_id does not match switch values")
    thresholds = policy.get("base_thresholds", {})
    tau_agree = float(thresholds.get("tau_agree", TAU_AGREE))
    tau_box = float(thresholds.get("tau_box", TAU_BOX))
    if not all(math.isfinite(value) and value >= 0 for value in (tau_agree, tau_box)):
        raise ValueError("base thresholds must be finite nonnegative values")
    return {"policy_id": policy_id, "safe_augment": safe,
            "base_thresholds": {"tau_agree": tau_agree, "tau_box": tau_box}, **switches}


def _confidence(attempt):
    return float(attempt["confidence"])


def _valid_route(attempt):
    if not _valid(attempt) or not isinstance(attempt.get("plate"), str) or not attempt["plate"]:
        return False
    confidence = _confidence(attempt)
    return math.isfinite(confidence) and 0.0 <= confidence <= 1.0


def _baseline_decision(record, tau_agree, tau_box):
    boxes = record.get("boxes", {})
    by_candidate = defaultdict(list)
    for rect in record.get("rects", []):
        by_candidate[int(rect["candidate_index"])].append(rect)
    candidate_count = int(record.get("candidate_count", 0))
    if not candidate_count:
        indices = [*by_candidate, *(int(key) for key in boxes)]
        candidate_count = max(indices, default=-1) + 1

    for ci in range(candidate_count):
        rects = sorted(by_candidate.get(ci, []), key=lambda row: int(row["rect_index"]))
        for rect in rects:
            routes = {"segment": rect.get("segment"), "reader": rect.get("reader"),
                      "box": boxes.get(str(ci))}
            for left, right in ROUTE_PAIRS:
                a, b = routes[left], routes[right]
                if (_valid_route(a) and _valid_route(b) and a["plate"] == b["plate"] and
                        min(_confidence(a), _confidence(b)) >= tau_agree):
                    chosen = a if left != "box" else b
                    return {"status": "OK", "plate": chosen["plate"],
                            "confidence": _confidence(chosen),
                            "route": left if left != "box" else right,
                            "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                            "event": f"baseline_{left}_{right}"}
        box = boxes.get(str(ci))
        if _valid_route(box) and _confidence(box) >= tau_box:
            return {"status": "OK", "plate": box["plate"], "confidence": _confidence(box),
                    "route": "box", "candidate_index": ci, "rect_index": None,
                    "event": "baseline_box_solo"}
    return None


def reference_decision(record, policy="000"):
    """Return first auto-pass for a preregistered policy; never branches on truth."""
    selected = _policy(policy)
    if any(record.get(key) for key in ("input_error", "decode_error", "box_decode_error", "live_error")):
        return {"status": "BAD_INPUT", "plate": None, "confidence": 0.0,
                "route": None, "candidate_index": None, "rect_index": None}

    thresholds = selected["base_thresholds"]
    if selected["safe_augment"]:
        baseline = _baseline_decision(record, thresholds["tau_agree"], thresholds["tau_box"])
        if baseline is not None:
            return baseline
        return _legacy_decision(record, selected)
    return _combined_decision(record, selected)


def _combined_decision(record, selected):
    boxes = record.get("boxes", {})
    by_candidate = defaultdict(list)
    for rect in record.get("rects", []):
        by_candidate[int(rect["candidate_index"])].append(rect)
    candidate_count = int(record.get("candidate_count", 0))
    if not candidate_count:
        indices = [*by_candidate, *(int(key) for key in boxes)]
        candidate_count = max(indices, default=-1) + 1
    tau_agree = selected["base_thresholds"]["tau_agree"]
    tau_box = selected["base_thresholds"]["tau_box"]
    for ci in range(candidate_count):
        rects = sorted(by_candidate.get(ci, []), key=lambda row: int(row["rect_index"]))
        for rect in rects:
            routes = {"segment": rect.get("segment"), "reader": rect.get("reader"),
                      "box": boxes.get(str(ci))}
            for left, right in ROUTE_PAIRS:
                a, b = routes[left], routes[right]
                if (_valid_route(a) and _valid_route(b) and a["plate"] == b["plate"] and
                        min(_confidence(a), _confidence(b)) >= tau_agree):
                    chosen = a if left != "box" else b
                    return {"status": "OK", "plate": chosen["plate"],
                            "confidence": _confidence(chosen),
                            "route": left if left != "box" else right,
                            "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                            "event": f"baseline_{left}_{right}"}
            segment, reader = routes["segment"], routes["reader"]
            if (selected["legacy_sr_agree"] and _valid_route(segment) and _valid_route(reader)
                    and segment["plate"] == reader["plate"]
                    and min(_confidence(segment), _confidence(reader)) >= LEGACY_SR):
                return {"status": "OK", "plate": segment["plate"],
                        "confidence": max(_confidence(segment), _confidence(reader)), "route": "segment",
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "event": "legacy_sr_agree"}
            solo = []
            if selected["legacy_solo_segment"] and _valid_route(segment) and _confidence(segment) >= LEGACY_SOLO_SEGMENT:
                solo.append((segment, "segment"))
            if selected["legacy_solo_reader"] and _valid_route(reader) and _confidence(reader) >= LEGACY_SOLO_READER:
                solo.append((reader, "reader"))
            if solo:
                chosen, route = solo[0]
                return {"status": "OK", "plate": chosen["plate"],
                        "confidence": _confidence(chosen), "route": route,
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "event": f"legacy_solo_{route}"}
        box = boxes.get(str(ci))
        if _valid_route(box) and _confidence(box) >= tau_box:
            return {"status": "OK", "plate": box["plate"], "confidence": _confidence(box),
                    "route": "box", "candidate_index": ci, "rect_index": None,
                    "event": "baseline_box_solo"}
    return {"status": "NO_AUTO", "plate": None, "confidence": 0.0,
            "route": None, "candidate_index": None, "rect_index": None, "event": None}


def _legacy_decision(record, selected):
    boxes = record.get("boxes", {})
    by_candidate = defaultdict(list)
    for rect in record.get("rects", []):
        by_candidate[int(rect["candidate_index"])].append(rect)
    candidate_count = int(record.get("candidate_count", 0))
    if not candidate_count:
        indices = [*by_candidate, *(int(key) for key in boxes)]
        candidate_count = max(indices, default=-1) + 1

    for ci in range(candidate_count):
        rects = sorted(by_candidate.get(ci, []), key=lambda row: int(row["rect_index"]))
        for rect in rects:
            routes = {"segment": rect.get("segment"), "reader": rect.get("reader"),
                      "box": boxes.get(str(ci))}
            segment, reader = routes["segment"], routes["reader"]
            if (selected["legacy_sr_agree"] and _valid_route(segment) and _valid_route(reader)
                    and segment["plate"] == reader["plate"]
                    and min(_confidence(segment), _confidence(reader)) >= LEGACY_SR):
                return {"status": "OK", "plate": segment["plate"],
                        "confidence": max(_confidence(segment), _confidence(reader)), "route": "segment",
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "event": "legacy_sr_agree"}

            solo = []
            if (selected["legacy_solo_segment"] and _valid_route(segment) and
                    _confidence(segment) >= LEGACY_SOLO_SEGMENT):
                solo.append((segment, "segment"))
            if (selected["legacy_solo_reader"] and _valid_route(reader) and
                    _confidence(reader) >= LEGACY_SOLO_READER):
                solo.append((reader, "reader"))
            if solo:
                chosen, route = solo[0]  # fixed production preference: segment first
                return {"status": "OK", "plate": chosen["plate"],
                        "confidence": _confidence(chosen), "route": route,
                        "candidate_index": ci, "rect_index": int(rect["rect_index"]),
                        "event": f"legacy_solo_{route}"}

    return {"status": "NO_AUTO", "plate": None, "confidence": 0.0,
            "route": None, "candidate_index": None, "rect_index": None, "event": None}


def _groups(record):
    split = "val" if record.get("split") == "val" else "devhard"
    groups = {split, f"{split}/{record.get('subset', 'unknown')}",
              f"{split}/{record.get('category', 'unknown')}"}
    brightness = record.get("brightness", record.get("darkness"))
    if record.get("subset") == "ccpd_db" and brightness is not None and float(brightness) <= 19:
        groups.add(f"{split}/ccpd_db_brightness_le_19")
    if record.get("subset") in {"ccpd_tilt", "ccpd_rotate"}:
        groups.add(f"{split}/ccpd_tilt_plus_rotate")
    color = record.get("plate_color", record.get("color", record.get("lead_color", "")))
    if str(color).lower() == "green":
        groups.add(f"{split}/green")
    return groups


def score_policy(records, policy, assignments=None):
    selected = _policy(policy)
    groups = defaultdict(lambda: {"n": 0, "accepted": 0, "accepted_correct": 0,
                                  "whole_plate_correct": 0, "routes": {}})
    for record in records:
        decision = reference_decision(record, selected)
        accepted = decision["status"] == "OK"
        auto_correct = accepted and decision["plate"] == record["truth"]
        fallback = _fallback(record, "agreement_then_reader") if not accepted else None
        final_plate = decision["plate"] if accepted else (fallback.get("plate") if fallback else None)
        whole_correct = final_plate == record["truth"]
        record_groups = _groups(record)
        if assignments is not None:
            half = assignments[path_key(record["path"])]
            split = "val" if record.get("split") == "val" else "devhard"
            record_groups.update({half, f"{half}/{split}",
                                  f"{half}/{split}/{record.get('subset', 'unknown')}",
                                  f"{half}/{split}/{record.get('category', 'unknown')}"})
        for group in record_groups:
            row = groups[group]
            row["n"] += 1
            row["accepted"] += int(accepted)
            row["accepted_correct"] += int(auto_correct)
            row["whole_plate_correct"] += int(whole_correct)
            if accepted:
                route_counts = row["routes"].setdefault(decision["route"], {"accepted": 0, "correct": 0})
                route_counts["accepted"] += 1
                route_counts["correct"] += int(auto_correct)
    for row in groups.values():
        row["auto_pass_precision"] = row["accepted_correct"] / row["accepted"] if row["accepted"] else None
        row["whole_plate_accuracy"] = row["whole_plate_correct"] / row["n"] if row["n"] else None
    return dict(groups)


def _qualifies(metrics):
    for split in SPLITS:
        row = metrics.get(split, {})
        if row.get("accepted", 0) <= 0 or row["accepted_correct"] * 1000 < row["accepted"] * 995:
            return False
    return True


def evaluate_family(records, base_thresholds=None, expected_baseline_count=None, assignments=None):
    records = list(records)
    base_thresholds = {"tau_agree": float((base_thresholds or {}).get("tau_agree", TAU_AGREE)),
                       "tau_box": float((base_thresholds or {}).get("tau_box", TAU_BOX))}
    results = {}
    for safe in (False, True):
        for index in range(8):
            short_id = f"{index:03b}"
            policy_id = f"safe-{short_id}" if safe else short_id
            policy = {"policy_id": policy_id, "safe_augment": safe,
                      "base_thresholds": base_thresholds,
                      "legacy_sr_agree": short_id[0] == "1",
                      "legacy_solo_segment": short_id[1] == "1",
                      "legacy_solo_reader": short_id[2] == "1"}
            results[policy_id] = score_policy(records, policy, assignments)
    eligible = [policy_id for policy_id, metrics in results.items() if _qualifies(metrics)]
    # Explicit tie order: integrated baseline, then fewer switches, then lexical id.
    selected_id = None
    best_count = max((sum(results[item][split]["accepted"] for split in SPLITS)
                      for item in eligible), default=0)
    if eligible:
        selected_id = min((item for item in eligible
                           if sum(results[item][split]["accepted"] for split in SPLITS) == best_count),
                          key=lambda value: (value not in {"000", "safe-000"},
                                             value.count("1"), value))
    baseline_accepted = sum(results["000"][split]["accepted"] for split in SPLITS)
    if expected_baseline_count is not None and baseline_accepted != int(expected_baseline_count):
        raise ValueError(f"baseline accepted {baseline_accepted} differs from base report {expected_baseline_count}")
    return {"schema": "three-route-auto-v1", "thresholds": {
                "tau_agree": base_thresholds["tau_agree"], "tau_box": base_thresholds["tau_box"],
                "legacy_sr": LEGACY_SR, "legacy_solo_segment": LEGACY_SOLO_SEGMENT,
                "legacy_solo_reader": LEGACY_SOLO_READER},
            "rule_order": "combined policies per rect: baseline S/R,S/B,R/B; optional legacy S/R (segment output, max confidence); optional legacy solo segment then reader; after all rects per candidate: baseline B solo. Safe augment first exhausts the complete baseline decision, then tries legacy S/R and solo rules in candidate/rect order.",
            "selection_scope": "exactly the preregistered sixteen policies; no threshold sweep",
            "whole_plate_fallback": "fixed existing agreement_then_reader; not varied for policy selection",
            "n": len(records), "baseline_policy": "000", "baseline_expected_accepted": expected_baseline_count,
            "policies": results,
            "eligible": eligible, "selected_policy": selected_id,
            "selected_total_accepted": best_count if selected_id is not None else None}


def read_trace(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    metadata = None
    records = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if "_meta" in row:
                if line_number != 1 or metadata is not None:
                    raise ValueError("trace metadata must appear once on the first line")
                metadata = row["_meta"]
            else:
                records.append(row)
    if metadata is None:
        raise ValueError("trace is missing its first-line _meta identity")
    return metadata, records


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_base_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    best = report.get("search", {}).get("best")
    if report.get("status") != "complete" or report.get("search", {}).get("complete") is not True or not isinstance(best, dict):
        raise ValueError("base report must contain a complete search.best result")
    thresholds = {"tau_agree": float(best["tau_agree"]), "tau_box": float(best["tau_box"])}
    expected_count = int(best.get("accepted", -1))
    if expected_count < 0 or not all(math.isfinite(value) and value >= 0 for value in thresholds.values()):
        raise ValueError("base report has invalid thresholds or accepted count")
    return thresholds, expected_count, best.get("threshold_counts"), report


def _load_or_create_assignments(records, split_json=None):
    if split_json:
        data = json.loads(Path(split_json).read_text(encoding="utf-8"))
        values = data.get("assignments", data.get("path_to_half", data)) if isinstance(data, dict) else None
        if not isinstance(values, dict):
            raise ValueError("split JSON must be a path-to-fit/hold mapping or contain assignments")
        assignments = {path_key(path): half for path, half in values.items()}
        if any(half not in {"fit", "hold"} for half in assignments.values()):
            raise ValueError("split assignments must be fit or hold")
        expected = {path_key(record["path"]) for record in records}
        if set(assignments) != expected:
            raise ValueError("split JSON paths do not exactly match trace paths")
        return assignments, "provided", _sha256(split_json)
    return split_assignments(records, seed=42), "generated_route_split_seed42", None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=r"D:\keshe2\algorithm\out\tmp_arch\three_v1\live.jsonl.gz")
    parser.add_argument("--base-report", default=r"D:\keshe2\algorithm\out\tmp_arch\three_v1\report.json")
    parser.add_argument("--split-json", help="optional shared path-to-fit/hold assignment JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    metadata, records = read_trace(args.trace)
    validate_records(records)
    counts = {split: sum(row["split"] == split for row in records) for split in SPLITS}
    if len(records) != 3100 or counts != {"val": 1000, "devhard": 2100}:
        raise ValueError(f"trace must be the fixed 3100 rows (1000 val/2100 devhard), got {len(records)} {counts}")
    if metadata.get("schema") != "three-route-live-v1":
        raise ValueError("trace metadata schema is not three-route-live-v1")
    base_thresholds, expected_count, expected_by_split, base_report = _load_base_report(args.base_report)
    assignments, split_method, split_sha = _load_or_create_assignments(records, args.split_json)
    result = evaluate_family(records, base_thresholds, expected_count, assignments)
    if isinstance(expected_by_split, dict):
        for split in SPLITS:
            expected = expected_by_split.get(split, {})
            actual = result["policies"]["000"][split]
            if expected and (int(expected.get("accepted", -1)) != actual["accepted"] or
                             int(expected.get("correct", -1)) != actual["accepted_correct"]):
                raise ValueError(f"baseline {split} accepted/correct differs from base report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.update({"trace": str(Path(args.trace).resolve()), "trace_sha256": _sha256(args.trace),
                   "trace_metadata_sha256": hashlib.sha256(json.dumps(
                       metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                   "trace_metadata_schema": metadata["schema"], "trace_rows": len(records),
                   "split_counts": counts, "base_report": str(Path(args.base_report).resolve()),
                   "base_report_sha256": _sha256(args.base_report),
                   "tool_source_sha256": _sha256(Path(__file__).resolve()),
                   "split_assignment_method": split_method, "split_assignment_sha256": split_sha,
                   "fit_hold_counts": {half: sum(value == half for value in assignments.values())
                                       for half in ("fit", "hold")}})
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(records), "baseline_accepted": sum(
        result["policies"]["000"].get(split, {}).get("accepted", 0) for split in SPLITS),
        "eligible_count": len(result["eligible"]), "selected_policy": result["selected_policy"],
        "trace_sha256": result["trace_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
