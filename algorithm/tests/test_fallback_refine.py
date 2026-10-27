from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

import pytest
from lpr_tools.eval.fallback_refine import (
    PREREG_PATH,
    RULES,
    _decision,
    _complete_fallback,
    _fallback_feature_view,
    _green_gate_minimum,
    _load_frozen_rule_id,
    _load_policy,
    _legacy_sr,
    _manual_best_sr,
    _rule_target,
    apply_frozen,
    run_refine,
)
from lpr_tools.eval.three_route_auto import reference_decision


def _attempt(plate, confidence, status="OK"):
    return {"plate": plate, "confidence": confidence, "status": status,
            "valid": status == "OK"}


def _row(truth="A", rects=None, boxes=None):
    return {"path": "D:/tiny/a.jpg", "split": "val", "subset": "ccpd_green",
            "category": "clear", "truth": truth, "brightness": 10,
            "candidate_count": 1, "rects": rects or [], "boxes": boxes or {}}


def test_implemented_rule_ids_match_frozen_preregistration():
    frozen = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    assert [rule["id"] for rule in RULES] == [rule["id"] for rule in frozen["rules"]]


def test_auto_policy_uses_report_selected_safe_policy_and_dynamic_thresholds(tmp_path):
    trace = tmp_path / "trace.jsonl.gz"
    trace.write_bytes(b"trace")
    report_path = tmp_path / "auto.json"
    report_path.write_text(json.dumps({
        "selected_policy": "safe-011", "trace_rows": 3100,
        "trace": str(trace), "eligible": ["safe-011"],
        "thresholds": {"tau_agree": .87, "tau_box": .96,
                       "legacy_sr": .82, "legacy_solo_segment": .93,
                       "legacy_solo_reader": .98},
    }), encoding="utf-8")
    report, selected = _load_policy(report_path, trace)
    assert report["selected_policy"] == "safe-011"
    assert selected == {"policy_id": "safe-011", "safe_augment": True,
                        "base_thresholds": {"tau_agree": .87, "tau_box": .96},
                        "legacy_sr_agree": False, "legacy_solo_segment": True,
                        "legacy_solo_reader": True}


def test_frozen_rule_reuse_binds_baseline_auto_and_all_input_hashes(tmp_path):
    baseline_auto = tmp_path / "baseline-auto.json"
    baseline_auto.write_text(json.dumps({
        "selected_policy": "000", "baseline_expected_accepted": 17,
        "policies": {"000": {
            "val": {"accepted": 9, "accepted_correct": 9},
            "devhard": {"accepted": 8, "accepted_correct": 8},
        }},
    }), encoding="utf-8")
    frozen_path = tmp_path / "frozen.json"
    hashes = {"trace_sha256": "trace", "features_sha256": "features",
              "split_assignment_sha256": "split", "preregistration_sha256": "prereg",
              "split_amendment_sha256": "amendment",
              "selection_protocol_sha256": "protocol", "guard_mode": "relative_default"}
    frozen_path.write_text(json.dumps({
        **hashes, "selected_rule": "G8_A", "selection_failed": False,
        "auto_policy": {"policy_id": "000"},
        "auto_precision_check": {"val": {"accepted": 9, "correct": 9},
                                 "devhard": {"accepted": 8, "correct": 8}},
        "auto_policy_report": str(baseline_auto),
        "auto_policy_report_sha256": hashlib.sha256(baseline_auto.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    frozen, selected = _load_frozen_rule_id(
        frozen_path, trace_sha="trace", features_sha="features", split_sha="split",
        prereg_sha="prereg", split_amendment_sha="amendment",
        selection_protocol_sha="protocol", guard_mode="relative_default")
    assert selected == "G8_A"
    assert frozen["selected_rule"] == selected
    with pytest.raises(ValueError, match="split_amendment_sha256"):
        _load_frozen_rule_id(
            frozen_path, trace_sha="trace", features_sha="features", split_sha="split",
            prereg_sha="prereg", split_amendment_sha="changed-amendment",
            selection_protocol_sha="protocol", guard_mode="relative_default")
    with pytest.raises(ValueError, match="selection_protocol_sha256"):
        _load_frozen_rule_id(
            frozen_path, trace_sha="trace", features_sha="features", split_sha="split",
            prereg_sha="prereg", split_amendment_sha="amendment",
            selection_protocol_sha="changed-protocol", guard_mode="relative_default")
    with pytest.raises(ValueError, match="guard_mode"):
        _load_frozen_rule_id(
            frozen_path, trace_sha="trace", features_sha="features", split_sha="split",
            prereg_sha="prereg", split_amendment_sha="amendment",
            selection_protocol_sha="protocol", guard_mode="strict_absolute")


def test_relative_green_guard_keeps_fit_default_eligible_when_absolute_rounding_would_not():
    assert _green_gate_minimum(250, 231, "relative_default") == 231
    assert _green_gate_minimum(250, 231, "strict_absolute") == 232


def test_legacy_sr_replay_uses_first_accepted_rect_and_production_confidence():
    row = _row(rects=[
        {"candidate_index": 0, "rect_index": 0,
         "segment": _attempt("A", .94), "reader": _attempt("B", .7)},
        {"candidate_index": 0, "rect_index": 1,
         "segment": _attempt("C", .83), "reader": _attempt("C", .91)},
    ])
    result = _legacy_sr(row, {"legacy_sr": .82, "legacy_solo_segment": .93,
                              "legacy_solo_reader": .98})
    assert result["route"] == "segment"
    assert result["candidate_index"] == 0
    assert result["rect_index"] == 0
    assert result["status"] == "OK"


def test_legacy_manual_fallback_keeps_later_best_and_prefers_segment_within_rect():
    thresholds = {"legacy_sr": .99, "legacy_solo_segment": .99,
                  "legacy_solo_reader": .99}
    row = _row(rects=[
        {"candidate_index": 0, "rect_index": 0,
         "segment": _attempt("EARLY", .6), "reader": None},
        {"candidate_index": 0, "rect_index": 1,
         "segment": _attempt("S1", .7), "reader": _attempt("R1", .98)},
        {"candidate_index": 0, "rect_index": 2,
         "segment": _attempt("LATER-S", .85), "reader": _attempt("R2", .9)},
        {"candidate_index": 0, "rect_index": 3,
         "segment": _attempt("LATER-ONLY-S", .8), "reader": None},
    ])
    best = _manual_best_sr(row)
    assert best["status"] == "LOW_CONFIDENCE"
    assert best["route"] == "segment"
    assert best["plate"] == "LATER-S"
    assert best["confidence"] == .85
    assert best["rect_index"] == 2
    prioritized_a = _legacy_sr(row, thresholds)
    assert prioritized_a["route"] == "reader"
    assert prioritized_a["confidence"] == .98

    only_segment = _row(rects=[
        {"candidate_index": 0, "rect_index": 0, "segment": _attempt("S0", .6), "reader": None},
        {"candidate_index": 0, "rect_index": 1, "segment": _attempt("S1", .83), "reader": None},
    ])
    fallback = _legacy_sr(only_segment, thresholds)
    assert fallback["status"] == "LOW_CONFIDENCE"
    assert fallback["plate"] == "S1"
    assert fallback["route"] == "segment"
    assert fallback["confidence"] == .83


@pytest.mark.parametrize("color", ["white", "black"])
def test_legacy_special_color_blocks_fallback_only_after_no_sr_auto_pass(color):
    thresholds = {"legacy_sr": .82, "legacy_solo_segment": .93,
                  "legacy_solo_reader": .98}
    no_pass = _row(rects=[{"candidate_index": 0, "rect_index": 0, "color": color,
                           "segment": _attempt("S", .5), "reader": _attempt("R", .6)}])
    assert _legacy_sr(no_pass, thresholds) is None
    accepted = _row(rects=[{"candidate_index": 0, "rect_index": 0, "color": color,
                            "segment": _attempt("A", .9), "reader": _attempt("A", .9)}])
    result = _legacy_sr(accepted, thresholds)
    assert result["status"] == "OK"
    assert result["plate"] == "A"


def test_annotated_a_and_m_fallback_keep_reference_text_confidence_and_selection():
    from lpr_tools.eval.three_route import _fallback

    row = _row(rects=[{"candidate_index": 2, "rect_index": 4,
                       "segment": _attempt("A", .9), "reader": _attempt("A", .8)}],
               boxes={"2": _attempt("B", .95)})
    for policy in ("agreement_then_reader", "majority"):
        actual = _complete_fallback(row, policy)
        text = _fallback(row, policy)
        assert actual["plate"] == text["plate"]
        assert actual["confidence"] == text["confidence"] == .8
        assert actual["route"] in {"segment", "reader"}
        assert actual["candidate_index"] == 2
        assert actual["rect_index"] == 4


def test_fallback_preserves_every_automatic_accept_field():
    row = _row(rects=[{"candidate_index": 0, "rect_index": 0,
                       "segment": _attempt("A", .95), "reader": _attempt("A", .94)}])
    policy = {"policy_id": "000", "safe_augment": False,
              "base_thresholds": {"tau_agree": .8, "tau_box": .9},
              "legacy_sr_agree": False, "legacy_solo_segment": False,
              "legacy_solo_reader": False}
    auto = reference_decision(row, policy)
    assert auto["status"] == "OK"
    fallbacks = {"A": _attempt("X", .7), "M": _attempt("Y", .8), "L": _attempt("Z", .9)}
    result = _decision(row, auto, fallbacks, {"candidates": [], "reader_lengths": [],
                                               "rect_colors": [], "valid_lengths": []},
                       next(rule for rule in RULES if rule["id"] == "L"))
    for field in ("status", "plate", "confidence", "route", "candidate_index", "rect_index", "event"):
        assert result[field] == auto[field]
    assert result["fallback_id"] is None


def test_no_legal_fallback_uses_experiment_no_read_status():
    result = _decision(_row(), {"status": "NO_AUTO", "plate": None, "confidence": 0.0,
                                "route": None, "candidate_index": None, "rect_index": None},
                       {"A": None, "M": None, "L": None},
                       {"candidates": [], "reader_lengths": [], "rect_colors": [],
                        "valid_lengths": [], "image_gray_mean": None},
                       next(rule for rule in RULES if rule["id"] == "A"))
    assert result["status"] == "NO_READ"
    assert result["plate"] is None


def test_green_or_eight_rule_reads_prediction_features_only():
    record = _row(truth="DO_NOT_READ", rects=[{"candidate_index": 0, "rect_index": 0,
                                                "color": "blue", "reader": _attempt("12345678", .7)}])
    features = {"image_gray_mean": 50, "candidates": [
        {"candidate_index": 0, "color": "blue", "tilt_angle_deg": 0,
         "layout_lengths": [8]}]}
    view = _fallback_feature_view(features, {"A": None, "M": None, "L": None}, record)
    fallbacks = {"A": _attempt("A", .7), "M": _attempt("M", .8), "L": None}
    rule = next(rule for rule in RULES if rule["id"] == "G8_A")
    assert _rule_target(rule, view, fallbacks) == "A"


def _write_jsonl_gz(path, meta, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"_meta": meta}, separators=(",", ":")) + "\n")
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def _synthetic_refinement_inputs(root):
    records, features = [], []
    subsets = [("val", "ccpd_base", 500), ("val", "ccpd_green", 500)]
    subsets.extend(("devhard", subset, 300) for subset in (
        "ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
        "ccpd_fn", "ccpd_challenge", "ccpd_weather"))
    index = 0
    for split, subset, count in subsets:
        for _ in range(count):
            path = f"D:/synthetic/{split}/{subset}/{index:04d}.jpg"
            truth = f"PLATE{index:04d}"
            bbox = [1, 2, 30, 12]
            attempt = {"status": "OK", "valid": True, "plate": truth,
                       "confidence": .95}
            record = {
                "path": path, "split": split, "index_split": "train",
                "subset": subset, "category": "clear", "brightness": 25,
                "truth": truth, "plate_color": "green" if subset == "ccpd_green" else "blue",
                "candidate_count": 1,
                "rects": [{"candidate_index": 0, "rect_index": 0,
                           "candidate_source": "cnn", "bbox": bbox,
                           "color": "green" if subset == "ccpd_green" else "blue",
                           "segment": dict(attempt), "reader": dict(attempt)}],
                "boxes": {"0": {**attempt, "source": "cnn", "bbox": bbox, "score": .9}},
            }
            records.append(record)
            features.append({
                "path": path, "candidate_count": 1,
                "image_sha256": hashlib.sha256(path.encode()).hexdigest(),
                "image_gray_mean": 80,
                "candidates": [{"candidate_index": 0, "source": "cnn", "score": .9,
                                "bbox": bbox,
                                "color": "green" if subset == "ccpd_green" else "blue",
                                "tilt_angle_deg": 0, "layout_lengths": [len(truth)],
                                "box_gray_mean": 80}],
            })
            index += 1
    trace = root / "trace.jsonl.gz"
    features_path = root / "features.jsonl.gz"
    _write_jsonl_gz(trace, {"schema": "three-route-live-v1"}, records)
    _write_jsonl_gz(features_path, {
        "schema": "three-route-prediction-features-v1",
        "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }, features)
    split_counts = {"val": (1000, 1000), "devhard": (2100, 2100)}
    policies = {}
    for policy_id in ("000", "safe-001"):
        policies[policy_id] = {
            split: {"accepted": counts[0], "accepted_correct": counts[1]}
            for split, counts in split_counts.items()
        }
    threshold_values = {"tau_agree": .8, "tau_box": .9,
                        "legacy_sr": .82, "legacy_solo_segment": .93,
                        "legacy_solo_reader": .98}
    auto_reports = {}
    for policy_id in ("000", "safe-001"):
        auto_report_path = root / f"auto-{policy_id.replace('-', '')}.json"
        auto_report_path.write_text(json.dumps({
            "selected_policy": policy_id, "trace_rows": 3100, "trace": str(trace),
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "eligible": [policy_id], "thresholds": threshold_values,
            "policies": policies,
        }), encoding="utf-8")
        auto_reports[policy_id] = auto_report_path
    return trace, features_path, auto_reports


def test_run_refine_then_apply_frozen_keeps_only_frozen_rule_and_baselines(tmp_path):
    trace, features, auto_reports = _synthetic_refinement_inputs(tmp_path)
    baseline = run_refine(trace, features, auto_reports["000"],
                          output_dir=tmp_path / "baseline")
    assert baseline["selected_rule"] is not None

    applied = apply_frozen(trace, features, auto_reports["safe-001"],
                           tmp_path / "baseline" / "fallback-refine.json",
                           tmp_path / "applied")

    assert applied["selection_reused_without_refit"] is True
    assert applied["selected_rule"] == baseline["selected_rule"]
    assert set(applied["fit_scores"]) == {baseline["selected_rule"], "A", "M", "L"}
    assert applied["split_assignment_sha256"] == baseline["split_assignment_sha256"]
    assert applied["auto_policy"]["policy_id"] == "safe-001"
    assert applied["guard_mode"] == baseline["guard_mode"] == "relative_default"
    with gzip.open(baseline["decisions_file"], "rt", encoding="utf-8") as stream:
        baseline_decisions = [json.loads(line) for line in stream][1:]
    with gzip.open(applied["decisions_file"], "rt", encoding="utf-8") as stream:
        applied_decisions = [json.loads(line) for line in stream][1:]
    assert [row["auto"] for row in applied_decisions] == [row["auto"] for row in baseline_decisions]
