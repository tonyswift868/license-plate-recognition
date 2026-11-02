from __future__ import annotations

import pytest
from lpr_tools.eval.e1_threshold import (
    _stable_threshold,
    candidate_thresholds,
    recommend,
    scan,
    validate_provenance,
)


def attempt(index, plate, confidence, valid=True):
    return {"attempt_index": index, "decoded": True, "plate_no": plate,
            "confidence": confidence, "format_valid": valid, "color": "blue"}


def test_threshold_candidates_include_every_boundary_and_just_above():
    values = candidate_thresholds([{"attempts": [attempt(0, "PLATE001", .8)]}])
    assert 0.0 in values and 1.0 in values and .8 in values
    assert __import__("math").nextafter(.8, __import__("math").inf) in values


def test_scan_replays_early_exit_for_each_threshold():
    records = [
        {"truth": "PLATE001", "lead_color": "blue", "candidate_count": 2,
         "attempts": [attempt(0, "PLATE002", .90), attempt(1, "PLATE001", .85)]},
        {"truth": "PLATE003", "lead_color": "blue", "candidate_count": 1,
         "attempts": [attempt(0, "PLATE003", .70)]},
    ]
    rows, recommendation = scan(records)
    at_80 = next(row for row in rows if row["threshold"] == .8)
    assert at_80["accepted"] == 1
    assert at_80["correct_accepted"] == 0
    assert recommendation is None


def test_stable_interior_threshold_preserves_maximum_coverage_plateau():
    records = [
        {"truth": "PLATE001", "lead_color": "blue", "candidate_count": 2,
         "attempts": [attempt(0, "PLATE002", .8), attempt(1, "PLATE001", .9)]},
        {"truth": "PLATE003", "lead_color": "blue", "candidate_count": 1,
         "attempts": [attempt(0, "PLATE003", .7)]},
    ]
    rows, recommendation = scan(records)
    assert recommendation is not None
    stable = _stable_threshold(records, recommendation)
    assert stable is not None
    assert stable["accepted"] == recommendation["accepted"]
    assert stable["correct_accepted"] == recommendation["correct_accepted"]
    assert recommendation["threshold"] < stable["threshold"] < stable["next_coverage_boundary"]


def test_recommendation_supports_a_stricter_conservative_precision_floor():
    rows = [
        {"threshold": .6, "accepted": 100, "correct_accepted": 99,
         "auto_pass_precision": .99},
        {"threshold": .8, "accepted": 80, "correct_accepted": 80,
         "auto_pass_precision": 1.0},
    ]
    assert recommend(rows, .99)["accepted"] == 100
    assert recommend(rows, .995)["accepted"] == 80


def test_no_valid_attempts_reports_zero_coverage_without_precision():
    record = {"truth": "PLATE001", "lead_color": "blue", "candidate_count": 1,
              "attempts": [attempt(0, "bad", .99, valid=False)]}
    rows, recommendation = scan([record])
    assert all(row["accepted"] == 0 and row["auto_pass_precision"] is None for row in rows)
    assert recommendation is None


def test_empty_or_missing_truth_is_rejected():
    with pytest.raises(ValueError):
        scan([])
    with pytest.raises(ValueError):
        scan([{"truth": "", "attempts": []}])


def test_provenance_rejects_non_validation_source_before_scanning(tmp_path):
    with pytest.raises(ValueError, match="fixed val1000"):
        validate_provenance([], tmp_path / "attempts.jsonl", {"selection": {"split": "test"}})
