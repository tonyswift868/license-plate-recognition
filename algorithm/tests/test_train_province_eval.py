"""Small contract tests for batch-12 trace parsing and route summaries."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from lpr_tools.train.eval_province_readers import _aggregate, _selected, standalone_trace
from lpr_tools.train.score_province_readers import _first_char_patch


def test_selects_valid_attempt_and_keeps_first_tie():
    attempts = [
        {"plate": "陕A11111", "confidence": 0.8, "status": "LOW_CONFIDENCE", "valid": True},
        {"plate": "皖A22222", "confidence": 0.99, "status": "INVALID_OUTPUT", "valid": False},
        {"plate": "甘A33333", "confidence": 0.8, "status": "OK", "valid": True},
    ]
    assert _selected(attempts) == "陕A11111"


def test_rejects_invalid_missing_nonfinite_and_empty_attempts():
    assert _selected([]) is None
    assert _selected([None, {"plate": "", "confidence": 0.9, "status": "OK"}]) is None
    assert _selected([{"plate": "陕A11111", "confidence": float("nan"), "status": "OK"}]) is None
    assert _selected([{"plate": "陕A11111", "confidence": 0.9,
                       "status": "INVALID_FORMAT", "valid": True}]) is None


def test_aggregates_split_subset_province_and_wrong_anwei_counts():
    rows = [
        {"truth": "陕A11111", "prediction": "皖A11111", "split": "val",
         "subset": "ccpd_db", "brightness": 19},
        {"truth": "皖A22222", "prediction": "皖A22222", "split": "devhard",
         "subset": "ccpd_tilt", "brightness": 80},
    ]
    metrics = _aggregate(rows)
    assert metrics["all"]["n"] == 2
    assert metrics["all"]["province_correct"] == 1
    assert metrics["all"]["plate_correct"] == 1
    assert metrics["all"]["wrong_as_anwei"] == 1
    assert metrics["non_anwei"]["n"] == 1
    assert metrics["non_anwei"]["province_accuracy"] == 0.0
    assert metrics["val/ccpd_db"]["n"] == 1
    assert metrics["dark_le_19"]["n"] == 1
    assert metrics["tilt_plus_rotate"]["n"] == 1
    assert metrics["devhard/ccpd_tilt"]["plate_accuracy"] == 1.0


def test_summarizes_real_three_route_trace_schema(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    row = {
        "truth": "陕A12345", "split": "val", "subset": "ccpd_base", "brightness": 90,
        "rects": [
            {"segment": {"plate": "皖A12345", "confidence": 0.7, "status": "OK", "valid": True},
             "reader": None},
            {"segment": {"plate": "陕A12345", "confidence": 0.8, "status": "OK", "valid": True},
             "reader": {"plate": "陕A12345", "confidence": 0.8,
                        "status": "LOW_CONFIDENCE", "valid": True}},
        ],
        "boxes": {
            "0": {"plate": "皖A12345", "confidence": 0.9, "status": "OK", "valid": True},
            "1": {"plate": "陕A12345", "confidence": 0.9, "status": "OK", "valid": True},
        },
    }
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"_meta": {"schema": "three-route-live-v1"}}) + "\n")
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = standalone_trace(trace)
    assert result["segment_cnn"]["metrics"]["all"]["province_correct"] == 1
    assert result["crnn"]["metrics"]["all"]["plate_correct"] == 1
    assert result["box"]["metrics"]["all"]["province_correct"] == 0
    assert result["box"]["metrics"]["all"]["wrong_as_anwei"] == 1


def test_char_crop_receives_full_label_for_green_color_context():
    seen = []

    def normalize(image, quad, label):
        seen.append(label)
        return label

    assert _first_char_patch(normalize, object(), [], "沪A123456") == "沪A123456"
    assert seen == ["沪A123456"]
