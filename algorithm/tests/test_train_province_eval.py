"""Small contract tests for batch-12 trace parsing and route summaries."""
from __future__ import annotations

import gzip
import json
import csv
from pathlib import Path
import pytest

from lpr_tools.train.eval_province_readers import (
    _aggregate, _selected, _timing_summary, standalone_trace, summarize_testset_composition,
    workflow_trace,
)
import lpr_tools.train.eval_province_readers as province_eval
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
    assert metrics["anwei"]["n"] == 1
    assert metrics["anwei/devhard"]["n"] == 1
    assert metrics["anwei/devhard/ccpd_tilt"]["n"] == 1
    assert metrics["val/ccpd_db"]["n"] == 1
    assert metrics["dark_le_19"]["n"] == 1
    assert metrics["val/ccpd_db_brightness_le_19"]["n"] == 1
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


def test_workflow_reads_real_auto_report_policy_and_split_coverage(tmp_path: Path):
    trace = tmp_path / "raw.jsonl.gz"
    expected = tmp_path / "expected.jsonl.gz"
    workflow_dir = tmp_path / "workflow"
    workflow_dir.mkdir()
    records = [
        {"path": "v.jpg", "truth": "陕A12345", "split": "val", "subset": "ccpd_base"},
        {"path": "d.jpg", "truth": "皖A12345", "split": "devhard", "subset": "ccpd_base"},
    ]
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    decisions = [{"path": "v.jpg", "status": "OK", "plate": "陕A12345"},
                 {"path": "d.jpg", "status": "NO_READ", "plate": None}]
    with gzip.open(expected, "wt", encoding="utf-8") as stream:
        for row in decisions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    auto_report = {"policies": {"011": {
        "val": {"n": 1, "accepted": 1, "accepted_correct": 1},
        "devhard": {"n": 1, "accepted": 0, "accepted_correct": 0},
    }}}
    auto_path = workflow_dir / "auto-review.json"
    auto_path.write_text(json.dumps(auto_report), encoding="utf-8")
    (workflow_dir / "workflow.json").write_text(json.dumps({
        "mode": "all", "fresh": True, "trace": str(trace),
        "expected_runtime": {"path": str(expected), "rows": 2},
        "auto_report": str(auto_path), "auto_policy": "011",
    }), encoding="utf-8")
    result = workflow_trace(workflow_dir)
    assert result["accepted_metrics"]["val"] == {
        "n": 1, "accepted": 1, "correct": 1, "precision": 1.0, "coverage": 1.0}
    assert result["accepted_metrics"]["devhard"]["coverage"] == 0.0
    assert result["accepted_metrics"]["combined"]["composition"] == {"val": 1, "devhard": 1}
    assert result["runtime_projection"]["rows"] == 2


def test_box_timing_summary_keeps_first_cold_call_and_nearest_rank_p95():
    summary = _timing_summary([12.0, 2.0, 3.0, 4.0])
    assert summary["calls"] == 4
    assert summary["mean_ms"] == 5.25
    assert summary["p95_nearest_rank_ms"] == 12.0
    assert summary["max_ms"] == 12.0
    assert summary["over_10ms"] == 1


def test_trace_box_timing_reads_every_candidate_and_checks_trace_match(tmp_path, monkeypatch):
    from lpr import boxcrop, io_utils, recognize

    class Reader:
        box_pad = (0.1, 0.2)
        calls = 0

        def read(self, crop):
            self.calls += 1
            return "陕A12345", [0.8, 0.9]

    reader = Reader()

    row = {"path": "fixture.jpg", "boxes": {
        "0": {"bbox": [1, 2, 10, 5], "plate": "陕A12345", "confidence": 0.8,
              "status": "OK", "valid": True},
        "1": {"bbox": [2, 3, 10, 5], "plate": "陕A12345", "confidence": 0.8,
              "status": "OK", "valid": True},
        "2": {"bbox": [3, 3, 10, 5], "plate": "陕A12345", "confidence": 0.8,
              "status": "OK", "valid": True},
    }}
    crops = []
    monkeypatch.setattr(province_eval, "_jsonl_gz", lambda _path: iter([row]))
    monkeypatch.setattr(io_utils, "imread", lambda _path: object())
    class Crop:
        def __init__(self, size):
            self.size = size

    def crop_box(image, bbox, px, py):
        crops.append((bbox, px, py))
        return Crop(0 if bbox[0] == 2 else 3)

    monkeypatch.setattr(boxcrop, "crop_box", crop_box)
    monkeypatch.setattr(recognize, "load_box_reader", lambda _path: reader)
    import torch
    import cv2
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(cv2, "setNumThreads", lambda _count: None)

    report = province_eval.time_box_trace_candidates(
        tmp_path / "trace.jsonl.gz", tmp_path / "reader.pt", tmp_path / "timing.json")
    assert report["calls"]["calls"] == 2
    assert report["trace_match"] is True
    assert report["skipped_bad_crops"] == 1
    assert len(crops) == 3
    assert reader.calls == 2
    with Path(report["csv"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [item["call_index"] for item in rows] == ["0", "1"]


def test_testset_composition_uses_manifest_counts_and_marks_missing_groups(tmp_path):
    manifest = tmp_path / "testset.csv"
    manifest.write_text(
        "path,plate_no,plate_color,category,source,quad,note\n"
        "a,1,blue,clear,ccpd_base,,\n"
        "b,1,blue,clear,ccpd_base,,\n"
        "c,1,green,clear,ccpd_green,,\n"
        "d,1,blue,night,ccpd_db,,\n"
        "e,1,blue,tilt,ccpd_tilt,,\n"
        "f,1,blue,tilt,ccpd_tilt,,\n"
        "g,1,blue,tilt,ccpd_rotate,,\n"
        "h,1,blue,dirty,ccpd_weather,,\n", encoding="utf-8")
    metrics = {}
    for split in ("val", "devhard"):
        for group, accuracy in {
            "ccpd_base": .8, "ccpd_green": .5, "ccpd_db_brightness_le_19": .9,
            "ccpd_tilt": .75, "ccpd_rotate": .45, "ccpd_weather": .4,
        }.items():
            metrics[f"{split}/{group}"] = {"plate_accuracy": accuracy}
    del metrics["devhard/ccpd_weather"]
    result = summarize_testset_composition(manifest, metrics)
    val = result["scenarios"]["val"]
    assert val["clear"]["estimated_accuracy"] == pytest.approx(.7)
    assert val["tilt"]["estimated_accuracy"] == pytest.approx(.65)
    assert val["night"]["estimated_accuracy"] == pytest.approx(.9)
    assert result["scenarios"]["devhard"]["dirty"]["estimable"] is False
    assert result["scenarios"]["devhard"]["dirty"]["estimated_accuracy"] is None
