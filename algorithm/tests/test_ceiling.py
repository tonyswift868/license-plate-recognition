from types import SimpleNamespace

import numpy as np
import pytest
from lpr.config import load_config
from lpr_tools.data.manifest import ManifestRow
from lpr_tools.eval import ceiling
from lpr_tools.eval.ceiling import (
    _ensure_output_available,
    _gt_rect,
    _iou,
    _runtime_overrides,
    _summarize,
)


def _record(subset="ccpd_db", brightness=19, hit=True, actual=True, gt=True):
    return {"subset": subset, "brightness": brightness, "top3_hit": hit,
            "decode_error": None,
            "cells": {name: {route: {"correct": value} for route, value in
                              (("segment_cnn", value), ("crnn", value))}
                      for name, value in (("ground_truth", gt), ("actual", actual))}}


def test_summary_reports_four_cells_subsets_dark_db_and_error_buckets():
    result = _summarize([_record(), _record("ccpd_tilt", 35, False, False, True)])
    assert set(result) == {"all", "ccpd_db", "ccpd_db_brightness_le_19",
                           "ccpd_tilt", "ccpd_tilt_plus_rotate"}
    dark = result["ccpd_db_brightness_le_19"]
    assert dark["ground_truth/segment_cnn"] == {"correct": 1, "n": 1, "accuracy": 1.0}
    assert dark["actual/crnn"]["accuracy"] == 1.0
    assert dark["actual_top3"] == {"hits": 1, "n": 1, "recall": 1.0}
    assert result["ccpd_tilt"]["actual_error_buckets"]["crnn"] == {
        "top3_localization_miss": 1, "top3_hit_but_route_wrong": 0,
        "top3_undefined": 0, "total_wrong": 1}


def test_ground_truth_warp_uses_manifest_quad_and_label_layout():
    cfg = load_config()
    image = np.zeros((100, 200, 3), np.uint8)
    row = ManifestRow("x.jpg", "京A12345", "blue", "night", "ccpd2019/ccpd_db",
                      ((20, 20), (180, 20), (180, 70), (20, 70)))
    rect = _gt_rect(image, row, cfg)
    assert rect.image.shape[:2] == (80, 252)
    assert rect.layout == 7
    assert rect.candidate.source == "ground_truth"
    assert np.isfinite(rect.homography).all()


def test_quad_and_bbox_iou_contract():
    assert _iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert _iou((0, 0, 10, 10), (20, 20, 1, 1)) == 0


def test_decode_error_remains_in_fixed_denominator_and_is_wrong():
    failed = _record("ccpd_db", 5, None, False, False)
    failed["decode_error"] = "decode_failed after retry"
    result = _summarize([failed])
    for cell in ("ground_truth/segment_cnn", "ground_truth/crnn", "actual/segment_cnn", "actual/crnn"):
        assert result["ccpd_db"][cell] == {"correct": 0, "n": 1, "accuracy": 0.0}
    assert result["ccpd_db_brightness_le_19"]["actual_top3"] == {"hits": 0, "n": 1, "recall": 0.0}


def test_existing_outputs_are_not_replaced_without_opt_in(tmp_path):
    report = tmp_path / "report.json"
    trace = tmp_path / "report.jsonl"
    _ensure_output_available(report)
    trace.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _ensure_output_available(report)
    _ensure_output_available(report, overwrite=True)


def test_runtime_thread_option_matches_effective_config_override():
    requested = {"runtime__model_path": "cnn.pt"}
    effective = _runtime_overrides(requested, 2)
    assert effective == {"runtime__model_path": "cnn.pt", "runtime__torch_threads": 2}
    assert requested == {"runtime__model_path": "cnn.pt"}


def _fake_recognizer():
    config = SimpleNamespace(
        segment=object(),
        postprocess=SimpleNamespace(min_confidence=0.8),
    )
    classifier = SimpleNamespace(predict_proba=lambda _chars: None)
    reader = SimpleNamespace(read=lambda _image: ("", []))
    return SimpleNamespace(config=config, classifier=classifier, plate_reader=reader)


def _fake_rect(name):
    return SimpleNamespace(candidate=SimpleNamespace(source=name), color="blue", image=name)


def _fake_decode(status, confidence, plate):
    return SimpleNamespace(status=SimpleNamespace(value=status), confidence=confidence, plate_no=plate)


@pytest.mark.parametrize("route", ["segment", "reader"])
def test_route_first_ok_wins_even_if_later_candidate_matches(monkeypatch, route):
    recognizer = _fake_recognizer()
    truth = "京A12345"
    if route == "segment":
        monkeypatch.setattr(ceiling, "_layouts_to_try", lambda _rect: [7])
        monkeypatch.setattr(ceiling, "segment", lambda *_args: SimpleNamespace(ok=True, layout=7, chars=[]))
        decodes = iter([_fake_decode("OK", 0.9, "京A12346"), _fake_decode("OK", 0.95, truth)])
        monkeypatch.setattr(ceiling, "decode", lambda *_args: next(decodes))
    else:
        reads = iter([("京A12346", [0.9] * 7), (truth, [0.95] * 7)])
        recognizer.plate_reader.read = lambda _image: next(reads)
    prediction, correct = (ceiling._segment_read if route == "segment" else ceiling._reader_read)(
        [_fake_rect("first"), _fake_rect("second")], SimpleNamespace(plate_no=truth), recognizer)
    assert prediction == "京A12346"
    assert correct is False


@pytest.mark.parametrize("route", ["segment", "reader"])
def test_route_fallback_ranks_status_then_confidence(monkeypatch, route):
    recognizer = _fake_recognizer()
    truth = "京A12345"
    if route == "segment":
        monkeypatch.setattr(ceiling, "_layouts_to_try", lambda _rect: [7])
        monkeypatch.setattr(ceiling, "segment", lambda *_args: SimpleNamespace(ok=True, layout=7, chars=[]))
        decodes = iter([_fake_decode("LOW_CONFIDENCE", 0.4, "京A12346"),
                        _fake_decode("LOW_CONFIDENCE", 0.7, truth)])
        monkeypatch.setattr(ceiling, "decode", lambda *_args: next(decodes))
    else:
        reads = iter([("京A12346", [0.4] * 7), (truth, [0.7] * 7)])
        recognizer.plate_reader.read = lambda _image: next(reads)
    prediction, correct = (ceiling._segment_read if route == "segment" else ceiling._reader_read)(
        [_fake_rect("first"), _fake_rect("second")], SimpleNamespace(plate_no=truth), recognizer)
    assert prediction == truth
    assert correct is True
