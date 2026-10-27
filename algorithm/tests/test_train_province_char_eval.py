"""Stub contract checks for CharCNN production-style segment evaluation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lpr_tools.train.eval_province_char import (
    _segment_decode, _segment_record, _summarize,
)


def _cfg():
    return SimpleNamespace(segment=object(), postprocess=object())


def test_segmentation_failure_counts_as_wrong_plate_in_panel_denominator():
    calls = []

    def segmenter(_rect, layout, _cfg):
        calls.append(layout)
        return SimpleNamespace(ok=False)

    result = _segment_decode(
        SimpleNamespace(), SimpleNamespace(), _cfg(), segmenter=segmenter,
        decoder=lambda *_: None, layout_getter=lambda _rect: [7, 8],
        ranker=lambda _decision: (0, 0.0))
    record = _segment_record(
        {"panel": "province_holdout", "province": "陕", "subset": "tilt"},
        "陕A12345", result)
    metrics = _summarize([record])["overall"]
    assert calls == [7, 8]
    assert record["status"] == "SEG_FAILED"
    assert metrics["n"] == 1
    assert metrics["plate_n"] == 1
    assert metrics["plate_correct"] == 0
    assert metrics["plate_accuracy"] == 0.0
    assert metrics["province_accuracy"] == 0.0


def test_legal_layout_tie_keeps_the_first_production_layout():
    layouts = []

    def segmenter(_rect, layout, _cfg):
        layouts.append(layout)
        return SimpleNamespace(ok=True, chars=np.zeros((7, 32, 32), dtype=np.uint8),
                               layout=layout)

    classifier = SimpleNamespace(predict_proba=lambda _chars: np.ones((7, 65), dtype=np.float32))
    plates = iter(("陕A12345", "皖A12345"))

    def decoder(*_args):
        return SimpleNamespace(status="LOW_CONFIDENCE", plate_no=next(plates), confidence=.8)

    result = _segment_decode(
        SimpleNamespace(color="unknown"), classifier, _cfg(), segmenter=segmenter,
        decoder=decoder, layout_getter=lambda _rect: [7, 8],
        ranker=lambda _decision: (2, .8))
    assert layouts == [7, 8]
    assert result["prediction"] == "陕A12345"
    assert result["attempted_layouts"] == [7, 8]
