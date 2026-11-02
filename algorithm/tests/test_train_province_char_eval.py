"""Stub contract checks for CharCNN production-style segment evaluation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from lpr_tools.train.eval_province_char import (
    _alnum_analysis,
    _position_top1,
    _segment_decode,
    _segment_record,
    _summarize,
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


def test_new_province_logit_can_steal_global_alnum_top1_but_not_conditioned_top1():
    from lpr.charset import CHAR_TO_INDEX, CHARSET, DIGIT_IDX, LETTER_IDX

    old = np.zeros((7, len(CHARSET)), dtype=np.float32)
    new = old.copy()
    a_idx = CHAR_TO_INDEX["A"]
    province_idx = CHAR_TO_INDEX["皖"]
    old[1, a_idx] = new[1, a_idx] = .8
    old[1, province_idx] = .2
    new[1, province_idx] = .9
    # The fine-tune changed a province-class score only; old alphanumeric
    # logits remain bitwise identical.
    alnum_idx = tuple(DIGIT_IDX) + tuple(LETTER_IDX)
    assert np.array_equal(old[:, alnum_idx], new[:, alnum_idx])

    old_audit = _position_top1(old, "L7")
    new_audit = _position_top1(new, "L7")
    assert old_audit["raw_top1"][1] == "A"
    assert new_audit["raw_top1"][1] == "皖"
    assert new_audit["position_conditioned_top1"][1] == "A"

    analysis = _alnum_analysis({"_classifier_probabilities": new,
                                "_selected_layout_name": "L7"}, "陕A12345")
    assert analysis["positions"][0]["truth"] == "A"
    assert analysis["positions"][0]["raw_top1"] == "皖"
    assert analysis["positions"][0]["position_conditioned_top1"] == "A"


def test_alnum_classifier_audit_skips_failed_or_mismatched_segmentation():
    failed = _alnum_analysis({"prediction": None}, "陕A12345")
    mismatch = _alnum_analysis({"_classifier_probabilities": np.zeros((6, 65)),
                                "_selected_layout_name": "L7"}, "陕A12345")
    assert failed["skip_reason"] == "segment_failed"
    assert mismatch["skip_reason"] == "char_count_mismatch"
