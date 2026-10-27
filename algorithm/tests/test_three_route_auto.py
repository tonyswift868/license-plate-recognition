from __future__ import annotations

import pytest

from lpr_tools.eval.three_route import replay_auto
from lpr_tools.eval.three_route_auto import (
    LEGACY_SOLO_READER,
    LEGACY_SOLO_SEGMENT,
    LEGACY_SR,
    TAU_AGREE,
    TAU_BOX,
    evaluate_family,
    reference_decision,
    score_policy,
)


def _attempt(plate, confidence, status="OK"):
    return {"plate": plate, "confidence": confidence, "status": status,
            "valid": status == "OK"}


def _rect(ci=0, ri=0, segment=None, reader=None):
    return {"candidate_index": ci, "rect_index": ri,
            "segment": segment, "reader": reader}


def _record(truth="A", rects=(), boxes=None, **extra):
    return {"path": "/sample.jpg", "split": "val", "truth": truth,
            "subset": "ccpd_db", "category": "clear", "brightness": 19,
            "rects": list(rects), "boxes": boxes or {}, "candidate_count": 1, **extra}


def test_baseline_matches_three_route_replay_and_archived_thresholds():
    assert (TAU_AGREE, TAU_BOX) == (0.8839523792266846, 0.9605637192726135)
    row = _record(rects=[_rect(segment=_attempt("A", .9), reader=_attempt("A", .91))],
                  boxes={"0": _attempt("A", .97)})
    expected = replay_auto(row, TAU_AGREE, TAU_BOX)
    actual = reference_decision(row, "000")
    assert actual["status"] == expected["status"]
    assert actual["plate"] == expected["plate"]
    assert actual["route"] == expected["selected_route"]
    assert actual["candidate_index"] == expected["candidate_index"]


def test_legacy_same_rect_gate_only_adds_predeclared_agreement():
    row = _record(rects=[_rect(segment=_attempt("A", .84), reader=_attempt("A", .95))])
    assert reference_decision(row, "000")["status"] == "NO_AUTO"
    out = reference_decision(row, "100")
    assert out["status"] == "OK"
    assert out["event"] == "legacy_sr_agree"
    assert out["route"] == "segment"
    assert min(.84, .95) >= LEGACY_SR
    assert out["confidence"] == .95  # pipeline._arbitrate keeps max pair confidence


def test_legacy_solo_thresholds_prefer_segment_when_both_qualify():
    row = _record(rects=[_rect(segment=_attempt("S", LEGACY_SOLO_SEGMENT),
                               reader=_attempt("R", LEGACY_SOLO_READER))])
    assert reference_decision(row, "011")["route"] == "segment"
    reader_only = _record(rects=[_rect(segment=_attempt("S", .90),
                                       reader=_attempt("R", LEGACY_SOLO_READER))])
    assert reference_decision(reader_only, "001")["route"] == "reader"


def test_rule_decision_is_independent_of_truth_and_preserves_rect_order():
    rects = [_rect(0, 0, _attempt("X", .99), _attempt("X", .99)),
             _rect(0, 1, _attempt("A", .99), _attempt("A", .99))]
    first = _record(truth="A", rects=rects)
    second = {**first, "truth": "different-label"}
    assert reference_decision(first, "000") == reference_decision(second, "000")
    assert reference_decision(first, "000")["rect_index"] == 0


def test_invalid_attempts_never_pass_even_with_zero_confidence_threshold():
    row = _record(rects=[_rect(segment=_attempt("A", 1.0, "INVALID_FORMAT"),
                               reader=_attempt("A", 1.0, "INVALID_FORMAT"))],
                  boxes={"0": _attempt("A", 1.0, "INVALID_FORMAT")})
    assert reference_decision(row, "111")["status"] == "NO_AUTO"


def test_family_is_exactly_sixteen_and_requires_each_split_precision():
    rows = []
    for split in ("val", "devhard"):
        rows.extend([
            _record(truth="A", split=split,
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))]),
            _record(truth="B", split=split,
                    rects=[_rect(segment=_attempt("wrong", .95), reader=_attempt("wrong", .95))]),
        ])
    result = evaluate_family(rows)
    assert set(result["policies"]) == ({f"{index:03b}" for index in range(8)} |
                                        {f"safe-{index:03b}" for index in range(8)})
    assert result["baseline_policy"] == "000"
    assert result["selected_policy"] is None  # every accepting policy is only 50% precise
    assert set(result["eligible"]) == set()


@pytest.mark.parametrize("truth", ["A", "X"])
def test_safe_augment_preserves_every_baseline_ok_field(truth):
    row = _record(truth=truth, rects=[_rect(segment=_attempt("X", .90),
                                             reader=_attempt("X", .91))])
    baseline = reference_decision(row, "000")
    safe = reference_decision(row, "safe-111")
    assert baseline["status"] == "OK"
    assert safe == baseline


def test_safe_augment_runs_legacy_only_after_complete_baseline_fails():
    row = _record(rects=[
        _rect(0, 0, _attempt("S", .84), _attempt("S", .84)),
        _rect(1, 0, _attempt("A", .99), _attempt("A", .99)),
    ], boxes={"0": _attempt("B", .5), "1": _attempt("B", .5)}, candidate_count=2)
    baseline = reference_decision(row, "000")
    safe = reference_decision(row, "safe-100")
    assert baseline["status"] == "OK"  # later baseline pass wins over earlier legacy candidate
    assert baseline["candidate_index"] == 1
    assert safe == baseline

    no_baseline = _record(rects=[_rect(segment=_attempt("S", .84), reader=_attempt("S", .84))])
    augmented = reference_decision(no_baseline, "safe-100")
    assert augmented["event"] == "legacy_sr_agree"
    assert augmented["confidence"] == .84


def test_policy_dictionary_accepts_fresh_base_report_thresholds():
    row = _record(rects=[_rect(segment=_attempt("A", .75), reader=_attempt("A", .8))])
    assert reference_decision(row, {"base_thresholds": {"tau_agree": .7, "tau_box": .9}})["status"] == "OK"


def test_dark_subgroup_is_ccpd_db_only_and_whole_accuracy_uses_fixed_fallback():
    db_dark = _record(truth="A", subset="ccpd_db", brightness=19,
                      rects=[_rect(segment=_attempt("wrong", .7), reader=_attempt("A", .7))])
    other_dark = _record(truth="A", subset="ccpd_tilt", brightness=1,
                         rects=[_rect(segment=_attempt("wrong", .7), reader=_attempt("A", .7))])
    metrics = score_policy([db_dark, other_dark], "000")
    assert metrics["val/ccpd_db_brightness_le_19"]["n"] == 1
    assert "val/ccpd_tilt_brightness_le_19" not in metrics
    assert metrics["val"]["whole_plate_correct"] == 2


def test_unknown_policy_id_is_rejected():
    with pytest.raises(ValueError):
        reference_decision(_record(), "0101")


def test_fit_hold_assignments_are_reported_but_do_not_change_auto_selection():
    rows = [_record(truth="A", split="val", path="/val.jpg",
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))]),
            _record(truth="A", split="devhard", path="/hard.jpg",
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))])]
    assignments = {"/val.jpg": "fit", "/hard.jpg": "hold"}
    without = evaluate_family(rows)
    with_halves = evaluate_family(rows, assignments=assignments)
    assert without["selected_policy"] == with_halves["selected_policy"]
    assert with_halves["policies"]["000"]["fit"]["n"] == 1
    assert with_halves["policies"]["000"]["hold"]["n"] == 1
