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


def test_family_is_exactly_thirty_two_and_requires_fit_split_precision():
    rows = []
    for split in ("val", "devhard"):
        rows.extend([
            _record(truth="A", split=split,
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))]),
            _record(truth="B", split=split,
                    rects=[_rect(segment=_attempt("wrong", .95), reader=_attempt("wrong", .95))]),
        ])
    result = evaluate_family(rows)
    assert set(result["fit_policies"]) == ({f"{index:03b}" for index in range(8)} |
        {f"safe-{index:03b}" for index in range(8)} |
        {f"p2-{index:03b}" for index in range(8)} |
        {f"p2-safe-{index:03b}" for index in range(8)})
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


def test_non_wan_subgroup_is_a_fit_metric_stratum():
    rows = [_record(truth="皖A12345", path="/wan.jpg"),
            _record(truth="沪A12345", path="/non-wan.jpg")]
    metrics = score_policy(rows, "000")
    assert metrics["val/non_wan"]["n"] == 1
    assert metrics["val/non_wan"]["accepted"] == 0


def test_unknown_policy_id_is_rejected():
    with pytest.raises(ValueError):
        reference_decision(_record(), "0101")


def test_full_trace_threshold_report_is_rejected_before_auto_selection(tmp_path):
    import json
    from lpr_tools.eval.three_route_auto import _load_base_report
    p = tmp_path / "full-report.json"
    p.write_text(json.dumps({"status":"complete", "search":{"complete":True,
        "best":{"tau_agree":.8,"tau_box":.9,"accepted":10}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="fit-only"):
        _load_base_report(p)


def test_fit_hold_selection_does_not_use_hold_truth():
    rows = [_record(truth="A", split="val", path="/val.jpg",
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))]),
            _record(truth="A", split="devhard", path="/hard.jpg",
                    rects=[_rect(segment=_attempt("A", .95), reader=_attempt("A", .95))])]
    assignments = {"/val.jpg": "fit", "/hard.jpg": "hold"}
    with_halves = evaluate_family(rows, assignments=assignments)
    changed_hold = evaluate_family([{**rows[0]}, {**rows[1], "truth": "not-A"}], assignments=assignments)
    assert with_halves["selected_policy"] == changed_hold["selected_policy"]
    assert with_halves["selected_policy"] is None  # fit lacks a devhard row; hold cannot make it eligible
    assert with_halves["fit_policies"]["000"]["val"]["n"] == 1
    assert with_halves["policies"]["000"]["hold"]["n"] == 1


def test_province_support_gate_rejects_single_route_and_ignores_truth():
    one = _record(truth="皖A12345", rects=[_rect(segment=_attempt("皖A12345", .99),
        reader=_attempt("沪A12345", .5))], boxes={"0": _attempt("粤A12345", .5)})
    assert reference_decision(one, "011")["status"] == "OK"
    rejected = reference_decision(one, "p2-011")
    assert rejected["status"] == "NO_AUTO"
    assert rejected["event"] == "province_single_route_rejected"
    two = {**one, "boxes": {"0": _attempt("皖B12345", .98)}}
    accepted = reference_decision(two, "p2-011")
    assert accepted["status"] == "OK"
    assert reference_decision({**two, "truth": "different"}, "p2-011") == accepted


def test_fit_selects_ungated_then_uses_only_fit_subgroup_trigger_for_gate():
    rows = []
    assignments = {}
    for split in ("val", "devhard"):
        for index in range(200):
            path = f"/{split}-{index}.jpg"
            # 011 adds one wrong single-route green plate among 200. It still
            # clears 99.5% globally, while p2 removes the single-route vote.
            truth = "皖A12345" if index else "粤A12345"
            segment = _attempt("皖A12345", .99)
            reader = _attempt("皖A12345" if index else "沪B12345", .5)
            rows.append(_record(path=path, split=split, truth=truth,
                subset="ccpd_green" if index == 0 else "ccpd_base",
                plate_color="green" if index == 0 else "blue",
                rects=[_rect(segment=segment, reader=reader)], boxes={"0": _attempt("粤C12345", .4)}))
            assignments[path] = "fit"
    result = evaluate_family(rows, assignments=assignments)
    assert result["ungated_fit_winner"] == "010"
    assert result["province_gate"]["watched_fit_subgroups_below_99"]["green"]["accepted"] > 0
    assert result["province_gate"]["adopted"] is True
    assert result["selected_policy"] == "p2-010"
