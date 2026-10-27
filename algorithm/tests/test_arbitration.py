"""Pure decision tests for the frozen three-route 011 + T20_A policy."""
from __future__ import annotations

import math

import pytest

from lpr.arbitration import _fallbacks, _valid, arbitrate, try_auto_candidate


def cfg(**updates):
    value = {
        "three_route_agree_min_confidence": 0.7,
        "three_route_box_solo_min_confidence": 0.95,
        "three_route_legacy_sr_agreement": False,
        "three_route_segment_solo": False,
        "three_route_reader_solo": False,
        "three_route_province_support_min_routes": 0,
        "fallback_predicted_tilt_min_deg": 20.0,
        "solo_min_confidence_segment": 0.8,
        "solo_min_confidence_reader": 0.9,
        "agree_min_confidence": 0.82,
        "nonok_fallback": "predicted_tilt_then_corrected",
    }
    value.update(updates)
    return value


def attempt(plate="京A12345", confidence=0.7, status="OK"):
    return {"status": status, "plate": plate, "confidence": confidence}


def rect(s=None, r=None, ci=0, ri=0, color="blue"):
    return {"candidate_index": ci, "rect_index": ri, "segment": s, "reader": r, "color": color}


@pytest.mark.parametrize("left,right,expected", [
    ("segment", "reader", "segment"),
    ("segment", "box", "segment"),
    ("reader", "box", "reader"),
])
def test_pair_order_and_left_route_confidence(left, right, expected):
    same = attempt(confidence=0.75)
    row = rect(same if left == "segment" or right == "segment" else None,
               same if left == "reader" or right == "reader" else None)
    box = same if "box" in (left, right) else attempt("皖B88888", .99)
    # If S/R are both present, their pair wins even where the intended test is later.
    result = try_auto_candidate(0, [row], box, cfg(), candidate_complete=True)
    assert result is not None
    assert result.status == "OK"
    assert result.route == expected
    assert result.confidence == .75


@pytest.mark.parametrize("confidence", [0.7, math.nextafter(0.7, 0.0)])
def test_pair_gate_uses_exact_greater_equal(confidence):
    result = try_auto_candidate(0, [rect(attempt(confidence=confidence), attempt(confidence=0.99))],
                                None, cfg(), candidate_complete=True)
    assert (result is not None) is (confidence >= 0.7)


def test_all_three_matching_routes_take_sr_pair_first():
    row = rect(attempt("京A11111", .71), attempt("京A11111", .99))
    result = try_auto_candidate(0, [row], attempt("京A11111", .98), cfg(), candidate_complete=True)
    assert result.route == "segment" and result.confidence == .71


@pytest.mark.parametrize("route,threshold", [("segment", .8), ("reader", .9)])
@pytest.mark.parametrize("passed", [True, False])
def test_each_solo_gate_has_exact_inclusive_boundary(route, threshold, passed):
    conf = threshold if passed else math.nextafter(threshold, 0.0)
    row = rect(attempt("京A11111", conf), attempt("沪A22222", conf))
    result = try_auto_candidate(0, [row], None,
                                cfg(**{f"three_route_{route}_solo": True}), candidate_complete=True)
    assert (result is not None) is passed
    if passed:
        assert result.route == route
        assert result.confidence == threshold


def test_both_solos_passing_same_rect_prioritizes_segment():
    result = try_auto_candidate(0, [rect(attempt("京A11111", .8), attempt("沪A22222", .9))], None,
                                cfg(three_route_segment_solo=True, three_route_reader_solo=True),
                                candidate_complete=True)
    assert result.route == "segment" and result.confidence == .8


def test_each_pair_uses_its_own_gate_and_invalid_or_nonfinite_attempt_is_ignored():
    row = rect(attempt("京A11111", .69), attempt("沪A22222", 1.0))
    result = try_auto_candidate(0, [row], attempt("沪A22222", .7), cfg(), candidate_complete=True)
    assert result.route == "reader"  # R/B is the first matching legal pair.
    bad = rect(attempt(confidence=float("nan")), attempt(confidence=float("inf")))
    assert try_auto_candidate(0, [bad], attempt(confidence=-.1), cfg(), candidate_complete=True) is None


def test_box_solo_is_delayed_until_candidate_complete_and_has_no_rect_index():
    box = attempt(confidence=.95)
    assert try_auto_candidate(0, [rect()], box, cfg(), candidate_complete=False) is None
    result = try_auto_candidate(0, [rect(), rect(ri=1)], box, cfg(), candidate_complete=True)
    assert result.status == "OK" and result.route == "box"
    assert result.candidate_index == 0 and result.rect_index is None


@pytest.mark.parametrize("confidence", [.95, math.nextafter(.95, 0.0)])
def test_box_solo_gate_uses_exact_greater_equal(confidence):
    result = try_auto_candidate(0, [rect()], attempt(confidence=confidence), cfg(), candidate_complete=True)
    assert (result is not None) is (confidence >= .95)


def test_first_rect_event_beats_later_rect_and_candidate():
    rows = [rect(attempt(confidence=.81), ci=0), rect(attempt(confidence=.99), ci=1)]
    result = try_auto_candidate(0, rows, None,
                                cfg(three_route_segment_solo=True), candidate_complete=True)
    assert result.candidate_index == 0 and result.rect_index == 0


def test_legacy_sr_gate_is_configurable_and_threshold_inclusive():
    row = rect(attempt(confidence=.82), attempt(confidence=.82))
    result = try_auto_candidate(0, [row], None,
                                cfg(three_route_agree_min_confidence=.9,
                                    three_route_legacy_sr_agreement=True, agree_min_confidence=.82),
                                candidate_complete=True)
    assert result.route == "segment" and result.confidence == .82
    assert try_auto_candidate(0, [row], None,
        cfg(three_route_agree_min_confidence=.9, three_route_legacy_sr_agreement=False),
        candidate_complete=True) is None
    assert try_auto_candidate(0, [row], None,
        cfg(three_route_agree_min_confidence=.9, three_route_legacy_sr_agreement=True,
            agree_min_confidence=math.nextafter(.82, 1.0)), candidate_complete=True) is None


def test_tilt_gate_is_inclusive_at_its_configured_float_boundary():
    from lpr.arbitration import _tilt_degrees
    q=[[0,0],[10,10*math.tan(math.radians(20))],
       [10,10+10*math.tan(math.radians(20))],[0,10]]
    angle=_tilt_degrees(q)
    assert abs(angle-20.0)<1e-12
    trace={"candidate_count":1,"candidate_corners":{"0":q},
        "rects":[rect(attempt(confidence=.5),attempt(confidence=.4))],"boxes":{"0":None}}
    common={"three_route_agree_min_confidence":1.,"fallback_predicted_tilt_min_deg":angle}
    assert arbitrate(trace,cfg(**common)).route=="reader"
    common["fallback_predicted_tilt_min_deg"]=math.nextafter(angle,math.inf)
    assert arbitrate(trace,cfg(**common)).route=="segment"


def test_manual_confidence_ties_keep_earlier_candidates():
    trace={"candidate_count":2,"candidate_corners":{"0":[[0,0],[10,10],[10,20],[0,10]]},
        "rects":[rect(None,attempt("京A11111",.4),ci=0),
                 rect(None,attempt("京A22222",.4),ci=1)],
        "boxes":{"0":attempt("京B11111",.3),"1":attempt("京B22222",.3)}}
    assert arbitrate(trace,cfg(three_route_agree_min_confidence=1.)).candidate_index==0
    trace['candidate_corners']={}
    decision=arbitrate(trace,cfg(three_route_agree_min_confidence=1.))
    assert decision.route=='box' and decision.candidate_index==0


def test_legacy_fallback_uses_old_solo_policy_even_when_new_switches_are_off():
    row = rect(attempt("京A11111", .85), None)
    _, _, legacy = _fallbacks([row], {}, cfg(three_route_segment_solo=False,
                                               solo_min_confidence_segment=.8), 1)
    assert legacy.status == "LOW_CONFIDENCE"
    assert (legacy.route, legacy.plate, legacy.confidence) == ("segment", "京A11111", .85)


def test_attempt_validity_matches_frozen_eval_default_status_rule():
    assert _valid(attempt())
    assert not _valid({"status": "LOW_CONFIDENCE", "plate": "A", "confidence": .8})
    assert _valid({"status": "LOW_CONFIDENCE", "valid": True, "plate": "A", "confidence": .8})
    assert not _valid({"status": "MYSTERY", "plate": "A", "confidence": .8})
    assert not _valid({"status": "OK", "valid": False, "plate": "A", "confidence": .8})
    assert not _valid({"status": "BAD_INPUT", "valid": True, "plate": "A", "confidence": .8})


def test_tilt_t20_a_and_missing_or_nonfinite_corners_fall_back_to_m():
    # M's first matching pair is S/B, while A uses the S/R pair and its minimum.
    row = rect(attempt("京A11111", .8), attempt("京A11111", .6))
    row["candidate_index"] = 0
    box = attempt("京A11111", .9)
    trace = {"candidate_count": 1, "rects": [row], "boxes": {"0": box},
             "candidates": [{"corners": [[0, 0], [10, 5], [10, 15], [0, 10]]}]}
    no_auto = cfg(three_route_agree_min_confidence=1.0,
                  three_route_box_solo_min_confidence=1.0)
    tilted = arbitrate(trace, no_auto)
    assert tilted.route == "reader" and tilted.confidence == .6  # A
    trace["candidates"][0]["corners"] = None
    missing = arbitrate(trace, no_auto)
    assert missing.route == "segment" and missing.confidence == .6  # M's first S/R pair
    trace["candidates"][0]["corners"] = [[0, 0], [10, float("nan")], [10, 15], [0, 10]]
    assert arbitrate(trace, no_auto).route == "segment"


@pytest.mark.parametrize("color", ["white", "black"])
def test_white_black_projection_overrides_manual_but_not_auto(color):
    manual = {"candidate_count": 1,
              "rects": [rect(attempt(confidence=.3), attempt(confidence=.4), color=color)],
              "boxes": {"0": None}}
    assert arbitrate(manual, cfg()).status == "UNSUPPORTED"
    raw = arbitrate(manual, cfg(), runtime=False)
    assert raw.status == "LOW_CONFIDENCE" and raw.plate == "京A12345"
    auto = {"candidate_count": 1, "rects": [rect(attempt(confidence=.8), color=color)], "boxes": {"0": None}}
    assert arbitrate(auto, cfg(three_route_segment_solo=True)).status == "OK"


def test_white_color_suppresses_only_l_fallback_before_runtime_projection():
    row = rect(attempt(confidence=.3), attempt(confidence=.4), color="white")
    a, m, legacy = _fallbacks([row], {}, cfg(), 1)
    assert a is not None and m is not None
    assert legacy is None


def test_invalid_format_is_retained_and_no_read_projects_to_seg_failed():
    invalid = attempt("bad", .73, "INVALID_FORMAT")
    trace = {"candidate_count": 1, "rects": [rect(invalid, None)], "boxes": {"0": None}}
    result = arbitrate(trace, cfg())
    assert (result.status, result.plate, result.route, result.confidence) == (
        "INVALID_FORMAT", "bad", "segment", .73)
    trace["rects"] = [rect({"status": "SEG_FAILED"}, None)]
    result = arbitrate(trace, cfg())
    assert result.status == "SEG_FAILED" and result.confidence == 0.0


def test_bad_input_no_plate_and_p2_veto_terminal_semantics():
    assert arbitrate({"input_error": "bad"}, cfg()).status == "BAD_INPUT"
    assert arbitrate({"candidate_count": 0, "rects": [], "boxes": {}}, cfg()).status == "NO_PLATE"
    # First auto event fails province support. Later candidate must not steal auto acceptance.
    row0 = rect(attempt("京A11111", .8), None, ci=0)
    row1 = rect(attempt("沪B22222", .99), attempt("沪B22222", .99), ci=1)
    sentinel = try_auto_candidate(0, [row0], attempt("沪C33333", .99),
                                  cfg(three_route_segment_solo=True, solo_min_confidence_segment=.8,
                                      three_route_province_support_min_routes=2), candidate_complete=True)
    assert sentinel.status == "NO_AUTO"
    final = arbitrate({"candidate_count": 2, "rects": [row0, row1], "boxes": {"0": None, "1": None}},
                      cfg(three_route_segment_solo=True, solo_min_confidence_segment=.8,
                          three_route_province_support_min_routes=2))
    assert final.status == "LOW_CONFIDENCE"
    assert final.candidate_index == 1
