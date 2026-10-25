from __future__ import annotations

import pytest

from lpr_tools.eval.arbitration import _rank_rule, replay_attempts


def _attempt(i, plate, confidence, valid=True, color="blue"):
    return {"attempt_index": i, "decoded": True, "plate_no": plate,
            "confidence": confidence, "format_valid": valid, "color": color}


def test_replay_returns_first_ok_even_when_later_confidence_is_higher():
    record = {"lead_color": "blue", "candidate_count": 2, "attempts": [
        _attempt(0, "京A12345", .91), _attempt(1, "京A12345", .99)]}
    result = replay_attempts(record, .80)
    assert result["selected_attempt_index"] == 0
    assert result["early_return_attempt_index"] == 0


def test_replay_threshold_changes_status_and_can_select_later_correct_result():
    record = {"lead_color": "blue", "candidate_count": 2, "attempts": [
        _attempt(0, "京A12346", .75), _attempt(1, "京A12345", .82)]}
    assert replay_attempts(record, .70)["plate_no"] == "京A12346"
    result = replay_attempts(record, .80)
    assert result["plate_no"] == "京A12345"
    assert result["selected_attempt_index"] == 1


def test_invalid_format_never_becomes_ok_and_special_color_wins_if_no_ok():
    invalid = {"lead_color": "white", "candidate_count": 1,
               "attempts": [_attempt(0, "bad", .99, valid=False)]}
    assert replay_attempts(invalid, .0)["status"] == "UNSUPPORTED"
    regular = {"lead_color": "blue", "candidate_count": 1,
               "attempts": [_attempt(0, "bad", .99, valid=False)]}
    assert replay_attempts(regular, .0)["status"] == "INVALID_FORMAT"


def test_replay_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        replay_attempts({"lead_color": "unknown", "candidate_count": 0, "attempts": []}, 1.1)


def test_conservative_rule_preserves_first_ok():
    record = {"lead_color": "blue", "candidate_count": 2, "attempts": [
        _attempt(0, "京A12345", .81, color="blue"),
        _attempt(1, "京A12346", .99, color="blue")]}
    selected = _rank_rule(record, "conservative_first_ok_then_color_layout", .80)
    assert selected["attempt_index"] == 0


def test_conservative_rule_only_when_no_ok_penalizes_unknown():
    record = {"lead_color": "blue", "candidate_count": 2, "attempts": [
        {**_attempt(0, "京AD12345", .79, color="unknown"), "layout": 8},
        {**_attempt(1, "京A12345", .78, color="blue"), "layout": 7}]}
    selected = _rank_rule(record, "conservative_first_ok_then_color_layout", .80)
    assert selected["plate_no"] == "京A12345"


def test_analyze_distinguishes_decode_beam_from_other_attempts():
    from lpr_tools.eval.arbitration import analyze
    wrong = {**_attempt(0, '京A12346', .7), 'layout': 7, 'candidate_index': 0,
             'rect_index': 0, 'seg_ok': True, 'candidates': [('京A12346', .7), ('京A12345', .3)]}
    record = {'truth': '京A12345', 'lead_color': 'blue', 'candidate_count': 1, 'attempts': [wrong]}
    result = analyze([record])
    assert result['wrong_with_other_correct_attempt']['wrong_with_correct_attempt'] == 0
    assert result['decoder_topk_diagnostic']['wrong_with_truth_in_any_decode_topk'] == 1
