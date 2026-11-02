import pytest
from lpr_tools.eval.fallback_selection import (
    _identity_verified,
    apply_rule,
    compare_record,
    score,
    select_reader,
    validate_params,
)

PARAMS = {"agree_min": 0.9, "solo_seg": 0.9, "solo_reader": 0.9, "prefer": "segment"}


def attempt(text, conf, valid=True, status="OK"):
    return {"plate": text, "confidence": conf, "valid": valid, "status": status}


def rect(candidate, seg=None, reader=None):
    return {"candidate_index": candidate, "segment": seg, "reader": reader}


def test_ok_is_entirely_preserved():
    record = {"rects": [rect(0, reader=attempt("R1", .99))]}
    baseline = {"plate": "B", "confidence": .81, "status": "OK", "extra": "keep"}
    assert apply_rule(record, baseline, "max_reader") == {
        "plate": "B", "confidence": .81, "status": "OK", "extra": "keep",
        "selection_source": "baseline_preserved", "changed": False}


def test_unsupported_is_preserved_even_with_reader():
    record = {"rects": [rect(0, reader=attempt("R1", .99))]}
    baseline = {"plate": None, "confidence": 0, "status": "UNSUPPORTED"}
    assert apply_rule(record, baseline, "max_reader")["status"] == "UNSUPPORTED"
    assert apply_rule(record, baseline, "max_reader")["plate"] is None


def test_first_candidate_reader_is_distinct_from_late_candidate_max():
    record = {"rects": [rect(1, reader=attempt("late", .99)),
                        rect(0, reader=attempt("first-candidate", .55))]}
    assert select_reader(record, "first_candidate_reader")["plate"] == "first-candidate"
    assert select_reader(record, "max_reader")["plate"] == "late"


def test_same_rect_agreement_supports_but_does_not_raise_status():
    record = {"rects": [rect(0, seg=attempt("pair", .6), reader=attempt("pair", .7)),
                        rect(1, reader=attempt("solo", .95))]}
    baseline = {"plate": "wrong", "confidence": .3, "status": "LOW_CONFIDENCE"}
    result = apply_rule(record, baseline, "agreement_then_reader")
    assert result["plate"] == "pair"
    assert result["confidence"] == .6
    assert result["status"] == "LOW_CONFIDENCE"
    assert "not calibrated probability" in result["confidence_note"]


def test_selection_does_not_depend_on_truth():
    record = {"truth": "first", "rects": [rect(0, reader=attempt("first", .6)),
                                               rect(2, reader=attempt("second", .8))]}
    before = compare_record(record, PARAMS)
    record["truth"] = "second"
    after = compare_record(record, PARAMS)
    assert before == after


def test_group_scoring_rescues_and_regressions():
    rows = [
        {"truth": "T", "split": "val", "subset": "ccpd_green", "metadata": {"brightness": 10},
         "results": {"baseline": {"plate": "X", "status": "LOW_CONFIDENCE"},
                     "max_reader": {"plate": "T", "status": "LOW_CONFIDENCE", "changed": True}}},
        {"truth": "T", "split": "hard", "subset": "ccpd_db", "metadata": {"brightness": 18},
         "results": {"baseline": {"plate": "T", "status": "LOW_CONFIDENCE"},
                     "max_reader": {"plate": "X", "status": "LOW_CONFIDENCE", "changed": True}}},
    ]
    result = score(rows, "max_reader")
    assert result["all"]["rescued_non_ok"] == 1
    assert result["all"]["regressed_non_ok"] == 1
    assert result["valgreen"]["n"] == 1
    assert result["db_brightness_le_19"]["n"] == 1
    assert result["all"]["auto_pass"] == 0
    assert result["all"]["whole_plate_correct"] == 1


def test_new_identity_schema_is_accepted_without_posthoc_migration():
    assert _identity_verified({"verified": True})
    assert _identity_verified({"audit": {"verified": True}})
    assert _identity_verified({"identity_migration": {"verified": True}})
    assert not _identity_verified({})


def test_non_rescuable_count_is_reported_for_baseline_too():
    row = {"truth": "T", "split": "val", "subset": "ccpd_base", "metadata": {},
           "results": {"baseline": {"plate": None, "status": "UNSUPPORTED"},
                       "max_reader": {"plate": None, "status": "UNSUPPORTED"}}}
    assert score([row], "baseline")["all"]["cannot_pure_fallback_rescue"] == 1
    assert score([row], "max_reader")["all"]["cannot_pure_fallback_rescue"] == 1


def test_params_reject_nonfinite_bool_and_unknown_prefer():
    with pytest.raises(ValueError):
        validate_params({**PARAMS, "agree_min": float("nan")})
    with pytest.raises(ValueError):
        validate_params({**PARAMS, "solo_seg": True})
    with pytest.raises(ValueError):
        validate_params({**PARAMS, "prefer": "oracle"})
