"""Agree daily must enforce its own precision and latency contract."""
import pytest
from lpr_tools.eval.daily import _acceptance, _decision_policy, _effective_config


def identity(mode="agree"):
    return {"effective_config": {"postprocess": {"arbitration": mode}}}


@pytest.mark.parametrize("p95,wrong,expected", [(200, 1, "PASS"), (200.01, 1, "PARTIAL_FAILURE"),
                                               (199, 2, "PARTIAL_FAILURE")])
def test_agree_dual_precision_and_latency(p95, wrong, expected):
    val = {"overall_g2": {"auto_pass": 100, "auto_pass_correct": 99}}
    reports = {str(i): {"overall_g2": {"auto_pass": 100, "auto_pass_correct": 100 - wrong}}
               for i in range(7)}
    result = _acceptance(identity(), val, reports, {"wall_time": {"p95_ms": p95}}, True, 13, 0)
    assert result["status"] == expected


def test_agree_does_not_pool_away_bad_val_or_accept_empty_denominator():
    hard = {str(i): {"overall_g2": {"auto_pass": 300, "auto_pass_correct": 300}} for i in range(7)}
    for passed, correct in ((100, 98), (0, 0)):
        val = {"overall_g2": {"auto_pass": passed, "auto_pass_correct": correct}}
        result = _acceptance(identity(), val, hard, {"wall_time": {"p95_ms": 50}}, True, 20, 0)
        assert not result["dual_precision_pass"]
        assert result["status"] == "PARTIAL_FAILURE"


def test_first_ok_keeps_300ms_and_actual_golden_gate():
    val = {"overall_g2": {}}
    hard = {str(i): val for i in range(7)}
    assert _decision_policy(identity("first_ok"))["g1_p95_limit_ms"] == 300
    assert _acceptance(identity("first_ok"), val, hard, {"wall_time": {"p95_ms": 250}},
                       True, 12, 0)["status"] == "PARTIAL_FAILURE"


def test_agree_without_reader_is_rejected_instead_of_silent_first_ok(tmp_path):
    config = tmp_path / "agree.toml"
    config.write_text('[runtime]\nplate_reader_model_path=""\n[postprocess]\narbitration="agree"\n')
    with pytest.raises(ValueError, match="requires a configured plate reader"):
        _effective_config(config)
