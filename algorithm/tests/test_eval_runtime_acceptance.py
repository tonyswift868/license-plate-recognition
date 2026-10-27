from lpr_tools.eval.runtime_acceptance import compare


def test_runtime_comparison_keeps_identity_exact_and_float_tolerance_separate():
    expected={"status":"OK","plate":"京A12345","confidence":.8,"route":"segment",
              "candidate_index":0,"rect_index":0}
    assert not compare({**expected,"confidence":.80000001},expected)
    wrong=compare({**expected,"route":"reader","candidate_index":1,"confidence":.81},expected)
    assert set(wrong)=={"route","candidate_index","confidence"}
    # Old baseline intentionally compares only the original three fields.
    assert not compare(expected,{k:expected[k] for k in ("status","plate","confidence")})
    assert "confidence" in compare({**expected,"confidence":.80000001}, expected, tolerance=0)
    assert "confidence" in compare({**expected,"confidence":float("nan")}, expected)
