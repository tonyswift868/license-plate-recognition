import pytest

from lpr_tools.eval.route_features import (
    _box_gray_mean,
    _candidate_agrees,
    _layout_lengths,
    _tilt_angle_deg,
    _validate_identity,
)


def test_tilt_is_mean_absolute_top_and_bottom_edge_angles():
    assert _tilt_angle_deg([(0, 0), (10, 0), (10, 5), (0, 5)]) == 0
    assert _tilt_angle_deg([(0, 0), (10, 2), (10, 7), (0, 5)]) == pytest.approx(
        11.309932474
    )


def test_candidate_identity_requires_count_index_bbox_source_and_score():
    class Candidate:
        source = "corner_net"
        bbox = (1, 2, 30, 12)
        score = 0.875

    _candidate_agrees(Candidate(), {"0": {
        "candidate_index": 0, "source": "corner_net", "bbox": [1, 2, 30, 12],
        "score": 0.875,
    }}, 0)
    with pytest.raises(ValueError, match="bbox mismatch"):
        _candidate_agrees(Candidate(), {"0": {
            "candidate_index": 0, "source": "corner_net", "bbox": [1, 2, 31, 12],
            "score": 0.875,
        }}, 0)


def test_layout_inference_uses_only_valid_attempt_prediction_lengths():
    row = {"rects": [
        {"candidate_index": 0, "segment": {"valid": True, "plate": "ABCDEFG"},
         "reader": {"valid": True, "plate": "ABCDEFGH"}},
        {"candidate_index": 0, "segment": {"valid": False, "plate": "BAD"}},
        {"candidate_index": 1, "reader": {"valid": True, "plate": "12345678"}},
    ]}
    assert _layout_lengths(row, 0) == [7, 8]
    assert _layout_lengths(row, 1) == [8]


def test_box_gray_mean_clips_bbox_and_returns_none_for_empty_intersection():
    import numpy as np

    gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert _box_gray_mean(gray, (-2, 1, 4, 2)) == pytest.approx(float(gray[1:3, 0:2].mean()))
    assert _box_gray_mean(gray, (8, 8, 2, 2)) is None


def test_identity_validation_fails_early_for_model_or_locator_config_mismatch():
    trace_meta = {
        "schema": "three-route-live-v1",
        "models": {"detector": {"sha256": "expected-model-sha"}},
        "config": {"effective_config": {
            "preprocess": {"scale": 1}, "locate": {"threshold": 2},
            "lanes": {"main": {}}, "runtime": {"max_candidates": 3},
        }},
    }
    actual = {"effective_config": trace_meta["config"]["effective_config"]}
    with pytest.raises(ValueError, match="detector model SHA"):
        _validate_identity(trace_meta, {"sha256": "different-model"}, actual)
    with pytest.raises(ValueError, match="preprocess"):
        _validate_identity(trace_meta, {"sha256": "expected-model-sha"}, {
            "effective_config": {**actual["effective_config"], "preprocess": {"scale": 2}}
        })
