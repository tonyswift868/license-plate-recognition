import math
import pytest
from lpr.config import load_config
from lpr.errors import ConfigError


def test_three_route_override_retains_old_default_and_full_precision():
    baseline = load_config("config/legacy_agree_v1.toml")
    candidate = load_config("config/three_route_v2.toml")
    assert baseline.postprocess.arbitration == "agree"
    assert baseline.runtime.box_reader_model_path == ""
    assert candidate.runtime.corner_model_path == "models/plate_det_v1.pt"
    assert candidate.runtime.box_reader_model_path == "models/box_reader_v2.pt"
    assert candidate.postprocess.three_route_agree_min_confidence == 0.7329292297363281
    assert candidate.postprocess.three_route_box_solo_min_confidence == 0.9414944052696228
    assert candidate.postprocess.three_route_segment_solo
    assert candidate.postprocess.three_route_reader_solo
    assert not candidate.postprocess.three_route_legacy_sr_agreement
    assert candidate.postprocess.three_route_province_support_min_routes == 0
    assert candidate.postprocess.fallback_predicted_tilt_min_deg == 20.0


@pytest.mark.parametrize("key,value", [
    ("three_route_agree_min_confidence", -0.01),
    ("three_route_agree_min_confidence", 1.01),
    ("three_route_box_solo_min_confidence", float("nan")),
    ("three_route_box_solo_min_confidence", float("inf")),
    ("three_route_legacy_sr_agreement", 1),
    ("three_route_segment_solo", "yes"),
    ("three_route_reader_solo", []),
    ("three_route_province_support_min_routes", 1),
    ("three_route_province_support_min_routes", 3),
    ("fallback_predicted_tilt_min_deg", -1),
    ("fallback_predicted_tilt_min_deg", float("inf")),
])
def test_new_three_route_keys_reject_invalid_values(key, value):
    with pytest.raises(ConfigError):
        load_config(**{f"postprocess__{key}": value})


@pytest.mark.parametrize("v", [0., 1.])
def test_three_route_confidence_endpoints_are_valid(v):
    cfg = load_config(postprocess__three_route_agree_min_confidence=v,
                      postprocess__three_route_box_solo_min_confidence=v)
    assert cfg.postprocess.three_route_agree_min_confidence == v
