"""Default migration keeps the frozen policy and an explicit legacy replay."""
import json
import tomllib

from lpr.config import ALGORITHM_DIR, load_config


def test_default_matches_frozen_three_route_configuration(monkeypatch):
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    current = load_config()
    assert current.model_dump() == load_config("config/three_route_v2.toml").model_dump()
    assert current.runtime.corner_model_path == "models/plate_det_v1.pt"
    assert current.runtime.box_reader_model_path == "models/box_reader_v2.pt"
    assert current.postprocess.arbitration == "three_route"
    assert current.postprocess.three_route_agree_min_confidence == 0.7329292297363281
    assert current.postprocess.three_route_box_solo_min_confidence == 0.9414944052696228
    assert current.postprocess.nonok_fallback == "predicted_tilt_then_corrected"
    assert current.postprocess.fallback_predicted_tilt_min_deg == 20.0


def test_legacy_override_contains_exactly_the_changed_values(monkeypatch):
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    legacy = load_config("config/legacy_agree_v1.toml")
    current = load_config()
    expected = {
        "runtime": {"corner_model_path": "models/corner_net_v1.pt", "box_reader_model_path": ""},
        "postprocess": {"arbitration": "agree", "nonok_fallback": "agreement_then_reader",
                        "three_route_agree_min_confidence": 1.0,
                        "three_route_box_solo_min_confidence": 1.0,
                        "three_route_segment_solo": False, "three_route_reader_solo": False,
                        "fallback_predicted_tilt_min_deg": 0.0},
    }
    with (ALGORITHM_DIR / "config/legacy_agree_v1.toml").open("rb") as stream:
        assert tomllib.load(stream) == expected
    for section, values in expected.items():
        for key, value in values.items():
            assert getattr(getattr(legacy, section), key) == value
            assert getattr(getattr(current, section), key) != value
    assert legacy.postprocess.three_route_legacy_sr_agreement is False
    assert legacy.postprocess.three_route_province_support_min_routes == 0


def test_default_frozen_preparation_has_no_inference_or_marker(tmp_path, monkeypatch, capsys):
    from lpr_tools.eval import frozen_once as tool
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    manifest = tmp_path / "empty.csv"
    manifest.write_text("path,plate_no,plate_color,category,source,quad,note\n", encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare must not build models or consume the marker")
    monkeypatch.setattr(tool, "make_recognizer", forbidden)
    monkeypatch.setattr(tool, "_create_once_marker", forbidden)
    output = tmp_path / "future.json"
    assert tool.main(["--manifest", str(manifest), "--out", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["prepared"] is True and result["inference_run"] is False
    assert result["config"].replace("\\", "/").endswith("/algorithm/config/default.toml")
    assert "--config None" not in result["run_command"]
    assert "--config" not in result["run_command"]
    assert not output.exists()