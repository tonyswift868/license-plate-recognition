"""Default migration keeps the frozen policy and an explicit legacy replay."""
import json
import tomllib

from lpr.config import ALGORITHM_DIR, load_config


def test_default_matches_frozen_three_route_configuration(monkeypatch):
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    current = load_config()
    expected = load_config("config/three_route_v2.toml").model_dump()
    expected["runtime"].update({
        "detector_input_size": 384,
        "plate_reader_model_path": "models/crnn_v4.pt",
        "box_reader_model_path": "models/box_reader_v3.pt",
    })
    expected["postprocess"].update({
        "three_route_agree_min_confidence": 0.7233636975288391,
        "three_route_box_solo_min_confidence": 0.9802792072296143,
        "solo_min_confidence_segment": 0.9719608426094055,
        "solo_min_confidence_reader": 0.991486668586731,
    })
    assert current.model_dump() == expected
    assert current.runtime.corner_model_path == "models/plate_det_v1.pt"
    assert current.runtime.detector_input_size == 384
    assert current.runtime.plate_reader_model_path == "models/crnn_v4.pt"
    assert current.runtime.box_reader_model_path == "models/box_reader_v3.pt"
    assert current.postprocess.arbitration == "three_route"
    assert current.postprocess.three_route_agree_min_confidence == 0.7233636975288391
    assert current.postprocess.three_route_box_solo_min_confidence == 0.9802792072296143
    assert current.postprocess.solo_min_confidence_segment == 0.9719608426094055
    assert current.postprocess.solo_min_confidence_reader == 0.991486668586731
    assert current.postprocess.nonok_fallback == "predicted_tilt_then_corrected"
    assert current.postprocess.fallback_predicted_tilt_min_deg == 20.0


def test_default_differs_from_archived_det256_by_exactly_seven_values(monkeypatch):
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    current = load_config()
    archived_path = ALGORITHM_DIR / "config/three_route_v2_det256.toml"
    archived = load_config("config/three_route_v2_det256.toml")
    expected_archive = {
        "runtime": {
            "detector_input_size": 0,
            "plate_reader_model_path": "models/crnn_v2.pt",
            "box_reader_model_path": "models/box_reader_v2.pt",
        },
        "postprocess": {
            "solo_min_confidence_segment": 0.938401460647583,
            "solo_min_confidence_reader": 0.9818063974380493,
            "three_route_agree_min_confidence": 0.7329292297363281,
            "three_route_box_solo_min_confidence": 0.9414944052696228,
        },
    }
    with archived_path.open("rb") as stream:
        assert tomllib.load(stream) == expected_archive

    current_values = current.model_dump()
    archived_values = archived.model_dump()
    differences = {
        section: {
            key: current_values[section][key]
            for key in current_values[section]
            if current_values[section][key] != archived_values[section][key]
        }
        for section in ("runtime", "postprocess")
    }
    differences = {section: values for section, values in differences.items() if values}
    assert differences == {
        "runtime": {
            "detector_input_size": 384,
            "plate_reader_model_path": "models/crnn_v4.pt",
            "box_reader_model_path": "models/box_reader_v3.pt",
        },
        "postprocess": {
            "solo_min_confidence_segment": 0.9719608426094055,
            "solo_min_confidence_reader": 0.991486668586731,
            "three_route_agree_min_confidence": 0.7233636975288391,
            "three_route_box_solo_min_confidence": 0.9802792072296143,
        },
    }


def test_legacy_override_contains_exactly_the_changed_values(monkeypatch):
    monkeypatch.delenv("LPR_CONFIG", raising=False)
    legacy = load_config("config/legacy_agree_v1.toml")
    current = load_config()
    expected = {
        "runtime": {"detector_input_size": 0, "corner_model_path": "models/corner_net_v1.pt",
                    "plate_reader_model_path": "models/crnn_v2.pt", "box_reader_model_path": ""},
        "postprocess": {"arbitration": "agree", "nonok_fallback": "agreement_then_reader",
                        "solo_min_confidence_segment": 0.938401460647583,
                        "solo_min_confidence_reader": 0.9818063974380493,
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
