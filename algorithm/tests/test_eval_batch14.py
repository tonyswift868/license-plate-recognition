from types import SimpleNamespace

import pytest
from lpr.types import RecogStatus
from lpr_tools.eval.bench_latency import benchmark_paths
from lpr_tools.eval.common import AttributedPlateRecognizer
from lpr_tools.eval.frozen_once import _canonical_marker, _create_once_marker, summarize


class Recorder:
    def __init__(self):
        self.images = []

    def recognize(self, image):
        self.images.append(image)
        return SimpleNamespace(status=RecogStatus.OK, stage_ms={
            "corner": 1.0, "reader": 2.0, "box_reader": 3.0,
        })


def test_no_truth_latency_uses_fixed_cyclic_order_and_reports_all_reader_stages():
    paths = ["a", "b"]
    decoded = []

    def decoder(path):
        decoded.append(path)
        # Already target-sized input avoids an unnecessary resize in this unit test.
        import numpy as np
        return np.zeros((2, 2, 3), dtype=np.uint8), None

    recognizer = Recorder()
    result = benchmark_paths(recognizer, paths, n=3, size=(2, 2), decoder=decoder)

    # Warm-ups cycle from index 0; the timed loop also resets to index 0, preserving legacy sampling.
    assert decoded == ["a", "b", "a", "b", "a", "a", "b", "a"]
    assert len(recognizer.images) == 8
    assert result["n"] == 3 and result["warmup"] == 5
    assert result["ground_truth_available"] is False
    assert set(("corner", "reader", "box_reader")) <= set(result["stage_ms"])
    assert "network_attribution" not in result


def test_frozen_summary_groups_and_auto_precision():
    rows = [SimpleNamespace(category="clear", plate_no="A"),
            SimpleNamespace(category="night", plate_no="B"),
            SimpleNamespace(category="tilt", plate_no="C"),
            SimpleNamespace(category="dirty", plate_no="D")]
    results = [SimpleNamespace(status=RecogStatus.OK, plate_no="A"),
               SimpleNamespace(status=RecogStatus.OK, plate_no="wrong"),
               SimpleNamespace(status=RecogStatus.NO_PLATE, plate_no=None),
               SimpleNamespace(status=RecogStatus.OK, plate_no="D")]

    report = summarize(rows, results)

    assert report["clear"]["plate_accuracy"] == 1
    assert report["night"]["auto_precision"] == 0
    assert report["tilt"]["auto_precision"] is None
    assert report["overall"]["auto_accepted"] == 3
    assert report["overall"]["auto_correct"] == 2


def test_frozen_once_marker_is_exclusive(tmp_path):
    marker = tmp_path / "receipt.json.frozen-once.json"
    _create_once_marker(marker, {"state": "in_progress"})
    with pytest.raises(FileExistsError):
        _create_once_marker(marker, {"state": "in_progress"})


def test_frozen_once_marker_is_stable_for_same_manifest(tmp_path):
    manifest = tmp_path / "frozen.csv"
    manifest.write_text("frozen manifest", encoding="utf-8")
    assert _canonical_marker(manifest) == _canonical_marker(manifest)


def test_frozen_preparation_resolves_config_and_never_builds_a_model(tmp_path, monkeypatch, capsys):
    import json

    from lpr_tools.eval import frozen_once as tool
    manifest=tmp_path/'empty.csv'
    manifest.write_text('path,plate_no,plate_color,category,source,quad,note\n',encoding='utf-8')
    monkeypatch.setattr(tool,'make_recognizer',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('inference forbidden')))
    monkeypatch.setattr(tool,'_create_once_marker',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('marker forbidden')))
    assert tool.main(['--manifest',str(manifest),'--config','config/three_route_v2.toml',
                      '--out',str(tmp_path/'future.json')])==0
    prepared=json.loads(capsys.readouterr().out)
    assert prepared['inference_run'] is False
    assert prepared['config'].replace('\\','/').endswith('/algorithm/config/three_route_v2.toml')


def test_attributed_recognizer_preserves_and_counts_box_reader():
    class BoxReader:
        version = "box-v2"

        def read(self, image):
            return "京A12345", [0.9] * 7

    reader = BoxReader()
    source = SimpleNamespace(config=object(), classifier=object(), corner_detector=None,
                             plate_reader=None, box_reader=reader)
    recognizer = AttributedPlateRecognizer(source)

    assert recognizer.box_reader.version == "box-v2"
    assert recognizer.box_reader.read(None)[0] == "京A12345"
    assert recognizer.box_reader_invoked_calls == 1
    recognizer._reset_observation()
    assert recognizer.box_reader_invoked_calls == 0


def test_box_reader_attempt_is_not_attributed_to_crnn(monkeypatch):
    from lpr.pipeline import PlateRecognizer

    monkeypatch.setattr(PlateRecognizer, "_build", lambda self, attempt, dbg: object())
    source = SimpleNamespace(config=object(), classifier=object(), corner_detector=None,
                             plate_reader=None, box_reader=None)
    recognizer = AttributedPlateRecognizer(source)
    attempt = SimpleNamespace(
        rect=SimpleNamespace(candidate=SimpleNamespace(source="net")),
        seg=None,
        plate_image=object(),
    )

    recognizer._build(attempt, None)

    assert recognizer.last_selection["crnn_selected"] is False


def test_legacy_attempt_without_plate_image_keeps_crnn_attribution(monkeypatch):
    from lpr.pipeline import PlateRecognizer

    monkeypatch.setattr(PlateRecognizer, "_build", lambda self, attempt, dbg: object())
    source = SimpleNamespace(config=object(), classifier=object(), corner_detector=None,
                             plate_reader=None, box_reader=None)
    recognizer = AttributedPlateRecognizer(source)
    attempt = SimpleNamespace(rect=SimpleNamespace(candidate=SimpleNamespace(source="net")), seg=None)

    recognizer._build(attempt, None)

    assert recognizer.last_selection["crnn_selected"] is True
