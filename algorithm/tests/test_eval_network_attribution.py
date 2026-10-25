from types import SimpleNamespace

import numpy as np

from lpr import pipeline
from lpr.config import load_config
from lpr.errors import failure_result
from lpr.pipeline import _Attempt
from lpr.types import DecodeResult, PlateCandidate, RectifiedPlate, RecogStatus as S, SegmentResult
from lpr_tools.data.manifest import ManifestRow, write_manifest
from lpr_tools.eval.common import AttributedPlateRecognizer
import lpr_tools.eval.eval_accuracy as eval_accuracy


def test_reader_invoked_but_not_selected_when_classic_candidate_wins(monkeypatch):
    image = np.zeros((100, 200, 3), np.uint8)
    net = PlateCandidate([(0, 0), (80, 0), (80, 30), (0, 30)], (0, 0, 80, 30), "blue", .9, "net")
    classic = PlateCandidate([(90, 0), (190, 0), (190, 30), (90, 30)], (90, 0, 100, 30), "blue", .8, "color")
    monkeypatch.setattr(pipeline, "preprocess", lambda *_: SimpleNamespace())
    monkeypatch.setattr(pipeline, "locate", lambda *_: [classic])
    monkeypatch.setattr(pipeline, "_rectifications", lambda _i, c, _cfg: [
        RectifiedPlate(np.zeros((2, 2, 3), np.uint8), 7, "blue", np.eye(3), c)])
    monkeypatch.setattr(pipeline, "segment", lambda *_: SegmentResult(False, 7, [], np.zeros((7, 2, 2), np.uint8), 0.))
    # The reader produces no valid decode, while the conventional path later succeeds.
    monkeypatch.setattr(pipeline, "_decode_text", lambda *_: None)
    # Inject a net candidate before the traditional candidate; its reader is called,
    # then the traditional candidate produces an OK result and wins final selection.
    class Corner:
        version = "corner-test"
        def detect(self, *_): return [net]
    class Reader:
        version = "reader-test"
        def __init__(self): self.calls = 0
        def read(self, *_): self.calls += 1; return "", []
    def rects(_i, c, _cfg):
        return [RectifiedPlate(np.zeros((2, 2, 3), np.uint8), 7, "blue", np.eye(3), c)]
    monkeypatch.setattr(pipeline, "_rectifications", rects)
    monkeypatch.setattr(pipeline, "segment", lambda r, *_: SegmentResult(
        r.candidate.source == "color", 7, [], np.zeros((7, 2, 2), np.uint8), 1.))
    monkeypatch.setattr(pipeline, "decode", lambda *_: DecodeResult(S.OK, "京A12345", 7, "L7", .99, [.99]*7, []))
    # _decode_text remains None, so the reader invocation on the net candidate is not selected.
    cfg = load_config()
    reader = Reader()
    base = pipeline.PlateRecognizer(cfg, SimpleNamespace(version="fake", predict_proba=lambda _: None),
                                    corner_detector=Corner(), plate_reader=reader)
    recognizer = AttributedPlateRecognizer(base)
    result = recognizer.recognize(image)
    assert reader.calls == 1
    assert recognizer.reader_invoked_calls == 1
    assert recognizer.corner_invoked_calls == 1
    assert "corner" in result.stage_ms and "reader" in result.stage_ms
    assert recognizer.last_selection == {"source": "color", "crnn_selected": False}
    assert result.plate_no == "京A12345"


def test_final_reader_attempt_is_reported_as_crnn_selected():
    candidate = PlateCandidate([(0, 0), (80, 0), (80, 30), (0, 30)],
                               (0, 0, 80, 30), "blue", .9, "color")
    rect = RectifiedPlate(np.zeros((2, 2, 3), np.uint8), 7, "blue", np.eye(3), candidate)
    decode = DecodeResult(S.OK, "京A12345", 7, "L7", .99, [.99] * 7, [])
    attempt = _Attempt(decode, rect, None)
    base = pipeline.PlateRecognizer(load_config(), SimpleNamespace(version="fake", predict_proba=lambda _: None))
    recognizer = AttributedPlateRecognizer(base)
    result = recognizer._build(attempt, None)
    assert result.plate_no == "京A12345"
    assert recognizer.last_selection == {"source": "color", "crnn_selected": True}


def test_reader_selected_from_pipeline_after_corner_candidate(monkeypatch):
    image = np.zeros((100, 200, 3), np.uint8)
    net = PlateCandidate([(0, 0), (80, 0), (80, 30), (0, 30)], (0, 0, 80, 30), "blue", .9, "net")
    monkeypatch.setattr(pipeline, "preprocess", lambda *_: SimpleNamespace())
    monkeypatch.setattr(pipeline, "locate", lambda *_: [])
    monkeypatch.setattr(pipeline, "_rectifications", lambda _i, c, _cfg: [
        RectifiedPlate(np.zeros((2, 2, 3), np.uint8), 7, "blue", np.eye(3), c)])
    monkeypatch.setattr(pipeline, "segment", lambda *_: SegmentResult(False, 7, [], np.zeros((7, 2, 2), np.uint8), 0.))
    monkeypatch.setattr(pipeline, "_decode_text", lambda *_: DecodeResult(
        S.OK, "京A12345", 7, "L7", .99, [.99] * 7, []))
    class Corner:
        version = "corner-test"
        def detect(self, *_): return [net]
    class Reader:
        version = "reader-test"
        def read(self, *_): return "京A12345", [.99] * 7
    base = pipeline.PlateRecognizer(load_config(), SimpleNamespace(version="fake", predict_proba=lambda _: None),
                                    corner_detector=Corner(), plate_reader=Reader())
    recognizer = AttributedPlateRecognizer(base)
    result = recognizer.recognize(image)
    assert result.plate_no == "京A12345"
    assert recognizer.last_selection == {"source": "net", "crnn_selected": True}
    assert recognizer.reader_invoked_calls == recognizer.corner_invoked_calls == 1


def test_no_output_does_not_leave_selection_from_previous_result(monkeypatch):
    candidate = PlateCandidate([(0, 0), (80, 0), (80, 30), (0, 30)],
                               (0, 0, 80, 30), "blue", .9, "net")
    rect = RectifiedPlate(np.zeros((2, 2, 3), np.uint8), 7, "blue", np.eye(3), candidate)
    decoded = DecodeResult(S.OK, "京A12345", 7, "L7", .99, [.99] * 7, [])
    base = pipeline.PlateRecognizer(load_config(), SimpleNamespace(version="fake", predict_proba=lambda _: None))
    recognizer = AttributedPlateRecognizer(base)
    recognizer._build(_Attempt(decoded, rect, None), None)
    assert recognizer.last_selection is not None
    monkeypatch.setattr(pipeline, "preprocess", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    result = recognizer.recognize(np.zeros((20, 20, 3), np.uint8))
    assert result.status == S.INTERNAL_ERROR
    assert recognizer.last_selection is None


def test_oracle_row_does_not_recount_previous_np_network_calls(tmp_path, monkeypatch):
    manifest = tmp_path / "rows.csv"
    write_manifest(manifest, [
        ManifestRow("np.png", "", "unknown", "np", "test"),
        ManifestRow("plate.png", "京A12345", "blue", "clear", "test",
                    ((0, 0), (100, 0), (100, 40), (0, 40))),
    ])
    monkeypatch.setattr(eval_accuracy, "decode_image", lambda *_: (np.zeros((50, 120, 3), np.uint8), None))
    monkeypatch.setattr(eval_accuracy.OracleRecognizer, "recognize",
                        lambda *_: failure_result(S.NO_PLATE))
    class StubRecognizer:
        version = "stub"
        config = load_config()
        classifier = object()
        last_selection = None
        reader_invoked_calls = 0
        corner_invoked_calls = 0
        def _reset_observation(self):
            self.last_selection = None
            self.reader_invoked_calls = self.corner_invoked_calls = 0
        def recognize(self, _image):
            self.reader_invoked_calls = self.corner_invoked_calls = 1
            self.last_selection = {"source": "net", "crnn_selected": True}
            result = failure_result(S.LOW_CONFIDENCE, stage_ms={"reader": 1., "corner": 1.})
            result.plate_no = "京A12345"  # actual pipeline result for this np row is a false positive
            return result
    report = eval_accuracy.evaluate(manifest, StubRecognizer(), tmp_path / "out", oracle=True)
    attribution = report["network_attribution"]
    assert attribution["reader_invoked_images"] == attribution["corner_invoked_images"] == 1
    assert attribution["reader_invoked_calls"] == attribution["corner_invoked_calls"] == 1

    assert attribution["selected_result_count"] == 1
    assert attribution["network_candidate_selected_count"] == attribution["crnn_selected_count"] == 1
    assert attribution["crnn_selected_correct"] == 0 and attribution["crnn_selected_accuracy"] == 0.
