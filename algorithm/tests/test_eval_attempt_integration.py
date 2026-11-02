"""The offline trace/replay must agree with real pipeline control flow."""
from types import SimpleNamespace

import numpy as np
import pytest
from lpr import pipeline
from lpr.config import load_config
from lpr.types import DecodeResult, PlateCandidate, RectifiedPlate, SegmentResult
from lpr.types import RecogStatus as S
from lpr_tools.eval import arbitration


@pytest.mark.parametrize('threshold', [0.0, 0.8, 0.95, 1.0])
@pytest.mark.parametrize('case', ['mixed', 'special', 'none', 'seg_failed', 'tie'])
def test_exhaustive_replay_matches_pipeline_across_thresholds(monkeypatch, threshold, case):
    image = np.zeros((120, 240, 3), np.uint8)
    cfg = load_config("config/legacy_agree_v1.toml", postprocess__min_confidence=threshold)
    cand = PlateCandidate([(0, 0), (200, 0), (200, 60), (0, 60)],
                          (0, 0, 200, 60), 'white' if case == 'special' else 'unknown', .9, 'color')
    other = PlateCandidate(cand.corners, cand.bbox, 'blue', .8, 'edge')
    candidates = [] if case == 'none' else [cand, other]
    def rectangles(_image, candidate, _cfg):
        ids = [0, 1] if candidate is cand else [2]
        return [RectifiedPlate(np.full((2, 2, 3), i, np.uint8),
                               0 if candidate is cand else 7, candidate.color, np.eye(3), candidate)
                for i in ids]
    def segmentation(rect, layout, _cfg):
        return SegmentResult(case != 'seg_failed', layout, [],
                             np.full((layout, 32, 32), int(rect.image[0, 0, 0]), np.uint8), 1.)
    def decoder(chars, layout, color, post_cfg):
        marker = int(chars[0, 0, 0])
        confidence = .7 if case == 'special' else (.85 if case == 'tie' or marker == 0 else .98)
        plate = ('京A12345' if marker == 2 else '沪B54321') if layout == 7 else '沪AD12345'
        status = S.OK if confidence >= post_cfg.min_confidence else S.LOW_CONFIDENCE
        return DecodeResult(status, plate, layout, 'L7' if layout == 7 else 'L8S', confidence,
                            [confidence] * layout, [(plate, 1.)])
    def pre(*_args):
        return SimpleNamespace()
    def loc(*_args):
        return candidates
    for module in (pipeline, arbitration):
        monkeypatch.setattr(module, 'preprocess', pre)
        monkeypatch.setattr(module, 'locate', loc)
        monkeypatch.setattr(module, 'segment', segmentation)
    monkeypatch.setattr(pipeline, '_rectifications', rectangles)
    monkeypatch.setattr(pipeline, 'decode', decoder)
    classifier = SimpleNamespace(version='isolated', predict_proba=lambda chars: chars)
    recognizer = pipeline.PlateRecognizer(cfg, classifier)
    actual = recognizer.recognize(image)
    row = SimpleNamespace(path='isolated.png', plate_no='京A12345', category='clear', plate_color='blue')
    trace = arbitration.trace_image(image, recognizer, row)
    replay = arbitration.replay_attempts(trace, threshold)
    assert (replay['plate_no'], replay['status'], replay['confidence']) == (
        actual.plate_no, actual.status.value, actual.confidence)
    if case not in {'none', 'seg_failed'}:
        # All layouts and rectification hypotheses must remain in the trace,
        # even when production returned OK on its very first attempt.
        assert len(trace['attempts']) == 5
        assert [(a['candidate_index'], a['rect_index'], a['layout']) for a in trace['attempts']] == [
            (0, 0, 7), (0, 0, 8), (0, 1, 7), (0, 1, 8), (1, 0, 7)]
