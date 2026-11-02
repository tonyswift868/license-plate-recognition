"""Runtime integration checks for three_route ordering and the official box crop path."""

from __future__ import annotations

import lpr.pipeline as pl
import numpy as np
from lpr.types import RecogStatus

from . import test_pipeline as _tp
from .test_pipeline import IMG, _cand, _dec, _recognizer_with

stages = _tp.stages  # shared pytest fixture; an import here looks unused to ruff (F401/F811)

S = RecogStatus


class _FakeBoxReader:
    name, version = "fake_box", "box_v2_test"
    box_pad = (0.08, 0.12)

    def __init__(self, text="陕A12345", confidence=0.99):
        self.text, self.confidence = text, confidence
        self.calls = []

    def read(self, image):
        self.calls.append(image.copy())
        return self.text, [self.confidence] * len(self.text)


def _three_route_cfg(**overrides):
    from lpr.config import load_config

    pp = {
        "arbitration": "three_route",
        "three_route_agree_min_confidence": 0.7,
        "three_route_box_solo_min_confidence": 0.95,
        "three_route_legacy_sr_agreement": False,
        "three_route_segment_solo": True,
        "three_route_reader_solo": True,
        "three_route_province_support_min_routes": 0,
        "fallback_predicted_tilt_min_deg": 20.0,
        "solo_min_confidence_segment": 0.9,
        "solo_min_confidence_reader": 0.9,
        "nonok_fallback": "predicted_tilt_then_corrected",
    }
    pp.update(overrides)
    return load_config(postprocess=pp)


def test_three_route_early_segment_solo_precedes_later_candidates(stages):
    plan, calls = stages
    plan["cands"] = [_cand(0), _cand(30)]
    plan["decode"] = [_dec(S.OK, 0.96)]
    box = _FakeBoxReader("陕B11111", 0.99)
    rec = _recognizer_with(_three_route_cfg(), box_reader=box)

    result = rec.recognize(IMG)

    assert result.status is S.OK and result.plate_no == "陕A12345"
    assert result.route == "segment" and result.candidate_index == 0 and result.rect_index == 0
    assert box.calls and len(box.calls) == 1
    assert [call for call in calls if call[0] == "rectify"] == [("rectify", 0)]
    assert result.model_version.endswith("+box_v2_test")


def test_three_route_box_read_uses_official_crop_and_once_per_candidate(stages, monkeypatch):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.4), _dec(S.LOW_CONFIDENCE, 0.5)]
    box = _FakeBoxReader()
    seen = []
    real_crop = pl.crop_box

    def two_rects(image, cand, cfg):
        return [pl.RectifiedPlate(np.zeros((80, 252, 3), np.uint8), 7, "blue", np.eye(3), cand),
                pl.RectifiedPlate(np.zeros((80, 252, 3), np.uint8), 7, "blue", np.eye(3), cand)]

    def capture_crop(image, bbox, pad_x, pad_y):
        seen.append((bbox, pad_x, pad_y))
        return real_crop(image, bbox, pad_x, pad_y)

    monkeypatch.setattr(pl, "crop_box", capture_crop)
    monkeypatch.setattr(pl, "rectify_candidates", two_rects)
    result = _recognizer_with(_three_route_cfg(), box_reader=box).recognize(IMG)

    assert result.status is S.OK and result.route == "box"
    assert len(box.calls) == len(seen) == 1
    assert seen == [((0, 0, 10, 3), 0.08, 0.12)]
    assert len([call for call in stages[1] if call[0] == "segment"]) == 2


def test_three_route_box_solo_works_without_rectification(stages, monkeypatch):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    box = _FakeBoxReader()
    monkeypatch.setattr(pl, "_rectifications", lambda *args: [])

    result = _recognizer_with(_three_route_cfg(), box_reader=box).recognize(IMG)

    assert result.status is S.OK and result.route == "box"
    assert result.rect_index is None and result.bbox == (0, 0, 10, 3)
    assert result.plate_image.shape == box.calls[0].shape


def test_old_first_ok_constructor_without_box_reader_remains_compatible(stages):
    from lpr.config import load_config

    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.OK, 0.95)]
    rec = _recognizer_with(load_config(postprocess__arbitration="first_ok"))

    result = rec.recognize(IMG)

    assert result.status is S.OK and getattr(result, "route", None) is None
