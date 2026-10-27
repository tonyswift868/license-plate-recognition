"""流水线编排测试（架构师）：用替身阶段验证候选回退、多版式尝试、状态优先级与特种车牌（4.4）。"""

from __future__ import annotations

import numpy as np
import pytest

import lpr.pipeline as pl
from lpr.errors import BadInputError
from lpr.types import (DecodeResult, PlateCandidate, PreprocessOut, RecogStatus, RectifiedPlate,
                       SegmentResult)

S = RecogStatus


@pytest.fixture
def recognizer():
    """旧单路线编排断言显式使用旧配置；新默认由三路测试覆盖。"""
    from lpr.config import load_config
    from lpr.testing import UniformClassifier
    return pl.PlateRecognizer(load_config("config/legacy_agree_v1.toml"), UniformClassifier())


def _cand(i: int, color: str = "blue") -> PlateCandidate:
    return PlateCandidate(corners=[(i, 0), (i + 10, 0), (i + 10, 3), (i, 3)], bbox=(i, 0, 10, 3),
                          color=color, score=1.0 - i / 10, source="color")


def _dec(status: RecogStatus, conf: float, plate: str = "陕A12345", layout: int = 7) -> DecodeResult:
    return DecodeResult(status=status, plate_no=plate, layout=layout, layout_name="L7", confidence=conf,
                        char_confidences=[conf] * layout, candidates=[(plate, conf)], vehicle_type_hint="small")


@pytest.fixture
def stages(monkeypatch):
    """可配置的替身阶段：plan 控制候选、版式、分割成败与依次返回的解码结果，calls 记录调用顺序。"""
    plan = {"cands": [], "layout": {}, "seg_ok": {}, "decode": []}
    calls: list[tuple] = []

    def fake_pre(image, cfg, roi=None):
        if image.shape[0] < 20:
            raise BadInputError("短边过小")
        return PreprocessOut(image, image[..., 0], image, 1.0, (0, 0))

    def fake_locate(pre, cfg, max_candidates=3):
        return plan["cands"][:max_candidates]

    def fake_rectify(image, cand, cfg):
        calls.append(("rectify", cand.bbox[0]))
        layout = plan["layout"].get(cand.bbox[0], 7)
        return RectifiedPlate(np.zeros((80, 252, 3), np.uint8), layout, cand.color, np.eye(3), cand)

    def fake_segment(rect, layout, cfg):
        calls.append(("segment", rect.candidate.bbox[0], layout))
        ok = plan["seg_ok"].get((rect.candidate.bbox[0], layout), True)
        n = layout if ok else 0
        return SegmentResult(ok, layout, [(0, 0, 1, 1)] * n, np.zeros((n, 32, 32), np.uint8), 1.0)

    def fake_decode(probs, layout, color, cfg):
        return plan["decode"].pop(0) if plan["decode"] else _dec(S.LOW_CONFIDENCE, 0.1)

    monkeypatch.setattr(pl, "preprocess", fake_pre)
    monkeypatch.setattr(pl, "locate", fake_locate)
    monkeypatch.setattr(pl, "rectify", fake_rectify)
    monkeypatch.setattr(pl, "rectify_candidates", None)
    monkeypatch.setattr(pl, "segment", fake_segment)
    monkeypatch.setattr(pl, "decode", fake_decode)
    return plan, calls


IMG = np.zeros((100, 200, 3), np.uint8)


def test_no_candidates_is_no_plate(recognizer, stages):
    r = recognizer.recognize(IMG)
    assert r.status is S.NO_PLATE and r.error_code == 3001


def test_preprocess_bad_input(recognizer, stages):
    assert recognizer.recognize(np.zeros((10, 200, 3), np.uint8)).status is S.BAD_INPUT


def test_ok_stops_early(recognizer, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0), _cand(1), _cand(2)]
    plan["decode"] = [_dec(S.OK, 0.95)]
    r = recognizer.recognize(IMG, debug=True)
    assert r.status is S.OK and r.plate_no == "陕A12345" and r.error_code == 0
    assert calls == [("rectify", 0), ("segment", 0, 7)]
    assert r.bbox == (0, 0, 10, 3) and r.plate_image is not None
    assert {"work", "plate_0", "chars"} <= set(r.debug)
    assert set(r.stage_ms) >= {"preprocess", "locate", "rectify", "segment", "recognize", "postprocess"}


def test_fallback_to_next_candidate(recognizer, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0), _cand(1)]
    plan["seg_ok"] = {(0, 7): False}
    plan["decode"] = [_dec(S.OK, 0.9, "陕B11111")]
    r = recognizer.recognize(IMG)
    assert r.status is S.OK and r.plate_no == "陕B11111" and r.bbox[0] == 1
    assert ("segment", 1, 7) in calls


def test_best_kept_by_status_then_confidence(recognizer, stages):
    plan, _ = stages
    plan["cands"] = [_cand(0), _cand(1), _cand(2)]
    plan["decode"] = [_dec(S.INVALID_FORMAT, 0.99, "X"), _dec(S.LOW_CONFIDENCE, 0.5, "陕A11111"),
                      _dec(S.LOW_CONFIDENCE, 0.7, "陕A22222")]
    r = recognizer.recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.plate_no == "陕A22222" and r.error_code == 3002


def test_all_segment_failed(recognizer, stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["seg_ok"] = {(0, 7): False}
    assert recognizer.recognize(IMG).status is S.SEG_FAILED


def test_unknown_layout_tries_both(recognizer, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0, "unknown")]
    plan["layout"] = {0: 0}
    recognizer.recognize(IMG)
    assert [c for c in calls if c[0] == "segment"] == [("segment", 0, 7), ("segment", 0, 8)]


def test_yellow_tries_other_layout(recognizer, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0, "yellow")]
    recognizer.recognize(IMG)
    assert [c for c in calls if c[0] == "segment"] == [("segment", 0, 7), ("segment", 0, 8)]


def test_blue_tries_single_layout(recognizer, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0, "blue")]
    recognizer.recognize(IMG)
    assert [c for c in calls if c[0] == "segment"] == [("segment", 0, 7)]


@pytest.mark.parametrize("color", ["white", "black"])
def test_special_plate_unsupported(recognizer, stages, color):
    plan, _ = stages
    plan["cands"] = [_cand(0, color)]
    r = recognizer.recognize(IMG)
    assert r.status is S.UNSUPPORTED and r.error_code == 3006


def test_rectify_hypotheses_fallback(recognizer, stages, monkeypatch):
    """同一候选的首选四角点分割失败时，改用下一个四角点假设。"""
    plan, calls = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.OK, 0.95, "陕A54321")]

    def fake_many(image, cand, cfg):
        rects = []
        for k in (0, 5):                                 # 用 bbox x 偏移区分两个假设
            c = _cand(cand.bbox[0] + k)
            rects.append(RectifiedPlate(np.zeros((80, 252, 3), np.uint8), 7, "blue", np.eye(3), c))
        return rects

    monkeypatch.setattr(pl, "rectify_candidates", fake_many)
    plan["seg_ok"] = {(0, 7): False}
    r = recognizer.recognize(IMG)
    assert r.status is S.OK and r.plate_no == "陕A54321" and r.bbox[0] == 5
    assert [c for c in calls if c[0] == "segment"] == [("segment", 0, 7), ("segment", 5, 7)]


def test_stage_exception_becomes_internal_error(recognizer, stages, monkeypatch):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    monkeypatch.setattr(pl, "rectify", lambda *a: 1 / 0)
    r = recognizer.recognize(IMG)
    assert r.status is S.INTERNAL_ERROR and r.error_code == 3099


def test_recognize_many_uses_vote(recognizer, stages, monkeypatch):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    seen = {}

    def fake_vote(results, *, min_frames, min_confidence):
        seen["n"] = len(results)
        out = results[0]
        out.frames_used = len(results)
        return out

    monkeypatch.setattr(pl, "vote", fake_vote)
    r = recognizer.recognize_many([IMG, IMG, IMG])
    assert seen["n"] == 3 and r.frames_used == 3 and r.elapsed_ms > 0


class _FakeCornerNet:
    name, version = "fake_corner", "fake_corner_v0"

    def __init__(self, cands):
        self.cands = cands

    def detect(self, image, max_candidates):
        return list(self.cands)


class _FakeReader:
    name, version = "fake_crnn", "fake_crnn_v0"

    def __init__(self, text, confs):
        self.text, self.confs, self.calls = text, confs, 0

    def read(self, plate_image):
        self.calls += 1
        return self.text, list(self.confs)


def _recognizer_with(cfg, box_reader=None, **nets):
    from lpr.testing import UniformClassifier

    return pl.PlateRecognizer(cfg, UniformClassifier(), box_reader=box_reader, **nets)


def test_corner_candidates_go_first_and_dedupe(cfg, stages):
    plan, calls = stages
    plan["cands"] = [_cand(0), _cand(30)]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.4), _dec(S.LOW_CONFIDENCE, 0.3), _dec(S.LOW_CONFIDENCE, 0.2)]
    net = _FakeCornerNet([_cand(60), _cand(1)])          # 第二个与传统候选 0 重叠，应被去重
    rec = _recognizer_with(cfg, corner_detector=net)
    r = rec.recognize(IMG)
    assert [c[1] for c in calls if c[0] == "rectify"] == [60, 1, 30]
    assert "corner" in r.stage_ms and r.model_version.endswith("+fake_corner_v0")


def test_plate_reader_rescues_seg_failed(cfg, stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["seg_ok"] = {(0, 7): False}
    reader = _FakeReader("陕A12345", [0.99] * 7)
    r = _recognizer_with(cfg, plate_reader=reader).recognize(IMG, debug=True)
    assert r.status is S.OK and r.plate_no == "陕A12345" and reader.calls == 1
    assert r.vehicle_type_hint == "small" and "reader" in r.stage_ms


def test_plate_reader_not_called_after_segment_ok(stages):
    from lpr.config import load_config

    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.OK, 0.95)]
    reader = _FakeReader("陕A99999", [0.99] * 7)
    cfg = load_config(postprocess__arbitration="first_ok")    # 旧规则：分割路线先出 OK 即返回
    r = _recognizer_with(cfg, plate_reader=reader).recognize(IMG)
    assert r.plate_no == "陕A12345" and reader.calls == 0


def test_plate_reader_low_conf_and_invalid(cfg, stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["seg_ok"] = {(0, 7): False}
    r = _recognizer_with(cfg, plate_reader=_FakeReader("陕A12345", [0.99, 0.5] + [0.99] * 5)).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.confidence == 0.5
    r = _recognizer_with(cfg, plate_reader=_FakeReader("陕I12345", [0.99] * 7)).recognize(IMG)
    assert r.status is S.INVALID_FORMAT
    r = _recognizer_with(cfg, plate_reader=_FakeReader("", [])).recognize(IMG)
    assert r.status is S.SEG_FAILED


# ---- agree 模式：两条路线读法一致才自动通过（10-26） ----

def _agree_cfg(**kw):
    from lpr.config import load_config

    base = {"arbitration": "agree", "agree_min_confidence": 0.0, "solo_min_confidence_segment": 1.01,
            "solo_min_confidence_reader": 1.01, "disagree_prefer": "reader", "nonok_fallback": "rank"}
    return load_config(postprocess={**base, **kw})


def test_agree_mode_both_routes_same_is_ok(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.6)]          # 分割路线单独不够 OK，一致后升为 OK
    reader = _FakeReader("陕A12345", [0.7] * 7)
    r = _recognizer_with(_agree_cfg(agree_min_confidence=0.5), plate_reader=reader).recognize(IMG)
    assert r.status is S.OK and r.plate_no == "陕A12345" and r.confidence == 0.7 and reader.calls == 1


def test_agree_mode_disagree_is_low_confidence_with_both_candidates(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.OK, 0.99)]                     # 旧规则下会直接 OK
    reader = _FakeReader("陕A12346", [0.99] * 7)
    r = _recognizer_with(_agree_cfg(), plate_reader=reader).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.plate_no == "陕A12346"     # disagree_prefer 默认 reader
    assert [c[0] for c in r.candidates][:2] == ["陕A12346", "陕A12345"]
    r = _recognizer_with(_agree_cfg(disagree_prefer="segment"), plate_reader=reader).recognize(IMG)
    assert r.plate_no == "陕A12345"


def test_agree_mode_solo_threshold_and_below_agree_threshold(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["seg_ok"] = {(0, 7): False}
    reader = _FakeReader("陕A12345", [0.97] * 7)
    r = _recognizer_with(_agree_cfg(), plate_reader=reader).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE                     # 只有一边、单边门槛默认关闭
    r = _recognizer_with(_agree_cfg(solo_min_confidence_reader=0.95), plate_reader=reader).recognize(IMG)
    assert r.status is S.OK
    plan["seg_ok"] = {}
    plan["decode"] = [_dec(S.OK, 0.3)]
    r = _recognizer_with(_agree_cfg(agree_min_confidence=0.5), plate_reader=reader).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.plate_no == "陕A12345"


def test_agree_mode_without_reader_keeps_first_ok(stages):
    plan, calls = stages
    plan["cands"] = [_cand(0), _cand(1)]
    plan["decode"] = [_dec(S.OK, 0.95)]
    r = _recognizer_with(_agree_cfg()).recognize(IMG)
    assert r.status is S.OK and calls == [("rectify", 0), ("segment", 0, 7)]


class _SeqReader(_FakeReader):
    """依次返回不同读法，模拟多块校正图上的 CRNN 结果。"""

    def __init__(self, outputs):
        super().__init__("", [])
        self.outputs = list(outputs)

    def read(self, plate_image):
        self.calls += 1
        return self.outputs.pop(0) if self.outputs else ("", [])


def test_fallback_prefers_same_rect_agreement(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0), _cand(1)]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.4, "陕A11111"), _dec(S.LOW_CONFIDENCE, 0.3, "陕A22222")]
    reader = _SeqReader([("陕A99999", [0.9] * 7), ("陕A22222", [0.6] * 7)])   # 候选 1 两路线同文
    cfg = _agree_cfg(agree_min_confidence=0.99, nonok_fallback="agreement_then_reader")
    r = _recognizer_with(cfg, plate_reader=reader).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.plate_no == "陕A22222" and r.confidence == 0.3
    assert r.bbox[0] == 1 and r.candidates[0][0] == "陕A22222"


def test_fallback_without_agreement_takes_top_reader(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0), _cand(1)]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.95, "陕A11111"), _dec(S.LOW_CONFIDENCE, 0.95, "陕A22222")]
    reader = _SeqReader([("陕A33333", [0.5] * 7), ("陕A44444", [0.7] * 7)])
    cfg = _agree_cfg(nonok_fallback="agreement_then_reader", disagree_prefer="segment")
    r = _recognizer_with(cfg, plate_reader=reader).recognize(IMG)
    assert r.status is S.LOW_CONFIDENCE and r.plate_no == "陕A44444" and r.confidence == 0.7
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.95, "陕A11111"), _dec(S.LOW_CONFIDENCE, 0.95, "陕A22222")]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.95, "陕A11111"), _dec(S.LOW_CONFIDENCE, 0.95, "陕A22222")]
    r = _recognizer_with(_agree_cfg(disagree_prefer="segment"),
                         plate_reader=_SeqReader([("陕A33333", [0.5] * 7), ("陕A44444", [0.7] * 7)])).recognize(IMG)
    assert r.plate_no in {"陕A11111", "陕A22222"}                 # rank 回退：仍按旧规则取分割读法


def test_fallback_does_not_touch_ok_or_unsupported(stages):
    plan, _ = stages
    plan["cands"] = [_cand(0)]
    plan["decode"] = [_dec(S.OK, 0.99)]
    cfg = _agree_cfg(nonok_fallback="agreement_then_reader")
    r = _recognizer_with(cfg, plate_reader=_FakeReader("陕A12345", [0.99] * 7)).recognize(IMG)
    assert r.status is S.OK and r.plate_no == "陕A12345"
    plan["cands"] = [_cand(0, "white")]
    plan["decode"] = [_dec(S.LOW_CONFIDENCE, 0.5)]
    r = _recognizer_with(cfg, plate_reader=_FakeReader("陕A54321", [0.9] * 7)).recognize(IMG)
    assert r.status is S.UNSUPPORTED
