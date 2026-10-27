"""流水线编排（架构师）：PlateRecognizer，见开发文档 4.4、5.1、5.6。

recognize* 永不抛异常：任何错误都通过 status 与 error_code 返回。
阶段函数尚未实现（抛 NotImplementedError）时返回 INTERNAL_ERROR，保证桩阶段也能端到端跑通。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger

from ._version import __version__
from .config import LprConfig, load_config
from .errors import DEFAULT_MESSAGES, ERROR_CODES, BadInputError, failure_result
from .io_utils import decode_image
from .locate import locate
from .plate_rules import match_layout, vehicle_type_hint
from .postprocess import decode
from .preprocess import preprocess
from .recognize import load_box_reader, load_classifier, load_corner_detector, load_plate_reader
from .recognize.base import BoxReader, CharClassifier, CornerDetector, PlateReader
from .boxcrop import crop_box
from .rectify import rectify
from .segment import segment

try:                                                     # 可选：同一候选的多个四角点假设（5.9）
    from .rectify import rectify_candidates
except ImportError:
    rectify_candidates = None
from .types import (Box, DecodeResult, PlateCandidate, PlateColor, RecogStatus, RecognitionResult,
                    RectifiedPlate, SegmentResult)
from .voting import vote

S = RecogStatus
_STATUS_RANK = {S.OK: 3, S.LOW_CONFIDENCE: 2, S.INVALID_FORMAT: 1}
_SPECIAL_COLORS = frozenset({"white", "black"})          # 特种车牌，暂不支持（Q12）
_BYTES_TYPES = (bytes, bytearray, memoryview)


@dataclass
class _Attempt:
    dec: DecodeResult
    rect: RectifiedPlate
    seg: SegmentResult | None                            # 整牌识别（CRNN）的结果没有分割
    plate_image: np.ndarray | None = None                 # route B 保留官方原框裁片
    color: str | None = None


class _StageTimer:
    """按阶段累计耗时（毫秒）；候选回退时同一阶段会执行多次，耗时累加。"""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = self.ms.get(stage, 0.0) + (time.perf_counter() - t) * 1000


def _check_image(image: object) -> str | None:
    """类型层面的输入检查；尺寸等检查由阶段一负责。返回 None 表示通过。"""
    if not isinstance(image, np.ndarray):
        return "输入不是图像数组"
    if image.size == 0:
        return "图像为空"
    if image.dtype != np.uint8:
        return f"图像类型须为 uint8，实际为 {image.dtype}"
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3)):
        return None
    return f"图像形状不支持：{image.shape}"


def _layouts_to_try(rect: RectifiedPlate) -> list[int]:
    layouts = [rect.layout] if rect.layout in (7, 8) else [7, 8]
    if rect.color in ("yellow", "unknown"):              # 黄绿双拼的大型新能源可能被判为黄色
        other = 8 if layouts[0] == 7 else 7
        if other not in layouts:
            layouts.append(other)
    return layouts


def _rectifications(image: np.ndarray, cand, cfg) -> list[RectifiedPlate]:
    """同一候选的校正假设，首个为首选；阶段三没有提供多假设函数时退回单个 rectify。"""
    if rectify_candidates is not None:
        rects = rectify_candidates(image, cand, cfg)
        if rects:
            return list(rects)
    return [rectify(image, cand, cfg)]


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _in_roi(cand: PlateCandidate, roi: Box | None) -> bool:
    if roi is None:
        return True
    x, y, w, h = cand.bbox
    cx, cy = x + w / 2, y + h / 2
    rx, ry, rw, rh = roi
    return rx <= cx < rx + rw and ry <= cy < ry + rh


def _merge_candidates(net: list[PlateCandidate], classic: list[PlateCandidate],
                      limit: int) -> list[PlateCandidate]:
    """四角点网络的候选在前，传统定位的候选补位；与已选候选 IoU ≥0.5 的视为重复。"""
    merged: list[PlateCandidate] = []
    for cand in [*net, *classic]:
        if len(merged) >= limit:
            break
        if all(_iou(cand.bbox, m.bbox) < 0.5 for m in merged):
            merged.append(cand)
    return merged


def _decode_text(text: str, confs: Sequence[float], color: PlateColor,
                 min_confidence: float) -> DecodeResult | None:
    """整牌识别输出 → DecodeResult；状态规则与阶段六一致（5.4、附录 A）。读不出或长度不对返回 None。"""
    if len(text) not in (7, 8) or len(confs) != len(text):
        return None
    name = match_layout(text)
    conf = float(min(confs))
    if name is None:
        status = S.INVALID_FORMAT
    else:
        status = S.OK if conf >= min_confidence else S.LOW_CONFIDENCE
    return DecodeResult(status=status, plate_no=text, layout=len(text), layout_name=name or "",
                        confidence=conf, char_confidences=[float(c) for c in confs],
                        candidates=[(text, conf)], vehicle_type_hint=vehicle_type_hint(color, name))


def _route_trace_attempt(attempt: _Attempt | None) -> dict | None:
    if attempt is None:
        return None
    return {"status": attempt.dec.status.value, "plate": attempt.dec.plate_no,
            "confidence": attempt.dec.confidence,
            "valid": attempt.dec.status in (S.OK, S.LOW_CONFIDENCE)}


def _rank(dec: DecodeResult) -> tuple[int, float]:
    return _STATUS_RANK.get(dec.status, 0), float(dec.confidence)


def _arbitrate(seg: _Attempt | None, rd: _Attempt | None, pp) -> _Attempt | None:
    """agree 模式：同一块校正图上分割路线（seg）与整牌识别（rd）的仲裁（E1）。

    两边读法相同、格式合法且较小置信度 ≥ agree_min_confidence → OK；
    否则单边置信度 ≥ 各自 solo 门槛才 OK；其余为 LOW_CONFIDENCE，备选里带上另一边的读法。"""
    s = seg.dec if seg is not None and seg.dec.status != S.INVALID_FORMAT else None
    r = rd.dec if rd is not None and rd.dec.status != S.INVALID_FORMAT else None
    if s is not None and r is not None and s.plate_no == r.plate_no \
            and min(s.confidence, r.confidence) >= pp.agree_min_confidence:
        return _Attempt(replace(s, status=S.OK, confidence=max(s.confidence, r.confidence)), seg.rect, seg.seg)
    valid = {"segment": seg if s is not None else None, "reader": rd if r is not None else None}
    solo = {"segment": valid["segment"] if s is not None and s.confidence >= pp.solo_min_confidence_segment else None,
            "reader": valid["reader"] if r is not None and r.confidence >= pp.solo_min_confidence_reader else None}
    flip = {"segment": "reader", "reader": "segment"}
    for pool, status in ((solo, S.OK), (valid, S.LOW_CONFIDENCE)):
        side = pp.disagree_prefer if pool[pp.disagree_prefer] is not None else flip[pp.disagree_prefer]
        pick, alt = pool[side], valid[flip[side]]
        if pick is None:
            continue
        cands = [*pick.dec.candidates[:1], *([(alt.dec.plate_no, alt.dec.confidence)] if alt is not None else []),
                 *pick.dec.candidates[1:]]
        cands = [c for i, c in enumerate(cands) if c[0] not in {x[0] for x in cands[:i]}][:max(pp.topk, 1)]
        return _Attempt(replace(pick.dec, status=status, candidates=cands), pick.rect, pick.seg)
    return max((a for a in (seg, rd) if a is not None), key=lambda a: _rank(a.dec), default=None)


def _fallback_text(pairs: list[tuple[_Attempt | None, _Attempt | None]], best: _Attempt) -> _Attempt:
    """agree 模式且没有任何结果自动通过时（03 第十批推荐的 agreement_then_reader）：
    先取同一块校正图上两路线读法相同的一对（按两边置信度较小值最高），否则取置信度最高的合法 CRNN 读法；
    状态保持原结果（LOW_CONFIDENCE 等），只替换返回的读法。同分取先出现者。"""
    agreed: tuple[_Attempt, float] | None = None
    top_reader: _Attempt | None = None
    for seg, rd in pairs:
        if rd is None or rd.dec.status == S.INVALID_FORMAT:
            continue
        if top_reader is None or rd.dec.confidence > top_reader.dec.confidence:
            top_reader = rd
        if seg is not None and seg.dec.status != S.INVALID_FORMAT and seg.dec.plate_no == rd.dec.plate_no:
            score = min(seg.dec.confidence, rd.dec.confidence)
            if agreed is None or score > agreed[1]:
                agreed = (rd, score)
    pick, conf = (agreed if agreed is not None
                  else (top_reader, top_reader.dec.confidence) if top_reader is not None else (None, 0.0))
    if pick is None:
        return best
    alts = [c for c in best.dec.candidates if c[0] != pick.dec.plate_no]
    dec = replace(pick.dec, status=best.dec.status, confidence=float(conf),
                  candidates=[(pick.dec.plate_no, float(conf)), *alts][:max(len(best.dec.candidates), 1)])
    return _Attempt(dec, pick.rect, None)


def _chars_mosaic(chars: np.ndarray) -> np.ndarray:
    return np.hstack(list(chars)) if len(chars) else np.zeros((32, 32), np.uint8)


class PlateRecognizer:
    """车牌识别器。实例在调用之间没有可变共享状态，可被多线程并发调用。"""

    def __init__(self, config: LprConfig, classifier: CharClassifier, *,
                 corner_detector: CornerDetector | None = None,
                 plate_reader: PlateReader | None = None,
                 box_reader: BoxReader | None = None) -> None:
        self.config = config
        self.classifier = classifier
        self.corner_detector = corner_detector           # 可选：四角点网络（K1 Plan B-1）
        self.plate_reader = plate_reader                 # 可选：整牌识别 CRNN（K1 Plan B-2）
        self.box_reader = box_reader                     # 可选：原框整牌识别（three_route）

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | None = None, **overrides: object) -> "PlateRecognizer":
        """加载配置与模型。模型缺失或字符表哈希不匹配时立即抛 ModelLoadError（启动即失败）。
        path 为空时先读环境变量 LPR_CONFIG，再缺省为 algorithm/config/default.toml。
        overrides 写法：runtime__classifier="svm"、postprocess__min_confidence=0.9。"""
        cfg = load_config(path, **overrides)
        import torch                                     # 延迟导入：FakeRecognizer 与 import lpr 不需要 torch

        torch.set_num_threads(cfg.runtime.torch_threads)
        rt = cfg.runtime
        classifier = load_classifier(rt.classifier, cfg.resolve_path(rt.model_path))
        corner = load_corner_detector(cfg.resolve_path(rt.corner_model_path)) if rt.corner_model_path else None
        reader = load_plate_reader(cfg.resolve_path(rt.plate_reader_model_path)) if rt.plate_reader_model_path else None
        box_reader = load_box_reader(cfg.resolve_path(rt.box_reader_model_path)) if rt.box_reader_model_path else None
        return cls(cfg, classifier, corner_detector=corner, plate_reader=reader, box_reader=box_reader)

    @property
    def version(self) -> str:
        """例如 "lpr-1.0.0+char_cnn_v1"，启用网络时追加其版本，写入日志与结果。"""
        parts = [self.classifier.version]
        parts += [m.version for m in (self.corner_detector, self.plate_reader, self.box_reader) if m is not None]
        return f"lpr-{__version__}+" + "+".join(parts)

    def warmup(self) -> None:
        """用空白图跑一遍，消除首帧延迟。应用启动时调用一次。"""
        try:
            self.classifier.predict_proba(np.zeros((8, 32, 32), np.uint8))
        except Exception as e:                           # 预热失败不影响启动，真实调用时会返回 INTERNAL_ERROR
            logger.warning("lpr warmup: 分类器预热失败：{}", e)
        self._recognize_safe(np.zeros((720, 1280, 3), np.uint8), None, False)

    def recognize(self, image: np.ndarray, *, lane_id: int | None = None,
                  debug: bool = False) -> RecognitionResult:
        """单帧识别。image 为 BGR uint8。debug=True 时 result.debug 带各阶段中间图。"""
        result = self._recognize_safe(image, lane_id, debug)
        self._log(result, lane_id)
        return result

    def recognize_bytes(self, data: bytes, *, lane_id: int | None = None) -> RecognitionResult:
        """jpg/png 字节流，便于直接接 FastAPI UploadFile。"""
        result = self._recognize_bytes_safe(data, lane_id)
        self._log(result, lane_id)
        return result

    def recognize_many(self, images: Sequence[np.ndarray | bytes], *,
                       lane_id: int | None = None) -> RecognitionResult:
        """同一车辆的多帧（元素可为 BGR 数组或 jpg/png 字节），内部逐帧识别后投票。"""
        t0 = time.perf_counter()
        try:
            frames = list(images) if images is not None else []
            if not frames:
                result = failure_result(S.BAD_INPUT, "图像无效：没有输入帧")
            else:
                results = [self._recognize_bytes_safe(x, lane_id) if isinstance(x, _BYTES_TYPES)
                           else self._recognize_safe(x, lane_id, False) for x in frames]
                result = vote(results, min_frames=self.config.voting.min_frames,
                              min_confidence=self.config.postprocess.min_confidence)
                stage_ms: dict[str, float] = {}
                for r in results:
                    for k, v in r.stage_ms.items():
                        stage_ms[k] = stage_ms.get(k, 0.0) + v
                result.stage_ms = stage_ms
        except NotImplementedError as e:
            result = failure_result(S.INTERNAL_ERROR, f"算法内部异常：{e}")
        except Exception as e:
            logger.exception("lpr recognize_many 异常 lane={}", lane_id)
            result = failure_result(S.INTERNAL_ERROR, f"算法内部异常：{type(e).__name__}")
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        result.model_version = self.version
        self._log(result, lane_id)
        return result

    # ---- 内部实现 ----

    def _recognize_bytes_safe(self, data: object, lane_id: int | None) -> RecognitionResult:
        t0 = time.perf_counter()
        try:
            image = decode_image(data) if isinstance(data, _BYTES_TYPES) else None
        except Exception:
            image = None
        decode_ms = (time.perf_counter() - t0) * 1000
        if image is None:
            return failure_result(S.BAD_INPUT, "图像无效：无法解码为 jpg/png", elapsed_ms=decode_ms,
                                  stage_ms={"decode": decode_ms}, model_version=self.version)
        result = self._recognize_safe(image, lane_id, False)
        result.stage_ms = {"decode": decode_ms, **result.stage_ms}
        result.elapsed_ms += decode_ms
        return result

    def _recognize_safe(self, image: object, lane_id: int | None, debug: bool) -> RecognitionResult:
        t0 = time.perf_counter()
        timer = _StageTimer()
        dbg: dict[str, np.ndarray] | None = {} if debug else None
        try:
            result = self._run(image, lane_id, timer, dbg)
        except NotImplementedError as e:
            result = failure_result(S.INTERNAL_ERROR, f"算法内部异常：{e}")
        except Exception as e:
            logger.exception("lpr 识别异常 lane={}", lane_id)
            result = failure_result(S.INTERNAL_ERROR, f"算法内部异常：{type(e).__name__}")
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        result.stage_ms = timer.ms
        result.model_version = self.version
        if debug:
            result.debug = dbg
        return result

    def _run(self, image: object, lane_id: int | None, timer: _StageTimer,
             dbg: dict[str, np.ndarray] | None) -> RecognitionResult:
        reason = _check_image(image)
        if reason is not None:
            return failure_result(S.BAD_INPUT, f"图像无效：{reason}")
        assert isinstance(image, np.ndarray)
        cfg = self.config

        with timer("preprocess"):
            try:
                pre = preprocess(image, cfg.preprocess, cfg.lane_roi(lane_id))
            except BadInputError as e:
                return failure_result(S.BAD_INPUT, f"图像无效：{e}")
        if dbg is not None:
            dbg["work"] = pre.work_bgr
            dbg["gray"] = pre.gray

        with timer("locate"):
            candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
        if self.corner_detector is not None:
            roi = cfg.lane_roi(lane_id)
            with timer("corner"):
                net = [c for c in self.corner_detector.detect(image, cfg.runtime.max_candidates) if _in_roi(c, roi)]
            candidates = _merge_candidates(net, candidates, cfg.runtime.max_candidates)
        if not candidates:
            return failure_result(S.NO_PLATE)

        # Isolate the new policy: legacy first_ok and agree retain their original
        # control flow and early-return behavior.
        if cfg.postprocess.arbitration == "three_route":
            return self._run_three_route(image, lane_id, timer, dbg, candidates)

        best: _Attempt | None = None
        agree = cfg.postprocess.arbitration == "agree" and self.plate_reader is not None
        pairs: list[tuple[_Attempt | None, _Attempt | None]] = []   # agree：每块校正图两路线的结果，供回退选读法
        lead_color = candidates[0].color
        for i, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
            with timer("rectify"):
                rects = _rectifications(image, cand, cfg.rectify)
            if i == 0:
                lead_color = rects[0].color
            if dbg is not None:
                dbg[f"plate_{i}"] = rects[0].image
            for rect in rects:                           # 四角点假设回退
                seg_best: _Attempt | None = None
                for layout in _layouts_to_try(rect):
                    with timer("segment"):
                        seg = segment(rect, layout, cfg.segment)
                    if not seg.ok:
                        continue
                    with timer("recognize"):
                        probs = self.classifier.predict_proba(seg.chars)
                    with timer("postprocess"):
                        dec = decode(probs, seg.layout, rect.color, cfg.postprocess)
                    if agree:                            # 先不返回，等整牌识别结果一起仲裁
                        if seg_best is None or _rank(dec) > _rank(seg_best.dec):
                            seg_best = _Attempt(dec, rect, seg)
                        if dec.status == S.OK:
                            break
                        continue
                    if best is None or _rank(dec) > _rank(best.dec):
                        best = _Attempt(dec, rect, seg)
                    if dec.status == S.OK:               # 提前结束
                        return self._build(best, dbg)
                if agree:
                    with timer("reader"):
                        text, confs = self.plate_reader.read(rect.image)
                    dec = _decode_text(text, confs, rect.color, cfg.postprocess.min_confidence)
                    rd_att = _Attempt(dec, rect, None) if dec is not None else None
                    pairs.append((seg_best, rd_att))
                    att = _arbitrate(seg_best, rd_att, cfg.postprocess)
                    if att is not None:
                        if best is None or _rank(att.dec) > _rank(best.dec):
                            best = att
                        if att.dec.status == S.OK:
                            return self._build(best, dbg)
                elif self.plate_reader is not None:      # 分割路线没出 OK：整牌识别兜底
                    with timer("reader"):
                        text, confs = self.plate_reader.read(rect.image)
                    dec = _decode_text(text, confs, rect.color, cfg.postprocess.min_confidence)
                    if dec is not None:
                        if best is None or _rank(dec) > _rank(best.dec):
                            best = _Attempt(dec, rect, None)
                        if dec.status == S.OK:
                            return self._build(best, dbg)

        if lead_color in _SPECIAL_COLORS:                # 走到这里说明没有 OK 结果
            return failure_result(S.UNSUPPORTED)
        if best is not None:
            if agree and cfg.postprocess.nonok_fallback == "agreement_then_reader":
                best = _fallback_text(pairs, best)
            return self._build(best, dbg)
        return failure_result(S.SEG_FAILED)

    def _run_three_route(self, image: np.ndarray, lane_id: int | None, timer: _StageTimer,
                         dbg: dict[str, np.ndarray] | None,
                         candidates: list[PlateCandidate]) -> RecognitionResult:
        """Three-route streaming decision; model reads stop at the first automatic OK."""
        from . import arbitration

        cfg = self.config
        pp = cfg.postprocess
        records: list[dict] = []
        segment_attempts: dict[tuple[int, int], _Attempt | None] = {}
        reader_attempts: dict[tuple[int, int], _Attempt | None] = {}
        box_attempts: dict[int, _Attempt | None] = {}
        rects_by_candidate: dict[int, list[RectifiedPlate]] = {}
        lead_rect_color: str | None = None
        stop_auto_search = False

        for ci, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
            with timer("rectify"):
                rects = _rectifications(image, cand, cfg.rectify)
            rects_by_candidate[ci] = rects
            if ci == 0:
                lead_rect_color = rects[0].color if rects else None
            if dbg is not None and rects:
                dbg[f"plate_{ci}"] = rects[0].image

            # Lazily compute B once, then reuse it across every rect for this candidate.
            box_att: _Attempt | None = None
            if self.box_reader is not None:
                with timer("box_crop"):
                    native_crop = crop_box(image, cand.bbox, *self.box_reader.box_pad)
                with timer("box_reader"):
                    text, confs = self.box_reader.read(native_crop)
                box_dec = _decode_text(text, confs, cand.color, pp.min_confidence)
                if box_dec is not None:
                    box_source = (rects[0] if rects else
                                  RectifiedPlate(native_crop, box_dec.layout, cand.color, np.eye(3), cand))
                    box_att = _Attempt(box_dec, box_source, None, native_crop, cand.color)
                else:
                    box_att = None
            box_attempts[ci] = box_att
            box_trace = _route_trace_attempt(box_att)

            candidate_records: list[dict] = []
            for ri, rect in enumerate(rects):
                seg_best: _Attempt | None = None
                for layout in _layouts_to_try(rect):
                    with timer("segment"):
                        seg = segment(rect, layout, cfg.segment)
                    if not seg.ok:
                        continue
                    with timer("recognize"):
                        probs = self.classifier.predict_proba(seg.chars)
                    with timer("postprocess"):
                        dec = decode(probs, seg.layout, rect.color, pp)
                    attempt = _Attempt(dec, rect, seg)
                    if seg_best is None or _rank(dec) > _rank(seg_best.dec):
                        seg_best = attempt
                    if dec.status == S.OK:
                        break

                rd_att: _Attempt | None = None
                if self.plate_reader is not None:
                    with timer("reader"):
                        text, confs = self.plate_reader.read(rect.image)
                    rd_dec = _decode_text(text, confs, rect.color, pp.min_confidence)
                    if rd_dec is not None:
                        rd_att = _Attempt(rd_dec, rect, None)
                segment_attempts[(ci, ri)] = seg_best
                reader_attempts[(ci, ri)] = rd_att
                record = {
                    "candidate_index": ci, "rect_index": ri,
                    "segment": _route_trace_attempt(seg_best),
                    "reader": _route_trace_attempt(rd_att),
                    "box": box_trace, "color": rect.color,
                }
                candidate_records.append(record)
                records.append(record)
                auto = (arbitration.try_auto_candidate(ci, candidate_records, box_trace, pp)
                        if not stop_auto_search else None)
                if auto is not None:
                    if auto.status != S.OK or auto.plate is None:
                        stop_auto_search = True       # p2 rejects later auto results, but keep collecting fallback inputs.
                    else:
                        chosen = self._three_route_attempt(auto, segment_attempts, reader_attempts,
                                                           box_attempts, rects_by_candidate)
                        chosen.dec = replace(chosen.dec, status=S.OK, confidence=float(auto.confidence))
                        result = self._build(chosen, dbg)
                        result.route = auto.route
                        result.candidate_index = auto.candidate_index
                        result.rect_index = auto.rect_index
                        return result

            if not stop_auto_search:
                auto = arbitration.try_auto_candidate(ci, candidate_records, box_trace, pp,
                                                      candidate_complete=True)
                if auto is not None:
                    if auto.status == S.OK and auto.plate is not None:
                        chosen = self._three_route_attempt(auto, segment_attempts, reader_attempts,
                                                           box_attempts, rects_by_candidate)
                        chosen.dec = replace(chosen.dec, status=S.OK, confidence=float(auto.confidence))
                        result = self._build(chosen, dbg)
                        result.route = auto.route
                        result.candidate_index = auto.candidate_index
                        result.rect_index = auto.rect_index
                        return result
                    stop_auto_search = True

        # Terminal fallback uses the same pure policy used by trace replay.
        trace = {"rects": records, "boxes": {str(i): _route_trace_attempt(a)
                    for i, a in box_attempts.items()},
                 "candidates": [{"corners": list(c.corners)} for c in candidates],
                 "candidate_count": min(len(candidates), cfg.runtime.max_candidates),
                 "corners_by_candidate": {str(i): list(c.corners) for i, c in enumerate(candidates)}}
        decision = arbitration.arbitrate(trace, pp, predicted_corners=(candidates[0].corners if candidates else None),
                                         runtime=True)
        if decision is not None and decision.status == S.OK and decision.plate is not None:
            chosen = self._three_route_attempt(decision, segment_attempts, reader_attempts,
                                               box_attempts, rects_by_candidate)
            chosen.dec = replace(chosen.dec, status=S.OK, confidence=float(decision.confidence))
            result = self._build(chosen, dbg)
            result.route = decision.route
            result.candidate_index = decision.candidate_index
            result.rect_index = decision.rect_index
            return result

        if decision is not None and decision.status == S.LOW_CONFIDENCE and decision.plate is not None:
            chosen = self._three_route_attempt(decision, segment_attempts, reader_attempts,
                                               box_attempts, rects_by_candidate)
            chosen.dec = replace(chosen.dec, status=S.LOW_CONFIDENCE,
                                 confidence=float(decision.confidence), plate_no=decision.plate)
            result = self._build(chosen, dbg)
            result.route = decision.route
            result.candidate_index = decision.candidate_index
            result.rect_index = decision.rect_index
            return result

        if decision is not None and decision.status == S.INVALID_FORMAT and decision.plate is not None:
            chosen = self._three_route_attempt(decision, segment_attempts, reader_attempts,
                                               box_attempts, rects_by_candidate)
            result = self._build(chosen, dbg)
            result.route = decision.route
            result.candidate_index = decision.candidate_index
            result.rect_index = decision.rect_index
            return result

        if decision is not None and decision.status == S.UNSUPPORTED:
            return failure_result(S.UNSUPPORTED)

        if lead_rect_color in _SPECIAL_COLORS:
            return failure_result(S.UNSUPPORTED)
        # Preserve the old S/R INVALID_FORMAT result when no valid manual choice exists.
        invalid = [a for a in (*segment_attempts.values(), *reader_attempts.values())
                   if a is not None and a.dec.status == S.INVALID_FORMAT]
        if invalid:
            pick = max(invalid, key=lambda a: _rank(a.dec))
            return self._build(pick, dbg)
        return failure_result(S.SEG_FAILED)

    @staticmethod
    def _three_route_attempt(decision, segments, readers, boxes, rects_by_candidate) -> _Attempt:
        ci, ri, route = decision.candidate_index, decision.rect_index, decision.route
        if route == "segment":
            return segments[(ci, ri)]
        if route == "reader":
            return readers[(ci, ri)]
        if route == "box":
            return boxes[ci]
        # Defensive fallback: malformed pure-function output is caught by the safe wrapper.
        raise ValueError(f"未知三路路线：{route}")

    @staticmethod
    def _build(a: _Attempt, dbg: dict[str, np.ndarray] | None) -> RecognitionResult:
        d, cand = a.dec, a.rect.candidate
        if dbg is not None and a.seg is not None:
            dbg["chars"] = _chars_mosaic(a.seg.chars)
        return RecognitionResult(
            status=d.status,
            error_code=ERROR_CODES[d.status],
            message=DEFAULT_MESSAGES[d.status],
            plate_no=d.plate_no,
            plate_color=a.color or a.rect.color,
            layout=d.layout,
            confidence=float(d.confidence),
            char_confidences=[float(c) for c in d.char_confidences],
            candidates=[(p, float(s)) for p, s in d.candidates],
            bbox=cand.bbox,
            corners=list(cand.corners),
            plate_image=a.plate_image if a.plate_image is not None else a.rect.image,
            elapsed_ms=0.0,
            stage_ms={},
            vehicle_type_hint=d.vehicle_type_hint,
        )

    @staticmethod
    def _log(result: RecognitionResult, lane_id: int | None) -> None:
        logger.info("lpr lane={} plate={} conf={:.3f} status={} frames={} ms={:.1f}",
                    lane_id, result.plate_no, result.confidence, result.status.value,
                    result.frames_used, result.elapsed_ms)
        logger.debug("lpr stage_ms={}", {k: round(v, 2) for k, v in result.stage_ms.items()})
