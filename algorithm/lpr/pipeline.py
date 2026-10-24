"""流水线编排（架构师）：PlateRecognizer，见开发文档 4.4、5.1、5.6。

recognize* 永不抛异常：任何错误都通过 status 与 error_code 返回。
阶段函数尚未实现（抛 NotImplementedError）时返回 INTERNAL_ERROR，保证桩阶段也能端到端跑通。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ._version import __version__
from .config import LprConfig, load_config
from .errors import DEFAULT_MESSAGES, ERROR_CODES, BadInputError, failure_result
from .io_utils import decode_image
from .locate import locate
from .postprocess import decode
from .preprocess import preprocess
from .recognize import load_classifier
from .recognize.base import CharClassifier
from .rectify import rectify
from .segment import segment
from .types import DecodeResult, RecogStatus, RecognitionResult, RectifiedPlate, SegmentResult
from .voting import vote

S = RecogStatus
_STATUS_RANK = {S.OK: 3, S.LOW_CONFIDENCE: 2, S.INVALID_FORMAT: 1}
_SPECIAL_COLORS = frozenset({"white", "black"})          # 特种车牌，暂不支持（Q12）
_BYTES_TYPES = (bytes, bytearray, memoryview)


@dataclass
class _Attempt:
    dec: DecodeResult
    rect: RectifiedPlate
    seg: SegmentResult


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


def _rank(dec: DecodeResult) -> tuple[int, float]:
    return _STATUS_RANK.get(dec.status, 0), float(dec.confidence)


def _chars_mosaic(chars: np.ndarray) -> np.ndarray:
    return np.hstack(list(chars)) if len(chars) else np.zeros((32, 32), np.uint8)


class PlateRecognizer:
    """车牌识别器。实例在调用之间没有可变共享状态，可被多线程并发调用。"""

    def __init__(self, config: LprConfig, classifier: CharClassifier) -> None:
        self.config = config
        self.classifier = classifier

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | None = None, **overrides: object) -> "PlateRecognizer":
        """加载配置与模型。模型缺失或字符表哈希不匹配时立即抛 ModelLoadError（启动即失败）。
        path 为空时先读环境变量 LPR_CONFIG，再缺省为 algorithm/config/default.toml。
        overrides 写法：runtime__classifier="svm"、postprocess__min_confidence=0.9。"""
        cfg = load_config(path, **overrides)
        import torch                                     # 延迟导入：FakeRecognizer 与 import lpr 不需要 torch

        torch.set_num_threads(cfg.runtime.torch_threads)
        classifier = load_classifier(cfg.runtime.classifier, cfg.resolve_path(cfg.runtime.model_path))
        return cls(cfg, classifier)

    @property
    def version(self) -> str:
        """例如 "lpr-1.0.0+char_cnn_v1"，写入日志与结果。"""
        return f"lpr-{__version__}+{self.classifier.version}"

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
        if not candidates:
            return failure_result(S.NO_PLATE)

        best: _Attempt | None = None
        lead_color = candidates[0].color
        for i, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
            with timer("rectify"):
                rect = rectify(image, cand, cfg.rectify)
            if i == 0:
                lead_color = rect.color
            if dbg is not None:
                dbg[f"plate_{i}"] = rect.image
            for layout in _layouts_to_try(rect):
                with timer("segment"):
                    seg = segment(rect, layout, cfg.segment)
                if not seg.ok:
                    continue
                with timer("recognize"):
                    probs = self.classifier.predict_proba(seg.chars)
                with timer("postprocess"):
                    dec = decode(probs, seg.layout, rect.color, cfg.postprocess)
                if best is None or _rank(dec) > _rank(best.dec):
                    best = _Attempt(dec, rect, seg)
                if dec.status == S.OK:                   # 提前结束
                    return self._build(best, dbg)

        if lead_color in _SPECIAL_COLORS:                # 走到这里说明没有 OK 结果
            return failure_result(S.UNSUPPORTED)
        if best is not None:
            return self._build(best, dbg)
        return failure_result(S.SEG_FAILED)

    @staticmethod
    def _build(a: _Attempt, dbg: dict[str, np.ndarray] | None) -> RecognitionResult:
        d, cand = a.dec, a.rect.candidate
        if dbg is not None:
            dbg["chars"] = _chars_mosaic(a.seg.chars)
        return RecognitionResult(
            status=d.status,
            error_code=ERROR_CODES[d.status],
            message=DEFAULT_MESSAGES[d.status],
            plate_no=d.plate_no,
            plate_color=a.rect.color,
            layout=d.layout,
            confidence=float(d.confidence),
            char_confidences=[float(c) for c in d.char_confidences],
            candidates=[(p, float(s)) for p, s in d.candidates],
            bbox=cand.bbox,
            corners=list(cand.corners),
            plate_image=a.rect.image,
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
