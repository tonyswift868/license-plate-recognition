"""联调桩（架构师）：供 M3、M7 在算法完成前并行开发，见开发文档 5.5。

答辩演示 R1 必须使用真实识别器 PlateRecognizer。
"""

from __future__ import annotations

import hashlib
import itertools
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .charset import NUM_CLASSES
from .errors import DEFAULT_MESSAGES, ERROR_CODES, failure_result
from .io_utils import decode_image, list_images
from .pipeline import _BYTES_TYPES, _check_image
from .types import PlateColor, RecogStatus, RecognitionResult, VehicleTypeHint

S = RecogStatus

DEFAULT_PLATES: list[tuple[str, PlateColor]] = [
    ("陕A12345", "blue"),
    ("陕AD12345", "green"),
    ("陕B8C9D2", "blue"),
    ("京A88888", "blue"),
    ("陕A6Y521", "yellow"),
]

_SUFFIX_STATUS = {"__LOW": S.LOW_CONFIDENCE, "__INVALID": S.INVALID_FORMAT}
_PREFIX_STATUS = {s.value: s for s in (S.NO_PLATE, S.SEG_FAILED, S.UNSUPPORTED, S.BAD_INPUT, S.INTERNAL_ERROR)}
_PLATE_BGR = {"blue": (170, 70, 10), "yellow": (20, 190, 235), "green": (130, 215, 120)}
_STAGES = ("preprocess", "locate", "rectify", "segment", "recognize", "postprocess")


def _vehicle_hint(color: str, layout: int) -> VehicleTypeHint:
    if layout == 8:
        return "small"
    return {"blue": "small", "yellow": "large"}.get(color, "unknown")


def _draw_plate(plate: str, color: str) -> np.ndarray:
    """画一张示意校正图（cv2.putText 不能写中文，只画省份字之后的部分）。"""
    w = 276 if len(plate) == 8 else 252
    img = np.full((80, w, 3), _PLATE_BGR.get(color, (128, 128, 128)), np.uint8)
    fg = (255, 255, 255) if color == "blue" else (0, 0, 0)
    cv2.rectangle(img, (3, 3), (w - 4, 76), fg, 2)
    text = plate[1:]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(img, text, ((w - tw) // 2 + 14, (80 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, fg, 3)
    return img


def _array_key(image: np.ndarray) -> str:
    return f"{image.shape}:{hashlib.sha1(image.tobytes()).hexdigest()}"


class UniformClassifier:
    """所有类别等概率的分类器。员工 A 的模型就绪前，可用它构造 PlateRecognizer 测流程：
    PlateRecognizer(load_config(), UniformClassifier())"""

    name = "uniform"
    version = "uniform"

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        return np.full((len(chars), NUM_CLASSES), 1.0 / NUM_CLASSES, np.float32)


class FakeRecognizer:
    """联调桩：方法签名与 PlateRecognizer 完全一致，不加载模型、不跑算法。

    三种用法（可组合）：
    1. 默认：按 plates 列表轮流返回 OK。
    2. script=["OK", "LOW_CONFIDENCE", "NO_PLATE"]：按顺序循环返回这些状态（车牌号仍按 plates 轮流）。
    3. samples_dir="演示图目录"：上传的图片与目录里某个文件内容相同时，按文件名决定结果：
         陕A12345.jpg           → OK，车牌 陕A12345（8 位按绿牌，其余按蓝牌）
         陕A12345__LOW.jpg      → LOW_CONFIDENCE，车牌 陕A12345
         陕A12345__INVALID.jpg  → INVALID_FORMAT
         NO_PLATE_1.jpg         → NO_PLATE（以状态名开头：NO_PLATE、SEG_FAILED、UNSUPPORTED、BAD_INPUT、INTERNAL_ERROR）
       不匹配的图片按 1、2 的规则处理。
    无法解码的字节、空数组等坏输入与真实识别器一样返回 BAD_INPUT。
    """

    def __init__(self, plates: Sequence[tuple[str, PlateColor]] = DEFAULT_PLATES, *,
                 script: Sequence[RecogStatus | str] | None = None,
                 samples_dir: str | os.PathLike[str] | None = None,
                 confidence: float = 0.97, low_confidence: float = 0.55,
                 latency_ms: float = 0.0) -> None:
        if not plates:
            raise ValueError("plates 不能为空")
        self.plates = list(plates)
        self.script = [S(s) for s in script] if script else None
        self.confidence = confidence
        self.low_confidence = low_confidence
        self.latency_ms = latency_ms
        self._lock = threading.Lock()
        self._plate_cycle = itertools.cycle(self.plates)
        self._script_cycle = itertools.cycle(self.script) if self.script else None
        self._samples: dict[str, str] = {}               # 内容哈希 → 文件名主干
        if samples_dir is not None:
            self._load_samples(Path(samples_dir))

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | None = None, **overrides: object) -> "FakeRecognizer":
        """与 PlateRecognizer.from_config 签名一致；桩不读配置。"""
        return cls()

    @property
    def version(self) -> str:
        return "lpr-fake"

    def warmup(self) -> None:
        pass

    def recognize(self, image: np.ndarray, *, lane_id: int | None = None,
                  debug: bool = False) -> RecognitionResult:
        if _check_image(image) is not None:
            return self._finish(failure_result(S.BAD_INPUT))
        assert isinstance(image, np.ndarray)
        return self._finish(self._make(self._samples.get(_array_key(image)), image.shape))

    def recognize_bytes(self, data: bytes, *, lane_id: int | None = None) -> RecognitionResult:
        image = decode_image(data) if isinstance(data, _BYTES_TYPES) else None
        if image is None:
            return self._finish(failure_result(S.BAD_INPUT, "图像无效：无法解码为 jpg/png"))
        stem = self._samples.get(hashlib.sha1(bytes(data)).hexdigest()) or self._samples.get(_array_key(image))
        return self._finish(self._make(stem, image.shape))

    def recognize_many(self, images: Sequence[np.ndarray | bytes], *,
                       lane_id: int | None = None) -> RecognitionResult:
        frames = list(images) if images is not None else []
        if not frames:
            return self._finish(failure_result(S.BAD_INPUT, "图像无效：没有输入帧"))
        first = frames[0]
        result = (self.recognize_bytes(first, lane_id=lane_id) if isinstance(first, _BYTES_TYPES)
                  else self.recognize(first, lane_id=lane_id))
        result.frames_used = len(frames)
        return result

    # ---- 内部实现 ----

    def _load_samples(self, root: Path) -> None:
        for p in list_images(root):
            data = p.read_bytes()
            self._samples[hashlib.sha1(data).hexdigest()] = p.stem
            image = decode_image(data)
            if image is not None:
                self._samples[_array_key(image)] = p.stem

    def _next(self) -> tuple[str, PlateColor, RecogStatus]:
        with self._lock:
            plate, color = next(self._plate_cycle)
            status = next(self._script_cycle) if self._script_cycle else S.OK
        return plate, color, status

    def _make(self, stem: str | None, shape: tuple[int, ...]) -> RecognitionResult:
        if stem is not None:
            for prefix, status in _PREFIX_STATUS.items():
                if stem.startswith(prefix):
                    return failure_result(status)
            status = S.OK
            for suffix, s in _SUFFIX_STATUS.items():
                if stem.endswith(suffix):
                    stem, status = stem[: -len(suffix)], s
            plate, color = stem, ("green" if len(stem) == 8 else "blue")
        else:
            plate, color, status = self._next()
        if status not in (S.OK, S.LOW_CONFIDENCE, S.INVALID_FORMAT):
            return failure_result(status)
        return self._plate_result(plate, color, status, shape)

    def _plate_result(self, plate: str, color: PlateColor, status: RecogStatus,
                      shape: tuple[int, ...]) -> RecognitionResult:
        h, w = shape[:2]
        bw, bh = max(1, int(w * 0.18)), max(1, int(h * 0.07))
        x, y = (w - bw) // 2, int(h * 0.7) - bh // 2
        corners = [(float(x), float(y)), (float(x + bw), float(y)),
                   (float(x + bw), float(y + bh)), (float(x), float(y + bh))]
        conf = {S.OK: self.confidence, S.LOW_CONFIDENCE: self.low_confidence}.get(status, 0.9)
        alt = plate[:-1] + ("8" if plate[-1] != "8" else "B")
        alt2 = plate[:-2] + ("0" if plate[-2] != "0" else "D") + plate[-1]
        return RecognitionResult(
            status=status,
            error_code=ERROR_CODES[status],
            message=DEFAULT_MESSAGES[status],
            plate_no=plate,
            plate_color=color,
            layout=len(plate),
            confidence=conf,
            char_confidences=[min(0.999, conf + 0.02)] * (len(plate) - 1) + [conf],
            candidates=[(plate, conf), (alt, round(conf * 0.3, 4)), (alt2, round(conf * 0.1, 4))],
            bbox=(x, y, bw, bh),
            corners=corners,
            plate_image=_draw_plate(plate, color),
            elapsed_ms=0.0,
            stage_ms={},
            vehicle_type_hint=_vehicle_hint(color, len(plate)),
        )

    def _finish(self, result: RecognitionResult) -> RecognitionResult:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
        result.elapsed_ms = self.latency_ms
        result.stage_ms = {k: self.latency_ms / len(_STAGES) for k in _STAGES}
        result.model_version = self.version
        return result
