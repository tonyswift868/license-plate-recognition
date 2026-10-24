"""数据契约（冻结，见开发文档第 5 章）。

规则：10-25 冻结后原则上只加字段，不改不删；改动须架构师确认并同步 FakeRecognizer。
坐标一律为原图像素坐标 (x, y)；图像一律 BGR uint8。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

PlateColor = Literal["blue", "yellow", "green", "white", "black", "unknown"]
VehicleTypeHint = Literal["small", "large", "unknown"]
Point = tuple[float, float]               # (x, y)，原图像素坐标
Box = tuple[int, int, int, int]           # (x, y, w, h)


class RecogStatus(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_PLATE = "NO_PLATE"
    SEG_FAILED = "SEG_FAILED"
    INVALID_FORMAT = "INVALID_FORMAT"
    UNSUPPORTED = "UNSUPPORTED"           # 特种车牌，暂不支持
    BAD_INPUT = "BAD_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class PreprocessOut:                      # 阶段一 → 阶段二
    work_bgr: np.ndarray                  # 缩放后的 BGR
    gray: np.ndarray                      # CLAHE 增强后的灰度
    hsv: np.ndarray
    scale: float                          # 工作图 = 原图（ROI 内）× scale
    offset: tuple[int, int]               # ROI 左上角在原图中的坐标


@dataclass
class PlateCandidate:                     # 阶段二 → 阶段三，坐标一律为原图坐标
    corners: list[Point]                  # 顺序 LT, RT, RB, LB（左上起顺时针）
    bbox: Box
    color: PlateColor
    score: float                          # 0~1
    source: str                           # "color" | "edge" | "merged"


@dataclass
class RectifiedPlate:                     # 阶段三 → 阶段四
    image: np.ndarray                     # BGR；7 位 252×80，8 位 276×80
    layout: int                           # 7 | 8 | 0（0 = 不确定，分割阶段两种都试）
    color: PlateColor
    homography: np.ndarray                # 3×3，原图 → 校正图
    candidate: PlateCandidate


@dataclass
class SegmentResult:                      # 阶段四 → 阶段五
    ok: bool
    layout: int
    boxes: list[Box]                      # 校正图坐标
    chars: np.ndarray                     # N×32×32 uint8，字符亮、背景暗，由 normalize_char 生成
    quality: float


@dataclass
class DecodeResult:                       # 阶段六 → 流水线（Phase 0 补充，见 5.9）
    status: RecogStatus                   # 只会是 OK | LOW_CONFIDENCE | INVALID_FORMAT
    plate_no: str
    layout: int                           # 7 | 8
    layout_name: str                      # "L7" | "L8S" | "L8L"，见附录 A
    confidence: float                     # min(char_confidences)
    char_confidences: list[float]
    candidates: list[tuple[str, float]]   # Top-k 备选车牌及得分
    vehicle_type_hint: VehicleTypeHint = "unknown"


@dataclass
class RecognitionResult:                  # 对外结果
    status: RecogStatus
    error_code: int                       # 0 或 3xxx，见 5.3
    message: str
    plate_no: str | None
    plate_color: PlateColor
    layout: int | None
    confidence: float                     # 0~1，定义见 5.4
    char_confidences: list[float]
    candidates: list[tuple[str, float]]   # Top-3 备选车牌及得分，供人工改牌下拉框
    bbox: Box | None
    corners: list[Point] | None
    plate_image: np.ndarray | None        # 校正后的车牌图（BGR）
    elapsed_ms: float
    stage_ms: dict[str, float]            # {"preprocess": .., "locate": .., ...}
    frames_used: int = 1
    vehicle_type_hint: VehicleTypeHint = "unknown"   # 车型提示，见 5.8
    model_version: str = ""
    debug: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        """去掉 ndarray 字段，结果可直接 JSON 序列化，用于统一响应体 data 与 WS 事件。"""
        return {
            "status": self.status.value,
            "error_code": int(self.error_code),
            "message": self.message,
            "plate_no": self.plate_no,
            "plate_color": self.plate_color,
            "layout": None if self.layout is None else int(self.layout),
            "confidence": round(float(self.confidence), 4),
            "char_confidences": [round(float(c), 4) for c in self.char_confidences],
            "candidates": [[str(p), round(float(s), 4)] for p, s in self.candidates],
            "bbox": None if self.bbox is None else [int(v) for v in self.bbox],
            "corners": None if self.corners is None else [[float(x), float(y)] for x, y in self.corners],
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "stage_ms": {k: round(float(v), 2) for k, v in self.stage_ms.items()},
            "frames_used": int(self.frames_used),
            "vehicle_type_hint": self.vehicle_type_hint,
            "model_version": self.model_version,
        }
