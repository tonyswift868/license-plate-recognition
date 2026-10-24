"""阶段六 后处理（员工 C，任务 C3；算法见开发文档 6.6、5.4）。"""

from __future__ import annotations

import numpy as np

from .config import PostprocessConfig
from .types import DecodeResult, PlateColor


def decode(probs: np.ndarray, layout: int, color: PlateColor, cfg: PostprocessConfig) -> DecodeResult:
    """按版式位置约束解码 → 正则校验 → 置信度 → Top-k 备选 → 状态。

    probs：layout×65 float32（分类器输出，每行和为 1）。layout：7 或 8。
    逐位置信度取约束集合内选中字符的原始 softmax 概率（不重新归一化）；confidence = min(逐位置信度)。
    status 只会是 OK、LOW_CONFIDENCE、INVALID_FORMAT；阈值用 cfg.min_confidence。
    """
    raise NotImplementedError("阶段六 decode 尚未实现（员工 C，任务 C3）")
