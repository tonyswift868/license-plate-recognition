"""阶段一 预处理（员工 B，任务 B1；算法见开发文档 6.1）。"""

from __future__ import annotations

import numpy as np

from .config import PreprocessConfig
from .types import Box, PreprocessOut


def preprocess(image: np.ndarray, cfg: PreprocessConfig, roi: Box | None = None) -> PreprocessOut:
    """校验 → 车道 ROI 裁剪 → 缩放到 cfg.work_width → CLAHE 灰度 + HSV。

    image：BGR uint8（流水线已排除 None、空图、非 uint8、维度不对的输入；单通道 H×W 也可能传进来，需转 3 通道）。
    roi：车道 ROI（原图坐标 x, y, w, h），None 表示全图。
    短边 < cfg.min_side 时抛 errors.BadInputError，流水线转为 BAD_INPUT。
    返回的 scale、offset 用于定位阶段把候选坐标换算回原图。
    """
    raise NotImplementedError("阶段一 preprocess 尚未实现（员工 B，任务 B1）")
