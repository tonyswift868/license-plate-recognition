"""阶段三 校正（员工 B，任务 B2；算法见开发文档 6.3）。"""

from __future__ import annotations

import numpy as np

from .config import RectifyConfig
from .types import PlateCandidate, RectifiedPlate


def rectify(image: np.ndarray, cand: PlateCandidate, cfg: RectifyConfig) -> RectifiedPlate:
    """在原图上精修四角点 → 透视变换到 cfg.size_7 / cfg.size_8 → 颜色判别。

    image：原图 BGR（不是工作图，保证分辨率）。
    layout：绿牌 8，蓝/黄牌 7，判断不了为 0（分割阶段 7、8 两种都试）。
    homography：3×3，原图坐标 → 校正图坐标。
    """
    raise NotImplementedError("阶段三 rectify 尚未实现（员工 B，任务 B2）")
