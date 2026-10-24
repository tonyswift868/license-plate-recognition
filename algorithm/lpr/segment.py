"""阶段四 分割（员工 B，任务 B3；算法见开发文档 6.4）。"""

from __future__ import annotations

from .config import SegmentConfig
from .types import RectifiedPlate, SegmentResult


def segment(rect: RectifiedPlate, layout: int, cfg: SegmentConfig) -> SegmentResult:
    """按指定版式（7 或 8）把校正图切成 layout 个字符。

    极性统一必须用 charnorm.unify_polarity，字符图必须用 charnorm.normalize_char 生成。
    成功：ok=True，boxes 有 layout 个（校正图坐标），chars 形状 layout×32×32 uint8。
    失败（空槽、字宽异常等）：ok=False，chars 为 0×32×32，流水线据此尝试下一个版式或候选。
    """
    raise NotImplementedError("阶段四 segment 尚未实现（员工 B，任务 B3）")
