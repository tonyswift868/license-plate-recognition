"""阶段二 定位（员工 B，任务 B1；算法见开发文档 6.2）。"""

from __future__ import annotations

from .config import LocateConfig
from .types import PlateCandidate, PreprocessOut


def locate(pre: PreprocessOut, cfg: LocateConfig, max_candidates: int = 3) -> list[PlateCandidate]:
    """颜色分支 + 边缘分支 → 形态学 → 轮廓 → 几何过滤 → 打分 → NMS → 前 max_candidates 个候选。

    返回按 score 降序排列的候选，坐标已换算回原图（用 pre.scale、pre.offset）；
    corners 顺序 LT, RT, RB, LB；color 为候选区域 HSV 多数投票结果。没有候选时返回空列表。
    """
    raise NotImplementedError("阶段二 locate 尚未实现（员工 B，任务 B1）")
