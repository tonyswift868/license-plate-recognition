"""四角点网络：直接回归车牌四个角点（员工 B，开发文档 14 章 K1 Plan B-1）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import PlateCandidate


class CornerNet:
    """实现 recognize.base.CornerDetector。用 PyTorch 从零训练，不用任何预训练权重。"""

    name = "corner_net"

    def __init__(self, version: str) -> None:
        self.version = version

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CornerNet":
        """加载 state_dict（torch.load(..., weights_only=True)），模型置为 eval 模式。"""
        raise NotImplementedError("CornerNet 尚未实现（员工 B，第七批）")

    def detect(self, image: np.ndarray, max_candidates: int) -> list[PlateCandidate]:
        """原图 BGR uint8 → 按 score 降序的候选（原图坐标，corners 为 LT,RT,RB,LB，source="net"）。"""
        raise NotImplementedError("CornerNet 尚未实现（员工 B，第七批）")
