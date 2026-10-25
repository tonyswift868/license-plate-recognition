"""整牌识别 CRNN+CTC（员工 A，开发文档 14 章 K1 Plan B-2）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CrnnReader:
    """实现 recognize.base.PlateReader。用 PyTorch 从零训练，不用任何预训练权重。"""

    name = "crnn"

    def __init__(self, version: str) -> None:
        self.version = version

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CrnnReader":
        """加载 state_dict（torch.load(..., weights_only=True)），模型置为 eval 模式。"""
        raise NotImplementedError("CrnnReader 尚未实现（员工 A）")

    def read(self, plate_image: np.ndarray) -> tuple[str, list[float]]:
        """校正后的整块车牌 BGR → (车牌字符串, 逐位置信度)；读不出时返回 ("", [])。"""
        raise NotImplementedError("CrnnReader 尚未实现（员工 A）")
