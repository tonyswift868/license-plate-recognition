"""CNN 推理封装（员工 A，任务 A2）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CnnClassifier:
    """实现 recognize.base.CharClassifier。"""

    name = "cnn"

    def __init__(self, version: str) -> None:
        self.version = version

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CnnClassifier":
        """加载 state_dict（torch.load(..., weights_only=True)），模型置为 eval 模式。"""
        raise NotImplementedError("CnnClassifier 尚未实现（员工 A，任务 A2）")

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        """N×32×32 uint8 → N×65 float32（softmax），推理包在 torch.inference_mode() 中。"""
        raise NotImplementedError("CnnClassifier 尚未实现（员工 A，任务 A2）")
