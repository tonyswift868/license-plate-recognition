"""模板匹配基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TemplateClassifier:
    """实现 recognize.base.CharClassifier。得分经温度 softmax 转为概率。"""

    name = "template"

    def __init__(self, version: str) -> None:
        self.version = version

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "TemplateClassifier":
        raise NotImplementedError("TemplateClassifier 尚未实现（员工 A，任务 A3）")

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TemplateClassifier 尚未实现（员工 A，任务 A3）")
