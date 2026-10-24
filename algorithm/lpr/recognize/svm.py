"""HOG+SVM 基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SvmClassifier:
    """实现 recognize.base.CharClassifier。HOG 用 cv2.HOGDescriptor，不用 scikit-image。"""

    name = "svm"

    def __init__(self, version: str) -> None:
        self.version = version

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "SvmClassifier":
        raise NotImplementedError("SvmClassifier 尚未实现（员工 A，任务 A3）")

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SvmClassifier 尚未实现（员工 A，任务 A3）")
