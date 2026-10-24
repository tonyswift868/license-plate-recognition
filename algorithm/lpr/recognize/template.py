"""模板匹配基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from ..charset import NUM_CLASSES


class TemplateClassifier:
    """实现 recognize.base.CharClassifier。得分经温度 softmax 转为概率。"""

    name = "template"

    def __init__(self, version: str, templates: np.ndarray, temperature: float = 0.1) -> None:
        self.version = version
        if templates.shape != (NUM_CLASSES, 32, 32) or not np.isfinite(templates).all():
            raise ValueError("templates must be finite 65 x 32 x 32")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.templates = np.array(templates, dtype=np.float32, copy=True)
        self.templates.setflags(write=False)
        self.temperature = float(temperature)

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "TemplateClassifier":
        with np.load(path, allow_pickle=False) as archive:
            return cls(version, archive["templates"], float(archive["temperature"].item()))

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        if chars.ndim != 3 or chars.shape[1:] != (32, 32) or chars.dtype != np.uint8:
            raise ValueError("chars must be N x 32 x 32 uint8")
        scores = np.empty((len(chars), NUM_CLASSES), dtype=np.float32)
        if not len(chars):
            return scores
        for i, image in enumerate(chars):
            image = image.astype(np.float32)
            for label, template in enumerate(self.templates):
                scores[i, label] = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        scores = (scores - scores.max(axis=1, keepdims=True)) / self.temperature
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)
