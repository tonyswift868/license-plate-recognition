"""模板匹配基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from ..charset import NUM_CLASSES


class TemplateClassifier:
    """实现 recognize.base.CharClassifier。得分经温度 softmax 转为概率。"""

    name = "template"

    def __init__(self, version: str, templates: np.ndarray, temperature: float = 0.1,
                 present_classes: np.ndarray | None = None) -> None:
        self.version = version
        if templates.shape != (NUM_CLASSES, 32, 32) or not np.isfinite(templates).all():
            raise ValueError("templates must be finite 65 x 32 x 32")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.templates = np.array(templates, dtype=np.float32, copy=True)
        self.templates.setflags(write=False)
        mask = (np.ones(NUM_CLASSES, dtype=bool) if present_classes is None
                else np.asarray(present_classes, dtype=bool))
        if mask.shape != (NUM_CLASSES,) or not mask.any():
            raise ValueError("present_classes must be a nonempty 65-element mask")
        self.present_classes = np.array(mask, dtype=bool, copy=True)
        self.present_classes.setflags(write=False)
        self.temperature = float(temperature)

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "TemplateClassifier":
        with np.load(path, allow_pickle=False) as archive:
            mask = archive["present_classes"] if "present_classes" in archive else None
            return cls(version, archive["templates"], float(archive["temperature"].item()), mask)

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        if chars.ndim != 3 or chars.shape[1:] != (32, 32) or chars.dtype != np.uint8:
            raise ValueError("chars must be N x 32 x 32 uint8")
        probs = np.zeros((len(chars), NUM_CLASSES), dtype=np.float32)
        if not len(chars):
            return probs
        for i, image in enumerate(chars):
            image = image.astype(np.float32)
            scores = np.empty(int(self.present_classes.sum()), dtype=np.float32)
            labels = np.flatnonzero(self.present_classes)
            for j, label in enumerate(labels):
                score = cv2.matchTemplate(image, self.templates[label], cv2.TM_CCOEFF_NORMED)[0, 0]
                scores[j] = float(score) if np.isfinite(score) else -1.0
            scores = (scores - scores.max()) / self.temperature
            row = np.exp(scores)
            total = float(row.sum())
            if not np.isfinite(total) or total <= 0:
                raise ValueError("template scores could not be normalized")
            probs[i, labels] = row / total
        return probs
