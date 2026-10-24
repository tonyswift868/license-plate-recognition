"""HOG+SVM 基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import joblib

from ..charset import NUM_CLASSES

HOG_SIZE = 324


def hog_features(chars: np.ndarray) -> np.ndarray:
    """Use a fresh descriptor per call to avoid sharing OpenCV mutable state."""
    if chars.ndim != 3 or chars.shape[1:] != (32, 32) or chars.dtype != np.uint8:
        raise ValueError("chars must be N x 32 x 32 uint8")
    if len(chars) == 0:
        return np.empty((0, HOG_SIZE), dtype=np.float32)
    if not hasattr(cv2, "HOGDescriptor"):
        raise RuntimeError("OpenCV build lacks cv2.HOGDescriptor; use an OpenCV 4.x build with HOG support")
    descriptor = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    return np.stack([descriptor.compute(np.ascontiguousarray(image)).reshape(-1) for image in chars])


class SvmClassifier:
    """实现 recognize.base.CharClassifier。HOG 用 cv2.HOGDescriptor，不用 scikit-image。"""

    name = "svm"

    def __init__(self, version: str, model) -> None:
        self.version = version
        if (not np.array_equal(model.classes_, np.arange(NUM_CLASSES))
                or model.n_features_in_ != HOG_SIZE or not model.probability):
            raise ValueError("expected probability SVC with all 65 classes and 324 HOG features")
        self.model = model

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "SvmClassifier":
        return cls(version, joblib.load(path))

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        features = hog_features(chars)
        if len(features) == 0:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        probs = self.model.predict_proba(features).astype(np.float32)
        return probs / probs.sum(axis=1, keepdims=True)
