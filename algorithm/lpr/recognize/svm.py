"""HOG+SVM 基线（员工 A，任务 A3；做法见开发文档 8.3）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

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
        classes = np.asarray(getattr(model, "classes_", ()))
        if (classes.ndim != 1 or len(classes) < 2 or
                not np.issubdtype(classes.dtype, np.integer) or
                np.any(classes < 0) or np.any(classes >= NUM_CLASSES) or
                len(np.unique(classes)) != len(classes) or
                getattr(model, "n_features_in_", None) != HOG_SIZE or
                not callable(getattr(model, "predict_proba", None))):
            raise ValueError("expected a probability classifier with at least two valid classes and 324 HOG features")
        self.classes = np.array(classes, dtype=np.int64, copy=True)
        self.classes.setflags(write=False)
        self.model = model

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> SvmClassifier:
        return cls(version, joblib.load(path))

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        features = hog_features(chars)
        if len(features) == 0:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        class_probs = np.asarray(self.model.predict_proba(features), dtype=np.float32)
        if class_probs.shape != (len(features), len(self.classes)):
            raise ValueError("SVM returned probabilities with an unexpected class shape")
        if not np.isfinite(class_probs).all() or np.any(class_probs < 0):
            raise ValueError("SVM returned invalid probabilities")
        totals = class_probs.sum(axis=1, keepdims=True)
        if np.any(totals <= 0) or not np.isfinite(totals).all():
            raise ValueError("SVM probability rows cannot be normalized")
        class_probs /= totals
        probs = np.zeros((len(features), NUM_CLASSES), dtype=np.float32)
        probs[:, self.classes] = class_probs
        return probs
