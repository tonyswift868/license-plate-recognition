"""Stage 1: validate, crop, downsample, enhance gray and preserve raw HSV."""
from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig
from .errors import BadInputError
from .types import Box, PreprocessOut


def preprocess(image: np.ndarray, cfg: PreprocessConfig, roi: Box | None = None) -> PreprocessOut:
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3))):
        raise BadInputError("expected nonempty uint8 gray/BGR image")
    if min(image.shape[:2]) < cfg.min_side:
        raise BadInputError(f"short side must be at least {cfg.min_side}")
    if cfg.work_width < 1 or cfg.blur_ksize < 1 or cfg.blur_ksize % 2 != 1 or cfg.clahe_grid < 1:
        raise BadInputError("invalid preprocessing parameters")
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image.reshape(image.shape[:2]), cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    x0, y0 = 0, 0
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise BadInputError("ROI must contain four integers")
        x, y, rw, rh = roi
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x+rw), min(h, y+rh)
        if rw <= 0 or rh <= 0 or x1 <= x0 or y1 <= y0:
            raise BadInputError("ROI has no intersection with image")
        image = image[y0:y1, x0:x1]
        if min(image.shape[:2]) < cfg.min_side:
            raise BadInputError(f"ROI short side must be at least {cfg.min_side}")
    scale = min(1.0, cfg.work_width / image.shape[1])
    if scale < 1:
        image = cv2.resize(image, (cfg.work_width, max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    else:
        image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(cfg.clahe_clip, (cfg.clahe_grid, cfg.clahe_grid)).apply(gray)
    gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return PreprocessOut(image, gray, cv2.cvtColor(image, cv2.COLOR_BGR2HSV), scale, (x0, y0))
