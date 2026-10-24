"""Shared directory dataset reader for CNN and baselines."""

from pathlib import Path

import cv2
import numpy as np

from lpr.charset import NUM_CLASSES
from lpr.io_utils import imread


def load_split(root: Path, split: str, *, max_per_class: int | None = None,
               seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Read already-normalized chars/<split>/<label>/*.png in charset order.

    Require all classes to avoid silently producing an incomplete model.
    Sample before decoding when a per-class cap is requested.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in range(NUM_CLASSES):
        paths = sorted((Path(root) / split / str(label)).glob("*.png"))
        if not paths:
            raise ValueError(f"missing samples for {split} class {label}")
        if max_per_class is not None and len(paths) > max_per_class:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_per_class, replace=False))]
        for path in paths:
            image = imread(path, cv2.IMREAD_GRAYSCALE)
            if image.shape != (32, 32) or image.dtype != np.uint8:
                raise ValueError(f"expected normalized 32 x 32 uint8: {path}")
            images.append(image)
            labels.append(label)
    return np.stack(images), np.asarray(labels, dtype=np.int64)
