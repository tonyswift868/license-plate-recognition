"""Shared directory dataset reader for CNN and baselines."""

from pathlib import Path
from concurrent.futures import Executor, ThreadPoolExecutor

import cv2
import numpy as np

from lpr.charset import NUM_CLASSES
from lpr.io_utils import imread

READ_WORKERS = 4
READ_BATCH_SIZE = 512


def _read_validated(path: Path) -> np.ndarray:
    image = imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (32, 32) or image.dtype != np.uint8:
        raise ValueError(f"expected normalized 32 x 32 uint8: {path}")
    return image


def _read_batched(paths: list[Path], executor: Executor, *, batch_size: int = READ_BATCH_SIZE):
    """Yield decoded images in input order while bounding queued work per batch."""
    for offset in range(0, len(paths), batch_size):
        batch = paths[offset:offset + batch_size]
        yield from executor.map(_read_validated, batch)


def load_split(root: Path, split: str, *, max_per_class: int | None = None,
               seed: int = 42, require_all: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Read already-normalized chars/<split>/<label>/*.png in charset order.

    Require all classes to avoid silently producing an incomplete model.
    Sample before decoding when a per-class cap is requested.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    # Select each class's sorted paths using the original seeded RNG before any
    # parallel reads; executor.map preserves this exact order.
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
        for label in range(NUM_CLASSES):
            paths = sorted((Path(root) / split / str(label)).glob("*.png"))
            if not paths and require_all:
                raise ValueError(f"missing samples for {split} class {label}")
            if max_per_class is not None and len(paths) > max_per_class:
                paths = [paths[i] for i in sorted(rng.choice(len(paths), max_per_class, replace=False))]
            images.extend(_read_batched(paths, executor))
            labels.extend([label] * len(paths))
    if not images:
        return np.empty((0, 32, 32), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    return np.stack(images), np.asarray(labels, dtype=np.int64)
