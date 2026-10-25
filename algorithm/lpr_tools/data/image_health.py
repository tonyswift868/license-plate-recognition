"""Image decode checks shared by test-set and evaluation tools."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_image(path: str | Path) -> tuple[np.ndarray | None, str | None]:
    """Decode an image from disk, returning (image, reason); reason is None on success."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        return None, f"read_error:{type(exc).__name__}:{exc}"
    if not data.size:
        return None, "empty_file"
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        return None, f"decode_error:{type(exc).__name__}"
    if image is None or image.size == 0:
        return None, "decode_failed"
    return image, None
