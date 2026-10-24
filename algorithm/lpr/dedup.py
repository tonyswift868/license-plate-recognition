"""Locked, lane-isolated duplicate detection (6.8); generic OpenCV dHash."""
from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from .config import LprConfig
from .types import RecogStatus as S, RecognitionResult


def dhash(image: np.ndarray | None) -> int | None:
    """64-bit difference hash; missing/invalid images never support similarity."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8:
        return None
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        else:
            edits += 1
            if edits > 1:
                return False
        j += 1
    return True


class DuplicateFilter:
    """Keep nonduplicate observations only; duplicate hits do not extend TTL."""
    def __init__(self, window_s: float = 30, dhash_max_hamming: int = 10) -> None:
        if not math.isfinite(window_s) or window_s < 0 or not 0 <= dhash_max_hamming <= 64:
            raise ValueError("invalid duplicate thresholds")
        self.window_s = window_s
        self.dhash_max_hamming = dhash_max_hamming
        self._lock = threading.Lock()
        self._history: list[tuple[int, str, int | None, float]] = []
        self._last: float | None = None

    @classmethod
    def from_config(cls, cfg: LprConfig) -> "DuplicateFilter":
        return cls(cfg.dedup.window_s, cfg.dedup.dhash_max_hamming)

    def check(self, lane_id: int, result: RecognitionResult, ts: float) -> tuple[bool, str]:
        with self._lock:
            if not math.isfinite(ts) or (self._last is not None and ts < self._last):
                raise ValueError("timestamps must be finite and monotonic")
            self._last = ts
            self._history = [entry for entry in self._history if ts - entry[3] <= self.window_s]
            if result.status not in {S.OK, S.LOW_CONFIDENCE} or not result.plate_no:
                return False, ""
            plate, hashed = result.plate_no, dhash(result.plate_image)
            lane_history = [entry for entry in self._history if entry[0] == lane_id]
            if any(old == plate for _, old, _, _ in lane_history):
                return True, "same_plate"
            for _, old, old_hash, _ in lane_history:
                if (hashed is not None and old_hash is not None
                        and _edit_distance_at_most_one(plate, old)
                        and (hashed ^ old_hash).bit_count() <= self.dhash_max_hamming):
                    return True, "similar_plate"
            self._history.append((lane_id, plate, hashed, ts))
            return False, ""

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._last = None
