"""Weighted per-character voting and locked streaming sessions (6.7)."""
from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from .config import LprConfig
from .errors import DEFAULT_MESSAGES, ERROR_CODES, failure_result
from .plate_rules import match_layout, vehicle_type_hint
from .types import RecognitionResult
from .types import RecogStatus as S


def _valid(r: RecognitionResult) -> bool:
    return (r.status in {S.OK, S.LOW_CONFIDENCE} and r.layout in (7, 8)
            and bool(r.plate_no) and len(r.plate_no) == r.layout
            and len(r.char_confidences) == r.layout
            and all(math.isfinite(p) and 0 <= p <= 1 for p in r.char_confidences))


def vote(results: Sequence[RecognitionResult], *, min_frames: int = 2,
         min_confidence: float = 0.80) -> RecognitionResult:
    """Vote using valid frames of the modal length, without mutating inputs.

    Ties in modal layout and character weight use first occurrence. A fallback
    single frame counts as one participating frame (zero when all are invalid).
    """
    if min_frames < 1 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid voting thresholds")
    frames = list(results)
    if not frames:
        return failure_result(S.BAD_INPUT, frames_used=0)
    valid = [r for r in frames if _valid(r)]
    if not valid:
        return replace(frames[0], frames_used=0)
    if len(valid) < min_frames:
        return replace(max(valid, key=lambda r: r.confidence), frames_used=1)
    layout = Counter(r.layout for r in valid).most_common(1)[0][0]
    valid = [r for r in valid if r.layout == layout]
    best = max(valid, key=lambda r: r.confidence)
    chars, confidences = [], []
    for i in range(layout):
        weights = defaultdict(float)
        for r in valid:
            weights[r.plate_no[i]] += r.char_confidences[i]
        char = max(weights, key=weights.get)
        chars.append(char)
        confidences.append(weights[char] / len(valid))
    plate = "".join(chars)
    name = match_layout(plate)
    confidence = min(confidences)
    status = (S.INVALID_FORMAT if name is None else
              S.OK if confidence >= min_confidence else S.LOW_CONFIDENCE)
    # A new plate can be assembled from several frames; stale single-frame
    # candidates would disagree with it. Publish the voted plate and its score.
    return replace(best, status=status, error_code=ERROR_CODES[status],
                   message=DEFAULT_MESSAGES[status], plate_no=plate, layout=layout,
                   confidence=confidence, char_confidences=confidences,
                   candidates=[(plate, confidence)], frames_used=len(valid),
                   vehicle_type_hint=vehicle_type_hint(best.plate_color, name))


class PlateVoteSession:
    """One session per lane. Timestamps are finite, monotonic seconds."""
    def __init__(self, agree_frames: int = 3, window_s: float = 2.0, *,
                 min_frames: int = 2, min_confidence: float = 0.80) -> None:
        if agree_frames < 1 or not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("invalid session thresholds")
        if min_frames < 1 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid voting thresholds")
        self.agree_frames = agree_frames
        self.window_s = window_s
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._frames: list[RecognitionResult] = []
        self._counts: Counter[str] = Counter()
        self._start: float | None = None
        self._last: float | None = None

    @classmethod
    def from_config(cls, cfg: LprConfig) -> PlateVoteSession:
        return cls(cfg.voting.agree_frames, cfg.voting.window_s,
                   min_frames=cfg.voting.min_frames, min_confidence=cfg.postprocess.min_confidence)

    def feed(self, result: RecognitionResult, ts: float) -> RecognitionResult | None:
        with self._lock:
            if not math.isfinite(ts) or (self._last is not None and ts < self._last):
                raise ValueError("timestamps must be finite and monotonic")
            if self._start is None:
                self._start = ts
            self._last = ts
            self._frames.append(result)
            if _valid(result):
                self._counts[result.plate_no] += 1
            if (self._counts and max(self._counts.values()) >= self.agree_frames
                    or ts - self._start > self.window_s):
                return self._finish()
            return None

    def _finish(self) -> RecognitionResult | None:
        result = (vote(self._frames, min_frames=self.min_frames, min_confidence=self.min_confidence)
                  if self._frames else None)
        self._clear()
        return result

    def _clear(self) -> None:
        self._frames.clear()
        self._counts.clear()
        self._start = self._last = None

    def flush(self) -> RecognitionResult | None:
        with self._lock:
            return self._finish()

    def reset(self) -> None:
        with self._lock:
            self._clear()
