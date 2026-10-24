"""Position-constrained decoding; confidence always uses unmodified probabilities."""
from __future__ import annotations

import math
import numpy as np

from .charset import CHARSET, CHAR_TO_INDEX, NUM_CLASSES
from .config import PostprocessConfig
from .plate_rules import LAYOUTS, PlateLayout, match_layout, vehicle_type_hint
from .types import DecodeResult, PlateColor, RecogStatus


def _log(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def _beam(probs: np.ndarray, weights: np.ndarray, hyp: PlateLayout, k: int):
    beam = [("", 0.0, 0.0)]  # plate, selection log-score, original log-score
    for i, allowed in enumerate(hyp.slots):
        choices = sorted(allowed, key=lambda j: (-weights[i, j], j))[:k]
        expanded = [(s + CHARSET[j], score + _log(float(weights[i, j])),
                     raw + _log(float(probs[i, j])))
                    for s, score, raw in beam for j in choices]
        beam = sorted(expanded, key=lambda item: (-item[1], item[0]))[:k]
    return beam


def decode(probs: np.ndarray, layout: int, color: PlateColor, cfg: PostprocessConfig) -> DecodeResult:
    """Choose a layout by sum(log p), then beam-search its Top-k alternatives.

    Candidate scores are geometric means of raw probabilities, ranked by the
    selection likelihood (including province prior). They are not confidence.
    Malformed probability matrices are programming errors and raise ValueError.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if layout not in LAYOUTS or probs.shape != (layout, NUM_CLASSES):
        raise ValueError("probs must have shape (7 or 8, 65), matching layout")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must be finite probabilities in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    if cfg.topk < 1 or not 0 <= cfg.min_confidence <= 1:
        raise ValueError("topk must be positive and min_confidence in [0, 1]")
    weights = probs.copy()
    for province, coefficient in cfg.province_prior.items():
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("province prior coefficients must be finite and positive")
        weights[0, CHAR_TO_INDEX[province]] *= coefficient
    attempts = []
    for hyp in LAYOUTS[layout]:
        indices = [max(allowed, key=lambda j: weights[i, j]) for i, allowed in enumerate(hyp.slots)]
        confidence = [float(probs[i, j]) for i, j in enumerate(indices)]
        attempts.append((sum(_log(p) for p in confidence), hyp, indices, confidence))
    _, hyp, indices, char_conf = max(attempts, key=lambda item: item[0])
    plate = "".join(CHARSET[j] for j in indices)
    confidence = min(char_conf)
    if match_layout(plate) != hyp.name:
        status = RecogStatus.INVALID_FORMAT
    else:
        status = RecogStatus.OK if confidence >= cfg.min_confidence else RecogStatus.LOW_CONFIDENCE
    beam = _beam(probs, weights, hyp, cfg.topk)
    candidates = [(s, math.exp(raw / layout)) for s, _, raw in beam]
    return DecodeResult(status, plate, layout, hyp.name, confidence, char_conf,
                        candidates, vehicle_type_hint(color, hyp.name))
