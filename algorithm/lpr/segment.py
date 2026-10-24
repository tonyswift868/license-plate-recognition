"""Stage 4: polarity, component cleanup, layout alignment and character crops."""
from __future__ import annotations

import cv2
import numpy as np

from . import charnorm
from .config import SegmentConfig
from .types import RectifiedPlate, SegmentResult


def binary_plate(rect: RectifiedPlate, cfg: SegmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return unified grayscale and cleaned binary; shared by segmentation and viz."""
    image = rect.image
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3))
            or min(image.shape[:2]) < 16):
        raise ValueError('invalid rectified image')
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[..., 0]
    else:
        gray = image
    gray = charnorm.unify_polarity(gray, rect.color)
    if cfg.binarize == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5)
    else:
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    h, w = binary.shape
    # Remember boundary-connected ink before row trimming disconnects a frame.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    border_ids = set()
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x == 0 or y == 0 or x+cw == w or y+ch == h or (cw > .85*w and ch > .75*h):
            border_ids.add(i)
    transitions = np.count_nonzero(binary[:, 1:] != binary[:, :-1], axis=1)
    active = np.flatnonzero((transitions >= 6) & (np.arange(h) >= .08*h) & (np.arange(h) < .92*h))
    if len(active):
        binary[:max(0, active[0]-1)] = 0
        binary[min(h, active[-1]+2):] = 0
    for i in range(1, n):
        if i in border_ids or stats[i, cv2.CC_STAT_AREA] < .002*w*h:
            binary[labels == i] = 0
    return gray, binary


def _aligned_slots(binary: np.ndarray, centers: list[float], char_w: float) -> list[tuple[int, int]]:
    h, w = binary.shape
    projection = np.count_nonzero(binary, axis=0)
    prefix = np.r_[0, projection.cumsum()]
    relative = np.asarray(centers)-.5
    best_score, best = -np.inf, []
    # Keep order and the enlarged gap after the second character while fitting
    # modest crop margins/scale errors. The small prior resolves equal-ink ties.
    for scale in np.linspace(.90, 1.08, 10):
        radius = w*char_w*scale*.65
        for offset in range(-round(.06*w), round(.06*w)+1, 2):
            xs = (.5+relative*scale)*w + offset
            left = np.clip(np.floor(xs-radius).astype(int), 0, w)
            right = np.clip(np.ceil(xs+radius).astype(int), 0, w)
            # No column may be counted by two slots during alignment.
            midpoints = np.round((xs[:-1]+xs[1:])/2).astype(int)
            right[:-1] = np.minimum(right[:-1], midpoints)
            left[1:] = np.maximum(left[1:], midpoints)
            amounts = prefix[right]-prefix[left]
            total = amounts.sum()
            boundary_ink = (projection[np.minimum(left, w-1)]+projection[np.maximum(right-1, 0)]).sum()
            score = (total + .1*len(centers)*amounts.min() - .5*boundary_ink
                     - .05*abs(offset)*h - .02*abs(scale-1)*w*h)
            if score > best_score:
                best_score, best = score, list(zip(left.tolist(), right.tolist()))
    return best


def segment(rect: RectifiedPlate, layout: int, cfg: SegmentConfig) -> SegmentResult:
    failure = SegmentResult(False, layout, [], np.empty((0, 32, 32), np.uint8), 0.)
    try:
        if layout not in (7, 8) or rect is None:
            return failure
        centers = cfg.layout7_centers if layout == 7 else cfg.layout8_centers
        char_w = cfg.layout7_char_w if layout == 7 else cfg.layout8_char_w
        if len(centers) != layout or char_w <= 0 or not np.isfinite(centers).all():
            return failure
        gray, binary = binary_plate(rect, cfg)
        h, w = binary.shape
        total = np.count_nonzero(binary)
        if total < layout*.002*w*h:
            return failure
        boxes, chars, ink_counts = [], [], []
        for left, right in _aligned_slots(binary, centers, char_w):
            ys, xs = np.where(binary[:, left:right] > 0)
            if len(xs) < .002*w*h:
                return failure
            x0, x1 = left+int(xs.min()), left+int(xs.max())+1
            y0, y1 = int(ys.min()), int(ys.max())+1
            cw, ch = x1-x0, y1-y0
            if not .015*w <= cw <= .15*w or not .32*h <= ch <= .9*h:
                return failure
            if boxes and x0 < boxes[-1][0]+boxes[-1][2]:
                return failure
            # Ink touching the slot's vertical edge indicates a severed glyph.
            # A tolerance of one pixel permits interpolation noise at the rim.
            touches = np.count_nonzero(binary[y0:y1, left]) + np.count_nonzero(binary[y0:y1, right-1])
            if touches > .40*ch:
                return failure
            boxes.append((x0, y0, cw, ch))
            chars.append(charnorm.normalize_char(gray[y0:y1, x0:x1]))
            ink_counts.append(len(xs))
        coverage = min(1., sum(ink_counts)/total)
        if coverage < .85:
            return failure
        heights = np.array([b[3] for b in boxes])
        if heights.min() < .65*heights.max():
            return failure
        quality = float(np.clip(coverage*(1-heights.std()/max(1, heights.mean())), 0, 1))
        return SegmentResult(True, layout, boxes, np.stack(chars), quality)
    except (ValueError, TypeError, AttributeError, IndexError, cv2.error, OverflowError):
        return failure
