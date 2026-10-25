"""Stage 4: polarity, component cleanup, layout alignment and character crops."""
from __future__ import annotations

import cv2
import numpy as np

from . import charnorm
from .config import SegmentConfig
from .types import RectifiedPlate, SegmentResult


def _unified_gray(rect: RectifiedPlate) -> np.ndarray:
    image = rect.image
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3))
            or min(image.shape[:2]) < 16):
        raise ValueError('invalid rectified image')
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[..., 0]
    else:
        gray = image
    return charnorm.unify_polarity(gray, rect.color)


def _threshold(gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    if cfg.binarize == 'adaptive':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, -5)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def binary_plate(rect: RectifiedPlate, cfg: SegmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return unified grayscale and cleaned binary; shared by segmentation and viz."""
    gray = _unified_gray(rect)
    binary = _threshold(gray, cfg)
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


def _fallback_binary_from_gray(gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Re-threshold unified grayscale, remove evidenced rim lines, then clean CCs.

    This independent mask can recover glyph ink if the primary cleanup removed a
    character connected to an interior plate-frame remnant.
    """
    binary = _threshold(gray, cfg)
    h, w = binary.shape
    rim_y, rim_x = max(2, round(.04*h)), max(2, round(.025*w))
    rim = np.zeros((h, w), np.uint8)
    rim[:rim_y] = 1
    rim[h-rim_y:] = 1
    rim[:, :rim_x] = 1
    rim[:, w-rim_x:] = 1
    edge_ink = (binary * rim).astype(np.uint8)

    # Detect long, near-horizontal/vertical runs in the outer rim. Erase only a
    # narrow stroke around the detected line so a connected character component
    # remains available to the subsequent CC cleanup.
    for y in list(range(rim_y)) + list(range(h-rim_y, h)):
        active = edge_ink[y] > 0
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if right-left >= .28*w:
                cv2.line(binary, (int(left), y), (int(right-1), y), 0, 3)
    for x in list(range(rim_x)) + list(range(w-rim_x, w)):
        active = edge_ink[:, x] > 0
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        for top, bottom in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if bottom-top >= .30*h:
                cv2.line(binary, (x, int(top)), (x, int(bottom-1)), 0, 3)

    # Hough catches diagonal plate borders which do not form long row/column runs.
    lines = cv2.HoughLinesP(edge_ink, 1, np.pi/180, threshold=max(10, round(.10*min(w, h))),
                            minLineLength=max(12, round(.20*min(w, h))), maxLineGap=round(.04*w))
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            dx, dy = abs(x2-x1), abs(y2-y1)
            horizontal = dx >= .25*w and dy <= max(3, .06*dx)
            vertical = dy >= .30*h and dx <= max(3, .06*dy)
            near_horizontal_edge = min(y1, y2) < rim_y or max(y1, y2) >= h-rim_y
            near_vertical_edge = min(x1, x2) < rim_x or max(x1, x2) >= w-rim_x
            if (horizontal and near_horizontal_edge) or (vertical and near_vertical_edge):
                cv2.line(binary, (x1, y1), (x2, y2), 0, 3)

    # Break border-connected components only after rim-line evidence is removed.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    border_ids = set()
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if x == 0 or y == 0 or x+cw == w or y+ch == h or (cw > .85*w and ch > .75*h):
            border_ids.add(i)
    transitions = np.count_nonzero(binary[:, 1:] != binary[:, :-1], axis=1)
    active = np.flatnonzero((transitions >= 6) & (np.arange(h) >= .08*h) & (np.arange(h) < .92*h))
    if len(active):
        binary[:max(0, active[0]-1)] = 0
        binary[min(h, active[-1]+2):] = 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        outside_text_rivet = (area < .012*w*h and ch < .16*h
                              and (y+ch <= .18*h or y >= .82*h))
        if i in border_ids or area < .002*w*h or outside_text_rivet:
            binary[labels == i] = 0
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]
        compact = area / max(1, cw*ch)
        in_separator_zone = .245*w <= cx <= .335*w and .30*h <= cy <= .70*h
        if (in_separator_zone and .0004*w*h <= area <= .008*w*h
                and cw <= .08*w and ch <= .30*h and compact >= .35):
            binary[labels == i] = 0
    return binary


def fallback_binary_plate(rect: RectifiedPlate, cfg: SegmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Independent unified-gray projection mask for calibration and fallback."""
    gray = _unified_gray(rect)
    return gray, _fallback_binary_from_gray(gray, cfg)


def _aligned_slots(binary: np.ndarray, centers: list[float], char_w: float,
                   *, projection_valleys: bool = False) -> list[tuple[int, int]]:
    h, w = binary.shape
    projection = np.count_nonzero(binary, axis=0)
    prefix = np.r_[0, projection.cumsum()]
    relative = np.asarray(centers)-.5
    scales = np.linspace(.90, 1.08, 10)
    offsets = np.arange(-round(.06*w), round(.06*w)+1, 2, dtype=np.int32)
    # Preserve the former scale-major/offset-minor search order so np.argmax's
    # first-tie rule returns exactly the same slots as the scalar implementation.
    trial_scales = np.repeat(scales, len(offsets))
    trial_offsets = np.tile(offsets, len(scales))
    radius = w*char_w*trial_scales[:, None]*.65
    xs = (.5+relative[None, :]*trial_scales[:, None])*w + trial_offsets[:, None]
    left = np.clip(np.floor(xs-radius).astype(int), 0, w)
    right = np.clip(np.ceil(xs+radius).astype(int), 0, w)
    # No column may be counted by two slots during alignment.
    midpoints = np.round((xs[:, :-1]+xs[:, 1:])/2).astype(int)
    right[:, :-1] = np.minimum(right[:, :-1], midpoints)
    left[:, 1:] = np.maximum(left[:, 1:], midpoints)
    amounts = prefix[right]-prefix[left]
    total = amounts.sum(axis=1)
    boundary_ink = (projection[np.minimum(left, w-1)]+
                    projection[np.maximum(right-1, 0)]).sum(axis=1)
    scores = (total + .1*len(centers)*amounts.min(axis=1) - .5*boundary_ink
              - .05*np.abs(trial_offsets)*h - .02*np.abs(trial_scales-1)*w*h)
    best = int(np.argmax(scores))
    slots = list(zip(left[best].tolist(), right[best].tolist()))
    if not projection_valleys or len(slots) < 2:
        return slots

    # Allow a small local adjustment around each predicted character boundary.
    # Select the lowest-ink column, using distance to the predicted midpoint to
    # break ties. This cuts through the observed gap rather than an arbitrary
    # center midpoint when mild horizontal skew shifts individual glyphs.
    adjusted = list(slots)
    for i in range(len(adjusted)-1):
        midpoint = round((adjusted[i][1]+adjusted[i+1][0])/2)
        radius_px = max(1, round(.025*w))
        lo, hi = max(1, midpoint-radius_px), min(w-1, midpoint+radius_px+1)
        values = projection[lo:hi]
        if not len(values):
            continue
        # Only move into a convincing valley inside the character band. A
        # uniformly inked green plate gap can contain genuine strokes, so the
        # full-height minimum alone is insufficient evidence for a safe cut.
        y0, y1 = round(.20*h), round(.80*h)
        center_projection = np.count_nonzero(binary[y0:y1], axis=0)
        center_values = center_projection[lo:hi]
        minimum = values.min()
        center_minimum = center_values.min()
        center_midpoint = center_projection[midpoint]
        local_median = float(np.median(center_values))
        if (center_minimum > 2 or local_median == 0
                or center_minimum > .35*local_median
                or center_minimum > .5*center_midpoint):
            continue
        candidates = np.flatnonzero((center_values == center_minimum) & (values == minimum))+lo
        if not len(candidates):
            candidates = np.flatnonzero(center_values == center_minimum)+lo
        cut = int(candidates[np.argmin(np.abs(candidates-midpoint))])
        adjusted[i] = (adjusted[i][0], cut)
        adjusted[i+1] = (cut, adjusted[i+1][1])
    return adjusted


def _crossing_edge_ink(binary: np.ndarray, y0: int, y1: int,
                       left: int, right: int) -> int:
    """Count rows where ink crosses a slot cut into its outside neighbor."""
    h, w = binary.shape
    left_ink = np.count_nonzero(binary[y0:y1, left])
    right_ink = np.count_nonzero(binary[y0:y1, right-1])
    if left > 0:
        left_ink = np.count_nonzero(binary[y0:y1, left] & binary[y0:y1, left-1])
    if right < w:
        right_ink = np.count_nonzero(binary[y0:y1, right-1] & binary[y0:y1, right])
    return int(left_ink + right_ink)


def _segment_binary(gray: np.ndarray, binary: np.ndarray, layout: int,
                    centers: list[float], char_w: float,
                    *, projection_valleys: bool = False) -> SegmentResult:
    failure = SegmentResult(False, layout, [], np.empty((0, 32, 32), np.uint8), 0.)
    h, w = binary.shape
    total = np.count_nonzero(binary)
    if total < layout*.002*w*h:
        return failure
    boxes, chars, ink_counts = [], [], []
    for left, right in _aligned_slots(binary, centers, char_w, projection_valleys=projection_valleys):
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
        # Only count strokes that continue across a cut. A complete glyph may
        # legitimately touch a slot edge while the adjacent outside column is
        # blank; counting one-sided edge ink rejects that valid case.
        if _crossing_edge_ink(binary, y0, y1, left, right) > .40*ch:
            return failure
        boxes.append((x0, y0, cw, ch))
        try:
            chars.append(charnorm.normalize_char(gray[y0:y1, x0:x1]))
        except (ValueError, cv2.error):
            return failure
        ink_counts.append(len(xs))
    coverage = min(1., sum(ink_counts)/total)
    if coverage < .85:
        return failure
    heights = np.array([b[3] for b in boxes])
    if heights.min() < .65*heights.max():
        return failure
    quality = float(np.clip(coverage*(1-heights.std()/max(1, heights.mean())), 0, 1))
    return SegmentResult(True, layout, boxes, np.stack(chars), quality)


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
        # Keep the original path first to preserve prior successes. Only failed
        # plates try a rim-trimmed mask, then local projection-valley boundaries.
        result = _segment_binary(gray, binary, layout, centers, char_w)
        if result.ok:
            return result
        rim_cleaned = _fallback_binary_from_gray(gray, cfg)
        result = _segment_binary(gray, rim_cleaned, layout, centers, char_w)
        if result.ok:
            return result
        result = _segment_binary(gray, binary, layout, centers, char_w, projection_valleys=True)
        if result.ok:
            return result
        result = _segment_binary(gray, rim_cleaned, layout, centers, char_w, projection_valleys=True)
        if result.ok:
            return result
        # Uneven illumination can erase strokes under Otsu. Try adaptive
        # thresholding only as a final fallback and retain every geometry and
        # quality guard used by the primary path.
        adaptive_cfg = cfg.model_copy(update={'binarize': 'adaptive'})
        adaptive = _fallback_binary_from_gray(gray, adaptive_cfg)
        result = _segment_binary(gray, adaptive, layout, centers, char_w)
        if result.ok:
            return result
        return _segment_binary(gray, adaptive, layout, centers, char_w, projection_valleys=True)
    except (ValueError, TypeError, AttributeError, IndexError, cv2.error, OverflowError):
        return failure
