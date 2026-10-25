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


def _glyph_like_border_component(x: int, y: int, cw: int, ch: int, area: int,
                                 width: int, height: int) -> bool:
    """Keep a large character-shaped CC that merely touches a clipped plate edge."""
    fill = area / max(1, cw*ch)
    return (ch >= .38*height and .025*width <= cw <= .18*width
            and area >= .002*width*height and .12 <= cw/max(1, ch) <= 1.25
            and .10*height <= y+ch and y <= .90*height and .08 <= fill <= .95)


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
        x, y, cw, ch, area = stats[i]
        clipped_glyph = i in border_ids and _glyph_like_border_component(
            int(x), int(y), int(cw), int(ch), int(area), w, h)
        if (i in border_ids and not clipped_glyph) or area < .002*w*h:
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
        clipped_glyph = i in border_ids and _glyph_like_border_component(
            int(x), int(y), int(cw), int(ch), int(area), w, h)
        if ((i in border_ids and not clipped_glyph) or area < .002*w*h
                or outside_text_rivet):
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
                   *, projection_valleys: bool = False,
                   valley_search_rel: float = .06,
                   expanded_fit: bool = False,
                   return_fit: bool = False):
    h, w = binary.shape
    projection = np.count_nonzero(binary, axis=0)
    prefix = np.r_[0, projection.cumsum()]
    relative = np.asarray(centers)-.5
    if expanded_fit:
        scales = np.linspace(.75, 1.25, 21)
        offsets = np.arange(-round(.12*w), round(.12*w)+1, 2, dtype=np.int32)
    else:
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
    right[:, :-1] = np.clip(np.minimum(right[:, :-1], midpoints), 0, w)
    left[:, 1:] = np.clip(np.maximum(left[:, 1:], midpoints), 0, w)
    amounts = prefix[right]-prefix[left]
    total = amounts.sum(axis=1)
    boundary_ink = (projection[np.minimum(left, w-1)]+
                    projection[np.maximum(right-1, 0)]).sum(axis=1)
    scores = (total + .1*len(centers)*amounts.min(axis=1) - .5*boundary_ink
              - .05*np.abs(trial_offsets)*h - .02*np.abs(trial_scales-1)*w*h)
    best = int(np.argmax(scores))
    slots = list(zip(left[best].tolist(), right[best].tolist()))
    fit = (float(trial_scales[best]), int(trial_offsets[best]))
    if return_fit:
        return slots, fit[0], fit[1], round(w*.06)
    if not projection_valleys or len(slots) < 2:
        return slots

    # Allow a small local adjustment around each predicted character boundary.
    # Select the lowest-ink column, using distance to the predicted midpoint to
    # break ties. This cuts through the observed gap rather than an arbitrary
    # center midpoint when mild horizontal skew shifts individual glyphs.
    adjusted = list(slots)
    for i in range(len(adjusted)-1):
        midpoint = round((adjusted[i][1]+adjusted[i+1][0])/2)
        # When a fitted cut lands on visible character ink, search farther for
        # the actual inter-character gap. The CCPD oracle examples show a
        # misplaced prior cut can bisect a glyph by 10-15 px; keep the narrow
        # window whenever the fitted boundary already lies in a clean valley.
        center_midpoint = center_projection = None
        band_y0, band_y1 = round(.20*h), round(.80*h)
        center_projection = np.count_nonzero(binary[band_y0:band_y1], axis=0)
        center_midpoint = int(center_projection[midpoint])
        radius_rel = valley_search_rel if center_midpoint > 2 else .025
        radius_px = max(1, round(radius_rel*w))
        lo, hi = max(1, midpoint-radius_px), min(w-1, midpoint+radius_px+1)
        values = projection[lo:hi]
        if not len(values):
            continue
        # Only move into a convincing valley inside the character band. A
        # uniformly inked green plate gap can contain genuine strokes, so the
        # full-height minimum alone is insufficient evidence for a safe cut.
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


def _fit_slots_to_text_extent(binary: np.ndarray, centers: list[float], char_w: float):
    """Estimate a bounded global slot scale/offset from the central text band."""
    h, w = binary.shape
    y0, y1 = round(.20*h), round(.80*h)
    projection = np.count_nonzero(binary[y0:y1], axis=0)
    active = np.flatnonzero(projection >= max(3, round(.06*(y1-y0))))
    if len(active) < max(4, len(centers)*2):
        return None
    expected_left = (centers[0] - char_w/2)*w
    expected_right = (centers[-1] + char_w/2)*w
    expected_span = max(1., expected_right-expected_left)
    actual_span = float(active[-1]-active[0]+1)
    span_ratio = actual_span/expected_span
    expected_mid = (expected_left+expected_right)/2
    actual_mid = (active[0]+active[-1])/2
    offset = abs(actual_mid-expected_mid)/w
    if not (span_ratio < .88 or span_ratio > 1.12 or offset > .035):
        return None
    if not .75 <= span_ratio <= 1.25 or offset > .12:
        return None
    fitted_centers = [0.5+(float(center)-0.5)*span_ratio+(actual_mid-expected_mid)/w
                      for center in centers]
    fitted_char_w = char_w*span_ratio
    if (not np.isfinite(fitted_centers).all() or not .015 <= fitted_char_w <= .25
            or any(not .015 < center < .985 for center in fitted_centers)):
        return None
    return fitted_centers, fitted_char_w, span_ratio, (actual_mid-expected_mid)/w


def _alignment_search_saturated(binary: np.ndarray, centers: list[float], char_w: float) -> bool:
    """Permit a larger search only when the nominal ink objective hits a bound."""
    _, scale, offset, max_offset = _aligned_slots(binary, centers, char_w, return_fit=True)
    return scale <= .900001 or scale >= 1.079999 or abs(offset) >= max(0, max_offset-1)


def _segment_binary(gray: np.ndarray, binary: np.ndarray, layout: int,
                    centers: list[float], char_w: float,
                    *, projection_valleys: bool = False,
                    expanded_fit: bool = False) -> SegmentResult:
    failure = SegmentResult(False, layout, [], np.empty((0, 32, 32), np.uint8), 0.)
    h, w = binary.shape
    total = np.count_nonzero(binary)
    if total < layout*.002*w*h:
        return failure
    boxes, ink_counts = [], []
    slots = _aligned_slots(binary, centers, char_w,
                           projection_valleys=projection_valleys,
                           expanded_fit=expanded_fit)
    for left, right in slots:
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
        ink_counts.append(len(xs))
    coverage = min(1., sum(ink_counts)/total)
    if coverage < .85:
        return failure
    heights = np.array([b[3] for b in boxes])
    if heights.min() < .65*heights.max():
        return failure
    quality = float(np.clip(coverage*(1-heights.std()/max(1, heights.mean())), 0, 1))
    try:
        # Character training uses the whole fixed slot across the plate's full
        # height (build_chars.py). Keep that context and aspect ratio for the
        # classifier; the tight ink box is only reported geometry. Normalize
        # only after every slot and plate-level quality gate has passed.
        chars = [charnorm.normalize_char(gray[:, left:right]) for left, right in slots]
    except (ValueError, cv2.error):
        return failure
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
        # Keep the original path first to preserve prior successes. Normalize
        # full-height training slots only after all slot and plate-level gates
        # pass, so a rejected geometry never creates partial classifier inputs.
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
        result = _segment_binary(gray, adaptive, layout, centers, char_w, projection_valleys=True)
        if result.ok:
            return result
        # Only refit the global slot prior when the central text band clearly
        # differs in scale or position from the nominal layout. Preserve all
        # per-slot geometry and coverage checks after this evidence-based fit.
        for fitted in (binary, rim_cleaned, adaptive):
            fit = _fit_slots_to_text_extent(fitted, centers, char_w)
            saturated = _alignment_search_saturated(fitted, centers, char_w)
            if fit is not None or saturated:
                fitted_centers, fitted_width = fit[:2] if fit is not None else (centers, char_w)
                result = _segment_binary(gray, fitted, layout, fitted_centers, fitted_width,
                                         projection_valleys=True, expanded_fit=saturated)
                if result.ok:
                    return result
        return result
    except (ValueError, TypeError, AttributeError, IndexError, cv2.error, OverflowError):
        return failure
