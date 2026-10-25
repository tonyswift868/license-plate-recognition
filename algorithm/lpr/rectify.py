"""Stage 3: refine a proposal in the original image and warp its four corners."""
from __future__ import annotations

import cv2
import numpy as np

from .config import RectifyConfig
from .errors import BadInputError
from .locate import box_iou, order_points
from .types import PlateCandidate, RectifiedPlate

# OpenCV hue intervals; these are colour classification rules, not proposal thresholds.
_COLOR_RANGES = {'blue': ((100, 70, 40), (124, 255, 255)),
                 'yellow': ((15, 90, 70), (35, 255, 255)),
                 'green': ((35, 50, 60), (90, 255, 255))}


def classify_color(image: np.ndarray) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Ignore the outer rim and count plate background pixels, including mixed plates.
    h, w = hsv.shape[:2]
    inner = hsv[max(0, h//10):max(1, h-h//10), max(0, w//20):max(1, w-w//20)]
    fractions = {color: float(np.mean(cv2.inRange(inner, np.uint8(lo), np.uint8(hi)) > 0))
                 for color, (lo, hi) in _COLOR_RANGES.items()}
    ranked = sorted(fractions.items(), key=lambda item: item[1], reverse=True)
    color, color_fraction = ranked[0]
    runner_up, runner_fraction = ranked[1]
    if color_fraction >= .15:
        # Daylight/white-balance shifts can move a true blue plate toward green
        # (or a green plate toward yellow). When the rectified crop has strong
        # evidence for both competing hues, retain uncertainty so the pipeline
        # tries both layouts instead of committing to the wrong character count.
        if runner_fraction >= max(.15, color_fraction*.40):
            return 'unknown'
        return color
    # White/black is a special-plate decision in the pipeline. Require neutral
    # pixels as well as the brightness vote so saturated low-light blue/green
    # backgrounds do not become unsupported black plates.
    neutral = inner[..., 1] < 35
    if np.mean(neutral & (inner[..., 2] > 175)) > .6:
        return 'white'
    if np.mean(neutral & (inner[..., 2] < 35)) > .6:
        return 'black'
    return 'unknown'


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two ordered convex quadrilaterals."""
    area_a = abs(cv2.contourArea(np.float32(a)))
    area_b = abs(cv2.contourArea(np.float32(b)))
    if area_a <= 0 or area_b <= 0:
        return 0.
    try:
        intersection, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    except cv2.error:
        return 0.
    union = area_a + area_b - float(intersection)
    return float(intersection / union) if union > 0 else 0.


def _quad_candidates(image: np.ndarray, cand: PlateCandidate, expand: float,
                    max_candidates: int = 8) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return legacy first accepted geometry plus a bounded set of alternatives."""
    h, w = image.shape[:2]
    x, y, bw, bh = cand.bbox
    x0, y0 = max(0, int(np.floor(x-bw*expand))), max(0, int(np.floor(y-bh*expand)))
    x1, y1 = min(w, int(np.ceil(x+bw*(1+expand)))), min(h, int(np.ceil(y+bh*(1+expand))))
    fallback = order_points(cv2.boxPoints(cv2.minAreaRect(np.float32(cand.corners))))
    if x1 <= x0 or y1 <= y0:
        raise BadInputError('candidate does not intersect image')
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(crop.shape[:2], np.uint8)
    ranges = [_COLOR_RANGES[cand.color]] if cand.color in _COLOR_RANGES else list(_COLOR_RANGES.values())
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.uint8(lo), np.uint8(hi))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    expected = abs(cv2.contourArea(np.float32(cand.corners)))
    accepted: list[np.ndarray] = []
    first: np.ndarray | None = None
    for branch in (mask, edge):
        mode = cv2.RETR_EXTERNAL if branch is mask else cv2.RETR_LIST
        contours = cv2.findContours(branch, mode, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if not .40*expected <= area <= 1.8*expected:
                continue
            cx, cy, cw, ch = cv2.boundingRect(contour)
            # A crop boundary is not evidence of a plate boundary.
            if cx <= 0 or cy <= 0 or cx+cw >= crop.shape[1] or cy+ch >= crop.shape[0]:
                continue
            if box_iou((cx+x0, cy+y0, cw, ch), cand.bbox) < .45:
                continue
            hull = cv2.convexHull(contour)
            for epsilon in (.01, .02, .035, .05):
                poly = cv2.approxPolyDP(hull, epsilon*cv2.arcLength(hull, True), True)
                if len(poly) == 4 and cv2.isContourConvex(poly) and cv2.contourArea(poly) > .75*area:
                    quad = order_points(poly.reshape(4, 2) + [x0, y0])
                    sides = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
                    if min(sides) > 2 and 1.8 < (sides[0]+sides[2])/(sides[1]+sides[3]) < 6:
                        fallback_area = abs(cv2.contourArea(np.float32(fallback)))
                        quad_area = abs(cv2.contourArea(np.float32(quad)))
                        consistent = (_quad_iou(fallback, quad) >= .80
                                      and quad_area <= fallback_area * 1.08)
                        if first is None and consistent:
                            first = quad
                        if all(_quad_iou(quad, old) < .995 for old in accepted):
                            accepted.append(quad)
                            if len(accepted) >= max_candidates-1:
                                break
            if len(accepted) >= max_candidates-1:
                break
        if len(accepted) >= max_candidates-1:
            break
    primary = first if first is not None else fallback
    candidates = [primary]
    for quad in [fallback, *accepted]:
        if all(_quad_iou(quad, old) < .995 for old in candidates):
            candidates.append(quad)
        if len(candidates) >= max_candidates:
            break
    return primary, candidates


def _linear_percentile_sorted(values: np.ndarray, quantile: float) -> float:
    """Match NumPy's default linear percentile for already-sorted small arrays."""
    position = quantile*(len(values)-1)
    lower = int(position)
    upper = min(len(values)-1, lower+1)
    fraction = position-lower
    return float(values[lower] + fraction*(values[upper]-values[lower]))


def _green_boundary_quad(image: np.ndarray, cand: PlateCandidate
                         ) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    """Fit a green plate's top/bottom boundaries from supported color pixels."""
    if cand.color != 'green':
        return None, {"reason": "candidate_color"}
    h, w = image.shape[:2]
    x, y, bw, bh = cand.bbox
    x0, x1 = max(0, int(x-.18*bw)), min(w, int(x+1.18*bw))
    y0, y1 = max(0, int(y-.18*bh)), min(h, int(y+1.18*bh))
    if x1-x0 < 20 or y1-y0 < 12:
        return None, {"reason": "small_roi"}
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    lo, hi = _COLOR_RANGES['green']
    mask = cv2.inRange(hsv, np.uint8(lo), np.uint8(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8)) > 0
    rh, rw = mask.shape
    edges = np.linspace(0, rw, 10).astype(int)
    top_points, bottom_points, supports = [], [], []
    for i in range(9):
        left, right = edges[i], edges[i+1]
        if right-left < 2:
            continue
        columns = [np.flatnonzero(mask[:, col]) for col in range(left, right)]
        valid = [ys for ys in columns if len(ys) >= .30*rh]
        supports.append(len(valid)/max(1, right-left))
        if len(valid) < max(2, int(.45*(right-left))):
            continue
        top = float(np.median([_linear_percentile_sorted(ys, .08) for ys in valid]))
        bottom = float(np.median([_linear_percentile_sorted(ys, .92) for ys in valid]))
        x_center = x0+(left+right-1)/2
        top_points.append((x_center, y0+top))
        bottom_points.append((x_center, y0+bottom))
    if len(top_points) < 6:
        return None, {"reason": "insufficient_bins", "bin_count": len(top_points)}

    def robust_line(points: list[tuple[float, float]]) -> tuple[float, float, float]:
        p = np.asarray(points, dtype=np.float64)
        slopes = [(p[j, 1]-p[i, 1])/(p[j, 0]-p[i, 0])
                  for i in range(len(p)) for j in range(i+1, len(p))
                  if p[j, 0] != p[i, 0]]
        slope = float(np.median(slopes))
        intercept = float(np.median(p[:, 1]-slope*p[:, 0]))
        inliers = np.abs(p[:, 1]-(slope*p[:, 0]+intercept)) <= max(2., .06*bh)
        if inliers.sum() >= 6:
            q = p[inliers]
            slopes = [(q[j, 1]-q[i, 1])/(q[j, 0]-q[i, 0])
                      for i in range(len(q)) for j in range(i+1, len(q))
                      if q[j, 0] != q[i, 0]]
            slope = float(np.median(slopes))
            intercept = float(np.median(q[:, 1]-slope*q[:, 0]))
            residual = np.abs(q[:, 1]-(slope*q[:, 0]+intercept))
        else:
            residual = np.abs(p[:, 1]-(slope*p[:, 0]+intercept))
        return slope, intercept, float(np.median(residual))

    top_m, top_b, top_residual = robust_line(top_points)
    bottom_m, bottom_b, bottom_residual = robust_line(bottom_points)
    if max(top_residual, bottom_residual) > .08*bh:
        return None, {"reason": "fit_residual", "top_residual": top_residual,
                      "bottom_residual": bottom_residual, "bin_count": len(top_points)}
    left = max(x0, min(x0+rw-1, min(p[0] for p in top_points+bottom_points)))
    right = max(left+1, min(x0+rw-1, max(p[0] for p in top_points+bottom_points)))
    quad = np.float32([[left, top_m*left+top_b], [right, top_m*right+top_b],
                       [right, bottom_m*right+bottom_b], [left, bottom_m*left+bottom_b]])
    area = abs(cv2.contourArea(quad))
    expected = abs(cv2.contourArea(np.float32(cand.corners)))
    if (not cv2.isContourConvex(quad) or not .40*expected <= area <= 1.8*expected):
        return None, {"reason": "geometry_gate"}
    support = float(np.median(supports)) if supports else 0.
    if support < .45:
        return None, {"reason": "weak_support", "fit_support": support}
    return quad, {"reason": "accepted", "fit_support": support,
                  "top_residual": top_residual, "bottom_residual": bottom_residual}


def _quad_glyph_evidence(image: np.ndarray) -> dict[str, float | int]:
    """Measure repeated glyph-like components and their placement in a warp."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y0, y1 = int(.10*h), int(.90*h)
    x0, x1 = int(.03*w), int(.97*w)
    center = gray[y0:y1, x0:x1]
    threshold, _ = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = center <= threshold
    # Estimate polarity from the center, but measure components in the complete
    # warp. A component touching the center window is not necessarily clipped by
    # the actual plate image; full-image coordinates distinguish those cases.
    ink = gray <= threshold if np.count_nonzero(dark) < dark.size/2 else gray > threshold
    n, _, stats, centers = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    glyphs: list[tuple[float, int]] = []
    edge_touch = 0
    for i in range(1, n):
        xx, yy, ww, hh, area = stats[i]
        if (hh < .30*h or hh > .98*h or ww < 2 or ww > .20*w
                or area < 10 or area > .20*w*h):
            continue
        cx = float(centers[i, 0])
        if cx < .02*w or cx > .98*w:
            continue
        glyphs.append((cx, int(hh)))
        edge_touch += int(yy <= 1 or yy+hh >= h-1)
    count = len(glyphs)
    if count >= 4:
        xs = np.sort([item[0] for item in glyphs])
        gaps = np.diff(xs)
        gap_cv = float(np.std(gaps)/(np.mean(gaps)+1e-6))
        coverage = float((xs[-1]-xs[0])/w)
        median_height = float(np.median([item[1] for item in glyphs])/h)
    else:
        gap_cv, coverage, median_height = 1., 0., 0.
    return {"glyph_count": count, "edge_touch_count": edge_touch,
            "gap_cv": gap_cv, "coverage": coverage,
            "median_height_ratio": median_height,
            "ink_fraction": float(np.mean(ink[y0:y1, x0:x1]))}


def _quad_pixel_quality(image: np.ndarray, layout: int) -> float:
    """Rank a rectified crop by repeated character-like pixel components."""
    evidence = _quad_glyph_evidence(image)
    count = int(evidence["glyph_count"])
    edge_touch = int(evidence["edge_touch_count"])
    gap_cv = float(evidence["gap_cv"])
    coverage = float(evidence["coverage"])
    median_height = float(evidence["median_height_ratio"])
    target = 8 if layout == 8 else 7
    score = float(min(count, target)) - 1.8*abs(count-target) - 1.8*edge_touch
    if count >= 4:
        score -= min(2., 2.*gap_cv)
        score += 1.5*max(0., 1.-abs(coverage-.72)/.72)
        score += max(0., 1.-abs(median_height-.65)/.65)
    ink_fraction = float(evidence["ink_fraction"])
    if ink_fraction < .015 or ink_fraction > .48:
        score -= 1.5
    return score


def _quad_rank_score(pixel_quality: float, geometry_iou: float) -> float:
    """Use the first accepted contour quad as a prior when pixel evidence ties."""
    return float(pixel_quality - 4.*(1.-geometry_iou))


def _warp_quad(image: np.ndarray, cand: PlateCandidate, quad: np.ndarray
               ) -> RectifiedPlate:
    target7 = np.float32([[0, 0], [251, 0], [251, 79], [0, 79]])
    matrix7 = cv2.getPerspectiveTransform(np.float32(quad), target7)
    plate7 = cv2.warpPerspective(image, matrix7, (252, 80))
    color = classify_color(plate7)
    layout = 8 if color == 'green' else (7 if color in ('blue', 'yellow') else 0)
    if layout == 8:
        target8 = np.float32([[0, 0], [275, 0], [275, 79], [0, 79]])
        matrix = cv2.getPerspectiveTransform(np.float32(quad), target8)
        plate = cv2.warpPerspective(image, matrix, (276, 80))
    else:
        matrix, plate = matrix7, plate7
    return RectifiedPlate(plate, layout, color, matrix, cand)


def _rank_quad_candidates(image: np.ndarray, cand: PlateCandidate, expand: float,
                          *, include_green_fit: bool = False
                          ) -> list[tuple[np.ndarray, RectifiedPlate, float]]:
    """Extract one bounded quad pool, warp each once, and rank by pixel evidence."""
    primary, candidates = _quad_candidates(image, cand, expand)
    ranked: list[tuple[np.ndarray, RectifiedPlate, float]] = []
    for quad in candidates:
        rect = _warp_quad(image, cand, quad)
        layouts = (7, 8) if rect.layout == 0 else (rect.layout,)
        pixel_score = max(_quad_pixel_quality(rect.image, layout) for layout in layouts)
        score = _quad_rank_score(pixel_score, _quad_iou(primary, quad))
        ranked.append((quad, rect, score))
    # Python's stable sort preserves the historical first-on-tie behavior.
    ranked.sort(key=lambda item: item[2], reverse=True)
    if include_green_fit and ranked:
        fitted, evidence = _green_boundary_quad(image, cand)
        if fitted is not None and all(_quad_iou(existing[0], fitted) < .995
                                      for existing in ranked):
            rect = _warp_quad(image, cand, fitted)
            glyphs = _quad_glyph_evidence(rect.image)
            count = int(glyphs['glyph_count'])
            # Require visible repeated text as well as a stable color boundary;
            # a green body panel without a plate-like text row cannot compete.
            if (count >= 4 and glyphs['edge_touch_count'] <= max(1, count//3)
                    and glyphs['gap_cv'] <= .95 and glyphs['coverage'] >= .35):
                quality = _quad_pixel_quality(rect.image, 8 if rect.layout == 8 else 7)
                geometry_iou = _quad_iou(primary, fitted)
                penalty = 2.*(evidence['top_residual'] + evidence['bottom_residual'])/max(1, cand.bbox[3])
                score = _quad_rank_score(quality, geometry_iou) + .75*evidence['fit_support'] - penalty
                alternatives = [*ranked[1:], (fitted, rect, score)]
                alternatives.sort(key=lambda item: item[2], reverse=True)
                ranked = [ranked[0], *alternatives]
    return ranked


def refine_quad(image: np.ndarray, cand: PlateCandidate, expand: float) -> np.ndarray:
    """Choose among bounded geometric candidates using only warped pixels."""
    return _rank_quad_candidates(image, cand, expand)[0][0]


def _validate_image_candidate(image: np.ndarray, cand: PlateCandidate) -> np.ndarray:
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3))):
        raise BadInputError('expected nonempty uint8 gray/BGR image')
    if (cand is None or np.asarray(cand.corners).shape != (4, 2)
            or not np.isfinite(cand.corners).all() or min(cand.bbox[2:]) <= 0
            or abs(cv2.contourArea(np.float32(cand.corners))) < 1):
        raise BadInputError('invalid plate candidate')
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image.reshape(image.shape[:2]), cv2.COLOR_GRAY2BGR)
    return image


def _finalize_warp(rect: RectifiedPlate, cand: PlateCandidate) -> RectifiedPlate:
    color, plate = rect.color, rect.image
    if color == 'white' and cand.color == 'unknown':
        evidence = _quad_glyph_evidence(plate)
        repeated_text = (evidence['glyph_count'] >= 5
                         and evidence['edge_touch_count'] <= max(1, evidence['glyph_count']//4)
                         and evidence['gap_cv'] <= .8
                         and evidence['coverage'] >= .45)
        if not repeated_text:
            color = 'unknown'
    layout = 8 if color == 'green' else (7 if color in ('blue', 'yellow') else 0)
    return RectifiedPlate(plate, layout, color, rect.homography, cand)


def rectify(image: np.ndarray, cand: PlateCandidate, cfg: RectifyConfig) -> RectifiedPlate:
    image = _validate_image_candidate(image, cand)
    _, rect, _ = _rank_quad_candidates(image, cand, cfg.expand_ratio)[0]
    return _finalize_warp(rect, cand)


def rectify_candidates(image: np.ndarray, cand: PlateCandidate,
                       cfg: RectifyConfig) -> list[RectifiedPlate]:
    """Return at most four pixel-ranked warps, preserving ``rectify`` first.

    The first result uses the same regular contour pool and stable ranking as
    ``rectify``. Remaining hypotheses come from that bounded pool, with one
    optional green-mask boundary fit. Pixel evidence and the contour primary
    provide the ranking; labels, segmentation, and classifier scores do not.
    """
    image = _validate_image_candidate(image, cand)
    ranked = _rank_quad_candidates(image, cand, cfg.expand_ratio, include_green_fit=True)
    # Green fitting only augments the alternatives; the first regular result
    # remains exactly the first result from rectify() and shares all regular
    # contour extraction and warps.
    return [_finalize_warp(rect, cand) for _, rect, _ in ranked[:4]]


def deskew_vertical(rect: RectifiedPlate, max_shear: float = .3,
                    step: float = .01) -> RectifiedPlate:
    """Opt-in residual vertical-shear search; rectify() never invokes this.

    The returned homography maps original-image points into the sheared image.
    A central crop and the minority Otsu class make the projection insensitive
    to plate polarity and prevent the outer frame from setting the score.
    """
    if (rect is None or not isinstance(rect.image, np.ndarray) or rect.image.size == 0
            or rect.image.dtype != np.uint8 or rect.image.ndim not in (2, 3)
            or (rect.image.ndim == 3 and rect.image.shape[2] not in (1, 3))
            or not isinstance(rect.homography, np.ndarray) or rect.homography.shape != (3, 3)
            or not np.isfinite(rect.homography).all()):
        raise BadInputError('invalid rectified image')
    if (not np.isfinite([max_shear, step]).all() or max_shear < 0
            or not .005 <= step <= .1 or max_shear > .3):
        raise ValueError('max_shear must be in [0, .3] and step in [.005, .1]')
    image = rect.image
    h, w = image.shape[:2]
    if min(h, w) < 16:
        raise BadInputError('rectified image is too small')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 and image.shape[2] == 3 else (
        image[..., 0] if image.ndim == 3 else image)
    # Otsu handles both color polarities. Only central pixels participate in
    # polarity selection, so a dark frame cannot outvote the character strokes.
    y0, y1 = int(.12*h), int(.88*h)
    x0, x1 = int(.06*w), int(.94*w)
    center = gray[y0:y1, x0:x1]
    threshold, _ = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = center <= threshold
    ink_is_dark = np.count_nonzero(dark) <= dark.size / 2
    ink_y, ink_x = np.where(dark if ink_is_dark else ~dark)
    # A weak geometric prior resolves repeated-glyph projection ties while
    # keeping the mandated valley search as the deciding score.
    if len(ink_x) > 16:
        slope = float(np.clip(np.polyfit(ink_y.astype(float), ink_x.astype(float), 1)[0],
                              -max_shear, max_shear))
    else:
        slope = 0.
    if abs(slope) < .05:
        return RectifiedPlate(image, rect.layout, rect.color,
                              rect.homography, rect.candidate)
    angles = np.arange(-max_shear, max_shear + step*.25, step, dtype=np.float64)
    if not np.any(np.isclose(angles, 0)):
        angles = np.append(angles, 0.)
    angles = np.unique(np.clip(angles, -max_shear, max_shear))

    def valley_score(k: float) -> float:
        affine = np.float32([[1., -k, k*(h-1)/2], [0., 1., 0.]])
        shifted = cv2.warpAffine(gray, affine, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        inner = shifted[y0:y1, x0:x1]
        mask = inner <= threshold if ink_is_dark else inner > threshold
        projection = np.count_nonzero(mask, axis=0).astype(np.float32)
        # Smooth only pixel noise, then reward valleys relative to their local
        # shoulders. Edges are already excluded by the central crop.
        smooth = cv2.GaussianBlur(projection[None, :], (0, 0), max(1., .008*w))[0]
        radius = max(3, int(round(.035*w)))
        valleys = []
        for i in range(radius, len(smooth)-radius):
            if smooth[i] <= smooth[i-1] and smooth[i] < smooth[i+1]:
                valleys.append(float(smooth[i]))
        if not valleys:
            return 0.
        valleys.sort()
        # Average several deepest valleys, normalized by total inner ink.
        # This rewards actual inter-character gaps and rejects a lone edge gap.
        count = 6 if rect.layout == 7 else 7 if rect.layout == 8 else 5
        valley_depth = 1. - float(np.mean(valleys[:count])) / max(1., float(smooth.mean()))
        return valley_depth - 1.2*abs(k-slope)

    scores = [valley_score(float(k)) for k in angles]
    best_k = float(angles[int(np.argmax(scores))])
    transform = np.array([[1., -best_k, best_k*(h-1)/2],
                          [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
    if abs(best_k) < step*.51 or scores[int(np.argmax(scores))] - valley_score(0.) < .015:
        return RectifiedPlate(image, rect.layout, rect.color,
                              rect.homography, rect.candidate)
    warped = cv2.warpPerspective(image, transform, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return RectifiedPlate(warped, rect.layout, rect.color,
                          transform @ rect.homography, rect.candidate)
