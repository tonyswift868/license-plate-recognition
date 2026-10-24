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
    color = max(fractions, key=fractions.get)
    if fractions[color] >= .15:
        return color
    if np.mean((inner[..., 1] < 35) & (inner[..., 2] > 175)) > .6:
        return 'white'
    if np.mean(inner[..., 2] < 35) > .6:
        return 'black'
    return 'unknown'


def refine_quad(image: np.ndarray, cand: PlateCandidate, expand: float) -> np.ndarray:
    """Prefer a colour contour, then edges; reject unrelated crop contours."""
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
                        return quad
    return fallback


def rectify(image: np.ndarray, cand: PlateCandidate, cfg: RectifyConfig) -> RectifiedPlate:
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3))):
        raise BadInputError('expected nonempty uint8 gray/BGR image')
    if (cand is None or np.asarray(cand.corners).shape != (4, 2)
            or not np.isfinite(cand.corners).all() or min(cand.bbox[2:]) <= 0
            or abs(cv2.contourArea(np.float32(cand.corners))) < 1):
        raise BadInputError('invalid plate candidate')
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image.reshape(image.shape[:2]), cv2.COLOR_GRAY2BGR)
    quad = refine_quad(image, cand, cfg.expand_ratio)
    # Classify from a first warp so an unknown proposal can still become L7/L8.
    target = np.float32([[0, 0], [251, 0], [251, 79], [0, 79]])
    matrix = cv2.getPerspectiveTransform(quad, target)
    plate = cv2.warpPerspective(image, matrix, cfg.size_7)
    color = classify_color(plate)
    layout = 8 if color == 'green' else (7 if color in ('blue', 'yellow') else 0)
    if layout == 8:
        width, height = cfg.size_8
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        matrix = cv2.getPerspectiveTransform(quad, target)
        plate = cv2.warpPerspective(image, matrix, cfg.size_8)
    # Residual rotation search is optional (6.3). Omitted so M is exactly source -> image.
    return RectifiedPlate(plate, layout, color, matrix, cand)


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
