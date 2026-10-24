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
                 'yellow': ((15, 70, 70), (35, 255, 255)),
                 'green': ((35, 30, 60), (90, 255, 255))}


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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
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
            if box_iou((cx+x0, cy+y0, cw, ch), cand.bbox) < .45:
                continue
            for epsilon in (.02, .035, .05):
                poly = cv2.approxPolyDP(contour, epsilon*cv2.arcLength(contour, True), True)
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
