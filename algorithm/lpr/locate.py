"""Stage 2: HSV and Sobel proposals, geometric scoring and original-space NMS."""
from __future__ import annotations

import cv2
import numpy as np

from .config import LocateConfig
from .types import PlateCandidate, PreprocessOut


def order_points(points: np.ndarray) -> np.ndarray:
    """LT, RT, RB, LB for the supported near-horizontal plates (tilt <=30 deg)."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    left, right = np.split(points[np.argsort(points[:, 0])], 2)
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.float32([left[0], right[0], right[1], left[1]])


def color_masks(hsv: np.ndarray, cfg: LocateConfig) -> dict[str, np.ndarray]:
    return {color: cv2.inRange(hsv, np.array(bounds[0], np.uint8), np.array(bounds[1], np.uint8))
            for color, bounds in (("blue", cfg.blue_hsv), ("yellow", cfg.yellow_hsv), ("green", cfg.green_hsv))}


def color_vote(hsv: np.ndarray, cfg: LocateConfig) -> tuple[str, float]:
    fractions = {color: float(np.count_nonzero(mask) / mask.size) for color, mask in color_masks(hsv, cfg).items()}
    color = max(fractions, key=fractions.get)
    ratio = fractions[color]
    return (color if ratio >= .15 else "unknown"), ratio


def detection_masks(pre: PreprocessOut, cfg: LocateConfig) -> dict[str, np.ndarray]:
    # Suppress isolated hue noise before closing can join it to a plate.
    masks = {name: cv2.medianBlur(mask, 5) for name, mask in color_masks(pre.hsv, cfg).items()}
    sobel = cv2.convertScaleAbs(cv2.Sobel(pre.gray, cv2.CV_16S, 1, 0, ksize=3))
    masks['edge'] = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kw, kh = [max(1, round(v*pre.work_bgr.shape[1])) for v in cfg.close_kernel_rel]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    opening = np.ones((3, 3), np.uint8)
    return {name: cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, opening)
            for name, mask in masks.items()}


def box_iou(a: tuple, b: tuple) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, right-x)*max(0, bottom-y)
    union = a[2]*a[3]+b[2]*b[3]-inter
    return float(inter / union) if union > 0 else 0.0


def locate(pre: PreprocessOut, cfg: LocateConfig, max_candidates: int = 3) -> list[PlateCandidate]:
    if (pre is None or max_candidates <= 0 or not np.isfinite(pre.scale) or pre.scale <= 0
            or not isinstance(pre.work_bgr, np.ndarray) or pre.work_bgr.size == 0
            or pre.gray.shape != pre.work_bgr.shape[:2] or pre.hsv.shape != pre.work_bgr.shape):
        return []
    image_area = pre.gray.size
    candidates = []
    for branch, mask in detection_masks(pre, cfg).items():
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            r = cv2.minAreaRect(contour)
            rw, rh = r[1]
            if min(rw, rh) <= 1:
                continue
            aspect, area_ratio = max(rw, rh)/min(rw, rh), rw*rh/image_area
            quad = order_points(cv2.boxPoints(r))
            delta = quad[1]-quad[0]
            angle = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
            if (not cfg.aspect_min <= aspect <= cfg.aspect_max
                    or not cfg.area_ratio_min <= area_ratio <= cfg.area_ratio_max
                    or angle > cfg.max_tilt_deg):
                continue
            matrix = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [135, 0], [135, 35], [0, 35]]))
            gray = cv2.warpPerspective(pre.gray, matrix, (136, 36))
            hsv = cv2.warpPerspective(pre.hsv, matrix, (136, 36), flags=cv2.INTER_NEAREST)
            color, ratio = color_vote(hsv, cfg)
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            transitions = np.count_nonzero(binary[9:27, 1:] != binary[9:27, :-1], axis=1).mean()
            aspect_score = max(0., 1-min(abs(aspect-v)/v for v in (3.14, 3.43)))
            transition_score = min(1., transitions/max(1, cfg.min_row_transitions))
            score = .35*aspect_score + .35*ratio + .30*transition_score
            original = quad/pre.scale + np.array(pre.offset)
            # Clamp to the ROI's extent; image dimensions outside the ROI are not available here.
            lower = np.array(pre.offset)
            upper = lower + np.array([pre.gray.shape[1]-1, pre.gray.shape[0]-1])/pre.scale
            original = np.clip(original, lower, upper)
            xy0 = np.floor(original.min(axis=0)).astype(int)
            xy1 = np.ceil(original.max(axis=0)).astype(int)
            bbox = (int(xy0[0]), int(xy0[1]), int(xy1[0]-xy0[0]+1), int(xy1[1]-xy0[1]+1))
            candidates.append(PlateCandidate([tuple(map(float, p)) for p in original], bbox, color,
                                             float(np.clip(score, 0, 1)), 'edge' if branch == 'edge' else 'color'))
    kept = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = next((c for c in kept if box_iou(c.bbox, candidate.bbox) > cfg.nms_iou), None)
        if duplicate is not None:
            if duplicate.source != candidate.source:
                # Edge closures can cover only the serial-number group. Retain
                # the enclosing colour proposal's geometry when merging evidence.
                if (candidate.source == 'color' and candidate.color == duplicate.color
                        and candidate.color != 'unknown'
                        and candidate.bbox[2]*candidate.bbox[3] > duplicate.bbox[2]*duplicate.bbox[3]):
                    duplicate.corners, duplicate.bbox = candidate.corners, candidate.bbox
                duplicate.source = 'merged'
        else:
            kept.append(candidate)
    return kept[:max_candidates]
