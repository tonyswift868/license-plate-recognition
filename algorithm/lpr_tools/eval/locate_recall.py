"""Top-3 localization recall, measured independently of recognition status."""
from __future__ import annotations

from lpr.config import LprConfig
from lpr.locate import locate
from lpr.preprocess import preprocess

TOP_K = 3
IOU_THRESHOLD = 0.5


def box_iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0., right-left)*max(0., bottom-top)
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union > 0 else 0.


def locate_match(image, truth_bbox, cfg: LprConfig) -> dict:
    """Input and candidate boxes are original-image pixels; no scaling of GT."""
    pre = preprocess(image, cfg.preprocess)
    candidates = locate(pre, cfg.locate, max_candidates=TOP_K)[:TOP_K]
    best = max((box_iou(candidate.bbox, truth_bbox) for candidate in candidates), default=0.)
    return {"hit": best >= IOU_THRESHOLD, "best_iou": best, "candidates": len(candidates)}


def summarize(matches: list[tuple[str, dict]], *, excluded_np: int = 0) -> dict:
    def summary(values):
        total, hits = len(values), sum(item["hit"] for item in values)
        return {"hits": hits, "total": total, "recall": hits/total if total else None}
    groups = sorted({key for key, _ in matches})
    return {"topk": TOP_K, "iou_threshold": IOU_THRESHOLD,
            "locator": "lpr.preprocess + lpr.locate (independent of classifier)",
            "overall": summary([item for _, item in matches]),
            "subsets": {key: summary([item for name, item in matches if name == key]) for key in groups},
            "excluded_np": excluded_np}
