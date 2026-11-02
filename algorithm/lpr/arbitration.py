"""Pure three-route plate arbitration (011 + T20_A).

This module deliberately has no image/model imports. Attempts are trace-shaped
mappings and configuration is read from the postprocess section supplied by the
caller, so frozen traces can be replayed with the same production policy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Decision:
    status: str
    plate: str | None
    confidence: float
    route: str | None
    candidate_index: int | None
    rect_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_PAIRS = (("segment", "reader"), ("segment", "box"), ("reader", "box"))
_ROUTE_KEYS = {"segment": "three_route_segment_solo", "reader": "three_route_reader_solo"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _confidence(attempt: Any) -> float | None:
    if not isinstance(attempt, Mapping):
        return None
    try:
        value = float(attempt["confidence"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def _valid(attempt: Any) -> bool:
    if not isinstance(attempt, Mapping):
        return False
    status = attempt.get("status", "OK")
    if not attempt.get("valid", status == "OK") or status in {"INVALID_FORMAT", "BAD_INPUT"}:
        return False
    return attempt.get("plate") is not None and _confidence(attempt) is not None


def _attempt(status: str, route: str | None = None, plate: str | None = None,
             confidence: float = 0.0, ci: int | None = None, ri: int | None = None) -> Decision:
    return Decision(status, plate, confidence, route, ci, ri)


def _ok(attempt: Mapping[str, Any], route: str, ci: int, ri: int | None,
        confidence: float | None = None) -> Decision:
    return _attempt("OK", route, attempt["plate"], _confidence(attempt) if confidence is None else confidence,
                    ci, ri)  # type: ignore[arg-type]


def _ordered_rects(rects: list[Mapping[str, Any]], ci: int | None = None) -> list[Mapping[str, Any]]:
    selected = [r for r in rects if ci is None or int(r["candidate_index"]) == ci]
    return sorted(selected, key=lambda r: (int(r["candidate_index"]), int(r["rect_index"])))


def _province_support(plate: str, ci: int, rect_index: int | None,
                      rects: list[Mapping[str, Any]], box: Mapping[str, Any] | None,
                      min_routes: int) -> bool:
    if min_routes <= 0:
        return True
    relevant = _ordered_rects(rects, ci)
    if rect_index is not None:
        relevant = [r for r in relevant if int(r["rect_index"]) == rect_index]
    for rect in relevant:
        routes = (rect.get("segment"), rect.get("reader"), box)
        support = {name for name, result in zip(("segment", "reader", "box"), routes, strict=False)
                   if _valid(result) and result["plate"].startswith(plate[0])}
        if len(support) >= min_routes:
            return True
    return False


def try_auto_candidate(candidate_index: int, rects: list[Mapping[str, Any]], box: Mapping[str, Any] | None,
                       postprocess_cfg: Any, *, candidate_complete: bool = False) -> Decision | None:
    """Try this candidate's auto routes using its currently available rects.

    Repeated calls are stateless and safe: pass the accumulated rect list in
    original rect order. The B-solo check is deferred until candidate_complete.
    """
    tau = float(_get(postprocess_cfg, "three_route_agree_min_confidence"))
    tau_box = float(_get(postprocess_cfg, "three_route_box_solo_min_confidence"))
    legacy_agree = bool(_get(postprocess_cfg, "three_route_legacy_sr_agreement"))
    solo_enabled = {route: bool(_get(postprocess_cfg, key)) for route, key in _ROUTE_KEYS.items()}
    solo_thresholds = {
        "segment": float(_get(postprocess_cfg, "solo_min_confidence_segment")),
        "reader": float(_get(postprocess_cfg, "solo_min_confidence_reader")),
    }
    legacy_tau = float(_get(postprocess_cfg, "agree_min_confidence"))
    province_min = int(_get(postprocess_cfg, "three_route_province_support_min_routes", 0))

    def accepted(result: Decision) -> Decision:
        if _province_support(result.plate or "", candidate_index, result.rect_index,
                             rects, box, province_min):
            return result
        return _attempt("NO_AUTO", ci=candidate_index, ri=result.rect_index)

    for rect in _ordered_rects(rects, candidate_index):
        ri = int(rect["rect_index"])
        routes = {"segment": rect.get("segment"), "reader": rect.get("reader"), "box": box}
        for left, right in _PAIRS:
            a, b = routes[left], routes[right]
            if (_valid(a) and _valid(b) and a["plate"] == b["plate"] and
                    min(_confidence(a), _confidence(b)) >= tau):
                return accepted(_ok(a, left, candidate_index, ri))
        s, r = routes["segment"], routes["reader"]
        if legacy_agree and _valid(s) and _valid(r) and s["plate"] == r["plate"] and min(
                _confidence(s), _confidence(r)) >= legacy_tau:
            return accepted(_ok(s, "segment", candidate_index, ri, max(_confidence(s), _confidence(r))))
        for route in ("segment", "reader"):
            result = routes[route]
            if (solo_enabled[route] and _valid(result) and
                    _confidence(result) >= solo_thresholds[route]):
                return accepted(_ok(result, route, candidate_index, ri))
    if candidate_complete and _valid(box) and _confidence(box) >= tau_box:
        if _province_support(box["plate"], candidate_index, None, rects, box, province_min):
            return _ok(box, "box", candidate_index, None)
        return _attempt("NO_AUTO", ci=candidate_index)
    return None


def _candidate_count(trace: Mapping[str, Any]) -> int:
    try:
        return max(0, int(trace.get("candidate_count", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _box_for(trace: Mapping[str, Any], ci: int) -> Mapping[str, Any] | None:
    boxes = trace.get("boxes", {})
    if isinstance(boxes, Mapping):
        value = boxes.get(str(ci), boxes.get(ci))
        return value if isinstance(value, Mapping) else None
    return None


def _best_single(rows: list[tuple[Mapping[str, Any], str, int, int | None, float]]) -> tuple | None:
    best = None
    for row in rows:
        if best is None or row[4] > best[4]:
            best = row
    return best


def _fallbacks(rects: list[Mapping[str, Any]], boxes: Mapping[str, Any],
               cfg: Any, candidate_count: int) -> tuple[Decision | None, Decision | None, Decision | None]:
    first_rect = next((r for r in _ordered_rects(rects) if int(r["candidate_index"]) == 0), None)
    special_color = first_rect is not None and first_rect.get("color") in {"white", "black"}
    # A: best same-rect S/R pair by min confidence, then best legal R.
    best_a = None
    best_r = None
    # M: first same-rect two-route agreement; otherwise best B, then best R.
    first_m = None
    valid_boxes = []
    for ci in range(candidate_count):
        b = boxes.get(str(ci), boxes.get(ci)) if isinstance(boxes, Mapping) else None
        if _valid(b):
            valid_boxes.append((b, "box", ci, None, _confidence(b)))
    for rect in _ordered_rects(rects):
        ci, ri = int(rect["candidate_index"]), int(rect["rect_index"])
        s, r = rect.get("segment"), rect.get("reader")
        if _valid(r):
            row = (r, "reader", ci, ri, _confidence(r))
            if best_r is None or row[4] > best_r[4]:
                best_r = row
        if _valid(s) and _valid(r) and s["plate"] == r["plate"]:
            score = min(_confidence(s), _confidence(r))
            if best_a is None or score > best_a[0]:
                best_a = (score, r, ci, ri)
        if first_m is None:
            current = {"segment": s, "reader": r,
                       "box": boxes.get(str(ci), boxes.get(ci)) if isinstance(boxes, Mapping) else None}
            for left, right in _PAIRS:
                a, b = current[left], current[right]
                if _valid(a) and _valid(b) and a["plate"] == b["plate"]:
                    selected, route = (a, left) if left != "box" else (b, right)
                    first_m = (selected, route, ci, ri, min(_confidence(a), _confidence(b)))
                    break
    a_decision = None
    if best_a is not None:
        score, result, ci, ri = best_a
        a_decision = _attempt("LOW_CONFIDENCE", "reader", result["plate"], score, ci, ri)
    elif best_r is not None:
        result, route, ci, ri, score = best_r
        a_decision = _attempt("LOW_CONFIDENCE", route, result["plate"], score, ci, ri)
    # Bmax ties retain the earliest candidate.
    best_b = _best_single(valid_boxes)
    if first_m is not None:
        result, route, ci, ri, score = first_m
        m_decision = _attempt("LOW_CONFIDENCE", route, result["plate"], score, ci, ri)
    elif best_b is not None:
        result, route, ci, ri, score = best_b
        m_decision = _attempt("LOW_CONFIDENCE", route, result["plate"], score, ci, ri)
    elif best_r is not None:
        result, route, ci, ri, score = best_r
        m_decision = _attempt("LOW_CONFIDENCE", route, result["plate"], score, ci, ri)
    else:
        m_decision = None

    # L replays old S/R auto order, then supplies the best legacy manual S/R.
    # L is an independent replay of the old S/R contract. New three-route
    # feature switches must not alter its historical policy.
    legacy_agree = True
    legacy_tau = float(_get(cfg, "agree_min_confidence"))
    solo_s = solo_r = True
    s_tau = float(_get(cfg, "solo_min_confidence_segment"))
    r_tau = float(_get(cfg, "solo_min_confidence_reader"))
    l_auto = None
    legacy_manual = None
    for rect in _ordered_rects(rects):
        ci, ri = int(rect["candidate_index"]), int(rect["rect_index"])
        s, r = rect.get("segment"), rect.get("reader")
        if legacy_agree and _valid(s) and _valid(r) and s["plate"] == r["plate"] and min(
                _confidence(s), _confidence(r)) >= legacy_tau:
            l_auto = _attempt("LOW_CONFIDENCE", "segment", s["plate"],
                              max(_confidence(s), _confidence(r)), ci, ri)
        elif solo_s and _valid(s) and _confidence(s) >= s_tau:
            l_auto = _attempt("LOW_CONFIDENCE", "segment", s["plate"], _confidence(s), ci, ri)
        elif solo_r and _valid(r) and _confidence(r) >= r_tau:
            l_auto = _attempt("LOW_CONFIDENCE", "reader", r["plate"], _confidence(r), ci, ri)
        if l_auto is not None:
            break
        choice = s if _valid(s) else r if _valid(r) else None
        route = "segment" if choice is s else "reader"
        if choice is not None:
            row = (choice, route, ci, ri, _confidence(choice))
            if legacy_manual is None or row[4] > legacy_manual[4]:
                legacy_manual = row
    if l_auto is not None:
        l_decision = l_auto
    elif special_color:
        # The old L policy suppresses only its own hand-read fallback here.
        l_decision = None
    elif a_decision is not None:
        l_decision = a_decision
    elif legacy_manual is not None:
        result, route, ci, ri, score = legacy_manual
        l_decision = _attempt("LOW_CONFIDENCE", route, result["plate"], score, ci, ri)
    else:
        l_decision = None
    return a_decision, m_decision, l_decision


def _candidate_zero_corners(trace: Mapping[str, Any], supplied: Any = None) -> Any:
    if supplied is not None:
        return supplied
    candidates = trace.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, Mapping):
            return candidate.get("corners", candidate.get("predicted_corners"))
    values = trace.get("candidate_corners")
    if isinstance(values, Mapping):
        return values.get("0", values.get(0))
    if isinstance(values, list) and values:
        return values[0]
    values = trace.get("corners_by_candidate")
    if isinstance(values, Mapping):
        return values.get("0", values.get(0))
    if isinstance(values, list) and values:
        return values[0]
    return None


def _tilt_degrees(corners: Any) -> float | None:
    try:
        if len(corners) != 4 or any(len(point) != 2 for point in corners):
            return None
        pts = [[float(v) for v in point] for point in corners]
        if not all(math.isfinite(v) for point in pts for v in point):
            return None
        lt, rt, rb, lb = pts
        top = math.degrees(math.atan2(rt[1] - lt[1], rt[0] - lt[0]))
        bottom = math.degrees(math.atan2(rb[1] - lb[1], rb[0] - lb[0]))
        angle = (abs(top) + abs(bottom)) / 2
        return angle if math.isfinite(angle) else None
    except (TypeError, ValueError, OverflowError):
        return None


def arbitrate(trace: Mapping[str, Any], postprocess_cfg: Any, *,
              predicted_corners: Any = None, runtime: bool = True) -> Decision:
    """Replay a trace and apply final manual fallback and runtime status projection."""
    if any(trace.get(key) for key in ("input_error", "decode_error", "box_decode_error", "live_error")):
        return _attempt("BAD_INPUT")
    rects = list(trace.get("rects", []))
    boxes = trace.get("boxes", {})
    count = _candidate_count(trace)
    if count == 0:
        # Allow a box-only trace that omits candidate_count.
        count = max((int(k) for k in boxes), default=-1) + 1 if isinstance(boxes, Mapping) else 0
    for ci in range(count):
        selected = try_auto_candidate(ci, rects, _box_for(trace, ci), postprocess_cfg,
                                      candidate_complete=True)
        if selected is not None:
            if selected.status != "NO_AUTO":
                return selected
            break
    a_decision, m_decision, l_decision = _fallbacks(rects, boxes, postprocess_cfg, count)
    angle = _tilt_degrees(_candidate_zero_corners(trace, predicted_corners))
    min_tilt = float(_get(postprocess_cfg, "fallback_predicted_tilt_min_deg"))
    tilt_policy = _get(postprocess_cfg, "nonok_fallback", "predicted_tilt_then_corrected") == "predicted_tilt_then_corrected"
    # T20_A is exactly A on a qualifying predicted tilt, M otherwise (including missing/bad corners).
    first_choice = (a_decision if angle is not None and angle >= min_tilt else m_decision) if tilt_policy else a_decision
    chosen = next((d for d in (first_choice, a_decision, m_decision, l_decision) if d is not None), None)
    if chosen is None:
        chosen = _attempt("NO_READ")
    if not runtime or chosen.status in {"OK", "BAD_INPUT"}:
        return chosen
    if count == 0:
        return _attempt("NO_PLATE")
    first_rect = next((r for r in _ordered_rects(rects) if int(r["candidate_index"]) == 0), None)
    if first_rect is not None and first_rect.get("color") in {"white", "black"}:
        return _attempt("UNSUPPORTED")
    if chosen.status != "NO_READ":
        return chosen
    invalid = None
    for rect in _ordered_rects(rects):
        for route in ("segment", "reader"):
            result = rect.get(route)
            score = _confidence(result)
            if isinstance(result, Mapping) and result.get("status") == "INVALID_FORMAT" and score is not None:
                row = _attempt("INVALID_FORMAT", route, result.get("plate"), score,
                               int(rect["candidate_index"]), int(rect["rect_index"]))
                if invalid is None or score > invalid.confidence:
                    invalid = row
    return invalid if invalid is not None else _attempt("SEG_FAILED")


__all__ = ["Decision", "arbitrate", "try_auto_candidate"]
