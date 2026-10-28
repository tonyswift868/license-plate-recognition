"""Exact fit-only search for the four production 011 automatic thresholds.

This module operates on three-route JSON trace records and does not import or
instantiate models.  The event builder follows ``arbitration.try_auto_candidate``:
candidate order; rect order; SR, SB, RB pairs; segment solo; reader solo; then
that candidate's box solo.  T20_A controls fallback after all auto events fail
and therefore does not participate in automatic threshold selection.
"""
from __future__ import annotations

from collections import OrderedDict
import bisect
import itertools
import math
import time
from typing import Callable, Iterable, Mapping

import numpy as np


AXES = ("agree", "box", "segment", "reader")
CONFIG_KEYS = {
    "agree": "three_route_agree_min_confidence",
    "box": "three_route_box_solo_min_confidence",
    "segment": "solo_min_confidence_segment",
    "reader": "solo_min_confidence_reader",
}
PAIRS = (("segment", "reader"), ("segment", "box"), ("reader", "box"))
PRECISION_NUMERATOR = 199
PRECISION_DENOMINATOR = 200
LEGACY_SR_AGREEMENT = False  # policy 011: only the legacy SR-agreement branch is disabled
LEGACY_SOLO_SEGMENT = True
LEGACY_SOLO_READER = True
RULE_FAMILY = "011 + T20_A"
SEED_VALUES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995,
               0.999, math.nextafter(1.0, math.inf))
SEED_FRACTIONS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def _attempt_confidence(attempt: object) -> float | None:
    if not isinstance(attempt, Mapping):
        return None
    status = attempt.get("status", "OK")
    if not attempt.get("valid", status == "OK") or status in {"INVALID_FORMAT", "BAD_INPUT"}:
        return None
    if attempt.get("plate") is None:
        return None
    try:
        value = float(attempt["confidence"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _box_for(boxes: Mapping, candidate_index: int):
    return boxes.get(str(candidate_index), boxes.get(candidate_index))


def _candidate_count(record: Mapping) -> int:
    try:
        return max(0, int(record.get("candidate_count", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _ordered_rects(record: Mapping, candidate_index: int) -> list[Mapping]:
    return sorted((rect for rect in record.get("rects", [])
                   if int(rect["candidate_index"]) == candidate_index),
                  key=lambda rect: int(rect["rect_index"]))


def _events(record: Mapping) -> list[dict]:
    """Return only data-derived automatic events in exact production order."""
    if any(record.get(key) for key in ("input_error", "decode_error", "box_decode_error", "live_error")):
        return []
    boxes = record.get("boxes", {})
    if not isinstance(boxes, Mapping):
        boxes = {}
    result = []
    for candidate_index in range(_candidate_count(record)):
        box = _box_for(boxes, candidate_index)
        box_confidence = _attempt_confidence(box)
        for rect in _ordered_rects(record, candidate_index):
            routes = {"segment": rect.get("segment"), "reader": rect.get("reader"), "box": box}
            confidences = {name: _attempt_confidence(routes[name])
                           for name in ("segment", "reader", "box")}
            # These are the three standard same-rect route pairs, including SR
            # even though policy 011 disables only the extra legacy SR gate.
            for left, right in PAIRS:
                a, b = routes[left], routes[right]
                ac, bc = confidences[left], confidences[right]
                if (ac is not None and bc is not None and a["plate"] == b["plate"]):
                    result.append({
                        "axis": "agree", "confidence": min(ac, bc),
                        "plate": a["plate"], "output_confidence": ac,
                        "route": left, "candidate_index": candidate_index,
                        "rect_index": int(rect["rect_index"]),
                        "event": f"agree_{left}_{right}",
                    })
            for route in ("segment", "reader"):
                confidence = confidences[route]
                if confidence is not None:
                    result.append({
                        "axis": route, "confidence": confidence,
                        "plate": routes[route]["plate"], "output_confidence": confidence,
                        "route": route, "candidate_index": candidate_index,
                        "rect_index": int(rect["rect_index"]),
                        "event": f"{route}_solo",
                    })
        # Production defers B solo until every rect for this candidate has
        # been checked, then advances to the next candidate.
        if box_confidence is not None:
            result.append({
                "axis": "box", "confidence": box_confidence,
                "plate": box["plate"], "output_confidence": box_confidence,
                "route": "box", "candidate_index": candidate_index,
                "rect_index": None, "event": "box_solo",
            })
    return result


def _threshold_tuple(thresholds: Mapping) -> tuple[float, float, float, float]:
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a mapping")
    values = []
    for axis in AXES:
        key = CONFIG_KEYS[axis]
        if key not in thresholds:
            raise ValueError(f"missing threshold {key}")
        try:
            value = float(thresholds[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid threshold {key}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid threshold {key}")
        values.append(value)
    return tuple(values)


def _replay_events(events: Iterable[Mapping], values: tuple[float, ...]) -> dict | None:
    limits = dict(zip(AXES, values))
    for event in events:
        if event["confidence"] >= limits[event["axis"]]:
            return {"status": "OK", "plate": event["plate"],
                    "confidence": event["output_confidence"], "route": event["route"],
                    "candidate_index": event["candidate_index"],
                    "rect_index": event["rect_index"], "event": event["event"]}
    return None


def replay_auto(record: Mapping, thresholds: Mapping) -> dict | None:
    """Replay the first accepted 011 event; truth is never consulted."""
    return _replay_events(_events(record), _threshold_tuple(thresholds))


def _score_event_rows(rows: list[Mapping], event_rows: list[list[dict]],
                      values: tuple[float, ...]) -> dict:
    counts = {name: {"n": 0, "accepted": 0, "correct": 0}
              for name in ("val", "devhard")}
    for row, events in zip(rows, event_rows):
        item = counts[row["split"]]
        item["n"] += 1
        decision = _replay_events(events, values)
        if decision is not None:
            item["accepted"] += 1
            item["correct"] += decision["plate"] == row["truth"]
    for item in counts.values():
        item["precision"] = item["correct"] / item["accepted"] if item["accepted"] else None
        item["incorrect"] = item["accepted"] - item["correct"]
    return counts


def _validate_records(records: Iterable[Mapping]) -> list[Mapping]:
    rows = list(records)
    if not rows:
        raise ValueError("threshold search requires nonempty records")
    seen = set()
    for row in rows:
        if row.get("split") not in {"val", "devhard"}:
            raise ValueError("records must use the fixed val/devhard splits")
        if not isinstance(row.get("truth"), str) or not row["truth"]:
            raise ValueError("record truth must be a nonempty plate string")
        path = str(row.get("path", "")).replace("\\", "/").casefold()
        if not path or path in seen:
            raise ValueError("records require nonempty unique paths")
        seen.add(path)
    if not any(row["split"] == "val" for row in rows) or not any(row["split"] == "devhard" for row in rows):
        raise ValueError("both val and devhard records are required")
    return rows


def score_records(records: Iterable[Mapping], thresholds: Mapping) -> dict:
    """Report exact auto-accept counts and precision for val/devhard."""
    rows = _validate_records(records)
    _threshold_tuple(thresholds)
    return _score_event_rows(rows, [_events(row) for row in rows], _threshold_tuple(thresholds))


def _qualifies(counts: Mapping) -> bool:
    return all(counts[name]["accepted"] > 0 and
               counts[name]["correct"] * PRECISION_DENOMINATOR >=
               counts[name]["accepted"] * PRECISION_NUMERATOR
               for name in ("val", "devhard"))


def _boundaries(rows: list[Mapping], event_rows: list[list[dict]] | None = None) -> dict[str, list[float]]:
    """Build the fixed fit grid under current production config constraints.

    Agree and box thresholds are validated in [0, 1], so their grids contain
    only observed fit confidences plus both legal endpoints. Solo thresholds
    allow values above one to disable a route, so their grids also include the
    frozen no-pass sentinel nextafter(1, +inf).
    """
    axes = {"agree": {0.0, 1.0}, "box": {0.0, 1.0},
            "segment": {0.0, math.nextafter(1.0, math.inf)},
            "reader": {0.0, math.nextafter(1.0, math.inf)}}
    event_rows = event_rows if event_rows is not None else [_events(row) for row in rows]
    for events in event_rows:
        for event in events:
            axes[event["axis"]].add(float(event["confidence"]))
    return {axis: sorted(axes[axis]) for axis in AXES}


def _seed_indices(values: list[float]) -> list[int]:
    indices = {0, len(values) - 1}
    indices.update(round((len(values) - 1) * fraction) for fraction in SEED_FRACTIONS)
    indices.update(min(len(values) - 1, bisect.bisect_left(values, target))
                   for target in SEED_VALUES)
    return sorted(indices)


class _FourAxisBits:
    """Compact ordered-event masks with bounded threshold-mask caches."""

    def __init__(self, rows: list[Mapping], event_rows: list[list[dict]] | None = None,
                 *, cache_limit: int = 32):
        self.rows = rows
        self.n = len(rows)
        self.all = (1 << self.n) - 1
        self.groups = {
            name: sum(1 << index for index, row in enumerate(rows) if row["split"] == name)
            for name in ("val", "devhard")
        }
        self.events = event_rows if event_rows is not None else [_events(row) for row in rows]
        self.slots = max((len(events) for events in self.events), default=0)
        self.confidence = {axis: np.full((self.slots, self.n), -np.inf, dtype=np.float64)
                           for axis in AXES}
        self.correct = [0] * self.slots
        for row_index, (row, events) in enumerate(zip(rows, self.events)):
            for slot, event in enumerate(events):
                self.confidence[event["axis"]][slot, row_index] = event["confidence"]
                if event["plate"] == row["truth"]:
                    self.correct[slot] |= 1 << row_index
        self.cache_limit = cache_limit
        self.caches = {axis: OrderedDict() for axis in AXES}

    def masks(self, axis: str, threshold: float) -> tuple[int, ...]:
        cache = self.caches[axis]
        threshold = float(threshold)
        if threshold in cache:
            cache.move_to_end(threshold)
            return cache[threshold]
        packed = np.packbits(self.confidence[axis] >= threshold, axis=1, bitorder="little")
        masks = tuple(int.from_bytes(row.tobytes(), "little") for row in packed)
        cache[threshold] = masks
        if len(cache) > self.cache_limit:
            cache.popitem(last=False)
        return masks

    def counts(self, thresholds: tuple[float, ...]) -> dict:
        masks = [self.masks(axis, threshold) for axis, threshold in zip(AXES, thresholds)]
        seen = correct = 0
        for slot in range(self.slots):
            accepted_here = masks[0][slot] | masks[1][slot] | masks[2][slot] | masks[3][slot]
            first = accepted_here & ~seen & self.all
            correct |= first & self.correct[slot]
            seen |= accepted_here
        return {name: {"accepted": (seen & group).bit_count(),
                       "correct": (correct & group).bit_count(),
                       "n": group.bit_count()}
                for name, group in self.groups.items()}

    def bounds(self, low: tuple[float, ...], high: tuple[float, ...]) -> dict:
        possible_masks = [self.masks(axis, threshold) for axis, threshold in zip(AXES, low)]
        guaranteed_masks = [self.masks(axis, threshold) for axis, threshold in zip(AXES, high)]
        possible_accept = guaranteed_accept = possible_correct_winner = 0
        for slot in range(self.slots):
            possible_here = 0
            guaranteed_here = 0
            possible_correct_here = 0
            for axis_index in range(len(AXES)):
                possible_here |= possible_masks[axis_index][slot]
                guaranteed_here |= guaranteed_masks[axis_index][slot]
                possible_correct_here |= (possible_masks[axis_index][slot] & self.correct[slot])
            # A later correct event cannot win after the earliest event that
            # passes even at the strictest corner. Earlier possible correct
            # events remain an optimistic (safe) upper bound for this row.
            possible_correct_winner |= possible_correct_here & ~guaranteed_accept & self.all
            possible_accept |= possible_here
            guaranteed_accept |= guaranteed_here
        guaranteed_wrong = guaranteed_accept & ~possible_correct_winner & self.all
        return {
            name: {"possible_accept": (possible_accept & group).bit_count(),
                   "guaranteed_accept": (guaranteed_accept & group).bit_count(),
                   "guaranteed_wrong": (guaranteed_wrong & group).bit_count(),
                   "possible_correct_winner": (possible_correct_winner & group).bit_count(),
                   "possible_correct": (possible_correct_winner & group).bit_count(),
                   "n": group.bit_count()}
            for name, group in self.groups.items()
        }


def search_fit(fit_records: Iterable[Mapping], *, progress: Callable[[dict], None] | None = None) -> dict:
    """Globally maximize fit auto coverage over the exact four-axis grid.

    The original two-axis search's first-seen tie semantics are retained: seed
    points use sorted-index ``itertools.product`` order, then a depth-first
    branch search bisects the widest axis (axis-order tie break), explores the
    low half first, and updates the winner only for strictly higher coverage.
    """
    started = time.perf_counter()
    rows = _validate_records(fit_records)
    event_rows = [_events(row) for row in rows]
    boundaries = _boundaries(rows, event_rows)
    values = [boundaries[axis] for axis in AXES]
    widths = tuple(len(axis) for axis in values)
    bits = _FourAxisBits(rows, event_rows)
    seed_axes = [_seed_indices(axis) for axis in values]
    seed_points = math.prod(len(indices) for indices in seed_axes)
    best = None
    for indices in itertools.product(*seed_axes):
        thresholds = tuple(values[axis][index] for axis, index in enumerate(indices))
        counts = bits.counts(thresholds)
        if _qualifies(counts):
            accepted = counts["val"]["accepted"] + counts["devhard"]["accepted"]
            if best is None or accepted > best["accepted"]:
                best = {"accepted": accepted, "thresholds": dict(zip(AXES, thresholds)),
                        "axis_indices": dict(zip(AXES, indices)), "counts": counts,
                        "first_seen_at": "seed"}

    if progress is not None:
        progress({"phase": "seed_complete", "seed_points": seed_points,
                  "best_accepted": best["accepted"] if best else None,
                  "elapsed_seconds": time.perf_counter() - started})

    initial = tuple(value for axis in values for value in (0, len(axis) - 1))
    stack = [initial]
    nodes = pruned = 0
    while stack:
        box = stack.pop()
        nodes += 1
        lower = tuple(box[2 * index] for index in range(len(AXES)))
        upper = tuple(box[2 * index + 1] for index in range(len(AXES)))
        low_thresholds = tuple(values[index][lower[index]] for index in range(len(AXES)))
        high_thresholds = tuple(values[index][upper[index]] for index in range(len(AXES)))
        bound = bits.bounds(low_thresholds, high_thresholds)

        feasible_possible = all(
            bound[name]["possible_accept"] > 0 and
            bound[name]["guaranteed_wrong"] * PRECISION_DENOMINATOR <= bound[name]["possible_accept"]
            for name in ("val", "devhard"))
        upper_total = bound["val"]["possible_accept"] + bound["devhard"]["possible_accept"]
        if (not feasible_possible) or (best is not None and upper_total <= best["accepted"]):
            pruned += 1
            continue

        spans = [upper[index] - lower[index] for index in range(len(AXES))]
        if not any(spans):
            high_counts = bits.counts(high_thresholds)
            if _qualifies(high_counts):
                accepted = high_counts["val"]["accepted"] + high_counts["devhard"]["accepted"]
                if best is None or accepted > best["accepted"]:
                    best = {"accepted": accepted,
                            "thresholds": dict(zip(AXES, high_thresholds)),
                            "axis_indices": dict(zip(AXES, upper)),
                            "counts": high_counts,
                            "first_seen_at": "branch_and_bound"}
            continue

        axis = max(range(len(AXES)), key=lambda index: spans[index])
        midpoint = (lower[axis] + upper[axis]) // 2
        low_box, high_box = list(box), list(box)
        low_box[2 * axis + 1] = midpoint
        high_box[2 * axis] = midpoint + 1
        # LIFO: push high first so the low half is visited first.
        stack.append(tuple(high_box))
        stack.append(tuple(low_box))
        if progress is not None and nodes % 10000 == 0:
            progress({"nodes": nodes, "pruned": pruned, "frontier": len(stack),
                      "best_accepted": best["accepted"] if best else None,
                      "elapsed_seconds": time.perf_counter() - started})

    complete = not stack
    return {
        "schema": "four-threshold-search-v1", "rule_family": RULE_FAMILY,
        "policy": {"legacy_sr_agreement": LEGACY_SR_AGREEMENT,
                   "segment_solo": LEGACY_SOLO_SEGMENT, "reader_solo": LEGACY_SOLO_READER},
        "axis_order": list(AXES), "threshold_config_keys": dict(CONFIG_KEYS),
        "selection_scope": "fit-only", "fit_rows": len(rows),
        "split_counts": {name: sum(row["split"] == name for row in rows)
                         for name in ("val", "devhard")},
        "boundary_counts": {axis: len(values[index]) for index, axis in enumerate(AXES)},
        "seed_axis_counts": {axis: len(seed_axes[index]) for index, axis in enumerate(AXES)},
        "seed_points": seed_points, "nodes": nodes, "pruned": pruned,
        "frontier_remaining": len(stack), "complete": complete,
        "optimality_gap_accepted": 0 if complete else None,
        "best": best, "first_fit_winner": best,
        "elapsed_seconds": time.perf_counter() - started,
        "mask_cache_limit_per_axis": bits.cache_limit,
    }


def _public_thresholds(indices: Mapping[str, int], boundaries: Mapping[str, list[float]]) -> dict:
    return {CONFIG_KEYS[axis]: boundaries[axis][indices[axis]] for axis in AXES}


def tighten_full(full_records: Iterable[Mapping], fit_records: Iterable[Mapping],
                 fit_report: Mapping, *, progress: Callable[[dict], None] | None = None) -> dict:
    """Apply the prescribed fit-grid one-bin tightening loop on full rows.

    Each round advances at most one observed fit boundary per axis in the
    fixed agree/box/segment/reader order. It checks and records the complete
    val/devhard counts after every single move, stops at the first feasible
    point, and never chooses an axis based on which full-set errors it fixes.
    """
    full = _validate_records(full_records)
    fit = _validate_records(fit_records)
    best = fit_report.get("best") if isinstance(fit_report, Mapping) else None
    if best is None or not fit_report.get("complete"):
        return {"success": False, "reason": "fit search has no complete feasible winner",
                "steps": [], "initial": None, "final": None}
    boundaries = _boundaries(fit)
    full_events = [_events(row) for row in full]
    indices = {axis: int(best["axis_indices"][axis]) for axis in AXES}
    current = {axis: boundaries[axis][indices[axis]] for axis in AXES}
    initial = dict(current)
    steps = []

    def snapshot():
        counts = _score_event_rows(full, full_events,
                                   tuple(current[axis] for axis in AXES))
        return {"thresholds": _public_thresholds(indices, boundaries),
                "axis_indices": dict(indices), "counts": counts,
                "qualifies": _qualifies(counts)}

    state = snapshot()
    initial_state = state
    if any(state["counts"][name]["accepted"] == 0 for name in ("val", "devhard")):
        return {"success": False, "reason": "initial full set has zero automatic passes",
                "initial": state, "steps": steps, "final": state,
                "auto_acceptance_loss": 0}
    if state["qualifies"]:
        return {"success": True, "reason": "fit winner meets full precision", "initial": state,
                "steps": steps, "final": state, "step_count": 0,
                "axis_boundary_moves": {axis: 0 for axis in AXES},
                "auto_acceptance_loss": 0}

    starting_accepted = sum(state["counts"][name]["accepted"] for name in ("val", "devhard"))
    while True:
        moved = False
        for axis in AXES:
            if indices[axis] >= len(boundaries[axis]) - 1:
                continue
            previous = current[axis]
            indices[axis] += 1
            current[axis] = boundaries[axis][indices[axis]]
            moved = True
            state = snapshot()
            steps.append({"step": len(steps) + 1, "axis": axis,
                          "from": previous, "to": current[axis], **state})
            if progress is not None:
                progress({"step": len(steps), "axis": axis,
                          "qualifies": state["qualifies"], "counts": state["counts"]})
            if any(state["counts"][name]["accepted"] == 0 for name in ("val", "devhard")):
                final_accepted = sum(state["counts"][name]["accepted"] for name in ("val", "devhard"))
                return {"success": False, "reason": f"{axis} next boundary leaves a split with zero passes",
                        "initial": initial_state,
                        "steps": steps, "final": state,
                        "step_count": len(steps),
                        "axis_boundary_moves": _moves(best["axis_indices"], indices),
                        "auto_acceptance_loss": starting_accepted - final_accepted}
            if state["qualifies"]:
                final_accepted = sum(state["counts"][name]["accepted"] for name in ("val", "devhard"))
                return {"success": True, "reason": "full precision reached by next-bin tightening",
                        "initial": initial_state,
                        "steps": steps, "final": state,
                        "step_count": len(steps),
                        "axis_boundary_moves": _moves(best["axis_indices"], indices),
                        "auto_acceptance_loss": starting_accepted - final_accepted}
        if not moved:
            return {"success": False, "reason": "all fit boundaries exhausted before full precision",
                    "initial": initial_state,
                    "steps": steps, "final": state, "step_count": len(steps),
                    "axis_boundary_moves": _moves(best["axis_indices"], indices),
                    "auto_acceptance_loss": starting_accepted -
                    sum(state["counts"][name]["accepted"] for name in ("val", "devhard"))}


def _moves(start: Mapping[str, int], end: Mapping[str, int]) -> dict[str, int]:
    return {axis: int(end[axis]) - int(start[axis]) for axis in AXES}
