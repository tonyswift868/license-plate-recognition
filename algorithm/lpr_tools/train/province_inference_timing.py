"""Small method wrapper for offline reader timing; never adds warmup calls."""
from __future__ import annotations

import math
import time


def instrument_method(instance, method_name, records, *, batch_size):
    """Wrap one bound method, recording one duration per original invocation."""
    original = getattr(instance, method_name)

    def timed(*args, **kwargs):
        try:
            size = batch_size(args, kwargs)
        except Exception:
            size = None
        started = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            records.append({"call_index": len(records), "batch_size": size,
                            "elapsed_ms": (time.perf_counter_ns() - started) / 1e6})

    setattr(instance, method_name, timed)
    return original


def _summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"calls": 0, "mean_ms": None, "p95_nearest_rank_ms": None, "max_ms": None}
    return {"calls": len(ordered), "mean_ms": sum(ordered) / len(ordered),
            "p95_nearest_rank_ms": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
            "max_ms": ordered[-1]}


def summarize(records, *, buckets=()):
    times = [float(row["elapsed_ms"]) for row in records]
    result = {"calls": _summary(times), "all_calls_elapsed_ms": times}
    if buckets:
        by_batch = {}
        for size in buckets:
            values = [float(row["elapsed_ms"]) for row in records
                      if row.get("batch_size") == size]
            by_batch[str(size)] = {**_summary(values), "all_calls_elapsed_ms": values}
        result["by_batch_size"] = by_batch
    return result
