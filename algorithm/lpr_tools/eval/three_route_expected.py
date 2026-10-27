"""Export flat stage expectations or expectations preserving runtime errors.

No model is loaded. The input decisions are already frozen; projecting the
existing error contract cannot promote any rejected row to automatic OK.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def runtime_projection(record, decision):
    """Keep auto/valid manual choices; preserve special-color/no-read errors."""
    if decision["status"] in {"OK", "BAD_INPUT"}:
        return dict(decision)
    def failure(status):
        return {"status": status, "plate": None, "confidence": 0.0,
                "route": None, "candidate_index": None, "rect_index": None}
    if int(record.get("candidate_count", 0)) == 0:
        return failure("NO_PLATE")
    first = next((r for r in record.get("rects", []) if r["candidate_index"] == 0), None)
    color = first.get("color") if first else None
    if color in {"white", "black"}:
        return failure("UNSUPPORTED")
    if decision["status"] != "NO_READ":
        return dict(decision)
    # No legal fallback: preserve the old S/R invalid-format result if any.
    # max with strict greater keeps the first route/rect on equal confidence.
    best = None
    for rect in record.get("rects", []):
        for route in ("segment", "reader"):
            attempt = rect.get(route)
            if attempt and attempt.get("status") == "INVALID_FORMAT":
                if best is None or float(attempt["confidence"]) > best["confidence"]:
                    best = {"status": "INVALID_FORMAT", "plate": attempt["plate"],
                            "confidence": float(attempt["confidence"]), "route": route,
                            "candidate_index": int(rect["candidate_index"]),
                            "rect_index": int(rect["rect_index"])}
    return best if best is not None else failure("SEG_FAILED")


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if "_meta" not in value:
                yield value


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_expected(trace_path, decisions_path, output_path, preserve_errors=True):
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(target)
    traces = {r["path"]: r for r in _rows(trace_path)}
    target.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    changed = []
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
        meta = {"schema": "three-route-flat-expected-v1", "preserve_runtime_errors": preserve_errors,
                "trace_sha256": _sha(trace_path), "decision_source_sha256": _sha(decisions_path),
                "source_sha256": _sha(__file__), "fields": ["path", "status", "plate", "confidence", "route"],
                "scope": "runtime error projection" if preserve_errors else "arbitration-stage decisions"}
        stream.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for row in _rows(decisions_path):
            path = row["path"]
            if path in seen or path not in traces:
                raise ValueError("duplicate/unmatched expected-output path")
            seen.add(path)
            choice = row["selected"]
            selected = runtime_projection(traces[path], choice) if preserve_errors else choice
            flat = {"path": path, "partition": row["partition"], **selected}
            flat = {key: flat.get(key) for key in ("path", "partition", "status", "plate", "confidence",
                                                  "route", "candidate_index", "rect_index")}
            if selected != choice:
                changed.append(path)
            stream.write(json.dumps(flat, ensure_ascii=False, separators=(",", ":")) + "\n")
    if seen != set(traces):
        raise ValueError("expected-output path set differs from trace")
    return {"path": str(target), "sha256": _sha(target), "rows": len(seen),
            "projected_rows": len(changed), "projection_paths": changed}
