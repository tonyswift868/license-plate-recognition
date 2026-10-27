"""Summarize batch-12 standalone route and integrated workflow traces."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def _jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if "_meta" not in value:
                yield value


def _selected(attempts):
    best = None
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        plate = attempt.get("plate", attempt.get("prediction"))
        try:
            confidence = float(attempt.get("confidence", attempt.get("min_confidence")))
        except (TypeError, ValueError):
            continue
        if (not isinstance(plate, str) or not plate or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
                or not attempt.get("valid", attempt.get("status") == "OK")
                or attempt.get("status") in {"INVALID_FORMAT", "BAD_INPUT"}):
            continue
        candidate = (confidence, plate)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best else None


def _groups(row):
    split = row.get("split", "unknown")
    subset = row.get("subset", "unknown")
    groups = ["all", f"subset/{subset}", f"split/{split}", f"{split}/{subset}"]
    truth = row.get("truth", "")
    if truth and truth[0] != "皖":
        groups.append("non_anwei")
        groups.append(f"non_anwei/{split}")
        groups.append(f"non_anwei/{split}/{subset}")
    if row.get("subset") == "ccpd_green":
        groups.append("green")
    if row.get("subset") == "ccpd_db" and row.get("brightness") is not None:
        try:
            if float(row["brightness"]) <= 19:
                groups.append("dark_le_19")
        except (TypeError, ValueError):
            pass
    if row.get("subset") in {"ccpd_tilt", "ccpd_rotate"}:
        groups.append("tilt_plus_rotate")
    return groups


def _aggregate(rows):
    groups = defaultdict(lambda: {"n": 0, "province_correct": 0, "plate_correct": 0,
                                  "predicted_anwei": 0, "wrong_as_anwei": 0})
    for row in rows:
        prediction = row.get("prediction")
        truth = row.get("truth", "")
        predicted_province = prediction[0] if isinstance(prediction, str) and prediction else None
        for name in _groups(row):
            item = groups[name]
            item["n"] += 1
            item["province_correct"] += int(predicted_province == truth[:1] and bool(truth))
            item["plate_correct"] += int(prediction == truth and bool(truth))
            item["predicted_anwei"] += int(predicted_province == "皖")
            item["wrong_as_anwei"] += int(predicted_province == "皖" and truth[:1] != "皖")
    for item in groups.values():
        n = item["n"]
        item["province_accuracy"] = item["province_correct"] / n if n else None
        item["plate_accuracy"] = item["plate_correct"] / n if n else None
    return dict(sorted(groups.items()))


def standalone_trace(path: Path):
    records = list(_jsonl_gz(path))
    routes = {name: [] for name in ("segment_cnn", "crnn", "box")}
    for row in records:
        segment = _selected(rect.get("segment") for rect in row.get("rects", []))
        reader = _selected(rect.get("reader") for rect in row.get("rects", []))
        box = _selected(row.get("boxes", {}).get(key) for key in sorted(
            row.get("boxes", {}), key=lambda value: int(value)))
        for name, prediction in (("segment_cnn", segment), ("crnn", reader), ("box", box)):
            routes[name].append({**row, "prediction": prediction})
    return {name: {"rows": len(values), "metrics": _aggregate(values)}
            for name, values in routes.items()}


def workflow_trace(directory: Path):
    workflow = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    trace_path = Path(workflow["trace"])
    records = list(_jsonl_gz(trace_path))
    source_by_path = {row["path"]: row for row in records}
    expected_info = workflow["expected_runtime"]
    expected_path = Path(expected_info["path"])
    decisions = list(_jsonl_gz(expected_path))
    if len(decisions) != len(records):
        raise ValueError("workflow runtime decisions and raw trace row count differ")
    rows = []
    accepted = accepted_correct = 0
    for decision in decisions:
        source = source_by_path.get(decision["path"])
        if source is None:
            raise ValueError(f"workflow decision path missing from its trace: {decision['path']}")
        prediction = decision.get("plate") if decision.get("status") in {"OK", "LOW_CONFIDENCE"} else None
        rows.append({**source, "prediction": prediction})
        if decision.get("status") == "OK":
            accepted += 1
            accepted_correct += int(decision.get("plate") == source.get("truth"))
    return {"rows": len(rows), "auto_accepted": accepted,
            "auto_correct": accepted_correct,
            "auto_precision": accepted_correct / accepted if accepted else None,
            "final_metrics": _aggregate(rows),
            "workflow": workflow}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-trace", type=Path,
                        help="three-route fresh raw.jsonl.gz containing S/R/B attempts")
    parser.add_argument("--workflow-dir", type=Path,
                        help="output directory from three_route_workflow --mode all --fresh")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.actual_trace is None and args.workflow_dir is None:
        parser.error("supply --actual-trace and/or --workflow-dir")
    result = {"schema": "batch12-reader-eval-summary-v1"}
    if args.actual_trace:
        result["actual_candidates"] = standalone_trace(args.actual_trace)
    if args.workflow_dir:
        result["three_route"] = workflow_trace(args.workflow_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()),
                      "actual_candidates": bool(args.actual_trace),
                      "three_route": bool(args.workflow_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
