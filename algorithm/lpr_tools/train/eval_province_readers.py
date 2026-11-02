"""Summarize batch-12 standalone route and integrated workflow traces."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path


def _jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if "_meta" not in value:
                yield value


def _selected_attempt(attempts):
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
        if best is None or confidence > best["confidence"]:
            best = {**attempt, "plate": plate, "confidence": confidence}
    return best


def _selected(attempts):
    best = _selected_attempt(attempts)
    return best["plate"] if best else None


def _groups(row):
    split = row.get("split", "unknown")
    subset = row.get("subset", "unknown")
    groups = ["all", f"subset/{subset}", f"split/{split}", f"{split}/{subset}"]
    truth = row.get("truth", "")
    if truth and truth[0] != "皖":
        groups.append("non_anwei")
        groups.append(f"non_anwei/{split}")
        groups.append(f"non_anwei/{split}/{subset}")
    elif truth and truth[0] == "皖":
        groups.extend(("anwei", f"anwei/{split}", f"anwei/{split}/{subset}"))
    if row.get("subset") == "ccpd_green":
        groups.append("green")
    if row.get("subset") == "ccpd_db" and row.get("brightness") is not None:
        try:
            if float(row["brightness"]) <= 19:
                groups.append("dark_le_19")
                groups.append(f"{split}/ccpd_db_brightness_le_19")
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
    selections = []
    for row in records:
        segment_attempts = [
            {**(rect.get("segment") or {}), "candidate_index": rect.get("candidate_index"),
             "rect_index": rect.get("rect_index"), "route": "segment"}
            for rect in row.get("rects", [])]
        reader_attempts = [
            {**(rect.get("reader") or {}), "candidate_index": rect.get("candidate_index"),
             "rect_index": rect.get("rect_index"), "route": "reader"}
            for rect in row.get("rects", [])]
        box_attempts = [
            {**(attempt or {}), "candidate_index": int(key), "rect_index": None,
             "route": "box"}
            for key, attempt in sorted(row.get("boxes", {}).items(),
                                       key=lambda item: int(item[0]))]
        chosen = {
            "segment_cnn": _selected_attempt(segment_attempts),
            "crnn": _selected_attempt(reader_attempts),
            "box": _selected_attempt(box_attempts),
        }
        selections.append({"path": row.get("path"), "truth": row.get("truth"),
                           "split": row.get("split"), "subset": row.get("subset"),
                           "candidate_attempts": {"segment_cnn": segment_attempts,
                                                  "crnn": reader_attempts,
                                                  "box": box_attempts},
                           "highest_confidence_legal": chosen})
        segment = chosen["segment_cnn"]["plate"] if chosen["segment_cnn"] else None
        reader = chosen["crnn"]["plate"] if chosen["crnn"] else None
        box = chosen["box"]["plate"] if chosen["box"] else None
        for name, prediction in (("segment_cnn", segment), ("crnn", reader), ("box", box)):
            routes[name].append({**row, "prediction": prediction})
    return {**{name: {"rows": len(values), "metrics": _aggregate(values)}
               for name, values in routes.items()},
            "selection_method": "highest-confidence legal attempt across all candidates; stable first on ties",
            "candidate_selection_records": selections}


def _timing_summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"calls": 0, "mean_ms": None, "p95_nearest_rank_ms": None, "max_ms": None}
    import math as _math
    return {"calls": len(ordered), "mean_ms": sum(ordered) / len(ordered),
            "p95_nearest_rank_ms": ordered[max(0, _math.ceil(.95 * len(ordered)) - 1)],
            "max_ms": ordered[-1], "over_10ms": sum(value > 10.0 for value in ordered)}


def summarize_testset_composition(manifest_path: Path, development_metrics: dict):
    """Estimate frozen scenarios using manifest counts only, never image bytes."""
    from collections import Counter

    composition = {name: Counter() for name in ("clear", "night", "tilt", "dirty")}
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"category", "source"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("test manifest must contain category and source columns")
        for row in reader:
            category, source = row["category"], row["source"].split("/")[-1]
            if category in composition:
                composition[category][source] += 1

    mappings = {
        "clear": {"ccpd_base": "ccpd_base", "ccpd_green": "ccpd_green"},
        "night": {"ccpd_db": "ccpd_db_brightness_le_19"},
        "tilt": {"ccpd_tilt": "ccpd_tilt", "ccpd_rotate": "ccpd_rotate"},
        "dirty": {"ccpd_weather": "ccpd_weather"},
    }
    output = {"schema": "batch12-testset-composition-estimate-v1",
              "manifest": str(Path(manifest_path).resolve()),
              "manifest_rows_by_category_source": {
                  category: dict(sorted(counts.items())) for category, counts in composition.items()},
              "source": "manifest metadata only; no test image or test label read",
              "night_mapping": "testset ccpd_db category is estimated with the frozen CCPD db brightness<=19 development subgroup; this is a proxy, not per-test-image brightness verification",
              "scenarios": {}}
    for split in ("val", "devhard"):
        split_metrics = {}
        for category, source_to_group in mappings.items():
            category_counts = composition[category]
            denominator = sum(category_counts.values())
            components, weighted_correct = {}, 0.0
            missing = []
            for source, group in source_to_group.items():
                count = category_counts.get(source, 0)
                if not count:
                    continue
                metric = development_metrics.get(f"{split}/{group}")
                accuracy_value = (metric.get("plate_accuracy", metric.get("whole_plate_accuracy"))
                                  if metric else None)
                if accuracy_value is None:
                    missing.append({"source": source, "development_group": f"{split}/{group}"})
                    continue
                accuracy = float(accuracy_value)
                components[source] = {"manifest_n": count, "development_group": f"{split}/{group}",
                                      "plate_accuracy": accuracy}
                weighted_correct += count * accuracy
            split_metrics[category] = {
                "estimated_accuracy": weighted_correct / denominator if denominator and not missing else None,
                "estimated_correct": weighted_correct if not missing else None,
                "manifest_n": denominator, "components": components,
                "estimable": denominator > 0 and not missing,
                "missing_development_groups": missing,
            }
        output["scenarios"][split] = split_metrics
    output["formulas"] = {
        "clear": "sum(manifest source count * matching development accuracy)/clear manifest count",
        "night": "db_brightness_le_19 development accuracy; labeled as proxy for the testset db night category",
        "tilt": "sum(manifest source count * matching development accuracy)/tilt manifest count",
        "dirty": "sum(manifest source count * matching development accuracy)/dirty manifest count",
    }
    return output


def time_box_trace_candidates(trace_path: Path, model_path: Path, output_path: Path):
    """Time every traced candidate read with the official CPU reader factory."""
    import cv2
    import torch
    from lpr.boxcrop import crop_box
    from lpr.io_utils import imread
    from lpr.recognize import load_box_reader

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reader = load_box_reader(model_path)
    pad_x, pad_y = reader.box_pad
    output_path = Path(output_path)
    csv_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed_values = []
    mismatches = []
    skipped_bad_crops = []
    records = 0
    with csv_path.open("w", encoding="utf-8", newline="") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=(
            "call_index", "path", "candidate_index", "trace_status", "trace_valid",
            "trace_plate", "trace_confidence", "read_plate", "read_min_confidence",
            "elapsed_ms", "text_matches", "confidence_abs_error"))
        writer.writeheader()
        for row in _jsonl_gz(trace_path):
            image = imread(row["path"])
            for key, attempt in sorted(row.get("boxes", {}).items(), key=lambda item: int(item[0])):
                crop = crop_box(image, attempt["bbox"], pad_x, pad_y)
                if crop.size == 0:
                    skipped_bad_crops.append({"path": row["path"], "candidate_index": int(key),
                                              "status": "BAD_CROP"})
                    continue
                started = time.perf_counter_ns()
                text, confidences = reader.read(crop)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                elapsed_values.append(elapsed)
                trace_conf = attempt.get("confidence")
                read_conf = min(confidences) if confidences else None
                error = abs(float(trace_conf) - float(read_conf)) if trace_conf is not None and read_conf is not None else None
                text_match = text == attempt.get("plate")
                if not text_match or (error is not None and error > 1e-7):
                    mismatches.append({"path": row["path"], "candidate_index": int(key),
                                       "text_matches": text_match, "confidence_abs_error": error})
                writer.writerow({"call_index": len(elapsed_values) - 1, "path": row["path"],
                                 "candidate_index": int(key), "trace_status": attempt.get("status"),
                                 "trace_valid": attempt.get("valid"), "trace_plate": attempt.get("plate"),
                                 "trace_confidence": trace_conf, "read_plate": text,
                                 "read_min_confidence": read_conf, "elapsed_ms": elapsed,
                                 "text_matches": text_match, "confidence_abs_error": error})
            records += 1
            del image
    summary = {"schema": "batch12-box-trace-read-timing-v1", "trace": str(Path(trace_path).resolve()),
               "model": str(Path(model_path).resolve()), "reader_threads": 2,
               "timing_scope": "reader.read only; excludes image decode and crop_box; no warmup calls",
               "image_rows": records, "calls": _timing_summary(elapsed_values),
               "all_calls_elapsed_ms": elapsed_values, "trace_output_mismatches": mismatches,
               "trace_match": not mismatches, "skipped_bad_crops": len(skipped_bad_crops),
               "skipped_bad_crop_rows": skipped_bad_crops, "csv": str(csv_path.resolve())}
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def workflow_trace(directory: Path):
    workflow = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    if workflow.get("mode") != "all" or workflow.get("fresh") is not True:
        raise ValueError("workflow.json must describe --fresh --mode all")
    trace_path = Path(workflow["trace"])
    records = list(_jsonl_gz(trace_path))
    source_by_path = {row["path"]: row for row in records}
    auto_info = workflow.get("auto_report")
    policy = workflow.get("auto_policy")
    accepted_metrics = None
    if auto_info and policy:
        auto = json.loads(Path(auto_info).read_text(encoding="utf-8"))
        chosen = auto.get("policies", {}).get(policy)
        if not isinstance(chosen, dict):
            raise ValueError("workflow selected automatic policy is absent from auto_report")
        accepted_metrics = {}
        for split in ("val", "devhard"):
            stats = chosen.get(split)
            if not isinstance(stats, dict):
                raise ValueError(f"automatic policy lacks {split} accepted metrics")
            n, accepted = int(stats["n"]), int(stats["accepted"])
            correct = int(stats["accepted_correct"])
            accepted_metrics[split] = {
                "n": n, "accepted": accepted, "correct": correct,
                "precision": correct / accepted if accepted else None,
                "coverage": accepted / n if n else None,
            }
        total_n = sum(item["n"] for item in accepted_metrics.values())
        total_accepted = sum(item["accepted"] for item in accepted_metrics.values())
        total_correct = sum(item["correct"] for item in accepted_metrics.values())
        accepted_metrics["combined"] = {
            "n": total_n, "accepted": total_accepted, "correct": total_correct,
            "precision": total_correct / total_accepted if total_accepted else None,
            "coverage": total_accepted / total_n if total_n else None,
            "composition": {name: item["n"] for name, item in accepted_metrics.items()
                            if name in {"val", "devhard"}},
        }

    runtime_metrics = None
    expected_info = workflow.get("expected_runtime")
    if expected_info:
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
        runtime_metrics = {"rows": len(rows), "auto_accepted": accepted,
                           "auto_correct": accepted_correct,
                           "auto_precision": accepted_correct / accepted if accepted else None,
                           "final_metrics": _aggregate(rows)}
    elif accepted_metrics is None:
        raise ValueError("workflow.json has neither expected_runtime nor auto_report/auto_policy")
    return {"rows": len(records), "accepted_metrics": accepted_metrics,
            "runtime_projection": runtime_metrics, "workflow": workflow}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-trace", type=Path,
                        help="three-route fresh raw.jsonl.gz containing S/R/B attempts")
    parser.add_argument("--workflow-dir", type=Path,
                        help="output directory from three_route_workflow --mode all --fresh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-box-model", type=Path,
                        help="optionally replay all trace candidate boxes through the official CPU box reader")
    parser.add_argument("--timing-out", type=Path,
                        help="JSON output for --time-box-model (per-call CSV is written alongside)")
    parser.add_argument("--testset-manifest", type=Path,
                        help="metadata-only frozen testset manifest for scenario composition estimates")
    args = parser.parse_args(argv)
    if args.actual_trace is None and args.workflow_dir is None:
        parser.error("supply --actual-trace and/or --workflow-dir")
    result = {"schema": "batch12-reader-eval-summary-v1"}
    if args.actual_trace:
        result["actual_candidates"] = standalone_trace(args.actual_trace)
    if args.workflow_dir:
        result["three_route"] = workflow_trace(args.workflow_dir)
        if args.testset_manifest:
            metrics = (result["three_route"].get("runtime_projection") or {}).get("final_metrics")
            if metrics is None:
                raise ValueError("scenario composition requires expected_runtime final metrics")
            result["three_route_testset_composition"] = summarize_testset_composition(
                args.testset_manifest, metrics)
    if args.time_box_model:
        if not args.actual_trace or not args.timing_out:
            parser.error("--time-box-model requires --actual-trace and --timing-out")
        result["box_reader_timing"] = time_box_trace_candidates(
            args.actual_trace, args.time_box_model, args.timing_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()),
                      "actual_candidates": bool(args.actual_trace),
                      "three_route": bool(args.workflow_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
