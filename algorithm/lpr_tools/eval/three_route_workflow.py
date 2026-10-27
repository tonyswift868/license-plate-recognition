"""One-process model rerun: fresh S/R/B trace, auto review, fit-only fallback.

Example (replace factory names with the delivered official adapters)::

  .venv/Scripts/python.exe -m lpr_tools.eval.three_route_workflow --fresh \
    --mode all --config algorithm/config/default.toml \
    --box-model algorithm/models/box_reader_v2.pt --box-loader module:factory \
    --detector-model algorithm/models/plate_det_v1.pt --detector-loader module:factory \
    --box-pad-x 0.08 --box-pad-y 0.12 --out-dir algorithm/out/new-three-rerun

Use the box model's registered pad. --mode auto runs only the automatic tool;
--mode fallback/all also fits the preregistered fallback on baseline-000 fit
rows and applies that frozen rule to the selected automatic policy. Every
phase is synchronous in the same process; no worker or inference subprocess.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _fresh_trace(output, options):
    from .three_route import run_live

    report = run_live(config=options["config"], index=options["index"],
        val_manifest=options["val_manifest"], devhard_manifest=options["devhard_manifest"],
        pad_x=options["pad_x"], pad_y=options["pad_y"], output=output,
        detector_model=options["detector_model"], box_model=options["box_model"],
        detector_loader=options["detector_loader"], box_loader=options["box_loader"], search=False)
    if report["status"] != "complete":
        raise ValueError("fresh model inference did not complete successfully")
    return output, output.with_suffix(".report.json")


def _auto_review(trace, base_report, target, split_path=None):
    from .three_route_auto import main

    args = ["--trace", str(trace), "--base-report", str(base_report), "--out", str(target)]
    if split_path:
        args.extend(["--split-json", str(split_path)])
    if main(args) != 0:
        raise ValueError("automatic policy review failed")
    return json.loads(target.read_text(encoding="utf-8"))


def _fit_threshold_report(trace, split_path, original_report, target):
    """Run the fixed exact threshold search on fit rows only, without opening hold labels."""
    from .three_route import exact_search, read_jsonl_gz, sha256, validate_records
    from .route_split import path_key

    records, metadata = read_jsonl_gz(trace)
    assignments_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    assignments = {path_key(path): half for path, half in assignments_doc["assignments"].items()}
    if set(assignments) != {path_key(row["path"]) for row in records}:
        raise ValueError("fit-only threshold split paths do not match the trace")
    fit_records = [row for row in records if assignments[path_key(row["path"])] == "fit"]
    hold_count = len(records) - len(fit_records)
    if not fit_records or not hold_count:
        raise ValueError("fit/hold threshold search requires nonempty halves")
    validate_records(fit_records)
    scan = exact_search(fit_records, progress=lambda event: print(
        json.dumps({"fit_threshold_progress": event}, separators=(",", ":")), flush=True))
    if not scan.get("complete") or not scan.get("best"):
        raise ValueError("fit-only exact threshold search has no feasible rule")
    original = json.loads(Path(original_report).read_text(encoding="utf-8"))
    payload = {"status": "complete", "selection_scope": "fit-only",
        "trace": str(Path(trace).resolve()), "trace_sha256": sha256(trace),
        "trace_metadata": metadata, "trace_metadata_sha256": hashlib.sha256(json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "original_trace_report": str(Path(original_report).resolve()),
        "original_trace_report_sha256": sha256(original_report),
        "original_full_search_ignored": "full-trace threshold search is not used for selection",
        "split_json": str(Path(split_path).resolve()),
        "split_sha256": sha256(split_path), "fit_rows": len(fit_records),
        "hold_rows_unscored": hold_count, "search": scan}
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite fit threshold report: {target}")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _features(trace, target, options):
    from .route_features import enrich_trace

    enrich_trace(trace_path=trace, config_path=options["config"], output_path=target,
                 detector_model=options["detector_model"], detector_loader=options["detector_loader"])
    return target


def _split(trace, features, target):
    import gzip
    from .route_split import write_split
    from .three_route import read_jsonl_gz, sha256

    records, _ = read_jsonl_gz(trace)
    hashes = {}
    with gzip.open(features, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "_meta" not in row:
                hashes[row["path"]] = row["image_sha256"]
    write_split(records, hashes, sha256(trace), target)
    return target


def _content_split(trace, target):
    """Auto-only reruns need content isolation but no locator replay/features."""
    from .route_split import write_split
    from .three_route import read_jsonl_gz, sha256
    records, _ = read_jsonl_gz(trace)
    hashes = {row["path"]: sha256(Path(row["path"])) for row in records}
    write_split(records, hashes, sha256(trace), target)
    return target


def _refine(trace, features, baseline_report, final_report, directory):
    from .fallback_refine import apply_frozen, run_refine
    from .three_route_expected import export_expected

    baseline = run_refine(trace, features, baseline_report, output_dir=directory)
    if baseline.get("selection_failed"):
        raise ValueError("no preregistered fallback met the fit-side green guard")
    final = apply_frozen(trace, features, final_report, directory / "fallback-refine.json", directory)
    expected = export_expected(trace, final["decisions_file"], directory / "expected-runtime.jsonl.gz")
    return {"baseline_rule": baseline["selected_rule"], "final_rule": final["selected_rule"],
            "expected_runtime": expected}


def run_workflow(output_dir, *, mode="all", fresh=False, trace=None, base_report=None,
                 features=None, config="algorithm/config/default.toml",
                 box_model=None, detector_model=None, box_loader=None, detector_loader=None,
                 pad_x=None, pad_y=None,
                 index=r"D:\keshe2\algorithm\data\index\ccpd_index.csv",
                 val_manifest=r"D:\keshe2\algorithm\data\index\val1000_v1.csv",
                 devhard_manifest=r"D:\keshe2\algorithm\data\index\devhard_v1.csv"):
    if mode not in {"auto", "fallback", "all"}:
        raise ValueError("mode must be auto, fallback, or all")
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty output directory: {directory}")
    if fresh and (pad_x is None or pad_y is None):
        raise ValueError("fresh inference requires the delivered model's box pad")
    if not fresh and (trace is None or base_report is None):
        raise ValueError("cached replay requires --trace and --base-report")
    options = dict(config=config, box_model=box_model, detector_model=detector_model,
                   box_loader=box_loader, detector_loader=detector_loader,
                   pad_x=pad_x, pad_y=pad_y, index=index, val_manifest=val_manifest,
                   devhard_manifest=devhard_manifest)
    directory.mkdir(parents=True, exist_ok=True)
    if fresh:
        trace, base_report = _fresh_trace(directory / "raw.jsonl.gz", options)
        gc.collect()
    else:
        trace, base_report = Path(trace).resolve(), Path(base_report).resolve()
    feature_path = split_path = None
    if mode != "auto" or features:
        feature_path = Path(features).resolve() if features else _features(
            trace, directory / "route_features.jsonl.gz", options)
        gc.collect()
        split_path = _split(trace, feature_path, directory / "split.json")
    else:
        split_path = _content_split(trace, directory / "split.json")
    auto_path = directory / "auto-review.json"
    fit_base_report = base_report
    if split_path:
        fit_base_report = _fit_threshold_report(trace, split_path, base_report,
                                                directory / "fit-threshold-report.json")
    automatic = _auto_review(trace, fit_base_report, auto_path, split_path)
    if automatic.get("selected_policy") is None:
        raise ValueError("no reviewed automatic policy satisfies both precision constraints")
    result = {"mode": mode, "fresh": fresh, "trace": str(trace),
              "base_report": str(base_report), "fit_base_report": str(fit_base_report),
              "auto_report": str(auto_path),
              "auto_policy": automatic["selected_policy"],
              "selected_total_accepted": automatic["selected_total_accepted"]}
    if mode != "auto":
        baseline = {**automatic, "selected_policy": "000",
                    "eligible": None,
                    "baseline_expected_accepted": sum(automatic["policies"]["000"][s]["accepted"]
                                                       for s in ("val", "devhard")),
                    "selected_total_accepted": sum(automatic["policies"]["000"][s]["accepted"]
                                                   for s in ("val", "devhard")),
                    "selection_override": "task1 freezes the original three-route acceptance set"}
        baseline_path = directory / "auto-baseline.json"
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        result.update(_refine(trace, feature_path, baseline_path, auto_path, directory))
        result["features"] = str(feature_path)
    (directory / "workflow.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "fallback", "all"), default="all")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--trace")
    parser.add_argument("--base-report")
    parser.add_argument("--features")
    parser.add_argument("--config", default="algorithm/config/default.toml")
    parser.add_argument("--box-model")
    parser.add_argument("--detector-model")
    parser.add_argument("--box-loader")
    parser.add_argument("--detector-loader")
    parser.add_argument("--box-pad-x", type=float)
    parser.add_argument("--box-pad-y", type=float)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    result = run_workflow(args.out_dir, mode=args.mode, fresh=args.fresh,
        trace=args.trace, base_report=args.base_report, features=args.features,
        config=args.config, box_model=args.box_model, detector_model=args.detector_model,
        box_loader=args.box_loader, detector_loader=args.detector_loader,
        pad_x=args.box_pad_x, pad_y=args.box_pad_y)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
