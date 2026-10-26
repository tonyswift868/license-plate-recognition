"""G2 evaluation v1, including explicit fake mode and ground-truth oracle.

Whole-plate equality includes rejected guesses; automatic acceptance requires OK.
Character accuracy is positional matches / sum(max(truth length, predicted length)).
Failure causes are status-based proxies, not proven causal localization recall.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from lpr.config import ALGORITHM_DIR, load_config
from lpr.errors import DEFAULT_MESSAGES, ERROR_CODES, failure_result
from lpr.io_utils import imwrite
from lpr.postprocess import decode
from lpr.segment import segment
from lpr.types import PlateCandidate, RectifiedPlate, RecognitionResult, RecogStatus as S
from lpr_tools.data.manifest import CATEGORIES, image_path, read_manifest, write_manifest
from lpr_tools.data.image_health import decode_image
from .common import (add_backend_args, config_for_args, effective_config_info,
                     make_recognizer, mode_name)
from .index_samples import (DEFAULT_SEED, DEFAULT_VALIDATION_N, FIXED_HARD_MANIFEST,
                           FIXED_VAL_MANIFEST, fixed_manifest_samples, path_key,
                           read_samples, sample_dev_hard, sample_validation)
from .locate_recall import locate_match, summarize as summarize_location


class OracleRecognizer:
    """Use annotated corners; never invoke locate or the stage-3 rectifier."""
    def __init__(self, recognizer):
        self.recognizer = recognizer

    def recognize(self, image, row):
        if row.category == "np":
            raise ValueError("oracle is undefined for no-plate images")
        if row.quad is None:
            raise ValueError("oracle needs quad ground truth on every plated row")
        if not hasattr(self.recognizer, "classifier"):
            return self.recognizer.recognize(image)
        start = time.perf_counter()
        cfg = self.recognizer.config
        layout = len(row.plate_no)  # oracle also receives true plate length
        w, h = cfg.rectify.size_7 if layout == 7 else cfg.rectify.size_8
        source = np.asarray(row.quad, np.float32)
        if abs(cv2.contourArea(source)) < 1:
            raise ValueError("degenerate oracle quad")
        target = np.asarray([(0,0),(w-1,0),(w-1,h-1),(0,h-1)], np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        plate = cv2.warpPerspective(image, transform, (w,h))
        x,y,bw,bh = cv2.boundingRect(source)
        candidate = PlateCandidate(list(row.quad), (x,y,bw,bh), row.plate_color, 1., "oracle")
        rect = RectifiedPlate(plate, layout, row.plate_color, transform, candidate)
        stages = {"oracle_warp": (time.perf_counter()-start)*1000}
        t = time.perf_counter()
        seg = segment(rect, layout, cfg.segment)
        stages["segment"] = (time.perf_counter()-t)*1000
        if not seg.ok:
            return failure_result(S.SEG_FAILED, elapsed_ms=(time.perf_counter()-start)*1000, stage_ms=stages)
        t = time.perf_counter()
        probs = self.recognizer.classifier.predict_proba(seg.chars)
        stages["recognize"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        dec = decode(probs, layout, row.plate_color, cfg.postprocess)
        stages["postprocess"] = (time.perf_counter()-t)*1000
        return RecognitionResult(dec.status, ERROR_CODES[dec.status], DEFAULT_MESSAGES[dec.status],
            dec.plate_no, row.plate_color, layout, dec.confidence, dec.char_confidences, dec.candidates,
            candidate.bbox, candidate.corners, plate, (time.perf_counter()-start)*1000, stages,
            vehicle_type_hint=dec.vehicle_type_hint, model_version=self.recognizer.version)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(samples):
    plated = [(row, result) for row, result in samples if row.category != "np"]
    correct = sum(result.plate_no == row.plate_no for row, result in plated)
    accepted = [(row, result) for row, result in plated if result.status == S.OK]
    char_correct = sum(sum(a == b for a,b in zip(row.plate_no, result.plate_no or "")) for row,result in plated)
    char_total = sum(max(len(row.plate_no),len(result.plate_no or "")) for row,result in plated)
    np_rows = [(row,result) for row,result in samples if row.category == "np"]
    false_positives = sum(bool(result.plate_no) for _,result in np_rows)
    return {"n": len(plated), "correct": correct, "plate_accuracy": _fraction(correct,len(plated)),
            "char_correct": char_correct, "char_total": char_total,
            "character_accuracy": _fraction(char_correct,char_total), "auto_pass": len(accepted),
            "auto_pass_rate": _fraction(len(accepted),len(plated)),
            "auto_pass_correct": sum(row.plate_no == result.plate_no for row,result in accepted),
            "auto_pass_precision": _fraction(sum(row.plate_no == result.plate_no for row,result in accepted),len(accepted)),
            "np_n": len(np_rows), "false_positives": false_positives,
            "false_positive_rate": _fraction(false_positives,len(np_rows))}


def failure_cause(row, result):
    if row.category == "np":
        return "localization" if result.plate_no else None
    if result.status == S.NO_PLATE:
        return "localization"
    if result.status == S.SEG_FAILED:
        return "segmentation"
    if result.plate_no and result.plate_no != row.plate_no:
        return "recognition"
    if result.status != S.OK:
        return "rejection"
    if result.plate_no != row.plate_no:
        return "recognition"
    return None


def evaluate(manifest, recognizer, out, *, oracle=False, index=None,
             locate_recall=False, sampling=None, stage_trace=None, reuse_stage_trace=None,
             truth_samples=None, effective_config=None):
    # Keep rows with missing or corrupt files in the denominator accounting;
    # they are explicitly recorded as BAD_INPUT results below.
    cv2.setNumThreads(1)
    rows = read_manifest(manifest, check_files=False)
    if not rows:
        raise ValueError("manifest is empty")
    if oracle and any(r.category != "np" and r.quad is None for r in rows):
        raise ValueError("oracle needs quad ground truth on every plated row")
    inherited_locations = None
    if reuse_stage_trace:
        inherited = {}
        with Path(reuse_stage_trace).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    inherited[item["path"]] = item
        expected_paths = [row.path for row in rows]
        if list(inherited) != expected_paths:
            raise ValueError("reused stage trace paths/order do not match manifest")
        inherited_locations = inherited
    source_samples = truth_samples if truth_samples is not None else (read_samples(index) if index is not None else [])
    lookup = {path_key(s.row.path): s for s in source_samples}
    if locate_recall:
        if index is None:
            raise ValueError("--locate-recall needs ccpd_index.csv via --index")
        for row in rows:
            truth = lookup.get(path_key(image_path(row, manifest)))
            if row.category != "np" and (truth is None or truth.bbox is None):
                raise ValueError(f"missing index bbox ground truth: {row.path}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples, failures, locations, decode_errors, decode_retries, traces = [], [], [], [], [], []
    subset_samples = {}
    locator_cfg = (getattr(recognizer, "config", None) or effective_config or load_config()) if locate_recall else None
    causes = Counter({key:0 for key in ("localization","segmentation","recognition","rejection")})
    oracle_recognizer = OracleRecognizer(recognizer)
    source_counts = Counter()
    source_correct = Counter()
    reader_invoked_images = corner_invoked_images = 0
    reader_invoked_calls = corner_invoked_calls = 0
    for i,row in enumerate(rows):
        image, decode_error = decode_image(image_path(row, manifest))
        if decode_error:
            first_error = decode_error
            gc.collect()
            image, decode_error = decode_image(image_path(row, manifest))
            decode_retries.append({"path": row.path, "first_error": first_error,
                                   "retry_error": decode_error})
        if decode_error:
            decode_errors.append({"path": row.path, "error": decode_error})
            result = failure_result(S.BAD_INPUT, f"invalid validation input: {decode_error}")
            samples.append((row, result))
            truth = lookup.get(path_key(image_path(row, manifest)))
            if truth is not None:
                subset_samples.setdefault(truth.stratum, []).append((row, result))
                if locate_recall and row.category != "np":
                    locations.append((truth.stratum, {"hit": False, "best_iou": 0., "candidates": 0}))
            causes["rejection"] += 1
            traces.append({"path": row.path, "category": row.category, "truth": row.plate_no,
                           "prediction": "", "plate_no": result.plate_no, "confidence": result.confidence,
                           "status": S.BAD_INPUT.value, "decode_error": decode_error,
                           "failure_cause": "rejection", "top3_hit": False if locate_recall else None})
            continue
        if hasattr(recognizer, "_reset_observation"):
            recognizer._reset_observation()
        else:
            if hasattr(recognizer, "last_selection"):
                recognizer.last_selection = None
            if hasattr(recognizer, "reader_invoked_calls"):
                recognizer.reader_invoked_calls = 0
            if hasattr(recognizer, "corner_invoked_calls"):
                recognizer.corner_invoked_calls = 0
        if oracle and row.category != "np":
            result = oracle_recognizer.recognize(image,row)
        else:
            result = recognizer.recognize(image)
        # Oracle plated rows bypass the final pipeline attempt; never attribute a previous
        # per-instance selection to an oracle result.
        selection = getattr(recognizer, "last_selection", None) if not oracle or row.category == "np" else None
        if selection is not None:
            source_counts["selected_result_count"] += 1
            if selection["crnn_selected"]:
                source_counts["crnn_selected_count"] += 1
                source_correct["crnn_selected_correct"] += result.plate_no == row.plate_no
            if selection["source"] == "net":
                source_counts["network_candidate_selected_count"] += 1
                source_correct["network_candidate_selected_correct"] += result.plate_no == row.plate_no
        reader_invoked_images += "reader" in result.stage_ms
        corner_invoked_images += "corner" in result.stage_ms
        reader_invoked_calls += getattr(recognizer, "reader_invoked_calls", 0)
        corner_invoked_calls += getattr(recognizer, "corner_invoked_calls", 0)
        samples.append((row,result))
        truth = lookup.get(path_key(image_path(row, manifest)))
        if truth is not None:
            subset_samples.setdefault(truth.stratum, []).append((row, result))
        if locate_recall and row.category != "np":
            if truth.bbox is None:
                raise ValueError(f"plated row has no index bbox: {row.path}")
            location = locate_match(image, truth.bbox, locator_cfg)
            locations.append((truth.stratum, location))
        else:
            location = None
        if location is None and inherited_locations is not None:
            inherited_row = inherited_locations[row.path]
            location = ({"hit": inherited_row["top3_hit"]}
                        if inherited_row.get("top3_hit") is not None else None)
        traces.append({"path": row.path, "category": row.category, "truth": row.plate_no,
                       "prediction": result.plate_no or "", "plate_no": result.plate_no,
                       "confidence": result.confidence, "status": result.status.value,
                       "selected_source": selection["source"] if selection else None,
                       "crnn_selected": selection["crnn_selected"] if selection else None,
                       "reader_invoked": "reader" in result.stage_ms,
                       "corner_invoked": "corner" in result.stage_ms,
                       "failure_cause": failure_cause(row, result),
                       "segmentation_failed": result.status == S.SEG_FAILED,
                       "top3_hit": location["hit"] if location is not None else None})
        cause = failure_cause(row,result)
        if cause:
            causes[cause] += 1
            filename = f"failure_{i:04d}.png"
            preview = image.copy()
            cv2.putText(preview, f"{cause}: {result.status.value}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,255),1)
            imwrite(out/filename,preview)
            if result.plate_image is not None:
                imwrite(out/f"failure_{i:04d}_plate.png",result.plate_image)
            failures.append({"path":row.path, "truth":row.plate_no, "category":row.category,
                             "cause":cause, "image":filename, "result":result.to_dict()})
    report = {"mode":recognizer.version, "oracle":oracle,
              "evaluation_config": effective_config_info(effective_config, getattr(recognizer, "config_path", None)) if effective_config else None,
              "network_attribution": {
                  "selection_source": "observed _Attempt at final _build; no-output rows are not selected",
                  "invocation_source": "stage_ms key means invoked only; it does not imply selected",
                  "selected_result_count": source_counts["selected_result_count"],
                  "crnn_selected_count": source_counts["crnn_selected_count"],
                  "crnn_selected_correct": source_correct["crnn_selected_correct"],
                  "crnn_selected_accuracy": _fraction(source_correct["crnn_selected_correct"], source_counts["crnn_selected_count"]),
                  "network_candidate_selected_count": source_counts["network_candidate_selected_count"],
                  "network_candidate_selected_correct": source_correct["network_candidate_selected_correct"],
                  "network_candidate_selected_accuracy": _fraction(source_correct["network_candidate_selected_correct"], source_counts["network_candidate_selected_count"]),
                  "reader_invoked_images": reader_invoked_images,
                  "corner_invoked_images": corner_invoked_images,
                  "reader_invoked_calls": reader_invoked_calls,
                  "corner_invoked_calls": corner_invoked_calls,
              },
              "manifest_rows": len(rows), "decoded_rows": len(samples) - len(decode_errors),
              "scored_rows": len(samples), "decode_retries": decode_retries,
              "decode_error_count": len(decode_errors), "decode_errors": decode_errors,
              "oracle_scope":"ground truth quad, color and length; np runs normally" if oracle else None,
              "character_metric":"positional matches / max(true length, predicted length)",
              "attribution":("failure causes are status-based proxies; Top-3 recall measured independently"
                             if locate_recall else "status-based proxy; not measured localization recall"),
              "sampling": sampling,
              "subsets": {key: metrics(value) for key, value in sorted(subset_samples.items())},
              "locate_recall": (summarize_location(locations, excluded_np=sum(row.category == "np" for row in rows))
                                if locate_recall else None),
              "locate_recall_method": "classic locate() Top-3 candidates; corner network fusion is not included"
                                     if locate_recall else None,
              "overall_g2":metrics([(r,s) for r,s in samples if r.category in {"clear","night","tilt","dirty"}]),
              "categories":{category:metrics([(r,s) for r,s in samples if r.category == category]) for category in CATEGORIES},
              "failure_causes":dict(causes),"failures":failures}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if stage_trace:
        trace_path = Path(stage_trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in traces),
                              encoding="utf-8")
    return report


def evaluate_locate_only(manifest, index, out, *, sampling=None, truth_samples=None, config=None,
                         config_arg=None):
    """Measure Top-3 localization directly from CCPD bbox labels, without a model."""
    rows = read_manifest(manifest, check_files=False)
    if not rows:
        raise ValueError("manifest is empty")
    if index is None:
        raise ValueError("--locate-only needs ccpd_index.csv via --index")
    lookup = ({path_key(sample.row.path): sample for sample in truth_samples}
              if truth_samples is not None else
              {path_key(sample.row.path): sample for sample in read_samples(index)})
    locations, decode_errors = [], []
    excluded_np = 0
    cfg = config or load_config()
    for row in rows:
        if row.category == "np":
            excluded_np += 1
            continue
        sample = lookup.get(path_key(image_path(row, manifest)))
        if sample is None or sample.bbox is None:
            raise ValueError(f"missing index bbox ground truth: {row.path}")
        image, decode_error = decode_image(image_path(row, manifest))
        if decode_error:
            decode_errors.append({"path": row.path, "error": decode_error})
            continue
        locations.append((sample.stratum, locate_match(image, sample.bbox, cfg)))
    report = {"mode": "locate-only", "oracle": False,
              "evaluation_config": effective_config_info(cfg, config_arg),
              "attribution": "Top-3 localization recall measured independently; no recognizer loaded",
              "sampling": sampling,
              "locate_recall": summarize_location(locations, excluded_np=excluded_np),
              "manifest_rows": len(rows), "decoded_rows": len(locations),
              "decode_error_count": len(decode_errors), "decode_errors": decode_errors}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def print_report(report):
    if report.get("mode") == "locate-only":
        location = report["locate_recall"]
        print("mode=locate-only oracle=False")
        print(f"Top-{location['topk']} locate recall (IoU >= {location['iou_threshold']}; np excluded)")
        for subset, values in [("overall", location["overall"]), *location["subsets"].items()]:
            print(f"{subset} {values['hits']}/{values['total']} {_percent(values['recall'])}")
        print(f"excluded_np={location['excluded_np']}")
        if "decode_error_count" in report:
            print(f"decoded_rows={report['decoded_rows']}/{report['manifest_rows']} decode_errors={report['decode_error_count']}")
        if report.get("sampling"):
            print("sampling=" + json.dumps({k:v for k,v in report["sampling"].items() if k != "paths"},
                                           ensure_ascii=False, sort_keys=True))
        return
    print(f"mode={report['mode']} oracle={report['oracle']}")
    print("category n plate_accuracy character_accuracy auto_pass_rate auto_pass_precision")
    for name,m in [("overall_g2",report["overall_g2"]),*report["categories"].items()]:
        print(f"{name} {m['n']} " + " ".join(_percent(m[key]) for key in
              ("plate_accuracy","character_accuracy","auto_pass_rate","auto_pass_precision")))
    m=report["categories"]["np"]
    print(f"np_false_positives={m['false_positives']}/{m['np_n']} rate={_percent(m['false_positive_rate'])}")
    print("failure_causes=" + json.dumps(report["failure_causes"],ensure_ascii=False))
    print("attribution=" + report["attribution"])
    print("character_metric=" + report["character_metric"])
    print(f"failure_examples={len(report['failures'])}")
    if "decode_error_count" in report:
        print(f"decoded_rows={report['decoded_rows']}/{report['manifest_rows']} decode_errors={report['decode_error_count']}")
    if report.get("sampling"):
        print("sampling=" + json.dumps({k:v for k,v in report["sampling"].items() if k != "paths"},
                                       ensure_ascii=False, sort_keys=True))
    if report.get("subsets"):
        print("subset n plate_accuracy auto_pass_precision")
        for subset, m in report["subsets"].items():
            print(f"{subset} {m['n']} {_percent(m['plate_accuracy'])} {_percent(m['auto_pass_precision'])}")
    if report.get("locate_recall"):
        location = report["locate_recall"]
        print(f"Top-{location['topk']} locate recall (IoU >= {location['iou_threshold']}; np excluded)")
        for subset, values in [("overall", location["overall"]), *location["subsets"].items()]:
            print(f"{subset} {values['hits']}/{values['total']} {_percent(values['recall'])}")


def append_scoreboard(path, report, *, split, classifier):
    path = Path(path)
    if path.resolve() == (ALGORITHM_DIR/"reports"/"scoreboard.md").resolve():
        raise ValueError("本批禁止修改正式 scoreboard.md；请传临时副本路径")
    if split not in {"val", "manifest-val"}:
        raise ValueError("积分板只追加验证集结果；演示请用 --split manifest-val")
    try:
        commit = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,
                                check=True,cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        commit = "unknown"
    categories = report["categories"]
    values = [_percent(categories[c]["plate_accuracy"]) for c in ("clear","night","tilt","dirty")]
    m=report["overall_g2"]
    location = report.get("locate_recall")
    recall = location["overall"]["recall"] if location else None
    sampling = report.get("sampling") or {}
    if sampling.get("mode") == "fixed_manifest":
        sample_note = f"; fixed manifest sha256={sampling['manifest_sha256']}; n={sampling['selected_n']}"
    elif sampling:
        sample_note = f"; seed={sampling['seed']}; n={sampling['selected_n']}"
    else:
        sample_note = ""
    line = "| " + " | ".join([datetime.now(timezone(timedelta(hours=9))).date().isoformat(),commit,classifier,*values,
        _percent(m["character_accuracy"]),_percent(recall),_percent(m["auto_pass_precision"]),"N/A",
        f"{report['mode']}; oracle={report['oracle']}; validation; status-based attribution{sample_note}"]) + " |\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:
        handle.write(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description="G2 evaluator v1: indexed validation sampling and Top-3 localization recall")
    parser.add_argument("--manifest", default="algorithm/testset/manifest.csv")
    parser.add_argument("--index", default=r"D:\keshe2\algorithm\data\index\ccpd_index.csv", help="CCPD index; relative images resolve beside CSV")
    parser.add_argument("--split", choices=("test", "val", "manifest-val"), default="test",
                        help="val samples only the index val split; manifest-val uses a supplied validation manifest")
    parser.add_argument("--n", type=int, help="explicitly resample this many validation rows (legacy exploratory mode)")
    parser.add_argument("--seed", type=int, help="seed for explicit exploratory resampling")
    parser.add_argument("--subset", action="append", help="restrict val sampling to a CCPD subset; repeatable")
    parser.add_argument("--dev-hard", action="store_true", help="use the fixed 7x300 development hard manifest")
    parser.add_argument("--fixed-manifest", help="override the versioned daily list (still validated against the CCPD index)")
    parser.add_argument("--exclude-manifest", default=r"D:\keshe2\algorithm\data\index\testset_manifest_v1.csv",
                        help="primary manifest of frozen test images excluded from val/dev-hard sampling; golden-only images are also excluded when golden_manifest.csv exists")
    parser.add_argument("--locate-recall", action="store_true", help="Top-3 original-space bbox IoU >= 0.5; requires --index")
    parser.add_argument("--locate-only", action="store_true", help="measure Top-3 localization without loading a classifier")
    add_backend_args(parser)
    parser.add_argument("--out", default="algorithm/out/eval")
    parser.add_argument("--scoreboard", metavar="TEMP_COPY", help="append validation row to a temporary scoreboard copy")
    parser.add_argument("--oracle", action="store_true", help="use quad, true color and length; skip locate/rectify")
    parser.add_argument("--stage-trace", help="write compact per-row JSONL trace for paired stage analysis")
    parser.add_argument("--reuse-stage-trace", help="inherit Top-3 labels from a matching normal trace without rerunning locate")
    args = parser.parse_args(argv)
    if args.scoreboard:
        if Path(args.scoreboard).resolve() == (ALGORITHM_DIR/"reports"/"scoreboard.md").resolve():
            parser.error("本批禁止修改正式 scoreboard.md；请传临时副本路径")
        if args.split not in {"val", "manifest-val"}:
            parser.error("--scoreboard 只允许验证清单：--split val 或 manifest-val")
    if (args.split == "val" or args.locate_recall or args.locate_only) and not Path(args.index).is_file():
        parser.error("需要 ccpd_index.csv；请用 --index 指定已有索引")
    if args.dev_hard and args.split != "val":
        parser.error("--dev-hard requires --split val")
    if args.locate_only and (args.scoreboard or args.oracle):
        parser.error("--locate-only cannot be combined with --scoreboard or --oracle")
    if args.stage_trace and not (args.locate_recall or args.reuse_stage_trace):
        parser.error("--stage-trace requires --locate-recall or --reuse-stage-trace")
    if args.reuse_stage_trace and not args.oracle:
        parser.error("--reuse-stage-trace is intended for --oracle runs")
    if args.subset and args.dev_hard:
        parser.error("--subset cannot be combined with --dev-hard")
    if args.fixed_manifest and (args.n is not None or args.seed is not None or args.subset):
        parser.error("--fixed-manifest cannot be combined with explicit resampling options")
    if args.split == "val" and not Path(args.exclude_manifest).is_file():
        parser.error(f"validation exclusion manifest not found: {args.exclude_manifest}")
    if args.n is not None and args.n < 1:
        parser.error("--n must be positive")
    try:
        cfg = config_for_args(args)
        manifest, sampling, selected = args.manifest, None, None
        if args.split == "val":
            fixed_path = Path(args.fixed_manifest) if args.fixed_manifest else (
                FIXED_HARD_MANIFEST if args.dev_hard else FIXED_VAL_MANIFEST)
            if args.n is None and args.seed is None and not args.subset:
                selected, sampling = fixed_manifest_samples(
                    args.index, fixed_path, purpose="dev-hard" if args.dev_hard else "val")
            else:
                exclusion_manifests = [Path(args.exclude_manifest)]
                golden_exclusion = ALGORITHM_DIR / "tests" / "data" / "golden" / "golden_manifest.csv"
                if golden_exclusion.is_file() and golden_exclusion.resolve() not in {
                        path.resolve() for path in exclusion_manifests}:
                    exclusion_manifests.append(golden_exclusion)
                count = args.n if args.n is not None else DEFAULT_VALIDATION_N
                seed = args.seed if args.seed is not None else DEFAULT_SEED
                if args.dev_hard:
                    selected, sampling = sample_dev_hard(args.index, n=count, seed=seed,
                                                         exclude_manifest=exclusion_manifests)
                    sampling["mode"] = "dev-hard-exploratory"
                else:
                    selected, sampling = sample_validation(args.index, n=count, seed=seed,
                        exclude_manifest=exclusion_manifests,
                        subsets=set(args.subset) if args.subset else None)
                    sampling["mode"] = "val-exploratory"
            manifest = Path(args.out)/"validation_manifest.csv"
            write_manifest(manifest, [sample.row for sample in selected], check_files=False)
        if args.locate_only:
            report = evaluate_locate_only(manifest, args.index, args.out, sampling=sampling,
                                          truth_samples=selected, config=cfg, config_arg=args.config)
            print_report(report)
            print(f"validation_manifest={manifest}")
            print(f"report={Path(args.out)/'report.json'}")
            return 0
        recognizer = make_recognizer(args, samples_dir=Path(manifest).resolve().parent, cfg=cfg)
        report = evaluate(manifest, recognizer, args.out, oracle=args.oracle,
                          index=args.index if args.split == "val" or args.locate_recall else None,
                          locate_recall=args.locate_recall, sampling=sampling,
                          stage_trace=args.stage_trace, reuse_stage_trace=args.reuse_stage_trace,
                          truth_samples=selected, effective_config=cfg)
        report["evaluation_config"] = effective_config_info(cfg, args.config)
        print_report(report)
        if args.split == "val":
            print(f"validation_manifest={manifest}")
        print(f"report={Path(args.out)/'report.json'}")
        if args.scoreboard:
            append_scoreboard(args.scoreboard, report, split="val", classifier=mode_name(args))
            print(f"scoreboard={args.scoreboard}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
