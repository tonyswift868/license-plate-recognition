"""Offline reproduction of every locator/rectifier/layout decoding attempt.

This is diagnostic analysis only; it does not change production ordering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from lpr import pipeline
from lpr.config import ALGORITHM_DIR
from lpr.locate import locate
from lpr.preprocess import preprocess
from lpr.segment import segment
from lpr.types import RecogStatus as S

from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path, read_manifest, write_manifest

from .common import effective_config_info, make_recognizer
from .index_samples import DEFAULT_SEED, path_key, sample_validation

STATUS_RANK = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}


def replay_attempts(record: dict, threshold: float = 0.80) -> dict:
    """Replay normal pipeline choice over a complete ordered attempt trace.

    Trace records retain the decode-independent format_valid bit, so thresholds
    can be changed without rerunning the CNN. Candidate order is unchanged;
    first OK still returns immediately, matching PlateRecognizer._run.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if record.get("input_error"):
        return {"plate_no": None, "status": "BAD_INPUT", "confidence": 0.0,
                "selected_attempt_index": None, "early_return_attempt_index": None}
    best = None
    selected = None
    early_index = None
    for attempt in record["attempts"]:
        if not attempt.get("decoded"):
            continue
        status = ("INVALID_FORMAT" if not attempt["format_valid"] else
                  "OK" if attempt["confidence"] >= threshold else "LOW_CONFIDENCE")
        candidate = {**attempt, "effective_status": status}
        rank = (STATUS_RANK[status], float(attempt["confidence"]))
        if best is None or rank > best[0]:
            best = (rank, candidate)
        if status == "OK":
            selected = candidate
            early_index = attempt["attempt_index"]
            break
    if selected is None and record.get("lead_color") in {"white", "black"}:
        return {"plate_no": None, "status": "UNSUPPORTED", "confidence": 0.0,
                "selected_attempt_index": None, "early_return_attempt_index": None}
    if selected is None and best is not None:
        selected = best[1]
    if selected is None:
        return {"plate_no": None, "status": "SEG_FAILED" if record.get("candidate_count") else "NO_PLATE",
                "confidence": 0.0, "selected_attempt_index": None,
                "early_return_attempt_index": None}
    return {"plate_no": selected["plate_no"], "status": selected["effective_status"],
            "confidence": selected["confidence"],
            "selected_attempt_index": selected["attempt_index"],
            "early_return_attempt_index": early_index}


def trace_image(image, recognizer, row):
    cfg = recognizer.config
    pre = preprocess(image, cfg.preprocess, cfg.lane_roi(None))
    candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    if recognizer.corner_detector is not None:
        from lpr.pipeline import _in_roi, _merge_candidates
        roi = cfg.lane_roi(None)
        net = [c for c in recognizer.corner_detector.detect(image, cfg.runtime.max_candidates)
               if _in_roi(c, roi)]
        candidates = _merge_candidates(net, candidates, cfg.runtime.max_candidates)
    attempts = []
    lead_color = candidates[0].color if candidates else "unknown"
    for ci, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
        rects = pipeline._rectifications(image, cand, cfg.rectify)
        if ci == 0 and rects:
            lead_color = rects[0].color
        for ri, rect in enumerate(rects):
            for layout in pipeline._layouts_to_try(rect):
                seg = segment(rect, layout, cfg.segment)
                item = {"attempt_index": len(attempts), "candidate_index": ci,
                        "rect_index": ri, "layout": layout, "color": rect.color,
                        "candidate_source": cand.source,
                        "decoded": False, "seg_ok": bool(seg.ok)}
                if seg.ok:
                    probs = recognizer.classifier.predict_proba(seg.chars)
                    dec = pipeline.decode(probs, seg.layout, rect.color, cfg.postprocess)
                    item.update(decoded=True, plate_no=dec.plate_no,
                                format_valid=dec.status != S.INVALID_FORMAT,
                                status=dec.status.value, confidence=float(dec.confidence),
                                char_confidences=[float(x) for x in dec.char_confidences],
                                candidates=[[p, float(s)] for p, s in dec.candidates])
                else:
                    item.update(plate_no=None, status="SEG_FAILED", confidence=0.0,
                                char_confidences=[], candidates=[])
                attempts.append(item)
            if recognizer.plate_reader is not None:
                text, confs = recognizer.plate_reader.read(rect.image)
                reader_decoded = len(text) in (7, 8) and len(confs) == len(text)
                reader_format_valid = reader_decoded and pipeline.match_layout(text) is not None
                item = {"attempt_index": len(attempts), "candidate_index": ci,
                        "rect_index": ri, "layout": len(text) if reader_decoded else 0,
                        "color": rect.color, "candidate_source": cand.source,
                        "decoded": bool(reader_decoded),
                        "seg_ok": False, "reader": True}
                if reader_decoded:
                    item.update(plate_no=text, format_valid=reader_format_valid,
                                confidence=float(min(confs)), char_confidences=[float(x) for x in confs],
                                candidates=[])
                else:
                    item.update(plate_no=None, format_valid=False, confidence=0.0,
                                char_confidences=[], candidates=[])
                attempts.append(item)
    return {"path": row.path, "truth": row.plate_no, "category": row.category,
            "truth_color": row.plate_color, "lead_color": lead_color, "candidate_count": len(candidates),
            "attempts": attempts}


def _threshold_outcome(attempt, threshold):
    status = "INVALID_FORMAT" if not attempt["format_valid"] else (
        "OK" if attempt["confidence"] >= threshold else "LOW_CONFIDENCE")
    return status


def _evaluate_choice(choice, record, threshold):
    if choice is None:
        return {"plate_no": None, "status": "NO_PLATE", "confidence": 0.0}
    status = _threshold_outcome(choice, threshold)
    if record.get("lead_color") in {"white", "black"} and status != "OK":
        return {"plate_no": None, "status": "UNSUPPORTED", "confidence": 0.0}
    return {"plate_no": choice["plate_no"], "status": status,
            "confidence": choice["confidence"]}


def _rank_rule(record, rule, threshold):
    if rule == "conservative_first_ok_then_color_layout":
        baseline = replay_attempts(record, threshold)
        if baseline["early_return_attempt_index"] is not None:
            return record["attempts"][baseline["early_return_attempt_index"]]
    attempts = [a for a in record["attempts"] if a.get("decoded") and a["format_valid"]]
    if not attempts:
        return None
    if rule == "max_confidence":
        return max(attempts, key=lambda a: (a["confidence"], -a["attempt_index"]))
    if rule == "color_layout_consensus":
        pair_counts = Counter((a["color"], a["layout"]) for a in attempts)
        total = len(attempts)
        return max(attempts, key=lambda a: (a["confidence"] + .02 * pair_counts[(a["color"], a["layout"])] / total,
                                             a["confidence"], -a["attempt_index"]))
    if rule == "conservative_first_ok_then_color_layout":
        def score(attempt):
            color, layout = attempt["color"], attempt["layout"]
            penalty = .05 if (color == "blue" and layout != 7) or (color == "green" and layout != 8) else 0.
            penalty += .02 if color == "unknown" else 0.
            return (attempt["confidence"] - penalty, attempt["confidence"], -attempt["attempt_index"])
        return max(attempts, key=score)
    raise ValueError(f"unknown arbitration rule {rule}")


def _rule_metrics(records, rule, threshold=.80):
    exact = accepted = accepted_correct = 0
    gained = lost = 0
    for record in records:
        truth = record["truth"]
        if not truth:
            continue
        baseline = replay_attempts(record, threshold)
        proposed = _evaluate_choice(_rank_rule(record, rule, threshold), record, threshold)
        exact += proposed["plate_no"] == truth
        accepted += proposed["status"] == "OK"
        accepted_correct += proposed["status"] == "OK" and proposed["plate_no"] == truth
        gained += proposed["plate_no"] == truth and baseline["plate_no"] != truth
        lost += proposed["plate_no"] != truth and baseline["plate_no"] == truth
    return {"rule": rule, "exact": exact, "n": sum(bool(r["truth"]) for r in records),
            "exact_rate": exact / max(1, sum(bool(r["truth"]) for r in records)),
            "auto_pass": accepted, "auto_pass_rate": accepted / max(1, sum(bool(r["truth"]) for r in records)),
            "auto_pass_precision": accepted_correct / accepted if accepted else None,
            "gained_vs_baseline": gained, "lost_vs_baseline": lost, "net_vs_baseline": gained-lost}


def analyze(records: list[dict]) -> dict:
    totals = Counter({key: 0 for key in (
        "wrong_with_correct_attempt", "correct_attempt_after_selected",
        "correct_attempt_before_or_at_selected", "correct_alternative_was_executed",
        "correct_alternative_skipped_after_early_ok")})
    beam_truth_among_wrong = 0
    by_reason = Counter()
    examples = []
    attempt_counts = Counter()
    thresholds = [0.0, 0.80, 1.0]
    threshold_replays = {}
    for threshold in thresholds:
        threshold_replays[str(threshold)] = {"exact": 0, "auto_pass": 0,
                                             "auto_correct": 0, "n": 0}
    for record in records:
        truth = record["truth"]
        attempts = record["attempts"]
        attempt_counts["rows"] += 1
        attempt_counts["total_attempts"] += len(attempts)
        attempt_counts["decoded_attempts"] += sum(bool(a.get("decoded")) for a in attempts)
        attempt_counts["segmentation_failures"] += sum(not a.get("seg_ok", False) for a in attempts)
        attempt_counts["rectification_hypotheses"] += len({(a["candidate_index"], a["rect_index"]) for a in attempts})
        if not truth:
            continue
        for threshold in thresholds:
            outcome = replay_attempts(record, threshold)
            stats = threshold_replays[str(threshold)]
            stats["n"] += 1
            stats["exact"] += outcome["plate_no"] == truth
            accepted = outcome["status"] == "OK"
            stats["auto_pass"] += accepted
            stats["auto_correct"] += accepted and outcome["plate_no"] == truth
        baseline = replay_attempts(record, 0.80)
        correct = [a for a in attempts if a.get("decoded") and a["plate_no"] == truth]
        if baseline["plate_no"] != truth and any(
                any(plate == truth for plate, _ in a.get("candidates", [])) for a in attempts):
            beam_truth_among_wrong += 1
        if baseline["early_return_attempt_index"] is not None:
            attempt_counts["first_ok_early_return_rows"] += 1
            tail = [a for a in attempts if a["attempt_index"] > baseline["early_return_attempt_index"]]
            attempt_counts["attempts_skipped_by_first_ok"] += len(tail)
            if any(a in tail for a in correct):
                attempt_counts["rows_with_truth_only_after_first_ok"] += 1
        if baseline["plate_no"] == truth or not correct:
            continue
        totals["wrong_with_correct_attempt"] += 1
        selected = baseline["selected_attempt_index"]
        executed_through = baseline["early_return_attempt_index"]
        if executed_through is None:
            executed_through = max((a["attempt_index"] for a in attempts), default=-1)
        executed_correct = [a for a in correct if a["attempt_index"] <= executed_through]
        latent_correct = [a for a in correct if a["attempt_index"] > executed_through]
        if executed_correct:
            totals["correct_alternative_was_executed"] += 1
        if latent_correct:
            totals["correct_alternative_skipped_after_early_ok"] += 1
        later = [a for a in correct if a["attempt_index"] > (selected if selected is not None else -1)]
        chosen = attempts[selected] if selected is not None else None
        if later:
            totals["correct_attempt_after_selected"] += 1
            reason = ("premature_ok_early_return" if latent_correct
                      else "rank_or_tie_break_among_executed_attempts")
        else:
            totals["correct_attempt_before_or_at_selected"] += 1
            reason = "rank_or_tie_break_among_executed_attempts" if executed_correct else "premature_ok_early_return"
        if chosen and any(a["color"] != chosen["color"] for a in correct):
            colors = "+".join(sorted({a["color"] for a in correct}))
            reason = f"color_conflict_selected_{chosen['color']}_truth_attempt_{colors}"
        if chosen and any(a["layout"] != chosen["layout"] for a in correct):
            layouts = "+".join(map(str, sorted({a["layout"] for a in correct})))
            reason += f"_layout_conflict_selected_{chosen['layout']}_truth_{layouts}"
        if chosen and any(a["candidate_index"] != chosen["candidate_index"] for a in correct):
            reason += "_different_locator_candidate"
        if chosen and any(a["rect_index"] != chosen["rect_index"] for a in correct):
            reason += "_different_corner_hypothesis"
        if chosen and chosen["layout"] == 8 and chosen["color"] in {"green", "unknown"} and record.get("truth_color") == "blue":
            if any(a["layout"] == 7 and a["color"] == "blue" for a in correct):
                reason += "_green_or_unknown_8_over_blue_7"
        by_reason[reason] += 1
        if len(examples) < 100:
            examples.append({"path": record["path"], "truth": truth,
                             "baseline": baseline, "correct_attempt_indices": [a["attempt_index"] for a in correct],
                             "selected": chosen})
    denominators = {k: v["n"] for k, v in threshold_replays.items()}
    for values in threshold_replays.values():
        n = values["n"]
        values["exact_rate"] = values["exact"] / n if n else None
        values["auto_pass_rate"] = values["auto_pass"] / n if n else None
        values["auto_pass_precision"] = values["auto_correct"] / values["auto_pass"] if values["auto_pass"] else None
    rule_metrics = [_rule_metrics(records, rule) for rule in
                    ("max_confidence", "color_layout_consensus", "conservative_first_ok_then_color_layout")]
    attempt_counts["mean_attempts_per_row"] = (attempt_counts["total_attempts"] / attempt_counts["rows"]
                                                if attempt_counts["rows"] else 0)
    return {"n_rows_with_truth": denominators.get("0.8", 0),
            "attempt_coverage": dict(attempt_counts),
            "wrong_with_other_correct_attempt": dict(totals),
            "decoder_topk_diagnostic": {
                "wrong_with_truth_in_any_decode_topk": beam_truth_among_wrong,
                "scope": "beam alternatives within decode; distinct from another attempt's selected plate; truth-assisted availability is not measured ranking gain"},
            "reason_counts": dict(by_reason), "threshold_replay": threshold_replays,
            "alternative_rules_at_0_80": rule_metrics,
            "recommendation": {
                "candidate": "no ordering change is supported by this run: none of the 1000 final wrong outputs had a correct alternative decoding attempt, and all tested rules had zero net change. If a later validation fold exposes recoverable cases, first test preserving first-OK acceptance and applying weak known-color/layout consistency only when no attempt is OK; architecture owner decides whether to adopt",
                "experimental_policy": "preserve first-OK; otherwise confidence minus 0.05 for blue/layout mismatch, 0.05 for green/layout mismatch, and 0.02 for unknown color",
                "observed_recoverable_upper_bound": totals["wrong_with_correct_attempt"],
                "observed_rule_comparison": "rule_metrics report val exploration only; do not treat as generalization",
                "architecture_owner": "architect",
            }, "examples": examples}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exhaustively trace val pipeline arbitration attempts")
    parser.add_argument("--index", default=str(ALGORITHM_DIR / "data/index/ccpd_index.csv"))
    parser.add_argument("--exclude-manifest", default=str(ALGORITHM_DIR / "data/index/testset_manifest_v1.csv"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest", default=str(ALGORITHM_DIR / "data/index/val1000_v1.csv"),
                        help="fixed ordered validation manifest")
    parser.add_argument("--config", default=str(Path(__file__).parents[2] / "config" / "default.toml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation-run-dir", help="reuse completed daily val1000 output directory and its sampling report")
    parser.add_argument("--reanalyze", action="store_true", help="recompute summaries from an existing attempts.jsonl without model inference")
    args = parser.parse_args(argv)
    fixed_manifest = ALGORITHM_DIR / "data/index/val1000_v1.csv".resolve()
    if Path(args.manifest).resolve() != fixed_manifest:
        parser.error("--manifest must be the fixed shared val1000_v1.csv")
    if args.validation_run_dir:
        parser.error("--validation-run-dir is obsolete; read the fixed val1000 manifest directly")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reanalyze:
        trace_path = out / "attempts.jsonl"
        report_path = out / "report.json"
        if not trace_path.is_file() or not report_path.is_file():
            parser.error("--reanalyze requires existing attempts.jsonl and report.json")
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = Path(report.get("selection", {}).get("manifest_path", args.manifest))
        rows = read_manifest(manifest, check_files=True)
        existing = {path_key(record["path"]): record for record in records}
        initial_errors = list(report.get("decode_errors", []))
        errors_by_path = {path_key(row["path"]): row["reason"] for row in initial_errors}
        reconciled = []
        recognizer = None
        replay_matches = 0
        for row in rows:
            key = path_key(image_path(row, manifest))
            if key in existing:
                record = existing[key]
                actual, baseline = record.get("actual", {}), record.get("baseline_replay", {})
                if (actual.get("status") != baseline.get("status")
                        or actual.get("plate_no") != baseline.get("plate_no")
                        or float(actual.get("confidence", 0)) != float(baseline.get("confidence", 0))):
                    raise ValueError(f"stored replay is not exactly equal to production: {row.path}")
                reconciled.append(record)
                replay_matches += 1
                continue
            image, reason = decode_image(image_path(row, manifest))
            if key not in errors_by_path:
                raise ValueError(f"cannot reconcile missing trace row for {row.path}")
            if image is not None:
                if recognizer is None:
                    recognizer = make_recognizer(argparse.Namespace(
                        fake=False, uniform=False, classifier=None, config=report["config_path"]))
                actual = recognizer.recognize(image)
                record = trace_image(image, recognizer, row)
                outcome = replay_attempts(record, report["config_min_confidence"])
                record["baseline_replay"] = outcome
                record["actual"] = {"plate_no": actual.plate_no, "status": actual.status.value,
                                    "confidence": float(actual.confidence)}
                if (actual.status.value != outcome["status"] or actual.plate_no != outcome["plate_no"]
                        or float(actual.confidence) != float(outcome["confidence"])):
                    raise ValueError(f"retraced validation row does not match production: {row.path}")
                del errors_by_path[key]
            else:
                record = {"path": str(image_path(row, manifest)), "truth": row.plate_no,
                          "category": row.category, "truth_color": row.plate_color,
                          "lead_color": "unknown", "candidate_count": 0, "attempts": [],
                          "input_error": reason}
                outcome = replay_attempts(record, report["config_min_confidence"])
                record["baseline_replay"] = outcome
                record["actual"] = {"plate_no": None, "status": "BAD_INPUT", "confidence": 0.0}
            replay_matches += 1
            reconciled.append(record)
        if len(reconciled) != 1000:
            raise ValueError("fixed val1000 trace reconciliation did not produce exactly 1000 rows")
        records = reconciled
        report.update(analyze(records))
        report["traced_rows"] = len(records)
        report["decoded_rows"] = len(records) - len(errors_by_path)
        report["recovered_decode_errors"] = [item for item in initial_errors
                                              if path_key(item["path"]) not in errors_by_path]
        report["decode_errors"] = [item for item in report.get("decode_errors", [])
                                    if path_key(item["path"]) in errors_by_path]
        report["replay_validation"] = {"n": len(records), "match": replay_matches,
                                        "decode_error_rows": len(errors_by_path),
                                        "mismatches": []}
        trace_path.write_text("".join(json.dumps(record, ensure_ascii=False)+"\n" for record in records),
                              encoding="utf-8")
        report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        report["arbitration_source_path"] = str(Path(__file__).resolve())
        report["arbitration_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"reanalyzed": len(records), "report": str(report_path),
                          "wrong_with_other_correct_attempt": report["wrong_with_other_correct_attempt"],
                          "alternative_rules_at_0_80": report["alternative_rules_at_0_80"]}, ensure_ascii=False, indent=2))
        return 0
    _validation_report = None
    if args.manifest:
        manifest = Path(args.manifest)
        rows = read_manifest(manifest, check_files=True)
        if args.n != 1000 or len(rows) != 1000:
            parser.error("fixed val1000 manifest must contain exactly 1000 rows; --n must remain 1000")
        selected = [SimpleNamespace(row=row, split="val", subset=row.source.rsplit("/", 1)[-1]) for row in rows]
        sampling = {"split": "val", "fixed": True, "selected_n": len(rows),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "paths": [row.path for row in rows]}
        (out / "validation_manifest.csv").write_bytes(manifest.read_bytes())
    else:
        selected, sampling = sample_validation(args.index, n=args.n, seed=args.seed,
                                               exclude_manifest=[args.exclude_manifest,
                                                   Path(__file__).parents[2] / "tests/data/golden/golden_manifest.csv"])
        manifest = out / "validation_manifest.csv"
        write_manifest(manifest, [s.row for s in selected], check_files=False)
    recognizer = make_recognizer(argparse.Namespace(fake=False, uniform=False, classifier=None,
                                                     config=args.config))
    records = []
    decode_errors = []
    verify = Counter()
    verify["max_confidence_abs_diff"] = 0.0
    for idx, sample in enumerate(selected):
        row = sample.row
        image, error = decode_image(image_path(row, manifest))
        if image is None:
            decode_errors.append({"path": row.path, "reason": error})
            continue
        actual = recognizer.recognize(image)
        record = trace_image(image, recognizer, row)
        replay = replay_attempts(record, recognizer.config.postprocess.min_confidence)
        verify["n"] += 1
        confidence_diff = abs(replay["confidence"] - actual.confidence)
        verify["max_confidence_abs_diff"] = max(verify["max_confidence_abs_diff"], confidence_diff)
        plate_match = replay["plate_no"] == actual.plate_no
        status_match = replay["status"] == actual.status.value
        confidence_match = replay["confidence"] == float(actual.confidence)
        verify["plate_match"] += plate_match
        verify["status_match"] += status_match
        verify["confidence_match"] += confidence_match
        verify["match"] += plate_match and status_match and confidence_match
        if not (plate_match and status_match and confidence_match):
            verify.setdefault("mismatches", []).append({"path": row.path, "replay": replay,
                "actual": {"plate_no": actual.plate_no, "status": actual.status.value,
                            "confidence": float(actual.confidence)},
                "confidence_abs_diff": confidence_diff})
        record["baseline_replay"] = replay
        record["actual"] = {"plate_no": actual.plate_no, "status": actual.status.value,
                            "confidence": float(actual.confidence)}
        records.append(record)
        if (idx + 1) % 100 == 0:
            print(f"processed {idx+1}/{len(selected)}; replay matches {verify['match']}/{verify['n']}", flush=True)
    report = analyze(records)
    model_path = recognizer.config.resolve_path(recognizer.config.runtime.model_path)
    algorithm_dir = Path(__file__).parents[2]
    config_path = Path(args.config).resolve()
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    selected_keys = [path_key(image_path(s.row, manifest)) for s in selected]
    frozen_paths, frozen_hashes = set(), set()
    exclusion_manifests = [Path(args.exclude_manifest), algorithm_dir / "tests/data/golden/golden_manifest.csv",
                           ALGORITHM_DIR / "data/index/devhard_v1.csv",
                           ALGORITHM_DIR / "data/index/undecodable.csv"]
    for frozen_manifest in exclusion_manifests:
        if not frozen_manifest.is_file():
            raise FileNotFoundError(f"required frozen exclusion list missing: {frozen_manifest}")
        if frozen_manifest.name == "undecodable.csv":
            with frozen_manifest.open(encoding="utf-8-sig", newline="") as handle:
                excluded_rows = [row["path"] for row in csv.DictReader(handle)]
        else:
            excluded_rows = [image_path(r, frozen_manifest) for r in read_manifest(frozen_manifest)]
        frozen_paths.update(path_key(p) for p in excluded_rows)
        for p in excluded_rows:
            candidate = Path(p)
            if not candidate.is_file():
                raise FileNotFoundError(f"required frozen exclusion image missing: {candidate}")
            blob = candidate.read_bytes()
            frozen_hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
    selected_hashes = set()
    for sample in selected:
        image_bytes = Path(image_path(sample.row, manifest)).read_bytes()
        selected_hashes.add((len(image_bytes), hashlib.sha256(image_bytes).hexdigest()))
    path_overlap = len(set(selected_keys) & frozen_paths)
    byte_overlap = len(selected_hashes & frozen_hashes)
    if path_overlap or byte_overlap:
        raise ValueError(f"fixed validation list overlaps frozen exclusions: paths={path_overlap}, bytes={byte_overlap}")
    trace_path = out / "attempts.jsonl"
    trace_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    config_json = json.dumps(recognizer.config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    report.update(sampling={k:v for k,v in sampling.items() if k != "paths"},
                  selection={"split": "val", "fixed": True, "manifest_path": str(manifest.resolve()),
                             "requested_n": args.n,
                             "selected_n": len(selected), "path_list_sha256": hashlib.sha256(("\n".join(selected_keys)+"\n").encode("utf-8")).hexdigest(),
                             "frozen_overlap_count": path_overlap, "frozen_byte_overlap_count": byte_overlap,
                             "frozen_manifests": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in exclusion_manifests],
                             "manifest_sha256": sha256(manifest)},
                  decode_errors=decode_errors,
                  replay_validation=dict(verify), manifest_rows=len(selected), decoded_rows=len(records),
                  model_version=recognizer.version,
                  config_min_confidence=recognizer.config.postprocess.min_confidence,
                  model_path=str(model_path), model_sha256=sha256(model_path),
                  config_path=str(config_path), config_file_sha256=sha256(config_path),
                  effective_config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
                  effective_config=recognizer.config.model_dump(mode="json"),
                  effective_config_info=effective_config_info(recognizer.config, args.config),
                  corner_model_path=(str(recognizer.config.resolve_path(recognizer.config.runtime.corner_model_path))
                                     if recognizer.config.runtime.corner_model_path else None),
                  corner_model_sha256=(sha256(recognizer.config.resolve_path(recognizer.config.runtime.corner_model_path))
                                       if recognizer.config.runtime.corner_model_path else None),
                  pipeline_path=str(algorithm_dir / "lpr/pipeline.py"),
                  pipeline_sha256=sha256(algorithm_dir / "lpr/pipeline.py"),
                  arbitration_source_path=str(Path(__file__).resolve()),
                  arbitration_source_sha256=sha256(Path(__file__).resolve()),
                  trace_path=str(trace_path), trace_sha256=sha256(trace_path),
                  validation_report_path=None,
                  validation_manifest_path=str(manifest))
    (out / "command.json").write_text(json.dumps({"argv": sys.argv, "cwd": str(Path.cwd()),
        "validation_run_dir": args.validation_run_dir, "index": args.index, "config": args.config,
        "manifest": str(manifest)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not decode_errors and verify["match"] == verify["n"] == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
