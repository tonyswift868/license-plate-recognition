"""Choose the E1 confidence threshold from exhaustive validation attempts only.

The arbitration trace must include every decode attempt, including attempts the
production pipeline would skip after its first OK result. Replaying those
attempts is necessary because changing the threshold changes early exit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

from lpr import PlateRecognizer
from lpr.config import ALGORITHM_DIR, load_config

from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path, read_manifest
from lpr_tools.eval.arbitration import replay_attempts
from lpr_tools.eval.index_samples import path_key


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_thresholds(records: list[dict]) -> list[float]:
    """All distinct decision boundaries and their just-above neighbors."""
    values = {0.0, 0.80, 1.0}
    for record in records:
        for attempt in record["attempts"]:
            confidence = float(attempt["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("attempt confidence must be finite and in [0, 1]")
            values.add(confidence)
            above = math.nextafter(confidence, math.inf)
            if above <= 1:
                values.add(above)
    return sorted(values)


def scan(records: list[dict]) -> tuple[list[dict], dict]:
    if not records:
        raise ValueError("validation attempt trace is empty")
    if any(not record.get("truth") for record in records):
        raise ValueError("each validation record needs a nonempty truth plate")
    rows = []
    for threshold in candidate_thresholds(records):
        accepted = correct = 0
        for record in records:
            result = replay_attempts(record, threshold)
            if result["status"] == "OK":
                accepted += 1
                correct += result["plate_no"] == record["truth"]
        rows.append({"threshold": threshold, "n": len(records), "accepted": accepted,
                     "correct_accepted": correct,
                     "auto_pass_rate": accepted / len(records),
                     "auto_pass_precision": correct / accepted if accepted else None})
    return rows, recommend(rows, .99)


def recommend(rows: list[dict], precision_floor: float) -> dict | None:
    eligible = [row for row in rows if row["auto_pass_precision"] is not None
                and row["auto_pass_precision"] >= precision_floor]
    return max(eligible, key=lambda row: (row["accepted"], -row["threshold"])) if eligible else None


def _metrics_at(records: list[dict], threshold: float) -> dict:
    accepted = correct = 0
    for record in records:
        result = replay_attempts(record, threshold)
        if result["status"] == "OK":
            accepted += 1
            correct += result["plate_no"] == record["truth"]
    return {"threshold": threshold, "n": len(records), "accepted": accepted,
            "correct_accepted": correct, "auto_pass_rate": accepted / len(records),
            "auto_pass_precision": correct / accepted if accepted else None}


def _stable_threshold(records: list[dict], recommendation: dict) -> dict | None:
    """Find an interior point with the exact same accepted set as a brittle cut."""
    lower = float(recommendation["threshold"])
    upper_values = [float(attempt["confidence"]) for record in records
                    for attempt in record["attempts"]
                    if attempt.get("decoded") and attempt.get("format_valid")
                    and float(attempt["confidence"]) > lower]
    if not upper_values:
        return None
    upper = min(upper_values)
    if not lower < upper:
        return None
    threshold = lower + (upper - lower) / 2
    if not lower < threshold < upper:
        return None
    metrics = _metrics_at(records, threshold)
    if (metrics["accepted"] != recommendation["accepted"]
            or metrics["correct_accepted"] != recommendation["correct_accepted"]
            or metrics["auto_pass_precision"] is None or metrics["auto_pass_precision"] < .99):
        return None
    metrics.update({"raw_cut_threshold": lower, "next_coverage_boundary": upper,
                    "selection_equivalent": True})
    return metrics


def _svg(rows: list[dict], recommendation: dict | None) -> str:
    """Write a self-contained SVG plot without a runtime plotting dependency."""
    width, height = 900, 600
    left, right, top, bottom = 88, 35, 48, 76
    pw, ph = width - left - right, height - top - bottom
    points = [(row["auto_pass_rate"], row["auto_pass_precision"])
              for row in rows if row["auto_pass_precision"] is not None]
    if not points:
        points = [(0., 0.)]
    def xy(x, y):
        return left + x * pw, top + (1 - y) * ph
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(lambda p: xy(*p), points))
    grid = []
    for tick in range(0, 11):
        v = tick / 10
        x, y = xy(v, v)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+pw}" y2="{y:.2f}" stroke="#ddd"/>')
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+ph}" stroke="#eee"/>')
        grid.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end">{v:.0%}</text>')
        grid.append(f'<text x="{x:.2f}" y="{top+ph+22}" text-anchor="middle">{v:.0%}</text>')
    marker = ""
    if recommendation is not None:
        x, y = xy(recommendation["auto_pass_rate"], recommendation["auto_pass_precision"])
        marker = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#d62728"/><text x="{x+10:.2f}" y="{y-10:.2f}">recommended</text>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13" fill="#222">'
            '<text x="450" y="25" text-anchor="middle" font-size="18">E1 validation precision vs automatic pass rate</text>'
            + "".join(grid) + f'<polyline fill="none" stroke="#1769aa" stroke-width="2" points="{polyline}"/>{marker}'
            + f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#222"/>'
            + f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#222"/>'
            + f'<text x="{left+pw/2}" y="{height-20}" text-anchor="middle">Automatic pass rate (coverage)</text>'
            + f'<text transform="translate(20 {top+ph/2}) rotate(-90)" text-anchor="middle">Automatic pass precision</text></g></svg>')


def validate_provenance(records: list[dict], trace_path: Path, source: dict) -> dict:
    selection = source.get("selection", {})
    n = len(records)
    replay = source.get("replay_validation", {})
    if selection.get("split") != "val" or selection.get("fixed") is not True or selection.get("selected_n") != 1000:
        raise ValueError("source trace must use the fixed val1000 manifest")
    decode_errors = source.get("decode_errors", [])
    if (source.get("manifest_rows") != n or source.get("decoded_rows", 0) + len(decode_errors) != n
            or n != 1000 or source.get("traced_rows", n) != n):
        raise ValueError("source trace must account for all 1000 fixed validation rows")
    error_paths = {path_key(item["path"]) for item in decode_errors}
    trace_error_paths = {path_key(record["path"]) for record in records if record.get("input_error")}
    if (error_paths != trace_error_paths or selection.get("frozen_overlap_count") != 0
            or any(not record.get("truth") for record in records)):
        raise ValueError("source contains unaccounted decode errors or frozen-set overlap")
    if selection.get("frozen_byte_overlap_count") != 0:
        raise ValueError("source fixed validation list has byte-identical frozen exclusions")
    if replay.get("n") != n or replay.get("match") != n:
        raise ValueError("source run must verify every replay against the real pipeline")
    if Path(source.get("trace_path", "")).resolve() != trace_path.resolve():
        raise ValueError("source report points to a different trace")
    trace_digest = sha256(trace_path)
    if source.get("trace_sha256") != trace_digest:
        raise ValueError("trace SHA256 differs from the source report")
    manifest_path = trace_path.parent / "validation_manifest.csv"
    if not manifest_path.is_file() or source.get("selection", {}).get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("validation manifest SHA256 differs from the source report")
    sample_paths = [path_key(record["path"]) for record in records]
    fixed_manifest = ALGORITHM_DIR / "data/index/val1000_v1.csv"
    if (Path(selection.get("manifest_path", "")).resolve() != fixed_manifest.resolve()
            or not fixed_manifest.is_file()
            or selection.get("manifest_sha256") != sha256(fixed_manifest)
            or manifest_path.read_bytes() != fixed_manifest.read_bytes()):
        raise ValueError("trace does not use the current frozen val1000 manifest")
    expected_list_hash = hashlib.sha256(("\n".join(sample_paths) + "\n").encode("utf-8")).hexdigest()
    if len(set(sample_paths)) != n or selection.get("path_list_sha256") != expected_list_hash:
        raise ValueError("trace paths are duplicated or do not match the source selection hash")
    frozen = [ALGORITHM_DIR / "data/index/testset_manifest_v1.csv",
              ALGORITHM_DIR / "tests" / "data" / "golden" / "golden_manifest.csv",
              ALGORITHM_DIR / "data/index/devhard_v1.csv",
              ALGORITHM_DIR / "data/index/undecodable.csv"]
    declared_manifests = {path_key(item["path"]): item["sha256"]
                          for item in selection.get("frozen_manifests", [])}
    frozen_paths = set()
    frozen_hashes = set()
    for manifest in frozen:
        if not manifest.is_file():
            raise ValueError(f"frozen exclusion manifest missing: {manifest}")
        if declared_manifests.get(path_key(manifest.resolve())) != sha256(manifest):
            raise ValueError(f"frozen exclusion manifest identity mismatch: {manifest}")
        if manifest.name == "undecodable.csv":
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                exclusion_rows = [row["path"] for row in csv.DictReader(handle)]
        else:
            exclusion_rows = [image_path(row, manifest) for row in read_manifest(manifest)]
        frozen_paths.update(path_key(path) for path in exclusion_rows)
        for path in exclusion_rows:
            path = Path(path)
            if not path.is_file():
                raise ValueError(f"frozen exclusion image missing: {path}")
            blob = path.read_bytes()
            frozen_hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
    sample_hashes = set()
    for record in records:
        blob = Path(record["path"]).read_bytes()
        sample_hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
    if set(sample_paths) & frozen_paths or sample_hashes & frozen_hashes:
        raise ValueError("validation trace overlaps frozen evaluation data by path or bytes")
    for record in records:
        actual, baseline = record.get("actual"), record.get("baseline_replay")
        if not actual or not baseline or actual.get("status") != baseline.get("status") \
                or actual.get("plate_no") != baseline.get("plate_no") \
                or float(actual.get("confidence", 0)) != float(baseline.get("confidence", 0)):
            raise ValueError("stored baseline differs from the exact pipeline replay")
    config, model = Path(source["config_path"]), Path(source["model_path"])
    config_hash, model_hash = source.get("config_file_sha256"), source.get("model_sha256")
    pipeline = ALGORITHM_DIR / "lpr" / "pipeline.py"
    corner = Path(source.get("corner_model_path", ""))
    if config_hash != sha256(config) or model_hash != sha256(model):
        raise ValueError("source config/model hash mismatch")
    expected_config = load_config(ALGORITHM_DIR / "config" / "default.toml")
    expected_corner = expected_config.resolve_path(expected_config.runtime.corner_model_path)
    if (config.resolve() != (ALGORITHM_DIR / "config" / "default.toml").resolve()
            or config_hash != sha256(ALGORITHM_DIR / "config" / "default.toml")
            or model.resolve() != expected_config.resolve_path(expected_config.runtime.model_path).resolve()
            or model_hash != sha256(expected_config.resolve_path(expected_config.runtime.model_path))
            or corner.resolve() != expected_corner.resolve()
            or not corner.is_file() or source.get("corner_model_sha256") != sha256(corner)
            or source.get("pipeline_path") != str(pipeline) or source.get("pipeline_sha256") != sha256(pipeline)):
        raise ValueError("trace was not produced by current CNN v2/corner default config")
    if "cnn_v2" not in source.get("model_version", "").lower():
        raise ValueError("source recognizer is not CNN v2")
    effective = source.get("effective_config")
    if effective != expected_config.model_dump(mode="json"):
        raise ValueError("trace effective configuration differs from current default.toml")
    canonical = json.dumps(effective, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != source.get("effective_config_sha256"):
        raise ValueError("effective configuration identity does not match the trace report")
    model_info = source.get("effective_config_info", {}).get("models", {})
    for key, expected in (("classifier", source["model_sha256"]), ("corner", source["corner_model_sha256"])):
        if not model_info.get(key, {}).get("exists") or model_info[key].get("sha256") != expected:
            raise ValueError(f"effective {key} model identity does not match the trace report")
    return {"split": "val", "seed": 42, "n": n, "frozen_overlap": 0,
            "trace_sha256": trace_digest, "path_list_sha256": expected_list_hash,
            "config_file_sha256": config_hash,
            "effective_config_sha256": source.get("effective_config_sha256"),
            "model_sha256": model_hash, "corner_model_sha256": source["corner_model_sha256"],
            "pipeline_sha256": source["pipeline_sha256"]}


def verify_runtime(records: list[dict], manifest: Path, threshold: float) -> dict:
    """Run the real recognizer at the proposed cutoff on the same ordered val rows."""
    rows = read_manifest(manifest)
    if len(rows) != len(records):
        raise ValueError("runtime verification manifest and attempt trace row counts differ")
    recognizer = PlateRecognizer.from_config(str(ALGORITHM_DIR / "config" / "default.toml"),
                                               postprocess__min_confidence=threshold)
    mismatches = []
    mismatch_count = 0
    accepted = correct = 0
    for i, (row, record) in enumerate(zip(rows, records, strict=False)):
        if path_key(image_path(row, manifest)) != path_key(record["path"]):
            raise ValueError(f"runtime verification path/order mismatch at row {i}")
        image, reason = decode_image(image_path(row, manifest))
        actual = (recognizer.recognize(image) if image is not None else
                  recognizer.recognize_bytes(Path(image_path(row, manifest)).read_bytes()))
        replay = replay_attempts(record, threshold)
        matches = (actual.status.value == replay["status"] and actual.plate_no == replay["plate_no"]
                   and float(actual.confidence) == float(replay["confidence"]))
        if not matches:
            mismatch_count += 1
            if len(mismatches) < 100:
                mismatches.append({"row": i, "path": row.path, "truth": record["truth"],
                                   "actual": {"plate_no": actual.plate_no, "status": actual.status.value,
                                              "confidence": float(actual.confidence), "decode_error": reason},
                                   "replay": replay})
        if actual.status.value == "OK":
            accepted += 1
            correct += actual.plate_no == record["truth"]
    return {"threshold": threshold, "n": len(records), "accepted": accepted,
            "correct_accepted": correct,
            "auto_pass_rate": accepted / len(records),
            "auto_pass_precision": correct / accepted if accepted else None,
            "mismatch_count": mismatch_count,
            "mismatches_first_100": mismatches,
            "verified": mismatch_count == 0}


def run(trace_path: str | Path, out: str | Path, *, runtime_check: bool = False) -> dict:
    trace_path, out = Path(trace_path), Path(out)
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_path = trace_path.parent / "report.json"
    if not source_path.is_file():
        raise ValueError("adjacent arbitration report.json is required to validate trace provenance")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    provenance = validate_provenance(records, trace_path, source)
    rows, recommendation = scan(records)
    conservative = recommend(rows, .995)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "curve.svg").write_text(_svg(rows, recommendation), encoding="utf-8")
    model = Path(source["model_path"])
    config = Path(source["config_path"])
    manifest = trace_path.parent / "validation_manifest.csv"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ALGORITHM_DIR.parent,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    conservative = recommend(rows, .995)
    runtime_validation = None
    stable_adjustment = None
    runtime_validations = {}
    if runtime_check:
        for name, selected, floor in (("precision_99", recommendation, .99),
                                      ("precision_995", conservative, .995)):
            if selected is None:
                runtime_validations[name] = None
                continue
            checked = verify_runtime(records, manifest, selected["threshold"])
            stable = None
            if (not checked["verified"] or checked["auto_pass_precision"] is None
                    or checked["auto_pass_precision"] < floor
                    or checked["accepted"] != selected["accepted"]
                    or checked["correct_accepted"] != selected["correct_accepted"]):
                stable = _stable_threshold(records, selected)
                if stable is None:
                    runtime_validations[name] = checked
                    raise ValueError(f"runtime disagrees at {name} threshold and no equivalent stable cut exists")
                checked = verify_runtime(records, manifest, stable["threshold"])
                if (not checked["verified"] or checked["auto_pass_precision"] is None
                        or checked["auto_pass_precision"] < floor
                        or checked["accepted"] != stable["accepted"]
                        or checked["correct_accepted"] != stable["correct_accepted"]):
                    runtime_validations[name] = checked
                    raise ValueError(f"runtime disagrees at stable {name} threshold")
                selected = stable
            runtime_validations[name] = checked
            if name == "precision_99":
                recommendation = selected
                runtime_validation = checked
            else:
                conservative = selected
        (out / "runtime_verification.json").write_text(
            json.dumps(runtime_validations, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"dataset": "validation only", "n": len(records), "provenance": provenance,
              "commit": commit,
              "trace": str(trace_path.resolve()), "trace_sha256": sha256(trace_path),
              "manifest": str(manifest.resolve()) if manifest.is_file() else None,
              "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
              "config": str(config), "config_sha256": sha256(config),
              "model": str(model), "model_sha256": sha256(model),
              "corner_model": source["corner_model_path"],
              "corner_model_sha256": source["corner_model_sha256"],
              "effective_config_sha256": source["effective_config_sha256"],
              "effective_config": source["effective_config"],
              "model_identities": source["effective_config_info"]["models"],
              "fixed_manifest_sha256": source["selection"]["manifest_sha256"],
              "recovered_decode_errors": source.get("recovered_decode_errors", []),
              "initial_threshold": source["config_min_confidence"],
              "default_0_80": next((row for row in rows if row["threshold"] == .80), None),
              "candidate_threshold_count": len(rows),
              "recommendation": recommendation,
              "conservative_recommendation": conservative,
              "stable_threshold_adjustment": stable_adjustment,
              "runtime_validation": runtime_validation,
              "runtime_validations": runtime_validations,
              "recommendation_caveat": "Empirical precision on this 1000-image validation sample; it does not guarantee 99% precision on future traffic.",
              "artifacts": {"curve_csv": "curve.csv", "curve_svg": "curve.svg"}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if recommendation is None:
        recommendation_line = "No threshold met empirical precision >=99%."
    else:
        recommendation_line = (f"Recommend min_confidence={recommendation['threshold']:.17g}: "
                               f"{recommendation['correct_accepted']}/{recommendation['accepted']} accepted plates "
                               f"correct ({recommendation['auto_pass_precision']:.4%} precision), "
                               f"{recommendation['accepted']}/{recommendation['n']} auto-pass "
                               f"({recommendation['auto_pass_rate']:.2%} coverage).")
    default = report["default_0_80"]
    conservative_line = ("none" if conservative is None else
                         f"{conservative['threshold']:.17g}: {conservative['correct_accepted']}/"
                         f"{conservative['accepted']} accepted correct, "
                         f"{conservative['auto_pass_precision']:.4%} precision, "
                         f"{conservative['auto_pass_rate']:.2%} auto-pass")
    (out / "summary.md").write_text(
        "# E1 confidence threshold scan\n\n"
        f"{recommendation_line}\n\n"
        f"At the current 0.80 threshold: {default['correct_accepted']}/{default['accepted']} "
        f"accepted plates correct ({default['auto_pass_precision']:.2%} precision), "
        f"{default['accepted']}/{default['n']} auto-pass ({default['auto_pass_rate']:.2%} coverage).\n\n"
        f"Conservative precision >=99.5% recommendation: {conservative_line}.\n\n"
        f"Scanned {report['candidate_threshold_count']} exact confidence cuts on {report['n']} "
        "fixed val images. The recommendation is an empirical result on this sample; "
        "it does not guarantee 99% precision on future traffic. The architecture owner decides whether "
        "to adopt it.\n\n"
        "Runtime verification on the same ordered images: "
        f"{('completed; mismatches=' + str(runtime_validation['mismatch_count'])) if runtime_validation else 'not requested'}.\n\n"
        "See [curve.svg](curve.svg), [curve.csv](curve.csv), [report.json](report.json), and "
        "[runtime_verification.json](runtime_verification.json).\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scan E1 min_confidence on an exhaustive val attempt trace")
    parser.add_argument("--trace", required=True, help="attempts.jsonl in arbitration's --out directory")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--verify-runtime", action="store_true",
                        help="run the actual CNN pipeline on the same val manifest at the recommended threshold")
    args = parser.parse_args(argv)
    report = run(args.trace, args.out, runtime_check=args.verify_runtime)
    command = (f".venv\\Scripts\\python.exe -m lpr_tools.eval.e1_threshold --trace {args.trace} "
               f"--out {args.out}" + (" --verify-runtime" if args.verify_runtime else ""))
    report["command"] = command
    out = Path(args.out)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
