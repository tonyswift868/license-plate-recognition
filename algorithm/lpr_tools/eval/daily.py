
"""Run the M2 daily validation, golden regression, and latency checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from lpr.config import ALGORITHM_DIR, DEFAULT_CONFIG_PATH, load_config

from lpr_tools.data.manifest import read_manifest, write_manifest

from .common import effective_config_info, model_identity
from .index_samples import FIXED_HARD_MANIFEST, FIXED_VAL_MANIFEST

ROOT = ALGORITHM_DIR.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INDEX = ALGORITHM_DIR / "data/index/ccpd_index.csv"
SCOREBOARD_HEADER = ("日期", "commit", "分类器", "清晰", "夜间", "倾斜", "污损", "字符准确率",
                     "Top-3 定位召回", "自动通过精度", "P95 ms", "备注")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_config(config_path: Path | None = None):
    """Resolve one config identity for a daily run and its child processes."""
    selected = Path(config_path or os.environ.get("LPR_CONFIG") or DEFAULT_CONFIG_PATH).resolve()
    cfg = load_config(selected)
    if cfg.postprocess.arbitration == "agree" and not cfg.runtime.plate_reader_model_path:
        raise ValueError("agree daily requires a configured plate reader; runtime would otherwise use first_ok")
    info = effective_config_info(cfg, str(selected))
    info.update(config_sha256=_sha(selected), classifier=cfg.runtime.classifier,
                classifier_model=model_identity(cfg, "model_path"))
    return selected, info


def _child_env(config_path: Path, base=None):
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(ALGORITHM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["LPR_CONFIG"] = str(config_path)
    return env


def _verify_marker_config(marker):
    saved = marker.get("effective_config")
    if not saved:
        raise ValueError("daily marker has no effective config identity; refusing to mix run configurations")
    selected_config, current = _effective_config(Path(saved["config_path"]))
    if current != saved:
        raise ValueError("effective config/model identity changed since daily validation; refusing to mix configurations")
    fixed = marker.get("fixed_manifests", {})
    if fixed:
        for name, manifest_path in (("val", FIXED_VAL_MANIFEST), ("dev-hard", FIXED_HARD_MANIFEST)):
            identity = fixed.get(name, {})
            if not manifest_path.is_file() or _sha(manifest_path) != identity.get("sha256"):
                raise ValueError(f"fixed {name} manifest changed since daily validation; refusing to resume")
        run_dir = Path(marker.get("run_dir", ""))
        for rel, expected in marker.get("run_manifests", {}).items():
            artifact = run_dir / rel
            if not artifact.is_file() or _sha(artifact) != expected:
                raise ValueError(f"daily run manifest changed since validation: {artifact}")
    return selected_config


def _attribution(reports):
    keys = ("network_candidate_selected_count", "network_candidate_selected_correct",
            "crnn_selected_count", "crnn_selected_correct", "reader_invoked_images",
            "reader_invoked_calls", "corner_invoked_images", "corner_invoked_calls")
    totals = {key: 0 for key in keys}
    for report in reports:
        values = report.get("network_attribution", {})
        for key in keys:
            totals[key] += int(values.get(key, 0) or 0)
    for stem in ("network_candidate_selected", "crnn_selected"):
        count = totals[f"{stem}_count"]
        totals[f"{stem}_accuracy"] = totals[f"{stem}_correct"] / count if count else None
    return totals


def _run(command: list[str], cwd: Path, log: Path, *, env=None) -> subprocess.CompletedProcess:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env)
    log.write_text("$ " + subprocess.list2cmdline(command) + "\n\n" + result.stdout +
                   ("\n[stderr]\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    return result


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value):
    return "N/A" if value is None else f"{value:.1%}"


def _decision_policy(info):
    pp = info.get("effective_config", {}).get("postprocess", {})
    mode = pp.get("arbitration", "first_ok")
    return {"arbitration": mode, "g1_p95_limit_ms": 200 if mode == "agree" else 300,
            "auto_pass_precision_min": .99 if mode == "agree" else None,
            "thresholds": {k: pp.get(k) for k in ("min_confidence", "agree_min_confidence",
                "solo_min_confidence_segment", "solo_min_confidence_reader", "disagree_prefer")}}


def _acceptance(info, val, reports, bench, golden_pass, golden_correct, input_errors):
    policy = _decision_policy(info)
    accepted = sum(r["overall_g2"].get("auto_pass", 0) for r in reports.values())
    correct = sum(r["overall_g2"].get("auto_pass_correct", 0) for r in reports.values())
    vm = val["overall_g2"]
    vp, vc = vm.get("auto_pass", 0), vm.get("auto_pass_correct", 0)
    precision_ok = policy["arbitration"] != "agree" or (
        vp > 0 and accepted > 0 and vc * 100 >= vp * 99 and correct * 100 >= accepted * 99)
    passed = (golden_pass and golden_correct is not None and golden_correct >= 13
              and bench["wall_time"]["p95_ms"] <= policy["g1_p95_limit_ms"]
              and len(reports) == 7 and precision_ok)
    return {"policy": policy, "dual_precision_pass": precision_ok,
            "hard_auto_pass": accepted, "hard_auto_pass_correct": correct,
            "hard_auto_pass_precision": correct / accepted if accepted else None,
            "status": "INPUT_ERRORS" if input_errors else ("PASS" if passed else "PARTIAL_FAILURE")}


def _scoreboard_line(date, commit, val, hard, bench, effective_config=None):
    cats = {name: [] for name in ("clear", "night", "tilt", "dirty")}
    cats["clear"].append(val["categories"]["clear"])
    for subset, category in (("ccpd_db", "night"), ("ccpd_tilt", "tilt"),
                             ("ccpd_rotate", "tilt"), ("ccpd_blur", "dirty"),
                             ("ccpd_challenge", "dirty"), ("ccpd_weather", "dirty")):
        if subset in hard.get("reports", {}):
            cats[category].append(hard["reports"][subset]["overall_g2"])
    def accuracy(parts):
        n = sum(part.get("n", 0) for part in parts)
        correct = sum(part.get("correct", 0) for part in parts)
        return correct / n if n else None
    metrics = val["overall_g2"]
    location = val.get("locate_recall")
    recall = location["overall"]["recall"] if location else None
    p95 = (bench.get("wall_time") or {}).get("p95_ms")
    attr = val.get("network_attribution", {})
    input_errors = val.get("decode_error_count", 0) + sum(
        report.get("decode_error_count", 0) for report in hard.get("reports", {}).values())
    note = (f"daily; val n={metrics['n']}; auto-pass={_fmt(metrics.get('auto_pass_rate'))}; "
            f"CRNN selected={attr.get('crnn_selected_count', 0)} "
            f"accuracy={_fmt(attr.get('crnn_selected_accuracy'))}; "
            f"network candidate selected={attr.get('network_candidate_selected_count', 0)}; "
            f"Top-3 classic-only; dev-hard 7x300; fixed-list decode errors={input_errors}; "
            f"golden={hard.get('golden_correct', 'N/A')}/20")
    policy = _decision_policy(effective_config or {})
    hard_n = sum(r["overall_g2"].get("n", 0) for r in hard.get("reports", {}).values())
    hard_pass = sum(r["overall_g2"].get("auto_pass", 0) for r in hard.get("reports", {}).values())
    hard_correct = sum(r["overall_g2"].get("auto_pass_correct", 0) for r in hard.get("reports", {}).values())
    note += (f"; arbitration={policy['arbitration']}; hard auto-pass={_fmt(hard_pass / hard_n if hard_n else None)}"
             f" precision={_fmt(hard_correct / hard_pass if hard_pass else None)}; "
             f"G1 limit={policy['g1_p95_limit_ms']}ms")
    if policy["arbitration"] == "agree":
        note += "; " + ", ".join(f"{k}={v}" for k, v in policy["thresholds"].items())
    mode = val.get("mode", "cnn")
    classifier = "cnn v1" if "char_cnn_v1" in mode else mode
    values = [date, commit, classifier,
              *(_fmt(accuracy(cats[name])) for name in ("clear", "night", "tilt", "dirty")),
              _fmt(metrics.get("character_accuracy")), _fmt(recall),
              _fmt(metrics.get("auto_pass_precision")),
              "N/A" if p95 is None else f"{p95:.1f}", note]
    return "| " + " | ".join(values) + " |"


def _error_analysis(trace: Path):
    errors, confusions, missing, extra = Counter(), Counter(), Counter(), Counter()
    no_prediction = 0
    if not trace.is_file():
        return {"position_errors": {}, "confusions_top20": []}
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truth, pred = row.get("truth", ""), row.get("prediction", "")
            if truth and not pred:
                no_prediction += 1
            for pos in range(max(len(truth), len(pred))):
                a = truth[pos] if pos < len(truth) else "<missing>"
                b = pred[pos] if pos < len(pred) else "<missing>"
                if a != b:
                    errors[str(pos + 1)] += 1
                    if a == "<missing>":
                        extra[str(pos + 1)] += 1
                    elif b == "<missing>":
                        missing[str(pos + 1)] += 1
                    else:
                        confusions[(a, b)] += 1
    return {"position_errors": dict(sorted(errors.items(), key=lambda kv: int(kv[0]))),
            "confusions_top20": [{"truth": a, "prediction": b, "count": n}
                                 for (a, b), n in confusions.most_common(20)],
            "no_prediction": no_prediction,
            "missing_positions": dict(sorted(missing.items(), key=lambda kv: int(kv[0]))),
            "extra_positions": dict(sorted(extra.items(), key=lambda kv: int(kv[0])))}


def _comparable(previous: dict, current: dict) -> bool:
    return all(previous.get(k) == current.get(k) for k in
               ("val_sampling", "hard_sampling", "metric_definition"))


def _identity_changes(previous: dict, current: dict):
    if not previous or not current:
        return {"available": False, "models": {},
                "note": "Previous or current run lacks detailed model identity metadata; do not infer unchanged weights."}
    models = {}
    for name, old_model, new_model in (
            ("classifier", previous.get("classifier_model", {}), current.get("classifier_model", {})),
            ("corner", previous.get("models", {}).get("corner", {}), current.get("models", {}).get("corner", {})),
            ("plate_reader", previous.get("models", {}).get("plate_reader", {}), current.get("models", {}).get("plate_reader", {}))):
        old_hash, new_hash = old_model.get("sha256"), new_model.get("sha256")
        old_path, new_path = old_model.get("path"), new_model.get("path")
        models[name] = {"changed": old_hash != new_hash or old_path != new_path,
                        "previous_path": old_path, "current_path": new_path,
                        "previous_sha256": old_hash, "current_sha256": new_hash}
    return {"available": True, "models": models,
            "config_path_changed": previous.get("config_path") != current.get("config_path"),
            "config_sha256_changed": previous.get("config_sha256") != current.get("config_sha256"),
            "effective_config_changed": previous.get("effective_config_sha256") != current.get("effective_config_sha256")}


def _compare(current: dict, root: Path):
    runs = sorted((p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file()),
                  key=lambda p: (p / "summary.json").stat().st_mtime, reverse=True) if root.exists() else []
    if runs:
        path = runs[0]
        old = _read_json(path / "summary.json")
        old_key, now_key = old.get("comparison_key", {}), current["comparison_key"]
        identities = _identity_changes(old.get("effective_config", {}), current.get("effective_config", {}))
        if _comparable(old_key, now_key):
            diffs = []
            for subset, now in current["metrics"].items():
                before = old.get("metrics", {}).get(subset, {})
                for metric in ("plate_accuracy", "character_accuracy", "auto_pass_rate", "auto_pass_precision"):
                    a, b = before.get(metric), now.get(metric)
                    if a is not None and b is not None:
                        delta = b - a
                        diffs.append({"subset": subset, "metric": metric, "previous": a,
                                      "current": b, "delta": delta,
                                      "alert": delta < -0.01 - 1e-9})
            return {"previous_run": path.name, "comparable": True,
                    "model_changed": old_key.get("model_sha256") != now_key.get("model_sha256"),
                    "config_changed": old_key.get("config_sha256") != now_key.get("config_sha256"),
                    "effective_config_changed": old_key.get("effective_config_sha256") != now_key.get("effective_config_sha256"),
                    "identity_changes": identities,
                    "comparisons": diffs,
                    "regressions_over_1pp": [x for x in diffs if x["alert"]]}
        reasons = [key for key in ("val_sampling", "hard_sampling", "metric_definition")
                   if old_key.get(key) != now_key.get(key)]
        return {"previous_run": path.name, "comparable": False,
                "note": "Latest completed run is not comparable: " + ", ".join(reasons),
                "model_changed": old_key.get("model_sha256") != now_key.get("model_sha256"),
                "config_changed": old_key.get("config_sha256") != now_key.get("config_sha256"),
                "effective_config_changed": old_key.get("effective_config_sha256") != now_key.get("effective_config_sha256"),
                "identity_changes": identities}
    return {"previous_run": None, "comparable": False,
            "note": "No previous run with matching sampling and metric definition."}


def _subset_metrics(report):
    result = {"val/overall": report["overall_g2"]}
    result.update({f"val/{k}": v for k, v in report["categories"].items()})
    result.update({f"val/subset/{k}": v for k, v in report.get("subsets", {}).items()})
    return result


def _partition_hard_rows(rows):
    pools = {}
    for row in rows:
        pools.setdefault(row["source"].split("/")[-1], []).append(row)
    return pools


def daily(out_root: Path, index: Path = INDEX, *, date: str | None = None, commit: str | None = None,
          defer_g1: bool = False, config_path: Path | None = None):
    if not PYTHON.is_file():
        raise FileNotFoundError(f"project interpreter missing: {PYTHON}")
    if not index.is_file():
        raise FileNotFoundError(f"CCPD index missing: {index}")
    commit = commit or subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip()
    date = date or datetime.now().astimezone().date().isoformat()
    out = out_root / f"{date}-{commit}"
    if out.exists():
        out = out_root / f"{date}-{commit}-{datetime.now().astimezone():%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    config_path, effective_config = _effective_config(config_path)
    env = _child_env(config_path)
    val_out = out / "val1000"
    val_cmd = [str(PYTHON), "-m", "lpr_tools.eval.eval_accuracy", "--split", "val",
               "--fixed-manifest", str(FIXED_VAL_MANIFEST), "--locate-recall", "--index", str(index), "--out", str(val_out),
               "--stage-trace", str(out / "val1000" / "trace.jsonl"), "--config", str(config_path)]
    val_run = _run(val_cmd, ROOT, out / "logs" / "val1000.log", env=env)
    if val_run.returncode:
        raise RuntimeError(f"val1000 failed ({val_run.returncode}); see {out / 'logs' / 'val1000.log'}")
    val_report = _read_json(val_out / "report.json")
    if val_report.get("manifest_rows") != 1000 or val_report.get("scored_rows") != 1000:
        raise RuntimeError("fixed val1000 manifest did not score all 1000 rows; see report.json")

    hard_reports = {}
    hard_status = {}
    hard_out = out / "dev-hard"
    hard_out.mkdir(parents=True, exist_ok=True)
    sample_all = hard_out / "all-hard-manifest.csv"
    sample_meta = hard_out / "sampling.json"
    command = [str(PYTHON), "-c",
               "from pathlib import Path; import sys,json; from lpr_tools.eval.index_samples import fixed_manifest_samples; "
               "from lpr_tools.data.manifest import write_manifest; "
               "selected,meta=fixed_manifest_samples(sys.argv[1],sys.argv[2],purpose='dev-hard'); "
               "write_manifest(sys.argv[3],[s.row for s in selected],check_files=False); "
               "Path(sys.argv[4]).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')",
               str(index), str(FIXED_HARD_MANIFEST), str(sample_all), str(sample_meta)]
    sample_run = _run(command, ROOT, out / "logs" / "sample-hard.log", env=env)
    if sample_run.returncode:
        raise RuntimeError(f"dev-hard sampling failed; see {out / 'logs' / 'sample-hard.log'}")
    hard_rows = read_manifest(sample_all, check_files=False)
    hard_pools = {}
    for row in hard_rows:
        hard_pools.setdefault(row.source.split("/")[-1], []).append(row)
    for subset in ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather"):
        target = out / "dev-hard" / subset
        target.mkdir(parents=True, exist_ok=True)
        rows = hard_pools.get(subset, [])
        if len(rows) != 300:
            raise RuntimeError(f"expected 300 rows for {subset}, got {len(rows)}")
        manifest = target / "subset_manifest.csv"
        write_manifest(manifest, rows, check_files=False)
        evaluate_cmd = [str(PYTHON), "-m", "lpr_tools.eval.eval_accuracy", "--manifest",
                        str(manifest), "--split", "manifest-val", "--out", str(target),
                        "--stage-trace", str(target / "trace.jsonl"),
                        "--config", str(config_path)]
        run = _run(evaluate_cmd, ROOT, out / "logs" / f"dev-{subset}.log", env=env)
        hard_status[subset] = run.returncode
        if run.returncode == 0:
            report = _read_json(target / "report.json")
            if report.get("manifest_rows") != 300 or report.get("scored_rows") != 300:
                raise RuntimeError(f"{subset} did not score all 300 fixed rows")
            hard_reports[subset] = report
    if len(hard_reports) != 7:
        raise RuntimeError("fixed dev-hard evaluation incomplete")
    val_input_errors = val_report.get("decode_error_count", 0)
    hard_input_errors = sum(report.get("decode_error_count", 0) for report in hard_reports.values())
    input_error_count = val_input_errors + hard_input_errors

    golden = _run([str(PYTHON), "-m", "pytest", "algorithm/tests/test_golden.py", "-q",
                   "--basetemp", str(out / "pytest-temp")], ROOT,
                  out / "logs" / "golden.log", env=env)
    golden_pass = golden.returncode == 0 and "passed" in golden.stdout and "skipped" not in golden.stdout
    golden_count = None
    if golden_pass:
        count_code = ("import csv; from lpr.pipeline import PlateRecognizer; from lpr.io_utils import imread; "
                      "from algorithm.tests.conftest import GOLDEN_DIR; "
                      "rows=list(csv.DictReader((GOLDEN_DIR/'labels.csv').open(encoding='utf-8',newline=''))); "
                      "rec=PlateRecognizer.from_config(); correct=0; "
                      "exec(\"for row in rows:\\n image=imread(GOLDEN_DIR/row['file'])\\n result=rec.recognize(image)\\n correct += (result.plate_no is None) if row['category']=='np' else (result.plate_no==row['plate_no'])\"); "
                      "print(f'{correct}/{len(rows)}')")
        count_run = _run([str(PYTHON), "-c", count_code], ROOT,
                         out / "logs" / "golden-count.log", env=env)
        if count_run.returncode == 0:
            golden_count = int(count_run.stdout.strip().splitlines()[-1].split("/")[0])
        else:
            golden_pass = False
    if defer_g1:
        marker = {"date": date, "commit": commit, "effective_config": effective_config,
                  "run_dir": str(out.resolve()),
                  "fixed_manifests": {"val": {"path": str(FIXED_VAL_MANIFEST), "sha256": _sha(FIXED_VAL_MANIFEST)},
                                      "dev-hard": {"path": str(FIXED_HARD_MANIFEST), "sha256": _sha(FIXED_HARD_MANIFEST)}},
                  "run_manifests": {"val1000/validation_manifest.csv": _sha(val_out / "validation_manifest.csv"),
                                    "dev-hard/all-hard-manifest.csv": _sha(sample_all)},
                  "hard_sampling": _read_json(sample_meta),
                  "golden_exit": golden.returncode,
                  "golden_pass": golden_pass, "golden_correct": golden_count,
                  "input_errors": input_error_count,
                  "val_input_errors": val_input_errors, "hard_input_errors": hard_input_errors,
                  "hard_status": hard_status, "hard_reports": sorted(hard_reports),
                  "g1": "deferred; run daily.py --g1-only <run-dir>"}
        (out / "pre-g1.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        return marker
    g1 = _run([str(PYTHON), "-m", "lpr_tools.eval.bench_latency", "--manifest",
               "algorithm/testset/manifest.csv", "--n", "100", "--size", "1280x720",
               "--config", str(config_path)],
              ROOT, out / "logs" / "g1.log", env=env)
    if g1.returncode:
        raise RuntimeError(f"G1 failed ({g1.returncode}); see {out / 'logs' / 'g1.log'}")
    bench = json.loads(g1.stdout)
    if not bench.get("acceptance_eligible") or bench.get("requested") != 320 or bench.get("decoded") != 320:
        raise RuntimeError("G1 input was not a fully decoded 320-image real-model run")
    hard_status["golden_exit"] = golden.returncode
    hard_status["golden_pass"] = golden_pass
    hard_status["golden_correct"] = golden_count
    hard_metrics = {subset: report["overall_g2"] for subset, report in hard_reports.items()}
    metrics = _subset_metrics(val_report)
    metrics.update({f"dev-hard/{k}": v for k, v in hard_metrics.items()})
    val_sampling = _read_json(val_out / "report.json").get("sampling", {})
    hard_sampling = _read_json(sample_meta)
    comparison_key = {"model_sha256": effective_config["classifier_model"]["sha256"],
                      "config_sha256": effective_config["config_sha256"],
                      "effective_config_sha256": effective_config["effective_config_sha256"],
                      "val_sampling": {"mode": "fixed_manifest", "manifest_sha256": val_sampling.get("manifest_sha256"),
                                       "paths_sha256": val_sampling.get("paths_sha256")},
                      "hard_sampling": {"mode": "fixed_manifest", "manifest_sha256": hard_sampling.get("manifest_sha256"),
                                        "paths_sha256": hard_sampling.get("paths_sha256"),
                                        "subsets": list(hard_reports)},
                      "metric_definition": "plate positional metrics v1"}
    comparison_key["val_sampling"]["run_manifest_sha256"] = _sha(val_out / "validation_manifest.csv")
    comparison_key["hard_sampling"]["run_manifest_sha256"] = _sha(sample_all)
    comparison = _compare({"comparison_key": comparison_key, "metrics": metrics,
                           "effective_config": effective_config}, out_root)
    line = _scoreboard_line(date, commit, val_report, {**hard_status, "reports": hard_reports}, bench, effective_config)
    acceptance = _acceptance(effective_config, val_report, hard_reports, bench,
                             golden_pass, golden_count, input_error_count)
    error_analysis = _error_analysis(out / "val1000" / "trace.jsonl")
    summary = {"date": date, "commit": commit, "directory": str(out), "scoreboard_header": SCOREBOARD_HEADER,
               "effective_config": effective_config,
               "network_attribution": _attribution([val_report, *hard_reports.values()]),
               "scoreboard_row": line, "metrics": metrics, "hard_status": hard_status,
               "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip(),
               "input_errors": input_error_count,
               "val_input_errors": val_input_errors, "hard_input_errors": hard_input_errors,
               "hard_completed": sorted(hard_reports), "golden_exit": golden.returncode,
               "g1": bench, "comparison_key": comparison_key, "comparison": comparison,
               **error_analysis,
               "acceptance": acceptance, "status": acceptance["status"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scoreboard-row.md").write_text("| " + " | ".join(SCOREBOARD_HEADER) + " |\n" +
                                           "| " + " | ".join("---" for _ in SCOREBOARD_HEADER) + " |\n" +
                                           line + "\n", encoding="utf-8")
    (out / "subsets.md").write_text(_format_subsets(metrics), encoding="utf-8")
    (out / "comparison.md").write_text(_format_comparison(comparison), encoding="utf-8")
    (out / "errors.md").write_text(_format_errors(summary), encoding="utf-8")
    (out / "summary.md").write_text(_format_summary(summary), encoding="utf-8")
    return summary


def recover_hard(run_dir: Path):
    """Retry only hard-set evaluations from the saved selected 7x300 manifest."""
    run_dir = run_dir.resolve()
    marker_path = run_dir / "pre-g1.json"
    marker = _read_json(marker_path)
    config_path = _verify_marker_config(marker)
    samples = read_manifest(run_dir / "dev-hard" / "all-hard-manifest.csv", check_files=False)
    pools = {}
    for row in samples:
        pools.setdefault(row.source.split("/")[-1], []).append(row)
    subsets = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather")
    reports, statuses = {}, {}
    env = _child_env(config_path)
    for subset in subsets:
        target = run_dir / "dev-hard" / subset
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "subset_manifest.csv"
        if manifest.exists():
            manifest.replace(target / "subset_manifest_attempt1.csv")
        rows = pools.get(subset, [])
        if len(rows) != 300:
            raise ValueError(f"saved sample has {len(rows)} rows for {subset}; expected 300")
        write_manifest(manifest, rows, check_files=False)
        log = run_dir / "logs" / f"dev-{subset}.log"
        if log.exists():
            log.replace(run_dir / "logs" / f"dev-{subset}-attempt1.log")
        command = [str(PYTHON), "-m", "lpr_tools.eval.eval_accuracy", "--manifest",
                   str(manifest), "--split", "manifest-val", "--out", str(target),
                   "--stage-trace", str(target / "trace.jsonl"),
                   "--config", str(config_path)]
        run = _run(command, ROOT, log, env=env)
        statuses[subset] = run.returncode
        if run.returncode == 0:
            report = _read_json(target / "report.json")
            if report.get("manifest_rows") != 300 or report.get("scored_rows") != 300:
                raise RuntimeError(f"{subset}: expected all 300 fixed rows to be scored")
            reports[subset] = report
    marker["hard_status"] = statuses
    marker["hard_reports"] = sorted(reports)
    marker["hard_recovery"] = {"reused_manifest": True, "selected_n": len(samples),
                               "decoded_reports": len(reports)}
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return marker


def finish_g1(run_dir: Path):
    run_dir = run_dir.resolve()
    marker = _read_json(run_dir / "pre-g1.json")
    if not marker:
        raise ValueError(f"no pre-G1 daily run in {run_dir}")
    config_path = _verify_marker_config(marker)
    env = _child_env(config_path)
    g1 = _run([str(PYTHON), "-m", "lpr_tools.eval.bench_latency", "--manifest",
               "algorithm/testset/manifest.csv", "--n", "100", "--size", "1280x720",
               "--config", str(config_path)],
              ROOT, run_dir / "logs" / "g1.log", env=env)
    if g1.returncode:
        raise RuntimeError(f"G1 failed ({g1.returncode}); see {run_dir / 'logs' / 'g1.log'}")
    bench = json.loads(g1.stdout)
    if not bench.get("acceptance_eligible") or bench.get("requested") != 320 or bench.get("decoded") != 320:
        raise RuntimeError("G1 input was not a fully decoded 320-image real-model run")
    val_report = _read_json(run_dir / "val1000" / "report.json")
    hard_reports = {subset: _read_json(run_dir / "dev-hard" / subset / "report.json")
                    for subset in marker["hard_reports"]}
    hard_status = {subset: 0 for subset in marker["hard_reports"]}
    hard_status.update(golden_exit=marker["golden_exit"], golden_pass=marker["golden_pass"],
                       golden_correct=marker["golden_correct"])
    metrics = _subset_metrics(val_report)
    metrics.update({f"dev-hard/{k}": report["overall_g2"] for k, report in hard_reports.items()})
    val_input_errors = val_report.get("decode_error_count", 0)
    hard_input_errors = sum(report.get("decode_error_count", 0) for report in hard_reports.values())
    input_error_count = val_input_errors + hard_input_errors
    effective_config = marker["effective_config"]
    val_sampling = _read_json(run_dir / "val1000" / "report.json").get("sampling", {})
    hard_sampling = marker.get("hard_sampling", {})
    key = {"model_sha256": effective_config["classifier_model"]["sha256"],
           "config_sha256": effective_config["config_sha256"],
           "effective_config_sha256": effective_config["effective_config_sha256"],
           "val_sampling": {"mode": "fixed_manifest", "manifest_sha256": val_sampling.get("manifest_sha256"),
                            "paths_sha256": val_sampling.get("paths_sha256"),
                            "run_manifest_sha256": _sha(run_dir / "val1000" / "validation_manifest.csv")},
           "hard_sampling": {"mode": "fixed_manifest", "manifest_sha256": hard_sampling.get("manifest_sha256"),
                             "paths_sha256": hard_sampling.get("paths_sha256"),
                             "subsets": marker["hard_reports"],
                             "run_manifest_sha256": _sha(run_dir / "dev-hard" / "all-hard-manifest.csv")},
           "metric_definition": "plate positional metrics v1"}
    out_root = run_dir.parent
    comparison = _compare({"comparison_key": key, "metrics": metrics,
                           "effective_config": marker["effective_config"]}, out_root)
    line = _scoreboard_line(marker["date"], marker["commit"], val_report,
                            {**hard_status, "reports": hard_reports}, bench, effective_config)
    acceptance = _acceptance(effective_config, val_report, hard_reports, bench,
                             marker["golden_pass"], marker["golden_correct"], input_error_count)
    summary = {"date": marker["date"], "commit": marker["commit"], "directory": str(run_dir),
               "effective_config": effective_config,
               "network_attribution": _attribution([val_report, *hard_reports.values()]),
               "scoreboard_header": SCOREBOARD_HEADER, "scoreboard_row": line,
               "metrics": metrics, "hard_status": hard_status, "hard_completed": marker["hard_reports"],
               "golden_exit": marker["golden_exit"], "g1": bench, "comparison_key": key,
               "comparison": comparison,
               "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip(),
               "input_errors": input_error_count,
               "val_input_errors": val_input_errors, "hard_input_errors": hard_input_errors,
               **_error_analysis(run_dir / "val1000" / "trace.jsonl"),
               "acceptance": acceptance, "status": acceptance["status"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "scoreboard-row.md").write_text("| " + " | ".join(SCOREBOARD_HEADER) + " |\n" +
        "| " + " | ".join("---" for _ in SCOREBOARD_HEADER) + " |\n" + line + "\n", encoding="utf-8")
    (run_dir / "subsets.md").write_text(_format_subsets(metrics), encoding="utf-8")
    (run_dir / "comparison.md").write_text(_format_comparison(comparison), encoding="utf-8")
    (run_dir / "errors.md").write_text(_format_errors(summary), encoding="utf-8")
    (run_dir / "summary.md").write_text(_format_summary(summary), encoding="utf-8")
    return summary


def _format_subsets(metrics):
    lines = ["| 子集 | n | 整牌准确率 | 字符准确率 | 自动通过率 | 自动通过精度 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in sorted(metrics.items()):
        lines.append(f"| {name} | {m.get('n', 0)} | {_fmt(m.get('plate_accuracy'))} | "
                     f"{_fmt(m.get('character_accuracy'))} | {_fmt(m.get('auto_pass_rate'))} | {_fmt(m.get('auto_pass_precision'))} |")
    return "\n".join(lines) + "\n"


def _format_errors(summary):
    lines = ["# Val 1000 逐位置错误与字符混淆", "",
             f"无预测样本：{summary['no_prediction']}", "",
             "| 字符位置 | 错误数 | 缺位 | 多位 |", "| ---: | ---: | ---: | ---: |"]
    positions = sorted(set(summary["position_errors"]) | set(summary["missing_positions"]) |
                       set(summary["extra_positions"]), key=int)
    for pos in positions:
        lines.append(f"| {pos} | {summary['position_errors'].get(pos, 0)} | "
                     f"{summary['missing_positions'].get(pos, 0)} | {summary['extra_positions'].get(pos, 0)} |")
    lines += ["", "| 真值 | 预测 | 次数 |", "| --- | --- | ---: |"]
    for row in summary["confusions_top20"]:
        lines.append(f"| {row['truth']} | {row['prediction']} | {row['count']} |")
    return "\n".join(lines) + "\n"


def _format_summary(summary):
    regressions = summary["comparison"].get("regressions_over_1pp", [])
    alert = ("\n⚠️ **检测到超过1个百分点的退步。**\n" if regressions else "")
    cfg = summary.get("effective_config", {})
    attr = summary.get("network_attribution", {})
    policy = _decision_policy(cfg)
    gate = policy["g1_p95_limit_ms"]
    def model_text(name):
        model = cfg.get("models", {}).get(name, {})
        return (f"{model.get('path') or 'disabled'} (SHA256 {model.get('sha256') or 'N/A'}, "
                f"version {model.get('version') or 'N/A'})")
    return (f"# 每日 M2 评测 {summary['date']}\n\n"
            f"- commit: `{summary['commit']}`（代码来源 `{summary['source_commit']}`）\n"
            f"- 配置: `{cfg.get('config_path', 'N/A')}`；有效配置 SHA256 `{cfg.get('effective_config_sha256', 'N/A')}`；"
            f"分类器 `{cfg.get('classifier', 'N/A')}`；分类器模型 `{cfg.get('classifier_model', {}).get('path', 'N/A')}`"
            f" SHA256 `{cfg.get('classifier_model', {}).get('sha256', 'N/A')}`\n"
            f"- 网络权重: corner {model_text('corner')}；plate reader {model_text('plate_reader')}\n"
            f"- 仲裁：`{policy['arbitration']}`；参数 `{policy['thresholds']}`\n"
            f"- 网络贡献（val + hard）：最终采用CRNN读法 {attr.get('crnn_selected_count', 0)}，"
            f"准确率 {_fmt(attr.get('crnn_selected_accuracy'))}；网络候选最终选中 "
            f"{attr.get('network_candidate_selected_count', 0)}；reader调用 "
            f"{attr.get('reader_invoked_calls', 0)} 次（覆盖 {attr.get('reader_invoked_images', 0)} 图），"
            f"corner调用 {attr.get('corner_invoked_calls', 0)} 次（覆盖 {attr.get('corner_invoked_images', 0)} 图）\n"
            f"- 状态: **{summary['status']}**；golden {summary['hard_status'].get('golden_correct')}/20，"
            f"门槛 13/20；hard 子集完成 {len(summary['hard_completed'])}/7\n"
            f"- G1: P95 {summary['g1'].get('wall_time', {}).get('p95_ms', float('nan')):.1f} ms；"
            f"{summary['g1'].get('decoded')}/{summary['g1'].get('requested')} 张已解码；"
            f"达到{gate}ms门槛：{summary['g1'].get('wall_time', {}).get('p95_ms', float('inf')) <= gate}\n"
            f"- dev-hard 自动通过精度 {_fmt(summary.get('acceptance', {}).get('hard_auto_pass_precision'))}；"
            f"agree双集合≥99%检查：{summary.get('acceptance', {}).get('dual_precision_pass', 'N/A')}\n"
            f"- val 自动通过率 {_fmt(summary['metrics']['val/overall'].get('auto_pass_rate'))}，"
            f"精度 {_fmt(summary['metrics']['val/overall'].get('auto_pass_precision'))}；"
            f"固定清单坏图：val {summary.get('val_input_errors', 0)}/1000，"
            f"dev-hard {summary.get('hard_input_errors', 0)}/2100；均按 BAD_INPUT 纳入对应分母\n"
            f"{alert}\n{summary['scoreboard_row']}\n\n"
            f"详细表：[`subsets.md`](subsets.md)、[`errors.md`](errors.md)、"
            f"[`comparison.md`](comparison.md)。\n")


def _format_comparison(comparison):
    if not comparison["comparable"]:
        return comparison["note"] + "\n"
    lines = [f"对比最近完成运行：{comparison['previous_run']}",
             f"模型变化：{'是' if comparison.get('model_changed') else '否'}；"
             f"配置文件变化：{'是' if comparison.get('config_changed') else '否'}；"
             f"有效配置变化：{'是' if comparison.get('effective_config_changed') else '否'}", "",
             "| 子集 | 指标 | 上次 | 本次 | 变化 | 标记 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    changes = comparison.get("identity_changes", {})
    if changes.get("note"):
        lines.insert(2, changes["note"])
    for name, info in changes.get("models", {}).items():
        if info["changed"]:
            lines.insert(2, (f"模型 `{name}` 变化：`{info.get('previous_path')}` "
                             f"({info.get('previous_sha256')}) → `{info.get('current_path')}` "
                             f"({info.get('current_sha256')})"))
    for item in comparison["comparisons"]:
        mark = "⚠️ 退步超过1个百分点" if item["alert"] else ""
        lines.append(f"| {item['subset']} | {item['metric']} | {_fmt(item['previous'])} | {_fmt(item['current'])} | "
                     f"{item['delta']:+.1%} | {mark} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ALGORITHM_DIR / "out" / "daily")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--config", type=Path, help="effective TOML config; overrides inherited LPR_CONFIG")
    parser.add_argument("--defer-g1", action="store_true", help="run validation and golden now; leave G1 for --g1-only")
    parser.add_argument("--g1-only", type=Path, help="finish a run created with --defer-g1")
    parser.add_argument("--recover-hard", type=Path, help="retry hard evaluations using the saved 7x300 selection")
    args = parser.parse_args(argv)
    try:
        resume_dir = args.recover_hard or args.g1_only
        if resume_dir and args.config:
            marker = _read_json(resume_dir / "pre-g1.json")
            saved_config = marker.get("effective_config", {}).get("config_path")
            if not saved_config or args.config.resolve() != Path(saved_config).resolve():
                raise ValueError("resumed runs must keep their saved --config; start a new daily run to change models")
        if args.recover_hard:
            summary = recover_hard(args.recover_hard)
        elif args.g1_only:
            summary = finish_g1(args.g1_only)
        else:
            summary = daily(args.out_root, args.index, defer_g1=args.defer_g1, config_path=args.config)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    if "hard_recovery" in summary:
        print(f"hard retry: {summary['hard_recovery']['decoded_reports']}/7 reports; G1 pending")
        return 0 if summary["hard_recovery"]["decoded_reports"] == 7 else 1
    if summary.get("g1") == "deferred; run daily.py --g1-only <run-dir>":
        print(f"G1 deferred; run={summary['commit']}; pending={summary['g1']}")
        return 0
    print(summary["scoreboard_row"])
    print(f"report={summary['directory']}")
    if summary["comparison"].get("regressions_over_1pp"):
        print("⚠️ REGRESSION >1 percentage point")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
