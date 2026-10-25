
"""Run the M2 daily validation, golden regression, and latency checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from lpr.config import ALGORITHM_DIR
from lpr_tools.data.manifest import read_manifest, write_manifest

ROOT = ALGORITHM_DIR.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
SCOREBOARD_HEADER = ("日期", "commit", "分类器", "清晰", "夜间", "倾斜", "污损", "字符准确率",
                     "Top-3 定位召回", "自动通过精度", "P95 ms", "备注")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _scoreboard_line(date, commit, val, hard, bench):
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
    note = (f"daily; val n={metrics['n']}; auto-pass={_fmt(metrics.get('auto_pass_rate'))}; "
            f"dev-hard 7x300; golden={hard.get('golden_correct', 'N/A')}/20")
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


def _compare(current: dict, root: Path):
    runs = sorted((p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file()),
                  key=lambda p: (p / "summary.json").stat().st_mtime, reverse=True) if root.exists() else []
    if runs:
        path = runs[0]
        old = _read_json(path / "summary.json")
        old_key, now_key = old.get("comparison_key", {}), current["comparison_key"]
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
                    "comparisons": diffs,
                    "regressions_over_1pp": [x for x in diffs if x["alert"]]}
        reasons = [key for key in ("val_sampling", "hard_sampling", "metric_definition")
                   if old_key.get(key) != now_key.get(key)]
        return {"previous_run": path.name, "comparable": False,
                "note": "Latest completed run is not comparable: " + ", ".join(reasons),
                "model_changed": old_key.get("model_sha256") != now_key.get("model_sha256"),
                "config_changed": old_key.get("config_sha256") != now_key.get("config_sha256")}
    return {"previous_run": None, "comparable": False,
            "note": "No previous run with identical model, config, and sampling fingerprint."}


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
          defer_g1: bool = False):
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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ALGORITHM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    val_out = out / "val1000"
    val_cmd = [str(PYTHON), "-m", "lpr_tools.eval.eval_accuracy", "--split", "val", "--n", "1000",
               "--seed", "42", "--locate-recall", "--index", str(index), "--out", str(val_out),
               "--stage-trace", str(out / "val1000" / "trace.jsonl")]
    val_run = _run(val_cmd, ROOT, out / "logs" / "val1000.log", env=env)
    if val_run.returncode:
        raise RuntimeError(f"val1000 failed ({val_run.returncode}); see {out / 'logs' / 'val1000.log'}")
    val_report = _read_json(val_out / "report.json")
    if val_report.get("decoded_rows") != 1000 or val_report.get("decode_error_count"):
        raise RuntimeError("val1000 incomplete or contains decode errors; see report.json")

    hard_reports = {}
    hard_status = {}
    hard_out = out / "dev-hard"
    hard_out.mkdir(parents=True, exist_ok=True)
    sample_all = hard_out / "all-hard-manifest.csv"
    sample_meta = hard_out / "sampling.json"
    command = [str(PYTHON), "-c",
               "from pathlib import Path; import sys,json; from lpr_tools.eval.index_samples import sample_dev_hard; "
               "from lpr_tools.data.manifest import write_manifest; "
               "selected,meta=sample_dev_hard(sys.argv[1],n=300,seed=42,exclude_manifest=[Path(sys.argv[2]),Path(sys.argv[3])]); "
               "write_manifest(sys.argv[4],[s.row for s in selected],check_files=False); "
               "Path(sys.argv[5]).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')",
               str(index), str(INDEX.parent / "testset_manifest_v1.csv"),
               str(ALGORITHM_DIR / "tests" / "data" / "golden" / "golden_manifest.csv"),
               str(sample_all), str(sample_meta)]
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
                        str(manifest), "--split", "manifest-val", "--out", str(target)]
        run = _run(evaluate_cmd, ROOT, out / "logs" / f"dev-{subset}.log", env=env)
        hard_status[subset] = run.returncode
        if run.returncode == 0:
            report = _read_json(target / "report.json")
            if report.get("decoded_rows") != 300 or report.get("decode_error_count"):
                raise RuntimeError(f"{subset} incomplete or contains decode errors")
            hard_reports[subset] = report

    golden = _run([str(PYTHON), "-m", "pytest", "algorithm/tests/test_golden.py", "-q"], ROOT,
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
        marker = {"date": date, "commit": commit, "golden_exit": golden.returncode,
                  "golden_pass": golden_pass, "golden_correct": golden_count,
                  "hard_status": hard_status, "hard_reports": sorted(hard_reports),
                  "g1": "deferred; run daily.py --g1-only <run-dir>"}
        (out / "pre-g1.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        return marker
    g1 = _run([str(PYTHON), "-m", "lpr_tools.eval.bench_latency", "--manifest",
               "algorithm/testset/manifest.csv", "--n", "100", "--size", "1280x720"],
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
    comparison_key = {"model_sha256": _sha(ALGORITHM_DIR / "models" / "char_cnn_v1.pt"),
                      "config_sha256": _sha(ALGORITHM_DIR / "config" / "default.toml"),
                      "val_sampling": {"mode": "val", "n": 1000, "seed": 42},
                      "hard_sampling": {"subsets": list(hard_reports), "n_each": 300, "seed": 42},
                      "metric_definition": "plate positional metrics v1"}
    comparison_key["val_sampling"]["paths_sha256"] = _sha(val_out / "validation_manifest.csv")
    comparison_key["hard_sampling"]["paths_sha256"] = _sha(sample_all)
    comparison = _compare({"comparison_key": comparison_key, "metrics": metrics}, out_root)
    line = _scoreboard_line(date, commit, val_report, {**hard_status, "reports": hard_reports}, bench)
    error_analysis = _error_analysis(out / "val1000" / "trace.jsonl")
    summary = {"date": date, "commit": commit, "directory": str(out), "scoreboard_header": SCOREBOARD_HEADER,
               "scoreboard_row": line, "metrics": metrics, "hard_status": hard_status,
               "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip(),
               "hard_completed": sorted(hard_reports), "golden_exit": golden.returncode,
               "g1": bench, "comparison_key": comparison_key, "comparison": comparison,
               **error_analysis,
               "status": "PASS" if golden_pass and golden_count is not None and golden_count >= 12
               and bench["wall_time"]["p95_ms"] <= 300
               and len(hard_reports) == 7 else "PARTIAL_FAILURE"}
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
    samples = read_manifest(run_dir / "dev-hard" / "all-hard-manifest.csv", check_files=False)
    pools = {}
    for row in samples:
        pools.setdefault(row.source.split("/")[-1], []).append(row)
    subsets = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather")
    reports, statuses = {}, {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ALGORITHM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
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
                   str(manifest), "--split", "manifest-val", "--out", str(target)]
        run = _run(command, ROOT, log, env=env)
        statuses[subset] = run.returncode
        if run.returncode == 0:
            report = _read_json(target / "report.json")
            if report.get("decoded_rows") != 300 or report.get("decode_error_count"):
                raise RuntimeError(f"{subset}: expected 300 decoded rows without errors")
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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ALGORITHM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    g1 = _run([str(PYTHON), "-m", "lpr_tools.eval.bench_latency", "--manifest",
               "algorithm/testset/manifest.csv", "--n", "100", "--size", "1280x720"],
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
    key = {"model_sha256": _sha(ALGORITHM_DIR / "models" / "char_cnn_v1.pt"),
           "config_sha256": _sha(ALGORITHM_DIR / "config" / "default.toml"),
           "val_sampling": {"mode": "val", "n": 1000, "seed": 42,
                            "paths_sha256": _sha(run_dir / "val1000" / "validation_manifest.csv")},
           "hard_sampling": {"subsets": marker["hard_reports"], "n_each": 300, "seed": 42,
                             "paths_sha256": _sha(run_dir / "dev-hard" / "all-hard-manifest.csv")},
           "metric_definition": "plate positional metrics v1"}
    out_root = run_dir.parent
    comparison = _compare({"comparison_key": key, "metrics": metrics}, out_root)
    line = _scoreboard_line(marker["date"], marker["commit"], val_report,
                            {**hard_status, "reports": hard_reports}, bench)
    summary = {"date": marker["date"], "commit": marker["commit"], "directory": str(run_dir),
               "scoreboard_header": SCOREBOARD_HEADER, "scoreboard_row": line,
               "metrics": metrics, "hard_status": hard_status, "hard_completed": marker["hard_reports"],
               "golden_exit": marker["golden_exit"], "g1": bench, "comparison_key": key,
               "comparison": comparison,
               "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip(),
               **_error_analysis(run_dir / "val1000" / "trace.jsonl"),
               "status": "PASS" if marker["golden_pass"] and marker["golden_correct"] >= 12
               and bench["wall_time"]["p95_ms"] <= 300
               and len(hard_reports) == 7 else "PARTIAL_FAILURE"}
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
    return (f"# 每日 M2 评测 {summary['date']}\n\n"
            f"- commit: `{summary['commit']}`（代码来源 `{summary['source_commit']}`）\n"
            f"- 状态: **{summary['status']}**；golden {summary['hard_status'].get('golden_correct')}/20，"
            f"门槛 12/20；hard 子集完成 {len(summary['hard_completed'])}/7\n"
            f"- G1: P95 {summary['g1'].get('wall_time', {}).get('p95_ms', float('nan')):.1f} ms；"
            f"{summary['g1'].get('decoded')}/{summary['g1'].get('requested')} 张已解码；"
            f"达到300ms门槛：{summary['g1'].get('wall_time', {}).get('p95_ms', float('inf')) <= 300}\n"
            f"- val 自动通过率 {_fmt(summary['metrics']['val/overall'].get('auto_pass_rate'))}，"
            f"精度 {_fmt(summary['metrics']['val/overall'].get('auto_pass_precision'))}\n"
            f"{alert}\n{summary['scoreboard_row']}\n\n"
            f"详细表：[`subsets.md`](subsets.md)、[`errors.md`](errors.md)、"
            f"[`comparison.md`](comparison.md)。\n")


def _format_comparison(comparison):
    if not comparison["comparable"]:
        return comparison["note"] + "\n"
    lines = [f"对比最近完成运行：{comparison['previous_run']}",
             f"模型变化：{'是' if comparison.get('model_changed') else '否'}；"
             f"配置变化：{'是' if comparison.get('config_changed') else '否'}", "",
             "| 子集 | 指标 | 上次 | 本次 | 变化 | 标记 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for item in comparison["comparisons"]:
        mark = "⚠️ 退步超过1个百分点" if item["alert"] else ""
        lines.append(f"| {item['subset']} | {item['metric']} | {_fmt(item['previous'])} | {_fmt(item['current'])} | "
                     f"{item['delta']:+.1%} | {mark} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ALGORITHM_DIR / "out" / "daily")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--defer-g1", action="store_true", help="run validation and golden now; leave G1 for --g1-only")
    parser.add_argument("--g1-only", type=Path, help="finish a run created with --defer-g1")
    parser.add_argument("--recover-hard", type=Path, help="retry hard evaluations using the saved 7x300 selection")
    args = parser.parse_args(argv)
    try:
        if args.recover_hard:
            summary = recover_hard(args.recover_hard)
        elif args.g1_only:
            summary = finish_g1(args.g1_only)
        else:
            summary = daily(args.out_root, args.index, defer_g1=args.defer_g1)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
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
