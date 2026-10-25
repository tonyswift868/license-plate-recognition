import json
import subprocess
import pytest

import lpr_tools.eval.daily as daily_module
from lpr_tools.eval.daily import (_compare, _error_analysis, _format_comparison,
                                 _partition_hard_rows, daily)
from lpr_tools.data.manifest import ManifestRow, read_manifest, write_manifest


def test_error_analysis_counts_positions_and_confusions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (
        {"truth": "京A12345", "prediction": "京B12346"},
        {"truth": "皖A12345", "prediction": "皖A1234"},
    )), encoding="utf-8")
    result = _error_analysis(trace)
    assert result["position_errors"] == {"2": 1, "7": 2}
    assert {tuple((item["truth"], item["prediction"], item["count"]))
            for item in result["confusions_top20"]} >= {("5", "6", 1), ("A", "B", 1)}
    assert result["missing_positions"] == {"7": 1}


def test_compare_requires_matching_sample_definition_but_allows_model_change(tmp_path):
    old = {"comparison_key": {"model_sha256": "old-model", "config_sha256": "cfg",
                              "val_sampling": {"n": 4, "seed": 42},
                              "hard_sampling": {"subsets": ["a"], "n_each": 3, "seed": 42}},
           "metrics": {"val/overall": {"plate_accuracy": .7}}}
    previous = tmp_path / "2026-10-25-a"
    previous.mkdir()
    (previous / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    current = {"comparison_key": {**old["comparison_key"], "model_sha256": "new-model"},
               "metrics": {"val/overall": {"plate_accuracy": .68}}}
    result = _compare(current, tmp_path)
    assert result["comparable"]
    assert result["regressions_over_1pp"][0]["delta"] == pytest.approx(-.02)
    assert "⚠️" in _format_comparison(result)
    current["comparison_key"]["val_sampling"] = {"n": 5, "seed": 42}
    assert not _compare(current, tmp_path)["comparable"]


def test_one_percentage_point_drop_is_not_marked_but_more_is(tmp_path):
    key = {"val_sampling": {"n": 1}, "hard_sampling": {"n": 1}, "metric_definition": "v1"}
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "summary.json").write_text(json.dumps({
        "comparison_key": key, "metrics": {"val/overall": {"plate_accuracy": .70}}}), encoding="utf-8")
    exactly_one = _compare({"comparison_key": key,
                            "metrics": {"val/overall": {"plate_accuracy": .69}}}, tmp_path)
    assert not exactly_one["regressions_over_1pp"]
    more = _compare({"comparison_key": key,
                     "metrics": {"val/overall": {"plate_accuracy": .68999}}}, tmp_path)
    assert more["regressions_over_1pp"]


def test_latest_incompatible_run_is_reported_without_searching_past_it(tmp_path):
    old = {"comparison_key": {"val_sampling": {"n": 1}, "hard_sampling": {"n": 1},
                              "metric_definition": "v1"}, "metrics": {}}
    older = tmp_path / "older"
    older.mkdir()
    (older / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    latest = tmp_path / "latest"
    latest.mkdir()
    changed = {"comparison_key": {**old["comparison_key"], "val_sampling": {"n": 2}}, "metrics": {}}
    (latest / "summary.json").write_text(json.dumps(changed), encoding="utf-8")
    current = {"comparison_key": old["comparison_key"], "metrics": {}}
    result = _compare(current, tmp_path)
    assert result["previous_run"] == "latest"
    assert not result["comparable"]
    assert "val_sampling" in result["note"]


def test_hard_source_partition_accepts_dataset_prefix():
    rows = [{"source": "ccpd2019/ccpd_db", "path": "db.jpg"},
            {"source": "ccpd2019/ccpd_blur", "path": "blur.jpg"}]
    result = _partition_hard_rows(rows)
    assert list(result) == ["ccpd_db", "ccpd_blur"]
    assert result["ccpd_db"][0]["path"] == "db.jpg"


def test_hard_manifest_round_trip_uses_canonical_encoding_and_columns(tmp_path):
    manifest = tmp_path / "subset_manifest.csv"
    rows = [ManifestRow("D:/images/one.jpg", "京A12345", "blue", "night",
                        "ccpd2019/ccpd_db", ((1., 2.), (3., 2.), (3., 4.), (1., 4.)), "daily")]
    write_manifest(manifest, rows, check_files=False)
    assert read_manifest(manifest, check_files=False) == rows
    assert not manifest.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize(("decoded", "expected"), [(6, 1), (7, 0)])
def test_recovery_exit_code_reflects_completed_hard_reports(tmp_path, monkeypatch, capsys,
                                                           decoded, expected):
    monkeypatch.setattr(daily_module, "recover_hard", lambda _: {
        "commit": "test", "g1": "deferred; run daily.py --g1-only <run-dir>",
        "hard_recovery": {"decoded_reports": decoded}})
    result = daily_module.main(["--recover-hard", str(tmp_path)])
    capsys.readouterr()
    assert result == expected


def test_daily_full_orchestration_writes_complete_reviewable_outputs(tmp_path, monkeypatch):
    root = tmp_path / "project"
    algorithm = root / "algorithm"
    (algorithm / "models").mkdir(parents=True)
    (algorithm / "config").mkdir()
    (algorithm / "models" / "char_cnn_v1.pt").write_bytes(b"model")
    (algorithm / "config" / "default.toml").write_text("cfg", encoding="utf-8")
    index = tmp_path / "ccpd_index.csv"
    index.write_text("index", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"stub")
    monkeypatch.setattr(daily_module, "ROOT", root)
    monkeypatch.setattr(daily_module, "ALGORITHM_DIR", algorithm)
    monkeypatch.setattr(daily_module, "PYTHON", python)
    monkeypatch.setattr(daily_module, "INDEX", index)

    metrics = {"n": 300, "correct": 210, "plate_accuracy": .7, "character_accuracy": .8,
               "auto_pass_rate": .2, "auto_pass_precision": .99, "char_correct": 1000,
               "char_total": 1200, "auto_pass": 60, "auto_pass_correct": 59}
    def report():
        return {"mode": "lpr-1+char_cnn_v1", "overall_g2": dict(metrics),
                "categories": {name: dict(metrics) for name in
                               ("clear", "night", "tilt", "dirty", "np")},
                "subsets": {"ccpd2019/ccpd_base": dict(metrics),
                            "ccpd2019/ccpd_green": dict(metrics)},
                "locate_recall": {"overall": {"recall": .9}},
                "decoded_rows": 1000, "decode_error_count": 0}

    def fake_run(command, cwd, log, *, env=None):
        from pathlib import Path
        command = [str(x) for x in command]
        if any("eval_accuracy" in item for item in command) and "--split" in command and "val" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text(json.dumps(report()), encoding="utf-8")
            (out / "validation_manifest.csv").write_text("sample\n", encoding="utf-8")
            trace = Path(command[command.index("--stage-trace") + 1])
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"truth": "京A12345", "prediction": "京B12345"}, ensure_ascii=False)+"\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if "sample_dev_hard" in " ".join(command):
            dest = Path(command[-2])
            dest.parent.mkdir(parents=True, exist_ok=True)
            lines = ["path,plate_no,plate_color,category,source,quad,note"]
            subsets = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather")
            for subset in subsets:
                for n in range(300):
                    lines.append(f"{subset}-{n}.jpg,京A12345,blue,night,ccpd2019/{subset},,fixture")
            dest.write_text("\n".join(lines), encoding="utf-8")
            Path(command[-1]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if "manifest-val" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            data = report(); data["decoded_rows"] = 300
            (out / "report.json").write_text(json.dumps(data), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if any("test_golden.py" in item for item in command):
            return subprocess.CompletedProcess(command, 0, "1 passed in 1s\n", "")
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, "12/20\n", "")
        if any("bench_latency" in item for item in command):
            bench = {"acceptance_eligible": True, "requested": 320, "decoded": 320,
                     "wall_time": {"p50_ms": 40.0, "p95_ms": 70.0, "max_ms": 80.0}}
            return subprocess.CompletedProcess(command, 0, json.dumps(bench), "")
        raise AssertionError(command)

    monkeypatch.setattr(daily_module, "_run", fake_run)
    monkeypatch.setattr(daily_module.subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "source123\n", ""))
    result = daily(tmp_path / "daily", index, date="2026-10-26", commit="test123")
    dest = tmp_path / "daily" / "2026-10-26-test123"
    assert result["status"] == "PASS"
    assert "val/subset/ccpd2019/ccpd_base" in result["metrics"]
    assert "val/subset/ccpd2019/ccpd_green" in result["metrics"]
    assert result["hard_completed"] == sorted(("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
                                               "ccpd_fn", "ccpd_challenge", "ccpd_weather"))
    assert len(result["scoreboard_row"].split("|")) == len(daily_module.SCOREBOARD_HEADER) + 2
    assert "| cnn v1 |" in result["scoreboard_row"]
    assert (dest / "summary.md").is_file() and (dest / "errors.md").is_file()
    assert "⚠️" not in (dest / "summary.md").read_text(encoding="utf-8")
