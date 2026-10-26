import json
import subprocess
import pytest

import lpr_tools.eval.daily as daily_module
from lpr_tools.eval.daily import (_compare, _error_analysis, _format_comparison,
                                 _partition_hard_rows, _attribution, _effective_config,
                                 _identity_changes, _verify_marker_config, daily)
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


def test_network_attribution_distinguishes_selected_results_and_invocations():
    report = {"network_attribution": {
        "network_candidate_selected_count": 2, "network_candidate_selected_correct": 1,
        "crnn_selected_count": 3, "crnn_selected_correct": 2,
        "reader_invoked_images": 5, "reader_invoked_calls": 8,
        "corner_invoked_images": 4, "corner_invoked_calls": 6}}
    totals = _attribution([report, report])
    assert totals["crnn_selected_count"] == 6
    assert totals["crnn_selected_accuracy"] == pytest.approx(2 / 3)
    assert totals["reader_invoked_images"] == 10
    assert totals["reader_invoked_calls"] == 16


def test_identity_comparison_reports_each_model_change_without_changing_sampling_key():
    old = {"classifier_model": {"path": "cnn-v1.pt", "sha256": "a"},
           "models": {"corner": {"path": None, "sha256": None},
                      "plate_reader": {"path": "reader-old.pt", "sha256": "b"}},
           "config_path": "old.toml", "config_sha256": "c", "effective_config_sha256": "d"}
    current = {"classifier_model": {"path": "cnn-v1.pt", "sha256": "a"},
               "models": {"corner": {"path": "corner.pt", "sha256": "e"},
                          "plate_reader": {"path": "reader-new.pt", "sha256": "f"}},
               "config_path": "new.toml", "config_sha256": "g", "effective_config_sha256": "h"}
    changes = _identity_changes(old, current)
    assert not changes["models"]["classifier"]["changed"]
    assert changes["models"]["corner"]["changed"]
    assert changes["models"]["plate_reader"]["changed"]
    assert changes["effective_config_changed"]


def test_deferred_marker_config_identity_rejects_changed_model(tmp_path):
    model = tmp_path / "reader.pt"
    model.write_bytes(b"reader-one")
    classifier = tmp_path / "classifier.pt"
    classifier.write_bytes(b"classifier")
    config = tmp_path / "daily.toml"
    config.write_text(f'[runtime]\nmodel_path = "{classifier.as_posix()}"\n'
                      f'plate_reader_model_path = "{model.as_posix()}"\n', encoding="utf-8")
    path, identity = _effective_config(config)
    assert identity["models"]["plate_reader"]["sha256"]
    assert identity["config_path"] == str(path)
    marker = {"effective_config": identity}
    assert _verify_marker_config(marker) == path
    model.write_bytes(b"reader-two")
    with pytest.raises(ValueError, match="identity changed"):
        _verify_marker_config(marker)


def test_deferred_marker_with_fixed_manifests_returns_config_path(tmp_path, monkeypatch):
    classifier = tmp_path / "classifier.pt"
    classifier.write_bytes(b"classifier")
    config = tmp_path / "daily.toml"
    config.write_text(f'[runtime]\nmodel_path = "{classifier.as_posix()}"\n', encoding="utf-8")
    fixed_val = tmp_path / "val.csv"
    fixed_hard = tmp_path / "hard.csv"
    fixed_val.write_bytes(b"val-fixed")
    fixed_hard.write_bytes(b"hard-fixed")
    run = tmp_path / "run"
    run.mkdir()
    val_copy = run / "val.csv"
    hard_copy = run / "hard.csv"
    val_copy.write_bytes(fixed_val.read_bytes())
    hard_copy.write_bytes(fixed_hard.read_bytes())
    monkeypatch.setattr(daily_module, "FIXED_VAL_MANIFEST", fixed_val)
    monkeypatch.setattr(daily_module, "FIXED_HARD_MANIFEST", fixed_hard)
    selected_config, identity = _effective_config(config)
    from lpr_tools.eval.daily import _sha
    marker = {"effective_config": identity, "run_dir": str(run),
              "fixed_manifests": {"val": {"sha256": _sha(fixed_val)},
                                  "dev-hard": {"sha256": _sha(fixed_hard)}},
              "run_manifests": {"val.csv": _sha(val_copy), "hard.csv": _sha(hard_copy)}}
    assert _verify_marker_config(marker) == selected_config
    fixed_hard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fixed dev-hard manifest changed"):
        _verify_marker_config(marker)


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
    config_path = tmp_path / "daily.toml"
    classifier = tmp_path / "classifier.pt"
    classifier.write_bytes(b"classifier")
    config_path.write_text(f'[runtime]\nmodel_path = "{classifier.as_posix()}"\n'
                           'corner_model_path = ""\nplate_reader_model_path = ""\n', encoding="utf-8")
    index = tmp_path / "ccpd_index.csv"
    index.write_text("index", encoding="utf-8")
    fixed_val = tmp_path / "val1000_v1.csv"
    fixed_hard = tmp_path / "devhard_v1.csv"
    fixed_val.write_text("val fixed fixture\n", encoding="utf-8")
    fixed_hard.write_text("hard fixed fixture\n", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"stub")
    monkeypatch.setattr(daily_module, "ROOT", root)
    monkeypatch.setattr(daily_module, "ALGORITHM_DIR", algorithm)
    monkeypatch.setattr(daily_module, "PYTHON", python)
    monkeypatch.setattr(daily_module, "INDEX", index)
    monkeypatch.setattr(daily_module, "FIXED_VAL_MANIFEST", fixed_val)
    monkeypatch.setattr(daily_module, "FIXED_HARD_MANIFEST", fixed_hard)

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
                "manifest_rows": 1000, "decoded_rows": 1000,
                "scored_rows": 1000, "decode_error_count": 0}

    observed = []
    def fake_run(command, cwd, log, *, env=None):
        from pathlib import Path
        command = [str(x) for x in command]
        observed.append((command, dict(env or {})))
        if any("eval_accuracy" in item for item in command) and "--split" in command and "val" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text(json.dumps(report()), encoding="utf-8")
            (out / "validation_manifest.csv").write_text("sample\n", encoding="utf-8")
            trace = Path(command[command.index("--stage-trace") + 1])
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"truth": "京A12345", "prediction": "京B12345"}, ensure_ascii=False)+"\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if "fixed_manifest_samples" in " ".join(command):
            dest = Path(command[-2])
            dest.parent.mkdir(parents=True, exist_ok=True)
            lines = ["path,plate_no,plate_color,category,source,quad,note"]
            subsets = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather")
            for subset in subsets:
                for n in range(300):
                    lines.append(f"{subset}-{n}.jpg,京A12345,blue,night,ccpd2019/{subset},,fixture")
            dest.write_text("\n".join(lines), encoding="utf-8")
            Path(command[-1]).write_text(json.dumps({"manifest_sha256": "fixed-hard-hash",
                                                     "paths_sha256": "hard-paths-hash"}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if "manifest-val" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            data = report(); data["manifest_rows"] = data["scored_rows"] = data["decoded_rows"] = 300
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
    original_compare = daily_module._compare
    def compare_with_identity(current, out_root):
        assert current["effective_config"]["effective_config_sha256"]
        return original_compare(current, out_root)
    monkeypatch.setattr(daily_module, "_compare", compare_with_identity)
    result = daily(tmp_path / "daily", index, date="2026-10-26", commit="test123",
                   config_path=config_path)
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
    effective_path = str(config_path.resolve())
    for command, child_env in observed:
        if "eval_accuracy" in command or "bench_latency" in command:
            assert command[command.index("--config") + 1] == effective_path
        if "eval_accuracy" in command and "--split" in command and "val" in command:
            assert "--n" not in command and "--seed" not in command
            assert command[command.index("--fixed-manifest") + 1] == str(fixed_val)
        if "eval_accuracy" in command or "bench_latency" in command or "test_golden.py" in command or "-c" in command:
            assert child_env["LPR_CONFIG"] == effective_path


def test_identity_comparison_does_not_treat_missing_legacy_metadata_as_unchanged():
    assert _identity_changes({}, {"config_path": "new.toml"})["available"] is False
    result = _identity_changes(
        {"models": {"corner": {"path": "corner.pt", "sha256": "old"}}},
        {"models": {"corner": {"path": "corner.pt", "sha256": "new"}}})
    assert result["available"] and result["models"]["corner"]["changed"]


@pytest.mark.parametrize("phase", ["--recover-hard", "--g1-only"])
def test_resuming_cannot_silently_ignore_a_different_explicit_config(tmp_path, phase):
    saved = tmp_path / "saved.toml"
    other = tmp_path / "other.toml"
    (tmp_path / "pre-g1.json").write_text(json.dumps({
        "effective_config": {"config_path": str(saved)}}), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        daily_module.main([phase, str(tmp_path), "--config", str(other)])
    assert caught.value.code == 2
