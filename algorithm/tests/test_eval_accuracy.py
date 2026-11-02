import hashlib
import json
from dataclasses import replace
from pathlib import Path

import lpr_tools.eval.eval_accuracy as ev
import lpr_tools.eval.index_samples as index_samples_module
import numpy as np
import pytest
from lpr.io_utils import imwrite
from lpr.pipeline import PlateRecognizer
from lpr.testing import FakeRecognizer, UniformClassifier
from lpr.types import RecogStatus as S
from lpr.types import SegmentResult
from lpr_tools.data.manifest import ManifestRow, read_manifest, write_manifest
from lpr_tools.eval.eval_accuracy import OracleRecognizer, append_scoreboard, evaluate, main

QUAD=((0.,0.),(199.,0.),(199.,119.),(0.,119.))


@pytest.fixture(autouse=True)
def isolate_eval_defaults(tmp_path, monkeypatch):
    """Keep CLI defaults and accidental model construction inside this test."""
    isolated_config = tmp_path / "isolated-eval.toml"
    isolated_config.write_text('[runtime]\nmodel_path = ""\n', encoding="utf-8")
    monkeypatch.setenv("LPR_CONFIG", str(isolated_config))
    algo = tmp_path / "algorithm"
    golden = algo / "tests" / "data" / "golden" / "golden_manifest.csv"
    golden.parent.mkdir(parents=True)
    golden.write_text("path,plate_no,plate_color,category,source,quad,note\n", encoding="utf-8")
    # --split val refuses to run without the frozen-test exclusion list; an empty one keeps tests portable.
    exclusion = algo / "data" / "index" / "testset_manifest_v1.csv"
    exclusion.parent.mkdir(parents=True)
    exclusion.write_text("path,plate_no,plate_color,category,source,quad,note\n", encoding="utf-8")
    monkeypatch.setattr(ev, "ALGORITHM_DIR", algo)
    make_recognizer = ev.make_recognizer

    def guarded_make_recognizer(args, *args_, **kwargs):
        assert args.fake or args.uniform, "unit tests must never load the configured real model"
        return make_recognizer(args, *args_, **kwargs)

    monkeypatch.setattr(ev, "make_recognizer", guarded_make_recognizer)
    read_samples_impl = index_samples_module.read_samples

    def guarded_read_samples(index_path):
        index_path = Path(index_path).resolve()
        assert index_path.is_relative_to(tmp_path.resolve()), \
            f"unit test attempted to read a non-temporary index: {index_path}"
        return read_samples_impl(index_path)

    monkeypatch.setattr(index_samples_module, "read_samples", guarded_read_samples)
    monkeypatch.setattr(ev, "read_samples", guarded_read_samples)


def fixture_manifest(tmp_path):
    specs=[("陕A12345","陕A12345","clear"),("陕B12345__LOW","陕B12345","night"),
           ("京A88888","京A88889","tilt"),("NO_PLATE_1","陕A12345","dirty"),
           ("SEG_FAILED_1","陕A12345","dirty"),("NO_PLATE_np","","np"),
           ("沪A12345","","np"),("粤A12345","粤A12345","local")]
    rows=[]
    for i,(stem,truth,category) in enumerate(specs):
        name=stem+".png"
        imwrite(tmp_path/name,np.full((120,200,3),i*25,np.uint8))
        rows.append(ManifestRow(name,truth,"blue",category,"constructed",QUAD))
    p=tmp_path/"manifest.csv"
    write_manifest(p,rows,check_files=True)
    return p


def test_metrics_and_failure_artifacts(tmp_path):
    manifest=fixture_manifest(tmp_path)
    report=evaluate(manifest,FakeRecognizer(samples_dir=tmp_path),tmp_path/"out")
    m=report["overall_g2"]
    assert m["n"]==5 and m["correct"]==2 and m["plate_accuracy"]==.4
    assert m["auto_pass"]==2 and m["auto_pass_precision"]==.5 and m["auto_pass_rate"]==.4
    assert m["char_total"]==35 and m["char_correct"]==20
    assert report["categories"]["np"]["false_positive_rate"]==.5
    assert report["categories"]["local"]["correct"]==1
    assert report["failure_causes"]=={"localization":2,"segmentation":1,"recognition":1,"rejection":1}
    assert len(report["failures"])==5
    assert len(list((tmp_path/"out").glob("failure_????.png")))==5
    saved=json.loads((tmp_path/"out/report.json").read_text(encoding="utf-8"))
    assert saved==report


def test_fixed_validation_decode_error_is_scored_as_bad_input(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    rows = read_manifest(manifest)
    write_manifest(manifest, [rows[0]], check_files=False)
    monkeypatch.setattr(ev, "decode_image", lambda _path: (None, "decode_failed"))
    trace = tmp_path / "bad-trace.jsonl"
    report = evaluate(manifest, FakeRecognizer(samples_dir=tmp_path), tmp_path / "bad-input", stage_trace=trace)
    assert report["manifest_rows"] == report["scored_rows"] == 1
    assert report["decoded_rows"] == 0 and report["decode_error_count"] == 1
    assert report["overall_g2"]["n"] == 1
    assert report["overall_g2"]["correct"] == report["overall_g2"]["auto_pass"] == 0
    assert report["failure_causes"]["rejection"] == 1
    row = json.loads(trace.read_text(encoding="utf-8"))
    assert row["plate_no"] is None and row["confidence"] == 0
    assert row["status"] == "BAD_INPUT"


def test_stage_trace_preserves_exact_text_confidence_and_failure_state(tmp_path):
    manifest = fixture_manifest(tmp_path)
    rec = FakeRecognizer(samples_dir=tmp_path)
    trace = tmp_path / "trace.jsonl"
    evaluate(manifest, rec, tmp_path / "out", stage_trace=trace)
    from lpr.io_utils import imread
    for traced, row in zip(map(json.loads, trace.read_text(encoding="utf-8").splitlines()), read_manifest(manifest), strict=False):
        result = rec.recognize(imread(tmp_path / row.path))
        assert traced["plate_no"] == result.plate_no
        assert traced["confidence"] == result.confidence
        assert traced["status"] == result.status.value


def test_manifest_cli_can_record_decisions_without_localization_measurement(tmp_path, capsys):
    manifest = fixture_manifest(tmp_path)
    trace = tmp_path / "decisions.jsonl"
    assert main(["--manifest", str(manifest), "--split", "manifest-val", "--fake",
                 "--out", str(tmp_path / "out"), "--stage-trace", str(trace)]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(read_manifest(manifest))
    assert all(row["top3_hit"] is None for row in rows)
    assert all("plate_no" in row and "confidence" in row for row in rows)


def test_cli_and_scoreboard(tmp_path,capsys):
    manifest=fixture_manifest(tmp_path)
    score=tmp_path/"scoreboard.md"
    score.write_text("temporary scoreboard\n",encoding="utf-8")
    assert main(["--manifest",str(manifest),"--fake","--out",str(tmp_path/"out"),
                 "--split","manifest-val","--scoreboard",str(score)])==0
    output=capsys.readouterr().out
    assert "overall_g2 5 40.00%" in output and "lpr-fake" in output
    assert len(score.read_text(encoding="utf-8").splitlines())==2
    # Supplying a missing temporary index exercises the CLI guard without
    # falling through to the real shared index/default exclusion manifests.
    protected = tmp_path/"algorithm"/"reports"/"scoreboard.md"
    for argv in (["--split","val","--index",str(tmp_path/"missing-index.csv")],
                 ["--scoreboard",str(protected)],
                 ["--scoreboard",str(score)]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code==2
    assert "ccpd_index.csv" in capsys.readouterr().err
    with pytest.raises(ValueError):
        append_scoreboard(tmp_path/"algorithm"/"reports"/"scoreboard.md",{},split="val",classifier="fake")


def test_scoreboard_fixed_manifest_note_uses_hash_without_seed(tmp_path):
    score = tmp_path / "scoreboard.md"
    report = {"categories": {name: {"plate_accuracy": 0.5} for name in ("clear", "night", "tilt", "dirty")},
              "overall_g2": {"character_accuracy": 0.5, "auto_pass_precision": 0.99},
              "locate_recall": None, "sampling": {"mode": "fixed_manifest",
                  "manifest_sha256": "a" * 64, "selected_n": 1000},
              "mode": "cnn", "oracle": False}
    append_scoreboard(score, report, split="val", classifier="cnn")
    row = score.read_text(encoding="utf-8")
    assert "fixed manifest sha256=" + "a" * 64 in row
    assert "seed=" not in row


def test_fixed_validation_cli_scoreboard_keeps_fixed_mode_and_rejects_resampling(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lpr_tools.data.manifest import read_manifest, write_manifest

    rows = read_manifest(fixture_manifest(tmp_path))
    index = tmp_path / "ccpd_index.csv"
    index.write_text("fixture index\n", encoding="utf-8")
    fixed = tmp_path / "val1000_v1.csv"
    fixed_rows = [replace(row, path=str((tmp_path / row.path).resolve())) for row in rows[:2]]
    write_manifest(fixed, fixed_rows, check_files=True)
    monkeypatch.setattr(ev, "FIXED_VAL_MANIFEST", fixed)
    meta = {"mode": "fixed_manifest", "purpose": "val", "manifest": str(fixed),
            "manifest_sha256": hashlib.sha256(fixed.read_bytes()).hexdigest(),
            "selected_n": len(fixed_rows), "paths_sha256": "fixed-paths-hash"}
    selected = [SimpleNamespace(row=row, dataset="ccpd2019", subset="ccpd_base",
                                split="val", bbox=(0., 0., 100., 50.), stratum="ccpd2019/ccpd_base")
                for row in fixed_rows]
    monkeypatch.setattr(ev, "fixed_manifest_samples", lambda _index, _path, purpose: (selected, meta))
    score = tmp_path / "scoreboard.md"

    assert main(["--split", "val", "--index", str(index), "--fixed-manifest", str(fixed),
                 "--fake", "--out", str(tmp_path / "out"), "--scoreboard", str(score)]) == 0
    line = score.read_text(encoding="utf-8")
    assert f"fixed manifest sha256={hashlib.sha256(fixed.read_bytes()).hexdigest()}" in line
    assert "seed=" not in line

    with pytest.raises(SystemExit) as caught:
        main(["--split", "val", "--index", str(index), "--fixed-manifest", str(fixed),
              "--n", "2", "--fake", "--out", str(tmp_path / "bad")])
    assert caught.value.code == 2


def test_fake_oracle(tmp_path):
    manifest=fixture_manifest(tmp_path)
    assert evaluate(manifest,FakeRecognizer(samples_dir=tmp_path),tmp_path/"out",oracle=True)["oracle"]
    write_manifest(tmp_path/"missing.csv",[ManifestRow("陕A12345.png","陕A12345","blue","clear","test")])
    with pytest.raises(ValueError,match="quad"):
        evaluate(tmp_path/"missing.csv",FakeRecognizer(),tmp_path/"bad",oracle=True)


def test_real_oracle_bypasses_localize_rectify(cfg,monkeypatch):
    import lpr.pipeline as pipeline
    def forbidden(*args,**kwargs):raise AssertionError("oracle must skip stage 2 and 3")
    monkeypatch.setattr(pipeline,"locate",forbidden)
    monkeypatch.setattr(pipeline,"rectify",forbidden)
    seen=[]
    def segment(rect,layout,cfg):
        seen.append((rect.image.shape,layout,rect.candidate.source))
        return SegmentResult(True,layout,[],np.zeros((layout,32,32),np.uint8),1.)
    monkeypatch.setattr(ev,"segment",segment)
    row=ManifestRow("unused","陕A12345","blue","clear","test",QUAD)
    oracle=OracleRecognizer(PlateRecognizer(cfg,UniformClassifier()))
    result=oracle.recognize(np.zeros((120,200,3),np.uint8),row)
    assert result.status==S.LOW_CONFIDENCE and seen==[((80,252,3),7,"oracle")]
    assert "locate" not in result.stage_ms and "rectify" not in result.stage_ms
    monkeypatch.setattr(ev,"segment",lambda *args:SegmentResult(False,7,[],np.zeros((0,32,32),np.uint8),0.))
    assert oracle.recognize(np.zeros((120,200,3),np.uint8),row).status==S.SEG_FAILED
    with pytest.raises(ValueError,match="degenerate"):
        oracle.recognize(np.zeros((120,200,3),np.uint8),replace(row,quad=((0.,0.),)*4))
