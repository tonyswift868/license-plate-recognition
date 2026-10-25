import json
import pytest

from lpr_tools.eval.stage_loss import analyze, render_markdown


def _write(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_paired_operational_buckets_and_character_confusions(tmp_path):
    normal = [
        {"path": "a", "category": "clear", "truth": "ABC1234", "prediction": "XYZ1234", "status": "OK", "top3_hit": False},
        {"path": "b", "category": "clear", "truth": "ABC1234", "prediction": "", "status": "SEG_FAILED", "top3_hit": True},
        {"path": "c", "category": "clear", "truth": "ABC1234", "prediction": "ABC1234", "status": "OK", "top3_hit": True},
    ]
    oracle = [
        {"path": "a", "category": "clear", "truth": "ABC1234", "prediction": "ABC1234", "status": "OK", "segmentation_failed": False},
        {"path": "b", "category": "clear", "truth": "ABC1234", "prediction": "", "status": "SEG_FAILED", "segmentation_failed": True},
        {"path": "c", "category": "clear", "truth": "ABC1234", "prediction": "ABX1234", "status": "OK", "segmentation_failed": False},
    ]
    n_path, o_path = tmp_path / "normal.jsonl", tmp_path / "oracle.jsonl"
    _write(n_path, normal)
    _write(o_path, oracle)

    result = analyze(n_path, o_path)
    assert result["mutually_exclusive_normal_error_buckets"] == {
        "localization_top3_miss": 1,
        "rectification_segmentation_operational": 1,
    }
    assert result["normal_oracle_exact_match_quadrants"] == {
        "oracle_only_correct": 1,
        "both_wrong": 1,
        "normal_only_correct": 1,
    }
    assert result["character_errors"]["oracle"]["substitutions_top20"] == [
        {"truth": "C", "prediction": "X", "count": 1}
    ]
    assert result["character_errors"]["oracle"]["substitution_position_counts"] == {"3": 1}
    assert result["character_errors"]["oracle"]["no_prediction_rows"] == 1
    assert "不是可加的因果分解" in render_markdown(result)


def test_trace_requires_same_paths_and_counts_decode_skips(tmp_path):
    normal = [{"path": "same", "category": "clear", "truth": "", "prediction": "", "decode_error": "decode_failed"}]
    oracle = [{"path": "same", "category": "clear", "truth": "", "prediction": "", "decode_error": "decode_failed"}]
    n_path, o_path = tmp_path / "normal.jsonl", tmp_path / "oracle.jsonl"
    _write(n_path, normal)
    _write(o_path, oracle)
    result = analyze(n_path, o_path)
    assert result["decode_mismatch_or_skip"] == 1
    assert result["plated_decoded_pairs"] == 0


def test_trace_rejects_mismatched_truth_or_category(tmp_path):
    normal = [{"path": "same", "category": "clear", "truth": "ABC1234", "prediction": "ABC1234"}]
    oracle = [{"path": "same", "category": "tilt", "truth": "ABC1234", "prediction": "ABC1234"}]
    n_path, o_path = tmp_path / "normal.jsonl", tmp_path / "oracle.jsonl"
    _write(n_path, normal)
    _write(o_path, oracle)
    with pytest.raises(ValueError, match="truth and category"):
        analyze(n_path, o_path)
import numpy as np

from lpr.errors import failure_result
from lpr.io_utils import imwrite
from lpr.types import RecogStatus
from lpr_tools.data.manifest import ManifestRow, write_manifest
from lpr_tools.eval.eval_accuracy import evaluate


class _FakeRecognizer:
    version = "fake"

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return failure_result(RecogStatus.NO_PLATE)


def test_evaluate_skips_and_counts_corrupt_image_without_aborting(tmp_path):
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    imwrite(good, np.zeros((24, 32, 3), dtype=np.uint8))
    bad.write_bytes(b"not an image")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [
        ManifestRow(good.name, "京A12345", "blue", "clear", "tmp"),
        ManifestRow(bad.name, "京A12346", "blue", "clear", "tmp"),
    ])
    recognizer = _FakeRecognizer()

    report = evaluate(manifest, recognizer, tmp_path / "out")

    assert report["manifest_rows"] == 2
    assert report["decoded_rows"] == 1
    assert report["decode_error_count"] == 1
    assert report["decode_errors"][0]["path"] == bad.name
    assert recognizer.calls == 1

