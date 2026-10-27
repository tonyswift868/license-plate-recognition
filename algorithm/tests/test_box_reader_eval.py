from __future__ import annotations

import json
import math

import pytest

from lpr_tools.train.eval_box_reader import (
    _actual_group_summary,
    apply_box_jitter,
    bbox_from_quad,
    choose_highest_legal,
    measured_box_deltas,
    _read_csv,
)


def test_actual_comparison_uses_actual_crnn_and_retains_gt_reference(tmp_path):
    path = tmp_path / "ceiling.json"
    path.write_text(json.dumps({"groups": {"devhard": {"all": {
        "actual/crnn": {"accuracy": .6, "correct": 6, "n": 10},
        "ground_truth/crnn": {"accuracy": .9, "correct": 9, "n": 10}}}}}), encoding="utf-8")
    result = _actual_group_summary([{"correct": True}], path, "devhard", "all")
    assert result["delta_vs_ceiling"] == pytest.approx(.4)
    assert result["ceiling_crnn_actual"]["accuracy"] == .6
    assert result["ceiling_crnn_gt"]["accuracy"] == .9


def test_gt_bbox_is_axis_aligned_envelope_and_applies_measured_jitter():
    bbox = bbox_from_quad("10 20 40 18 42 38 12 40")
    assert bbox == (10.0, 18.0, 32.0, 22.0)
    shifted = apply_box_jitter(bbox, (0.1, -0.2, math.log(1.1), math.log(0.9)))
    assert shifted == pytest.approx((13.2, 13.6, 35.2, 19.8))


def test_bbox_parser_rejects_malformed_or_degenerate_quad():
    with pytest.raises(ValueError):
        bbox_from_quad("1 2 3")
    with pytest.raises(ValueError):
        bbox_from_quad("1 1 1 1 1 1 1 1")


def test_highest_legal_reader_uses_minimum_character_confidence():
    values = [
        ("皖A12345", [0.98, 0.97, 0.88, 0.99, 0.95, 0.96, 0.94]),
        ("皖A1234", [1.0] * 7),  # illegal length
        ("皖O12345", [1.0] * 7),  # illegal frozen plate layout
        ("皖B23456", [0.98, 0.97, 0.91, 0.99, 0.95, 0.96, 0.94]),
    ]
    assert choose_highest_legal(values) == ("皖B23456", 0.91)


def test_safe_tune_csv_requires_fixed_groups_and_disjoint_folds(tmp_path):
    import csv

    path = tmp_path / "tune.csv"
    fields = ("path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        names = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
        for subset_index, subset in enumerate(names):
            for index in range(300):
                global_index = subset_index * 300 + index
                writer.writerow({"path": f"D:/synthetic/{global_index}.jpg", "label": "皖A12345",
                                 "subset": subset, "brightness": "20",
                                 "quad": "0 0 100 0 100 40 0 40",
                                 "sha256": f"{global_index:064x}",
                                 "calibration_fold": "fit" if index < 150 else "report"})
    assert len(_read_csv(path)) == 2100

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["sha256"] = rows[0]["sha256"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unique source paths and content hashes"):
        _read_csv(path)


def test_measured_jitter_uses_only_matched_training_rows(tmp_path):
    path = tmp_path / "geometry.jsonl"
    training = {
        "partition": "train", "matched": True, "subset": "db",
        "gt_quad_px": [[0, 0], [100, 0], [100, 40], [0, 40]],
        "actual_quad_px": [[10, 4], [110, 4], [110, 44], [10, 44]],
    }
    heldout = {**training, "partition": "tune"}
    path.write_text("\n".join(json.dumps(row) for row in (training, heldout)), encoding="utf-8")
    samples = measured_box_deltas(path)
    assert len(samples["db"]) == 1
    assert samples["db"][0] == pytest.approx((0.1, 0.1, 0.0, 0.0))
    assert "tune" not in samples
