from __future__ import annotations

import csv
import json
import random

import pytest

from lpr_tools.train.box_data import (
    bbox_from_quad,
    box_delta,
    iter_rows,
    jitter_bbox,
    load_measured_jitter,
    sample_jitter,
)


def test_bbox_delta_and_jitter_preserve_measured_width_height_errors():
    gt = "10 20 110 20 110 60 10 60"
    candidate = "20 24 130 24 130 72 20 72"
    assert bbox_from_quad(gt) == (10.0, 20.0, 100.0, 40.0)
    delta = box_delta(gt, candidate)
    assert delta == pytest.approx((0.1, 0.1, __import__("math").log(1.1), __import__("math").log(1.2)))
    assert jitter_bbox((10, 20, 100, 40), delta) == pytest.approx((20, 24, 110, 48))


def test_measured_sampler_uses_training_matched_rows_and_is_deterministic(tmp_path):
    path = tmp_path / "geometry.jsonl"
    records = [
        {"subset": "tilt", "partition": "train", "matched": True,
         "gt_quad_px": [[0, 0], [100, 0], [100, 40], [0, 40]],
         "proposal_quad_px": [[10, 4], [120, 4], [120, 52], [10, 52]]},
        {"subset": "db", "partition": "train", "matched": False},
        {"subset": "tilt", "partition": "tune", "matched": True,
         "gt_quad_px": [[0, 0], [100, 0], [100, 40], [0, 40]],
         "proposal_quad_px": [[90, 0], [190, 0], [190, 40], [90, 40]]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    samples = load_measured_jitter(path)
    assert set(samples) == {"tilt"}
    assert samples["tilt"][0] == pytest.approx((0.1, 0.1, __import__("math").log(1.1), __import__("math").log(1.2)))
    assert sample_jitter(random.Random(42), "tilt", samples) == samples["tilt"][0]


def test_iter_rows_streams_csv_and_invalid_box_is_rejected(tmp_path):
    path = tmp_path / "rows.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label"))
        writer.writeheader()
        writer.writerow({"path": "a.jpg", "label": "皖A12345"})
        writer.writerow({"path": "b.jpg", "label": "皖A54321"})
    assert [row["path"] for row in iter_rows(path)] == ["a.jpg", "b.jpg"]
    with pytest.raises(ValueError):
        jitter_bbox((1, 1, 0, 2), (0, 0, 0, 0))
