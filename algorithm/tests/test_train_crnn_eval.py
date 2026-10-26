import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpr_tools.train import eval_crnn


def _dataset(root: Path, synthetic=False):
    (root / "val").mkdir(parents=True)
    rows = []
    for i, source in enumerate(["ground_truth", "pipeline_candidate:0:0"]):
        image = np.full((80, 252, 3), i, np.uint8)
        cv2.imencode(".png", image)[1].tofile(root / f"val/{i}.png")
        rows.append([f"val/{i}.png", "皖A12345", 7, "blue",
                     "font_synthetic" if synthetic else source, ""])
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)


def test_crop_evaluation_separates_sources_and_wrong_auto_pass(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _dataset(data)
    model = tmp_path / "dummy.pt"
    model.write_bytes(b"test stub, never an acceptance model")
    class Reader:
        def read(self, image):
            return ("皖A12345" if int(image[0, 0, 0]) == 0 else "皖A12346", [.9] * 7)
    monkeypatch.setattr(eval_crnn, "load_plate_reader", lambda _: Reader())
    report = eval_crnn.evaluate(data, model, tmp_path / "result")
    assert report["groups"]["ground_truth"]["plate_accuracy"] == 1
    assert report["groups"]["pipeline_candidate"]["plate_accuracy"] == 0
    assert report["groups"]["all"]["auto_pass_precision"] == .5
    assert report["groups"]["all"]["char_correct"] == 13
    assert report["groups"]["all"]["char_total"] == 14
    assert report["read_latency_ms"]["n"] == 2


def test_synthetic_validation_is_rejected(tmp_path):
    _dataset(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="synthetic validation"):
        eval_crnn.evaluate(tmp_path, tmp_path / "unused.pt", tmp_path / "out")
