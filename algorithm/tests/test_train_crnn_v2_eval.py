import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.recognize.crnn import CrnnNet
from lpr_tools.train.eval_crnn_v2 import evaluate


def _write_index(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"file": "gt.png", "orig_path": "a.jpg", "source": "ground_truth", "layout": "7",
         "plate_no": "皖A12345", "crop_path": "gt.png", "subset": "db", "brightness": "19",
         "partition": "tune", "calibration_fold": "fit"},
        {"file": "pipe.png", "orig_path": "b.jpg", "source": "pipeline_candidate:0:0",
         "layout": "7", "plate_no": "皖A12345", "crop_path": "pipe.png", "subset": "blur",
         "brightness": "60", "partition": "tune", "calibration_fold": "report"},
    ]
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for name in ("gt.png", "pipe.png"):
        ok, encoded = cv2.imencode(".png", np.full((80, 252, 3), 120, np.uint8))
        assert ok
        encoded.tofile(path.parent / name)


def test_v2_eval_uses_real_reader_and_reports_required_groups(tmp_path):
    index = tmp_path / "index.csv"
    _write_index(index)
    model_path = tmp_path / "crnn.pt"
    torch.save(CrnnNet().state_dict(), model_path)

    report = evaluate(index, model_path, tmp_path / "out")

    assert report["groups"]["overall"]["n"] == 2
    assert report["groups"]["GT"]["n"] == 1
    assert report["groups"]["pipeline"]["n"] == 1
    assert report["groups"]["db_GT"]["n"] == 1
    assert report["groups"]["blur_GT"]["n"] == 0
    assert report["groups"]["db_dark_GT"]["n"] == 1
    assert len(report["groups"]["overall"]["confidence_bins"]) == 6
    assert report["read_latency_ms"]["n"] == 2
    with (tmp_path / "out" / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_index_fold_filter_and_gt_source_suffix(tmp_path):
    from lpr_tools.train.eval_crnn_v2 import _is_gt, load_index

    index = tmp_path / "index.csv"
    _write_index(index)
    rows = load_index(index, fold="fit", gt_only=True)
    assert [row["file"] for row in rows] == ["gt.png"]
    assert _is_gt("example:ground_truth")
    assert not _is_gt("pipeline_candidate")
