from __future__ import annotations

import csv

import pytest

from lpr_tools.train.calibrate_box_reader import (
    _report_fold,
    read_calibration_rows,
    training_pool_provenance,
)


def _write_folds(path, *, duplicate_sha=False):
    fields = ("path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold in ("fit", "report"):
            for index in range(1050):
                identity = index if duplicate_sha or fold == "fit" else index + 2000
                writer.writerow({
                    "path": f"D:/synthetic/{identity}.jpg",
                    "label": "皖A12345", "subset": "db", "brightness": "20",
                    "quad": "0 0 100 0 100 40 0 40",
                    "sha256": f"{identity:064x}", "calibration_fold": fold,
                })


def test_calibration_fold_requires_1050_disjoint_paths_and_hashes(tmp_path):
    source = tmp_path / "tune.csv"
    _write_folds(source)
    fit, report = read_calibration_rows(source)
    assert (len(fit), len(report)) == (1050, 1050)
    assert {row["calibration_fold"] for row in fit} == {"fit"}
    assert {row["calibration_fold"] for row in report} == {"report"}

    _write_folds(source, duplicate_sha=True)
    with pytest.raises(ValueError, match="share paths or source bytes"):
        read_calibration_rows(source)


def test_manifest_training_provenance_identifies_pool_not_calibration_index(tmp_path):
    import hashlib

    pool = tmp_path / "pool.csv"
    contents = b"source\ntrain-only\n"
    pool.write_bytes(contents)
    actual = training_pool_provenance(pool)
    assert actual == {"path": str(pool.resolve()), "sha256": hashlib.sha256(contents).hexdigest()}
    with pytest.raises(FileNotFoundError, match="training pool index"):
        training_pool_provenance(tmp_path / "missing.csv")


def test_positive_temperature_keeps_report_string_predictions():
    import torch

    from lpr.charset import NUM_CLASSES

    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((24, 1, NUM_CLASSES + 1), generator=generator)
    rows = [{"label": "皖A12345"}]
    report = _report_fold(rows, [logits], temperature=2.5)
    assert report["argmax_string_invariant"] is True
    assert len(report["predictions"]) == 1
    assert report["ctc_nll_before"] != report["ctc_nll_after"]
