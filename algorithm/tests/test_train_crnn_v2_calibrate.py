import csv
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize.crnn import CrnnNet, CrnnReader
from lpr_tools.train.calibrate_crnn_v2 import (
    _batched_ctc_loss,
    _ctc_loss,
    _read_rows,
    _validate_fold_integrity,
    calibrate,
    fit_temperature,
)


def _confident_logits(label: str, scale: float = 8.0) -> torch.Tensor:
    # Repeated character/blank path gives the known seven-character transcription.
    time_steps = len(label) * 2 - 1
    logits = torch.full((time_steps, 1, NUM_CLASSES + 1), -scale)
    for pos, char in enumerate(label):
        logits[pos * 2, 0, CHAR_TO_INDEX[char]] = scale
        if pos * 2 + 1 < time_steps:
            logits[pos * 2 + 1, 0, NUM_CLASSES] = scale
    return logits


def test_ctc_temperature_is_bounded_and_improves_fit_nll():
    labels = ["皖A12345", "沪B54321"]
    logits = [_confident_logits(label) for label in labels]

    fitted = fit_temperature(logits, labels)

    assert 0.05 <= fitted["temperature"] <= 10.0
    assert fitted["fit_nll_after"] <= fitted["fit_nll_before"]
    assert fitted["n"] == 2
    for item, temp in zip(logits, [fitted["temperature"]] * len(logits)):
        assert torch.equal(item.argmax(-1), (item / temp).argmax(-1))


def test_batched_ctc_nll_matches_per_sample_sum_mean_and_has_temperature_gradient():
    labels = ["皖A12345", "沪B54321"]
    logits = [_confident_logits(label).double() for label in labels]
    log_temperature = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    temperature = float(log_temperature.detach().exp())
    expected = torch.stack([_ctc_loss(item, [label], temperature)
                            for item, label in zip(logits, labels)]).mean()

    actual = _batched_ctc_loss(logits, labels, log_temperature.exp())
    actual.backward()

    torch.testing.assert_close(actual.detach(), expected, rtol=1e-10, atol=1e-10)
    assert log_temperature.grad is not None and torch.isfinite(log_temperature.grad)


def test_calibration_reader_selects_only_fit_fold_and_rejects_final_partition(tmp_path):
    path = tmp_path / "index.csv"
    fields = ["file", "orig_path", "source", "layout", "plate_no", "crop_path", "subset",
              "brightness", "partition", "calibration_fold"]
    base = {"file": "x.png", "orig_path": "orig.jpg", "source": "pipeline:0",
            "layout": "7", "plate_no": "皖A12345", "crop_path": "x.png", "subset": "db",
            "brightness": "10", "partition": "tune", "calibration_fold": "fit"}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(base)
        writer.writerow({**base, "file": "y.png", "calibration_fold": "report"})

    rows = _read_rows(path, "fit")
    assert len(rows) == 1 and rows[0]["source"] == "pipeline:0"

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerow({**base, "file": "z.png", "partition": "val1000"})
    try:
        _read_rows(path, "fit")
    except ValueError as exc:
        assert "final devhard/val1000" in str(exc)
    else:
        raise AssertionError("final partition must never enter calibration")


def test_smoke_calibration_exports_loadable_temperature_candidate(tmp_path):
    index = tmp_path / "tune.csv"
    fields = ["file", "orig_path", "source", "layout", "plate_no", "crop_path", "subset",
              "brightness", "partition", "calibration_fold"]
    rows = [
        {"file": "fit.png", "orig_path": "fit.jpg", "source": "pipeline:fit", "layout": "7",
         "plate_no": "皖A12345", "crop_path": "fit.png", "subset": "db", "brightness": "10",
         "partition": "tune", "calibration_fold": "fit"},
        {"file": "report.png", "orig_path": "report.jpg", "source": "ground_truth", "layout": "7",
         "plate_no": "沪B54321", "crop_path": "report.png", "subset": "blur", "brightness": "50",
         "partition": "tune", "calibration_fold": "report"},
    ]
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        ok, encoded = cv2.imencode(".png", np.full((80, 252, 3), 100, np.uint8))
        assert ok
        encoded.tofile(tmp_path / row["crop_path"])
    model_path = tmp_path / "base.pt"
    torch.save(CrnnNet().state_dict(), model_path)

    result = calibrate(index, model_path, tmp_path / "candidate")

    candidate = tmp_path / "candidate" / "crnn_v2_temperature.pt"
    saved = torch.load(candidate, map_location="cpu", weights_only=True)
    assert saved["ctc_temperature"].ndim == 0
    assert result["report_fold"]["argmax_string_invariant"] is True
    loaded = CrnnReader.from_file(candidate, version="smoke")
    assert loaded.temperature == float(saved["ctc_temperature"].item())
    assert result["fit_fold"]["temperature"] == loaded.temperature
    assert result["report_fold"]["temperature"] == loaded.temperature


def test_fold_integrity_fails_closed_and_checks_adjacent_audit(tmp_path):
    index = tmp_path / "tune.csv"
    index.write_text("fake tune rows\n", encoding="utf-8")
    fit = [{"orig_path": "images/A.jpg"}]
    report = [{"orig_path": "images/B.jpg"}]

    with pytest.raises(ValueError, match="share original"):
        _validate_fold_integrity(index, fit, [{"orig_path": "images\\a.jpg"}])

    audit_path = tmp_path / "audit.json"
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    audit = {"output_sha256": {"tune": digest},
             "content_checks": {"fit_report_sha_overlap": 0}}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    evidence = _validate_fold_integrity(index, fit, report)
    assert evidence["origin_path_overlap"] == 0
    assert evidence["adjacent_audit_verified"] is True
    assert evidence["tune_sha256"] == digest

    audit["output_sha256"]["tune"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_fold_integrity(index, fit, report)

    audit["output_sha256"]["tune"] = digest
    audit["content_checks"]["fit_report_sha_overlap"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="zero fit/report SHA overlap"):
        _validate_fold_integrity(index, fit, report)
