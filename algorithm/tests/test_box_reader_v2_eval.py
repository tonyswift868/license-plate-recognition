from __future__ import annotations

import csv
import hashlib
import json

from lpr_tools.train.calibrate_box_reader import validate_v2_training_history
from lpr_tools.train.eval_box_reader import _groups, _read_csv, legal_prediction


TUNE_FIELDS = ("path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold")
HARD = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")


def _write_tune(path):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TUNE_FIELDS)
        writer.writeheader()
        for group_index, subset in enumerate((*HARD, "green")):
            for index in range(300):
                fold = "fit" if index < 150 else "report"
                row_id = group_index * 300 + index
                writer.writerow({
                    "path": f"D:/safe/{row_id}.jpg",
                    "label": "皖AD12345" if subset == "green" else "皖A12345",
                    "subset": subset, "brightness": "20",
                    "quad": "0 0 100 0 100 40 0 40",
                    "sha256": hashlib.sha256(str(row_id).encode()).hexdigest(),
                    "calibration_fold": fold,
                })


def test_v2_tune_requires_equal_seven_hard_groups_and_eight_char_green(tmp_path):
    path = tmp_path / "tune.csv"
    _write_tune(path)
    rows = _read_csv(path)
    assert len(rows) == 2400
    assert sum(row["calibration_fold"] == "fit" for row in rows) == 1200
    assert sum(row["subset"] == "ccpd_green" and len(row["label"]) == 8 for row in rows) == 300


def test_v2_tune_rejects_a_seven_character_green_row(tmp_path):
    path = tmp_path / "tune.csv"
    _write_tune(path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["label"] = "皖A12345"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TUNE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        _read_csv(path)
    except ValueError as exc:
        assert "8-character" in str(exc)
    else:
        raise AssertionError("green 7-character tune row was accepted")


def test_tune_groups_report_seven_and_eight_character_plate_strata():
    rows = [
        {"subset": "ccpd_db", "label": "皖A12345", "brightness": 20, "correct": True},
        {"subset": "ccpd_green", "label": "皖AD12345", "brightness": 20, "correct": False},
    ]
    groups = _groups(rows)
    assert len(groups["all"]) == 2
    assert len(groups["all_7char"]) == 1
    assert len(groups["all_8char"]) == 1
    assert len(groups["ccpd_green_8char"]) == 1


def test_raw_green_plate_is_still_subject_to_frozen_legal_layout():
    assert legal_prediction("皖AD12345", [0.9] * 8) == ("皖AD12345", 0.9)
    assert legal_prediction("皖AO12345", [0.9] * 8) is None


def test_v2_calibration_accepts_only_full_balanced_60_epoch_history(tmp_path):
    train = tmp_path / "cache-index.csv"
    tune = tmp_path / "tune.csv"
    audit = tmp_path / "audit.json"
    history_path = tmp_path / "history.json"
    best = tmp_path / "best.pt"
    train.write_text("path,label\na.jpg,皖A12345\n", encoding="utf-8")
    tune.write_text("path,label\nb.jpg,皖A12345\n", encoding="utf-8")
    best.write_bytes(b"selected checkpoint")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    names = (*HARD, "green", "base")
    counts = {name: 100 for name in names}
    groups = {name: 100 for name in names}
    audit_data = {
        "schema": "box-reader-v2-metadata-v1",
        "passed": True,
        "source_hashes": {"packed_cache_index_sha256": sha(train)},
        "training": {"unique_rows": 900, "by_group": counts,
                     "train_tune_sha_overlap": 0, "train_protected_sha_overlap": 0,
                     "train_batch8_holdout_sha_overlap": 0},
        "tune": {"train_tune_sha_overlap": 0, "tune_protected_sha_overlap": 0,
                 "green_batch8_holdout_sha_overlap": 0, "fold_sha_overlap": 0,
                 "prior_batch8_hard_tune_sha_overlap": 2100},
        "balanced_epoch": {"group_exposures": {name: 100 for name in names},
                           "exposures_each_epoch": 900},
        "outputs": {"tune.csv": {"sha256": sha(tune)},
                    "sampler.json": {"sha256": hashlib.sha256(b"sampler").hexdigest()}},
    }
    audit.write_text(json.dumps(audit_data) + "\n", encoding="utf-8")
    epochs = [{"epoch": epoch, "samples": 900, "unique_samples": 900,
               "group_exposures": groups, "tune": {"accuracy": 0.5}}
              for epoch in range(1, 61)]
    history = {
        "recipe": {"device": "cuda", "smoke": False, "epochs": 60,
                   "train": str(train), "tune": str(tune), "audit": str(audit)},
        "history": epochs, "epochs_completed": 60, "best_epoch": 1,
        "best_sha256": sha(best), "training_unique_samples": 900,
        "exposures_per_epoch": 900, "group_source_counts": counts,
        "total_exposures": 54_000, "train_sha256": sha(train),
        "tune_sha256": sha(tune), "audit_sha256": sha(audit),
        "sampler_sha256": hashlib.sha256(b"sampler").hexdigest(),
    }
    history_path.write_text(json.dumps(history), encoding="utf-8")
    result = validate_v2_training_history(history_path, best, train, tune, audit)
    assert result["epochs"] == 60 and result["training_unique_samples"] == 900

    history["history"][12]["group_exposures"] = {**groups, "green": 103}
    history["history"][12]["samples"] = 903
    history_path.write_text(json.dumps(history), encoding="utf-8")
    try:
        validate_v2_training_history(history_path, best, train, tune, audit)
    except ValueError as exc:
        assert "group-balanced" in str(exc)
    else:
        raise AssertionError("imbalanced epoch was accepted")
