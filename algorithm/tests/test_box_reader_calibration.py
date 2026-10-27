from __future__ import annotations

import csv

import pytest

from lpr_tools.train.calibrate_box_reader import (
    _report_fold,
    read_calibration_rows,
    training_pool_provenance,
    validate_training_history,
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


def _write_pool_audit(path, pool):
    import hashlib
    import json

    path.write_text(json.dumps({"output_hashes": {
        "pool.csv": hashlib.sha256(pool.read_bytes()).hexdigest()}}), encoding="utf-8")


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


def test_release_history_requires_thirty_complete_cuda_epochs(tmp_path):
    import hashlib
    import json

    train = tmp_path / "pool.csv"
    tune = tmp_path / "tune.csv"
    train.write_text("path,label\na,皖A12345\nb,皖A23456\nc,皖A34567\n", encoding="utf-8")
    tune.write_text("path,label\na,皖A12345\n", encoding="utf-8")
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")
    records = [{"epoch": epoch, "samples": 3, "unique_samples": 3,
                "tune": {"accuracy": epoch / 100}} for epoch in range(1, 31)]
    history = {
        "recipe": {"smoke": False, "device": "cuda", "epochs": 30,
                   "train": str(train.resolve()), "tune": str(tune.resolve()),
                   "out": str(tmp_path.resolve())},
        "history": records, "best_epoch": 30, "total_exposures": 90,
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "tune_sha256": hashlib.sha256(tune.read_bytes()).hexdigest(),
    }
    history_path = tmp_path / "history.json"
    pool_audit_path = tmp_path / "pool-audit.json"
    _write_pool_audit(pool_audit_path, train)
    history_path.write_text(json.dumps(history), encoding="utf-8")
    evidence = validate_training_history(history_path, best, train, tune,
        pool_audit_path=pool_audit_path, expected_pool_count=3)
    assert evidence["epochs"] == 30
    assert evidence["total_exposures"] == 90

    history["history"] = records[:29]
    history_path.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 30 complete epochs"):
        validate_training_history(history_path, best, train, tune,
            pool_audit_path=pool_audit_path, expected_pool_count=3)


def test_history_accepts_packed_cache_only_when_audit_binds_original_pool(tmp_path):
    import hashlib
    import json

    pool = tmp_path / "pool.csv"
    tune = tmp_path / "tune.csv"
    cache = tmp_path / "cache" / "index.csv"
    audit_path = cache.parent / "audit.json"
    pool_audit_path = tmp_path / "pool-audit.json"
    train_rows = "path,label\na,皖A12345\nb,皖A23456\nc,皖A34567\n"
    pool.write_text(train_rows, encoding="utf-8")
    _write_pool_audit(pool_audit_path, pool)
    tune.write_text("path,label\na,皖A12345\n", encoding="utf-8")
    cache.parent.mkdir(parents=True)
    fields = ("path", "label", "subset", "brightness", "sha256", "quad", "cache_pack",
              "pack_offset", "pack_length", "x0", "y0", "orig_w", "orig_h")
    with cache.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(3):
            writer.writerow({key: "0" for key in fields})
    tar = cache.parent / "crops.tar"
    tar.write_bytes(b"ustar-test")
    pool_sha = hashlib.sha256(pool.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    audit = {"pool_sha256": pool_sha, "cache_index_sha256": cache_sha,
             "pool_count": 3, "quality": 65,
             "cache_tar_path": str(tar.resolve()), "cache_tar_sha256": "a" * 64,
             "tar_bytes": tar.stat().st_size, "index_bytes": cache.stat().st_size,
             "total_cache_bytes": tar.stat().st_size + cache.stat().st_size}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    best = tmp_path / "best.pt"
    best.write_bytes(b"checkpoint")
    history_path = tmp_path / "history.json"
    records = [{"epoch": epoch, "samples": 3, "unique_samples": 3,
                "tune": {"accuracy": epoch / 100}} for epoch in range(1, 31)]
    history = {"recipe": {"smoke": False, "device": "cuda", "epochs": 30,
                          "train": str(cache.resolve()), "tune": str(tune.resolve()),
                          "out": str(tmp_path.resolve()), "packed_cache": True,
                          "decode_jpeg": False},
               "history": records, "best_epoch": 30, "total_exposures": 90,
               "train_sha256": cache_sha,
               "tune_sha256": hashlib.sha256(tune.read_bytes()).hexdigest()}
    history_path.write_text(json.dumps(history), encoding="utf-8")

    evidence = validate_training_history(history_path, best, pool, tune,
        cache_index_path=cache, cache_audit_path=audit_path,
        pool_audit_path=pool_audit_path, expected_pool_count=3)
    assert evidence["packed_cache"] is True
    assert evidence["training_samples_per_epoch"] == 3
    assert evidence["cache_audit"]["cache_index_sha256"] == cache_sha

    audit["pool_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="original audited pool"):
        validate_training_history(history_path, best, pool, tune,
            cache_index_path=cache, cache_audit_path=audit_path,
            pool_audit_path=pool_audit_path, expected_pool_count=3)

    audit["pool_sha256"] = pool_sha
    audit["cache_index_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="cache index SHA-256"):
        validate_training_history(history_path, best, pool, tune,
            cache_index_path=cache, cache_audit_path=audit_path,
            pool_audit_path=pool_audit_path, expected_pool_count=3)

    audit["cache_index_sha256"] = cache_sha
    audit["pool_count"] = 2
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="cache count, cache index rows"):
        validate_training_history(history_path, best, pool, tune,
            cache_index_path=cache, cache_audit_path=audit_path,
            pool_audit_path=pool_audit_path, expected_pool_count=3)


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
