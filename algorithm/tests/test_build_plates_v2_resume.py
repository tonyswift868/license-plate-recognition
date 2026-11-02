"""Resume and preflight-cache contracts for the v2 corpus builder."""
import csv
import json
from pathlib import Path

import pytest
from lpr_tools.data import build_plates_v2 as builder
from lpr_tools.data.ccpd import INDEX_FIELDS


def _identity(tag="same"):
    return {"index_sha256": tag, "v1_labels_sha256": "v1-labels",
            "v1_ledger_sha256": "v1-ledger", "config_sha256": "config",
            "model_sha256": "model", "exclusions_sha256": "frozen", "limit": 8}


def _resume_tree(tmp_path, *, status="building"):
    v1, target = tmp_path / "v1", tmp_path / "out"
    for split in ("train", "val"):
        (v1 / split).mkdir(parents=True)
    (v1 / "train" / "a.jpg").write_bytes(b"base crop")
    target.mkdir()
    (target / "train").mkdir()
    (target / "val").mkdir()
    (target / "train" / "a.jpg").hardlink_to(v1 / "train" / "a.jpg")
    (target / "labels.csv").write_bytes(b"")
    (target / "build_state.json").write_text(json.dumps({
        "status": status, "identity": _identity(), "base_v1_labels": 1,
    }), encoding="utf-8")
    return v1, target, [["train/a.jpg"]]


def test_resume_accepts_matching_identity_and_samefile_base_link(tmp_path):
    v1, target, rows = _resume_tree(tmp_path)
    state = builder._validate_resume(target, v1, rows, _identity())
    assert state["status"] == "building"
    assert (target / "train/a.jpg").samefile(v1 / "train/a.jpg")


@pytest.mark.parametrize("identity", [_identity("changed-index"), {**_identity(), "model_sha256": "changed-model"}])
def test_resume_rejects_changed_input_identity(tmp_path, identity):
    v1, target, rows = _resume_tree(tmp_path)
    with pytest.raises(ValueError, match="identity differs"):
        builder._validate_resume(target, v1, rows, identity)


def test_resume_rejects_unknown_file_and_non_v1_replacement(tmp_path):
    v1, target, rows = _resume_tree(tmp_path)
    (target / "train" / "untracked.jpg").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="unknown or non-v1 image"):
        builder._validate_resume(target, v1, rows, _identity())
    (target / "train" / "untracked.jpg").unlink()
    (target / "train" / "a.jpg").unlink()
    (target / "train" / "a.jpg").write_bytes(b"not the v1 file")
    with pytest.raises(ValueError, match="unknown or non-v1 image"):
        builder._validate_resume(target, v1, rows, _identity())


def test_matching_preflight_cache_reuses_saved_hash_without_rehashing_original(tmp_path, monkeypatch):
    source = tmp_path / "hard.jpg"
    source.write_bytes(b"verified source bytes")
    index = tmp_path / "index.csv"
    row = dict.fromkeys(INDEX_FIELDS, "")
    row.update(path=str(source), dataset="ccpd2019", subset="ccpd_blur", split="test",
               has_plate="1", plate_no="皖A12345", layout="7", plate_color="blue",
               quad="1 1 20 1 20 10 1 10", bbox="1 1 20 10")
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    target = tmp_path / "target"
    target.mkdir()
    signature = builder._sha(source)
    first, skipped = builder._preflight_sources(index, target, _identity(), set(), set(), set(), 0)
    assert len(first) == 1 and not skipped
    assert (target / "preflight-meta.json").is_file()

    real_sha = builder._sha
    def fail_if_original(path):
        if Path(path).resolve() == source.resolve():
            raise AssertionError("cache hit must not hash original source again")
        return real_sha(path)
    monkeypatch.setattr(builder, "_sha", fail_if_original)
    second, skipped = builder._preflight_sources(index, target, _identity(), set(), set(), set(), 0)
    assert len(second) == len(first) == 1 and not skipped
    assert {key: value for key, value in second[0].items() if key != "__sha256"} == {
        key: value for key, value in first[0].items() if key != "__sha256"
    }
    assert tuple(second[0]["__sha256"]) == signature


def test_preflight_cache_rejects_changed_identity(tmp_path):
    source = tmp_path / "hard.jpg"
    source.write_bytes(b"verified source bytes")
    index = tmp_path / "index.csv"
    row = dict.fromkeys(INDEX_FIELDS, "")
    row.update(path=str(source), dataset="ccpd2019", subset="ccpd_blur", split="test",
               has_plate="1", plate_no="皖A12345", layout="7", plate_color="blue",
               quad="1 1 20 1 20 10 1 10", bbox="1 1 20 10")
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    target = tmp_path / "target"
    target.mkdir()
    builder._preflight_sources(index, target, _identity(), set(), set(), set(), 0)
    with pytest.raises(ValueError, match="cache identity or digest differs"):
        builder._preflight_sources(index, target, _identity("different"), set(), set(), set(), 0)
