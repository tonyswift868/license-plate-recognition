import csv
import hashlib
import json
from pathlib import Path

import pytest

from lpr_tools.train import prepare_crnn_v2_holdout as prep


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_choose_holdout_is_deterministic_global_and_filters_seen_hashes():
    candidates = {
        subset: [(f"{subset}/{i}", digest(f"{subset}-{i}")) for i in range(8)]
        for subset in prep.SUBSETS
    }
    shared = digest("same content in two subsets")
    candidates["blur"].append(("blur/shared", shared))
    candidates["db"].append(("db/shared", shared))
    protected = {digest("blur-0")}
    v1_seen = {digest("tilt-0")}

    one, eligible_one = prep.choose_holdout(candidates, protected, v1_seen,
                                             per_subset=3, fit_per_subset=1, seed=42)
    two, eligible_two = prep.choose_holdout(candidates, protected, v1_seen,
                                             per_subset=3, fit_per_subset=1, seed=42)
    assert one == two
    assert eligible_one == eligible_two
    selected_hashes = [item["sha256"] for item in one.values()]
    assert len(selected_hashes) == len(set(selected_hashes)) == 3 * len(prep.SUBSETS)
    assert not set(selected_hashes) & protected
    assert not set(selected_hashes) & v1_seen
    assert {item["calibration_fold"] for item in one.values()} == {"fit", "report"}
    for subset in prep.SUBSETS:
        assert sum(v["subset"] == subset and v["calibration_fold"] == "fit" for v in one.values()) == 1
        assert sum(v["subset"] == subset and v["calibration_fold"] == "report" for v in one.values()) == 2


@pytest.mark.parametrize(("basename", "expected"), [
    ("01-02-03-18-05.jpg", 18),
    ("01-02-03-20-05.jpg", 20),
    ("not-a-ccpd-name.jpg", None),
])
def test_ccpd_brightness_reads_penultimate_field(basename, expected):
    assert prep.ccpd_brightness(basename) == expected


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_keeps_origin_crops_together_and_filters_protected_and_v1(tmp_path, monkeypatch):
    data = tmp_path / "v2"
    v1 = tmp_path / "v1"
    protected_file = tmp_path / "protected.jpg"
    safe_file = tmp_path / "safe.jpg"
    v1_seen_file = tmp_path / "v1seen.jpg"
    protected_file.write_bytes(b"protected bytes")
    safe_file.write_bytes(b"safe bytes")
    v1_seen_file.write_bytes(b"v1 train bytes")

    labels_rows = []
    ledger = {}

    def add_origin(relative, payload, source, *, split="train", plate="A0000"):
        original = tmp_path / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(payload)
        rel = "train/" if split == "train" else "val/"
        rel += f"{len(labels_rows):05d}.jpg"
        labels_rows.append({"file": rel, "orig_path": str(original), "source": source,
                            "layout": "single", "plate_no": plate})
        ledger[str(original)] = {"sha256": hashlib.sha256(payload).hexdigest(), "split": split}
        return original

    # Two candidate origins per subset suffice for a tiny deterministic holdout.
    for subset in prep.SUBSETS:
        for i in range(2):
            add_origin(f"hard/{subset}/{i}-02-03-18-05.jpg", f"{subset}-{i}".encode(),
                       f"hard_train:{subset}:ground_truth")
    selected_origin = next(Path(row["orig_path"]) for row in labels_rows
                           if row["source"] == "hard_train:blur:ground_truth")
    selected_sha = hashlib.sha256(selected_origin.read_bytes()).hexdigest()
    # All pipeline crops from a selected original must follow the same tune fold.
    labels_rows.append({"file": "train/pipeline-copy.jpg", "orig_path": str(selected_origin),
                        "source": "hard_train:blur:pipeline:legacy", "layout": "single", "plate_no": "A0000"})
    # A different origin with the same bytes must be withheld with the selected SHA.
    alias = tmp_path / "alias" / "same.jpg"
    alias.parent.mkdir(parents=True)
    alias.write_bytes(selected_origin.read_bytes())
    labels_rows.append({"file": "train/alias.jpg", "orig_path": str(alias),
                        "source": "hard_train:blur:ground_truth", "layout": "single", "plate_no": "A0000"})
    ledger[str(alias)] = {"sha256": selected_sha, "split": "train"}
    v1_origin = add_origin("v1seen/01-02-03-18-05.jpg", v1_seen_file.read_bytes(), "base_train")
    v1_hard_alias = add_origin("hard/blur/v1-alias-02-03-18-05.jpg", v1_seen_file.read_bytes(),
                                "hard_train:blur:ground_truth")
    val_protected = add_origin("val/protected.jpg", protected_file.read_bytes(), "ground_truth", split="val")
    val_protected_alias = add_origin("val/protected-alias.jpg", protected_file.read_bytes(), "ground_truth", split="val")
    add_origin("val/safe.jpg", safe_file.read_bytes(), "ground_truth", split="val")

    labels_path = data / "labels.csv"
    write_csv(labels_path, ["file", "orig_path", "source", "layout", "plate_no"], labels_rows)
    ledger_path = data / "source_hashes.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    v1_labels = v1 / "labels.csv"
    write_csv(v1_labels, ["file", "orig_path"], [{"file": "train/seen.jpg", "orig_path": str(v1_origin)}])

    protected_manifest = tmp_path / "protected.csv"
    write_csv(protected_manifest, ["path"], [{"path": str(val_protected)}])
    monkeypatch.setattr(prep, "PROTECTED", (("testset", protected_manifest),))
    excluded_file = tmp_path / "excluded.txt"
    excluded_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(prep, "V1_EXCLUDED", excluded_file)
    monkeypatch.setattr(prep, "V2_LABELS_SHA", prep.sha256_file(labels_path))
    monkeypatch.setattr(prep, "V2_LEDGER_SHA", prep.sha256_file(ledger_path))

    out = tmp_path / "out"
    audit = prep.prepare(data, v1, out, per_subset=2, fit_per_subset=1)
    with (out / "tune.csv").open(encoding="utf-8", newline="") as stream:
        tune = list(csv.DictReader(stream))
    with (out / "train.csv").open(encoding="utf-8", newline="") as stream:
        train = list(csv.DictReader(stream))
    with (out / "val.csv").open(encoding="utf-8", newline="") as stream:
        val = list(csv.DictReader(stream))

    assert len(audit["selected"]["fold_origins_by_subset"]) == 7
    assert len(tune) == 16  # 14 selected origins, plus one pipeline crop and one SHA alias.
    selected_rows = [row for row in tune if row["orig_path"] == str(selected_origin)]
    assert len({row["calibration_fold"] for row in selected_rows}) == 1
    assert {row["partition"] for row in selected_rows} == {"tune"}
    alias_rows = [row for row in tune if row["orig_path"] == str(alias)]
    assert len(alias_rows) == 1
    assert alias_rows[0]["calibration_fold"] == selected_rows[0]["calibration_fold"]
    assert not any(row["orig_path"] == str(alias) for row in train)
    assert not any(row["orig_path"] == str(v1_origin) for row in tune)
    assert not any(row["orig_path"] == str(v1_hard_alias) for row in tune)
    assert [row["orig_path"] for row in val] == [str(tmp_path / "val/safe.jpg")]
    assert str(val_protected_alias) not in {row["orig_path"] for row in val}
    assert all({"file", "orig_path", "source", "layout", "plate_no", "crop_path",
                "subset", "brightness", "partition"}.issubset(row) for row in tune + train + val)
