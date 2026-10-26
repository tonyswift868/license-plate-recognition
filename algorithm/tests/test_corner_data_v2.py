import csv
import hashlib
import json
from pathlib import Path

import pytest

from lpr_tools.synth_ccpd import (HARD_SUBSETS, _file_facts, _load_hash_cache,
                                  _normalized_path, _stable_image_sha, prepare_hard_pool)


FIELDS = ["path", "dataset", "subset", "split", "has_plate", "plate_no", "layout",
          "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
          "blurriness", "area", "skip_reason"]


def test_normalized_path_unifies_slashes_and_case() -> None:
    assert _normalized_path("C:/Temp/AbC.jpg") == _normalized_path("c:\\temp\\abc.JPG")


def test_persistent_hash_cache_rejects_stale_malformed_and_truncated_entries(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    changed = tmp_path / "changed.jpg"
    malformed_key = tmp_path / "bad-key.jpg"
    invalid_sha = tmp_path / "bad-sha.jpg"
    for path in (good, changed, malformed_key, invalid_sha):
        path.write_bytes(path.name.encode())

    def record(path: Path, key: str | None = None, sha: str | None = None) -> dict:
        size, mtime = _file_facts(path)
        return {"key": key or _normalized_path(str(path)), "path": str(path),
                "size": size, "mtime_ns": mtime,
                "sha256": sha or hashlib.sha256(path.read_bytes()).hexdigest()}

    good_row = record(good)
    changed_row = record(changed)
    bad_key_row = record(malformed_key, key="wrong/path")
    bad_sha_row = record(invalid_sha, sha="z" * 64)
    changed.write_bytes(b"new content changes the file facts")
    cache = tmp_path / "hashes.jsonl"
    cache.write_text("\n".join(json.dumps(row) for row in
                               (good_row, changed_row, bad_key_row, bad_sha_row))
                     + "\n{truncated", encoding="utf-8")
    digests, facts = _load_hash_cache(cache)
    assert set(digests) == {_normalized_path(str(good))}
    assert facts[_normalized_path(str(good))] == _file_facts(good)
    digest, stable_facts = _stable_image_sha(changed)
    assert digest == hashlib.sha256(changed.read_bytes()).hexdigest()
    assert stable_facts == _file_facts(changed)


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(path: Path, subset: str, split: str = "train") -> dict[str, str]:
    return {"path": str(path), "dataset": "fixture", "subset": subset, "split": split,
            "has_plate": "1", "plate_no": "皖A12345", "layout": "7", "plate_color": "blue",
            "quad": "10 10 90 10 90 30 10 30", "bbox": "10 10 80 20", "tilt_h": "90",
            "tilt_v": "90", "brightness": "100", "blurriness": "20", "area": "10",
            "skip_reason": ""}


def test_hard_pool_excludes_exact_bytes_and_keeps_partitions_disjoint(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    index_rows = []
    for subset in HARD_SUBSETS:
        for i in range(12):
            path = image_dir / f"{subset}-{i}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image-{subset}-{i}".encode())
            index_rows.append(_row(path, subset))
    # One direct path, one byte-identical renamed copy, a fixed-val duplicate,
    # and an NP-train duplicate exercise each exclusion boundary.
    clone = image_dir / "copy-of-forbidden.jpg"
    clone.write_bytes((image_dir / "ccpd_db-1.jpg").read_bytes())
    index_rows[1]["path"] = str(clone)
    index_path = tmp_path / "ccpd_index.csv"
    _write_csv(index_path, index_rows, FIELDS)
    frozen = tmp_path / "frozen.csv"
    golden = tmp_path / "golden.csv"
    devhard = tmp_path / "devhard.csv"
    undecodable = tmp_path / "undecodable.csv"
    excluded_direct = image_dir / "ccpd_db-0.jpg"
    excluded_golden = image_dir / "ccpd_db-1.jpg"
    excluded_devhard = image_dir / "ccpd_fn-2.jpg"
    excluded_bad = image_dir / "ccpd_tilt-3.jpg"
    for path, image in ((frozen, excluded_direct), (golden, excluded_golden),
                        (devhard, excluded_devhard), (undecodable, excluded_bad)):
        _write_csv(path, [{"path": str(image)}], ["path"])

    seventh = tmp_path / "seventh"
    train_path, val_path = image_dir / "safe-train.jpg", image_dir / "safe-val.jpg"
    np_train_path, np_val_path = image_dir / "safe-np-train.jpg", image_dir / "safe-np-val.jpg"
    train_path.write_bytes(b"safe-train")
    val_path.write_bytes((image_dir / "ccpd_challenge-4.jpg").read_bytes())
    np_train_path.write_bytes((image_dir / "ccpd_weather-5.jpg").read_bytes())
    np_val_path.write_bytes(b"safe-np-validation")
    train = _row(train_path, "ccpd_base")
    train_val = {**_row(val_path, "ccpd_base", "val"), "plate_no": "皖B12345"}
    np_train = {**_row(np_train_path, "ccpd_np"), "has_plate": "0", "quad": "", "plate_no": ""}
    np_val = {**_row(np_val_path, "ccpd_np", "val"), "has_plate": "0", "quad": "", "plate_no": ""}
    for name, rows in (("train_sampled_30000.csv", [train]), ("val_1000.csv", [train_val]),
                       ("np_train.csv", [np_train]), ("np_val.csv", [np_val])):
        _write_csv(seventh / name, rows, list(rows[0]))
    np_train_sha = hashlib.sha256((seventh / "np_train.csv").read_bytes()).hexdigest()
    np_val_sha = hashlib.sha256((seventh / "np_val.csv").read_bytes()).hexdigest()
    (seventh / "np-split-audit.json").write_text(json.dumps({
        "all_selected_decoded_valid": True,
        "csv": {"np_train.csv_sha256": np_train_sha, "np_val.csv_sha256": np_val_sha}},
        indent=2), encoding="utf-8")
    v1_csv = tmp_path / "selected_train.csv"
    _write_csv(v1_csv, [train, np_train], list(dict.fromkeys([*train.keys(), *np_train.keys()])))

    out = tmp_path / "out"
    audit = prepare_hard_pool(output_dir=out, index_path=index_path, frozen_path=frozen,
                              golden_path=golden, devhard_path=devhard,
                              undecodable_path=undecodable, seventh_data_dir=seventh,
                              v1_training_csv=v1_csv)
    with (out / "hard_pool_train.csv").open(encoding="utf-8", newline="") as handle:
        hard_train = list(csv.DictReader(handle))
    with (out / "hard_tuning_holdout.csv").open(encoding="utf-8", newline="") as handle:
        hard_val = list(csv.DictReader(handle))
    paths_train = {row["path"] for row in hard_train}
    paths_val = {row["path"] for row in hard_val}
    assert str(clone) not in paths_train | paths_val
    assert str(excluded_direct) not in paths_train | paths_val
    assert str(excluded_devhard) not in paths_train | paths_val
    assert str(excluded_bad) not in paths_train | paths_val
    assert paths_train.isdisjoint(paths_val)
    assert len({row["subset"] for row in hard_val}) == len(HARD_SUBSETS)
    assert all(row["split"] == "train" and row["source_split"] == "train" for row in hard_train)
    assert all(row["split"] == "val" for row in hard_val)
    assert audit["excluded_unique_sha256"] == 4
    assert audit["disjointness"]["selected_forbidden_sha_intersection"] == 0
    assert audit["disjointness"]["training_vs_tuning_content_intersection"] == 0
    assert audit["safe_partitions"]["safe_val"]["selected"] == 1
    assert audit["safe_partitions"]["np_train"]["selected"] == 1
    assert audit["safe_partitions"]["np_val"]["selected"] == 1
    assert audit["safe_partitions"]["corner_net_v1_selected_training_set"]["count"] == 2
    assert audit["safe_partitions"]["np_safe_partition"]["all_selected_decoded_valid"] is True
    saved_audit = json.loads((out / "hard_pool_audit.json").read_text(encoding="utf-8"))
    assert saved_audit["hard_pool_by_subset"]["ccpd_db"]["available_after_exclusion_and_byte_dedup"] == 10
    expected_csvs = {"hard_pool_train.csv", "hard_tuning_holdout.csv", "safe_train.csv", "safe_val.csv",
                     "np_train_safe.csv", "np_val_safe.csv"}
    for name in expected_csvs:
        assert (out / name).is_file()
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() in saved_audit["csv_sha256"].values()
    assert set(saved_audit["manifest_sha256"]) == {"frozen", "golden", "devhard", "undecodable"}
    provenance = saved_audit["safe_partitions"]["np_safe_partition"]["inherited_decode_audit"]
    assert provenance["audit_train_csv_sha256"] == np_train_sha
    (seventh / "np_train.csv").write_bytes((seventh / "np_train.csv").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="NP decode proof"):
        prepare_hard_pool(output_dir=out, index_path=index_path, frozen_path=frozen,
                          golden_path=golden, devhard_path=devhard,
                          undecodable_path=undecodable, seventh_data_dir=seventh,
                          v1_training_csv=v1_csv)
