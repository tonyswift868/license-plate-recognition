import csv
from pathlib import Path

import pytest

from lpr_tools.data.ccpd import INDEX_FIELDS
from lpr_tools.data.manifest import ManifestRow, write_manifest
from lpr_tools.data.pick_golden import LABEL_FIELDS, pick_golden


def _make_index(tmp_path: Path, *, missing_bucket: str | None = None,
                extra_base: int = 0) -> Path:
    raw = tmp_path / "raw"
    entries = []
    specs = [
        ("ccpd2019", "ccpd_base", 10, "1"),
        ("ccpd2020", "ccpd_green", 2, "1"),
        ("ccpd2019", "ccpd_db", 2, "1"),
        ("ccpd2019", "ccpd_tilt", 2, "1"),
        ("ccpd2019", "ccpd_blur", 2, "1"),
        ("ccpd2019", "ccpd_np", 2, "0"),
    ]
    for dataset, subset, count, has_plate in specs:
        if subset == "ccpd_base":
            count += extra_base
        for i in range(count):
            # Train distractors must never be selected.
            for split in ("test", "train"):
                path = raw / dataset / subset / split / f"img-{i}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{subset}-{split}-{i}".encode())
                row = {field: "" for field in INDEX_FIELDS}
                row.update(path=str(path.relative_to(tmp_path)), dataset=dataset, subset=subset,
                           split=split, has_plate=has_plate,
                           plate_no="" if has_plate == "0" else "皖A12345",
                           plate_color="green" if dataset == "ccpd2020" else "blue",
                           skip_reason="no_plate" if has_plate == "0" else "")
                entries.append(row)
    if missing_bucket:
        entries = [r for r in entries if r["subset"] != missing_bucket or r["split"] != "test"]
    index = tmp_path / "ccpd_index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(entries)
    return index


def test_pick_golden_exact_buckets_labels_and_relative_paths(tmp_path):
    index = _make_index(tmp_path)
    out = tmp_path / "picked"
    labels = pick_golden(index, out)
    assert len(labels) == 20
    with (out / "labels.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == LABEL_FIELDS
        rows = list(reader)
    assert len(rows) == 20
    assert sum(row["category"] == "clear" for row in rows) == 12
    assert sum(row["category"] == "night" for row in rows) == 2
    assert sum(row["category"] == "tilt" for row in rows) == 2
    assert sum(row["category"] == "dirty" for row in rows) == 2
    no_plate = [row for row in rows if row["category"] == "np"]
    assert len(no_plate) == 2 and all(row["plate_no"] == "" for row in no_plate)
    assert len({row["file"] for row in rows}) == 20
    assert len(list(out.glob("*.jpg"))) == 20
    assert all("train" not in row["file"] for row in rows)


def test_pick_golden_shortage_leaves_no_output(tmp_path):
    index = _make_index(tmp_path, missing_bucket="ccpd_blur")
    out = tmp_path / "not-created"
    with pytest.raises(ValueError, match="ccpd_blur: need 2"):
        pick_golden(index, out)
    assert not out.exists()


def test_pick_golden_refuses_existing_output_and_protected_path(tmp_path):
    index = _make_index(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        pick_golden(index, out)
    assert sentinel.read_text() == "keep"

    protected = Path(__file__).resolve().parent / "data" / "golden"
    with pytest.raises(ValueError, match="protected"):
        pick_golden(index, protected)


def _write_rows(index: Path, rows: list[dict[str, str]]) -> None:
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(index: Path) -> list[dict[str, str]]:
    with index.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pick_golden_seed_is_reproducible_with_extra_candidates(tmp_path):
    index = _make_index(tmp_path, extra_base=6)
    first = pick_golden(index, tmp_path / "first")
    second = pick_golden(index, tmp_path / "second")
    assert first == second


def test_pick_golden_rejects_cross_split_path_alias_without_output(tmp_path):
    index = _make_index(tmp_path)
    rows = _read_rows(index)
    test_row = next(r for r in rows if r["subset"] == "ccpd_base" and r["split"] == "test")
    train_row = next(r for r in rows if r["subset"] == "ccpd_base" and r["split"] == "train")
    train_row["path"] = "raw/ccpd2019/ccpd_base/test/img-0.jpg"
    # A different lexical spelling must canonicalize to the same image too.
    test_row["path"] = "raw/ccpd2019/ccpd_base/test/../test/img-0.jpg"
    _write_rows(index, rows)
    out = tmp_path / "conflict-output"
    with pytest.raises(ValueError, match="multiple splits"):
        pick_golden(index, out)
    assert not out.exists()


def test_pick_golden_excludes_invalid_plate_labels_without_output(tmp_path):
    index = _make_index(tmp_path)
    rows = _read_rows(index)
    for row in rows:
        if row["subset"] == "ccpd_base" and row["split"] == "test":
            row["plate_no"] = "not-a-plate"
    _write_rows(index, rows)
    out = tmp_path / "invalid-label-output"
    with pytest.raises(ValueError, match="ccpd_base: need 10"):
        pick_golden(index, out)
    assert not out.exists()


def test_pick_golden_rejects_same_image_selected_through_alias(tmp_path):
    index = _make_index(tmp_path)
    rows = _read_rows(index)
    db = next(r for r in rows if r["subset"] == "ccpd_db" and r["split"] == "test")
    db["path"] = "raw/ccpd2019/ccpd_base/test/../test/img-0.jpg"
    _write_rows(index, rows)
    out = tmp_path / "alias-output"
    with pytest.raises(ValueError, match="selected more than once"):
        pick_golden(index, out)
    assert not out.exists()


def test_pick_golden_missing_source_leaves_no_output(tmp_path):
    index = _make_index(tmp_path)
    rows = _read_rows(index)
    for row in rows:
        if row["subset"] == "ccpd_base" and row["split"] == "test":
            (index.parent / row["path"]).unlink()
    out = tmp_path / "missing-source-output"
    with pytest.raises(FileNotFoundError, match="missing source"):
        pick_golden(index, out)
    assert not out.exists()


def test_pick_golden_manifest_limits_sources_and_uses_reserved_base_val(tmp_path):
    specs = [
        ("ccpd_base", "clear", 120, "blue", "皖A12345"),
        ("ccpd_green", "clear", 30, "green", "皖AD12345"),
        ("ccpd_db", "night", 4, "blue", "皖A12345"),
        ("ccpd_tilt", "tilt", 4, "blue", "皖A12345"),
        ("ccpd_blur", "dirty", 4, "blue", "皖A12345"),
        ("ccpd_np", "np", 20, "unknown", ""),
    ]
    rows = []
    for subset, category, count, color, plate_no in specs:
        for n in range(count):
            path = tmp_path / "raw" / subset / f"{subset}-{n:03}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"frozen-{subset}-{n}".encode())
            rows.append(ManifestRow(str(path), plate_no, color, category, subset))
    # This tempting unmanifested base-val candidate must never be selected.
    outsider = tmp_path / "raw" / "ccpd_base" / "outside.jpg"
    outsider.write_bytes(b"outside-manifest")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    out = tmp_path / "golden"
    labels = pick_golden(None, out, manifest=manifest)
    assert len(labels) == 20
    assert sum(r["source"] == "ccpd_base" and r["category"] == "clear" for r in labels) == 10
    assert sum(r["source"] == "ccpd_green" and r["category"] == "clear" for r in labels) == 2
    assert {r["source"] for r in labels} == {spec[0] for spec in specs}
    frozen_paths = {str(Path(r.path).resolve()) for r in rows}
    copied_bytes = {(out / row["file"]).read_bytes() for row in labels}
    expected_bytes = {Path(path).read_bytes() for path in frozen_paths}
    assert copied_bytes <= expected_bytes
    assert b"outside-manifest" not in copied_bytes


def test_pick_golden_hybrid_uses_manifest_for_base_and_official_test_fallback(tmp_path):
    index = _make_index(tmp_path, extra_base=110)
    index_rows = _read_rows(index)
    manifest_rows = []
    for row in index_rows:
        if row["subset"] != "ccpd_base" or row["split"] != "test":
            continue
        manifest_rows.append(ManifestRow(row["path"], row["plate_no"], row["plate_color"],
                                         "clear", "ccpd_base"))
    manifest = tmp_path / "frozen.csv"
    write_manifest(manifest, manifest_rows)

    out = tmp_path / "hybrid"
    labels = pick_golden(index, out, manifest=manifest)
    assert len(labels) == 20
    assert sum(r["source"] == "ccpd_base" and r["category"] == "clear" for r in labels) == 10
    assert all(r["source"] == "ccpd_base" for r in labels[:10])
    with (out / "golden_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_reader = csv.DictReader(handle)
        assert tuple(manifest_reader.fieldnames) == ("path", "plate_no", "plate_color", "category", "source", "quad", "note")
        golden_rows = list(manifest_reader)
    assert len(golden_rows) == 20
    assert sum(row["note"] == "frozen_manifest" for row in golden_rows) == 10
    assert sum(row["note"] == "golden_only" for row in golden_rows) == 10
