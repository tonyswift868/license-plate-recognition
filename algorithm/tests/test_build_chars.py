import csv
import json
import shutil
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpr.io_utils import imread, imwrite
from lpr.types import SegmentResult
from lpr_tools.data.ccpd import INDEX_FIELDS, parse_ccpd_name
from lpr_tools.train.build_chars import _iter_index, build_chars


def _name(plate_indices="0_0_22_27_27_33_16"):
    return f"025-0_0-0&0_251&79-251&79_0&79_0&0_251&0-{plate_indices}-37-15.jpg"


def _row(path, split="train", plate="皖AY339S", *, quad="0 0 251 0 251 79 0 79", **changes):
    row = dict.fromkeys(INDEX_FIELDS, "")
    row.update(path=str(path), dataset="ccpd2019", subset="ccpd_base", split=split,
               has_plate="1", plate_no=plate, layout=str(len(plate)), plate_color="blue",
               quad=quad, bbox="0 0 252 80", skip_reason="")
    row.update(changes)
    return row


def _write_index(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _image(path, size=(252, 80), value=0):
    image = np.full((size[1], size[0], 3), value, np.uint8)
    for x in range(12, size[0]-12, 30):
        cv2.rectangle(image, (x, 20), (x+10, 60), (255, 255, 255), -1)
    imwrite(path, image)


def test_build_chars_filters_excludes_caps_and_uses_shared_normalization(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / _name()
    _image(source)
    record = parse_ccpd_name(source.name)
    assert record.plate_no == "皖AY339S"
    index = tmp_path / "ccpd_index.csv"
    excluded = raw / "excluded.jpg"
    _image(excluded, value=3)
    test_source = raw / "test.jpg"
    _image(test_source, value=7)
    _write_index(index, [
        _row(source),
        _row(excluded, split="val", plate="沪B12345"),
        _row(test_source, split="test", plate="津C12345"),
        _row(source, split="train", has_plate="0"),
        _row(source, split="train", skip_reason="placeholder"),
    ])
    manifest = tmp_path / "testset.csv"
    manifest.write_text("path\n" + str(excluded) + "\n", encoding="utf-8")
    out = tmp_path / "chars"
    counts = build_chars(index, raw, out, exclude_manifest=manifest, max_per_class=1)
    assert sum(counts.values()) == 6
    assert len(list((out / "train").glob("*/*.png"))) == 6
    assert not (out / "val").exists()
    rows = list(csv.DictReader((out / "chars_index.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    assert {r["split"] for r in rows} == {"train"}
    assert all(imread(out / r["path"]).shape == (32, 32, 3) for r in rows)


def test_build_chars_rejects_cross_split_duplicate_content(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    first, second = raw / "one.jpg", raw / "two.jpg"
    _image(first)
    shutil.copyfile(first, second)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(first, "train"), _row(second, "val", "沪B12345")])
    with pytest.raises(ValueError, match="leakage across splits"):
        build_chars(index, raw, tmp_path / "out", skip_invalid=True)


def test_build_chars_rejects_train_test_duplicate_before_writing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    first, second = raw / "train.jpg", raw / "test.jpg"
    _image(first)
    shutil.copyfile(first, second)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(first, "train"), _row(second, "test")])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="leakage across splits"):
        build_chars(index, raw, out)
    assert not out.exists()


@pytest.mark.parametrize("manifest_relative_base", ["manifest", "raw"])
def test_build_chars_rejects_train_copy_of_excluded_manifest_image(tmp_path, manifest_relative_base):
    raw = tmp_path / "raw"
    manifest_dir = tmp_path / "manifests"
    raw.mkdir()
    manifest_dir.mkdir()
    excluded = (manifest_dir if manifest_relative_base == "manifest" else raw) / "heldout.jpg"
    train_copy = raw / "renamed-train-copy.jpg"
    _image(excluded)
    shutil.copyfile(excluded, train_copy)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(train_copy, "train"), _row(excluded, "val", "沪B12345")])
    manifest = manifest_dir / "testset.csv"
    manifest.write_text("path\nheldout.jpg\n", encoding="utf-8")
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="image content matches exclude manifest"):
        build_chars(index, raw, out, exclude_manifest=manifest,
                    skip_invalid=True, workers=2)
    assert not out.exists()


def test_build_chars_allows_test_candidate_matching_exclude_manifest_content(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    test_image = raw / "test.jpg"
    train_copy = raw / "train-copy.jpg"
    _image(test_image)
    shutil.copyfile(test_image, train_copy)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(train_copy, "train"), _row(test_image, "test")])
    manifest = tmp_path / "testset.csv"
    manifest.write_text("path\n" + str(test_image) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image content matches exclude manifest"):
        build_chars(index, raw, tmp_path / "out", exclude_manifest=manifest)

    # The manifest's own test-split row is allowed to share its content hash;
    # with no train/val alias, this build completes and emits only train data.
    index = tmp_path / "test-only-index.csv"
    _write_index(index, [_row(test_image, "test")])
    counts = build_chars(index, raw, tmp_path / "test-only-out", exclude_manifest=manifest)
    assert sum(counts.values()) == 0


def test_build_chars_class_cap_reserves_validation_quota(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    train_image, val_image = raw / "train.jpg", raw / "val.jpg"
    _image(train_image, value=0)
    _image(val_image, value=1)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(train_image, "train"), _row(val_image, "val")])
    out = tmp_path / "out"
    counts = build_chars(index, raw, out, max_per_class=2)
    rows = list(csv.DictReader((out / "chars_index.csv").open(encoding="utf-8")))
    province_rows = [r for r in rows if r["char"] == "皖"]
    assert counts[0] == 2
    assert {r["split"] for r in province_rows} == {"train", "val"}


def test_build_chars_rejects_invalid_quad(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "bad.jpg"
    _image(source)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(source, quad="0 0 1 1")])
    with pytest.raises(ValueError, match="invalid quad"):
        build_chars(index, raw, tmp_path / "out")


def test_build_chars_emits_seven_and_eight_slot_samples_with_green_polarity(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    seven = raw / "seven.jpg"
    eight = raw / "eight.jpg"
    _image(seven)
    green = 255 - np.asarray(imread(seven))
    green = cv2.resize(green, (276, 80))
    imwrite(eight, green)
    index = tmp_path / "index.csv"
    eight_plate = "皖AD12345"
    _write_index(index, [
        _row(seven, "train", "皖AY339S"),
        _row(eight, "val", eight_plate, quad="0 0 275 0 275 79 0 79", plate_color="green"),
    ])
    out = tmp_path / "out"
    counts = build_chars(index, raw, out)
    rows = list(csv.DictReader((out / "chars_index.csv").open(encoding="utf-8")))
    train, val = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    assert len(train) == 7 and len(val) == 8 and sum(counts.values()) == 15
    assert {r["plate_no"] for r in val} == {eight_plate}
    for row in rows:
        crop = imread(out / row["path"], cv2.IMREAD_GRAYSCALE)
        assert crop.shape == (32, 32)
        assert crop.max() > 0
    # The inverted green source must normalize dark source glyphs to bright ink.
    green_crop = imread(out / val[0]["path"], cv2.IMREAD_GRAYSCALE)
    assert np.count_nonzero(green_crop > 128) > 0


def test_build_chars_rejects_nonempty_output_and_duplicate_source_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "one.jpg"
    _image(source)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(source), _row(source)])
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="output directory is not empty"):
        build_chars(index, raw, out)
    assert (out / "sentinel").read_text(encoding="utf-8") == "keep"
    out_empty = tmp_path / "out-empty"
    with pytest.raises(ValueError, match="duplicate source path"):
        build_chars(index, raw, out_empty)
    assert not any(out_empty.iterdir()) if out_empty.exists() else True


def test_missing_test_image_is_ignored_but_its_path_still_blocks_train(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    train_image = raw / "train.jpg"
    _image(train_image)
    missing_test = raw / "missing-test.jpg"
    index = tmp_path / "index.csv"
    _write_index(index, [_row(train_image), _row(missing_test, "test")])
    counts = build_chars(index, raw, tmp_path / "out")
    assert sum(counts.values()) == 7


def test_skip_invalid_records_both_quad_boundary_errors_but_still_rejects_leakage(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    bad_x, bad_y, valid = raw / "bad-x.jpg", raw / "bad-y.jpg", raw / "valid.jpg"
    for path in (bad_x, bad_y, valid):
        _image(path)
    index = tmp_path / "index.csv"
    _write_index(index, [
        _row(bad_x, quad="0 0 252 0 252 79 0 79"),
        _row(bad_y, quad="0 0 251 0 251 80 0 80"),
        _row(valid),
    ])
    out = tmp_path / "skip-out"
    counts = build_chars(index, raw, out, skip_invalid=True)
    assert sum(counts.values()) == 7
    rejected = list(csv.DictReader((out / "chars_rejected.csv").open(encoding="utf-8")))
    assert [(r["row"], r["split"], Path(r["path"]).name) for r in rejected] == [
        ("2", "train", bad_x.name), ("3", "train", bad_y.name)]
    assert "point outside source image" in rejected[0]["reason"]
    stats = json.loads((out / "chars_stats.json").read_text(encoding="utf-8"))
    assert stats["rejected_count"] == 2 and stats["skip_invalid"] is True

    # Even when malformed rows are skippable, cross-split content reuse is fatal.
    other = raw / "same-content-val.jpg"
    shutil.copyfile(valid, other)
    _write_index(index, [_row(valid, "train"), _row(other, "val")])
    with pytest.raises(ValueError, match="leakage across splits"):
        build_chars(index, raw, tmp_path / "leak-out", skip_invalid=True)


def test_preflight_workers_produce_identical_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    train_image, val_image = raw / "train.jpg", raw / "val.jpg"
    _image(train_image, value=0)
    _image(val_image, value=1)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(train_image, "train"), _row(val_image, "val")])
    out_one, out_two = tmp_path / "out-one", tmp_path / "out-two"
    build_chars(index, raw, out_one, workers=1)
    build_chars(index, raw, out_two, workers=2)
    assert (out_one / "chars_index.csv").read_bytes() == (out_two / "chars_index.csv").read_bytes()
    assert (out_one / "chars_stats.json").read_bytes() == (out_two / "chars_stats.json").read_bytes()


def test_index_reader_is_strict_and_streams_large_csv(tmp_path):
    index = tmp_path / "large-index.csv"
    row = dict.fromkeys(INDEX_FIELDS, "")
    row.update(path="x.jpg", dataset="ccpd2019", subset="ccpd_base", split="train",
               has_plate="1", plate_no="皖AY339S", layout="7", plate_color="blue",
               quad="0 0 251 0 251 79 0 79", bbox="0 0 252 80", skip_reason="")
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows([row] * 12000)
    tracemalloc.start()
    count = sum(1 for _ in _iter_index(index))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert count == 12000
    assert peak < 2_000_000

    bad_header = tmp_path / "bad-header.csv"
    bad_header.write_text("split,path\ntrain,x.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CCPD index columns"):
        next(_iter_index(bad_header))
    short_row = tmp_path / "short-row.csv"
    short_row.write_text(",".join(INDEX_FIELDS) + "\n" + ",".join([""] * (len(INDEX_FIELDS)-1)) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CCPD index row width"):
        next(_iter_index(short_row))


def test_ground_truth_quad_extracts_plate_from_offset_scene(tmp_path):
    from lpr.charnorm import normalize_char
    from lpr.config import load_config

    cfg = load_config()
    # A nonuniform scene makes ignoring or reordering the GT corners observable.
    plate = np.random.default_rng(12).integers(0, 256, (80, 252), dtype=np.uint8)
    scene = np.zeros((140, 340, 3), np.uint8)
    scene[25:105, 40:292] = plate[..., None]
    source = tmp_path / "offset.png"
    imwrite(source, scene)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(source, quad="40 25 291 25 291 104 40 104")])
    out = tmp_path / "chars"
    build_chars(index, tmp_path, out)
    with (out / "chars_index.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for position in (0, 6):
        center = cfg.segment.layout7_centers[position] * 252
        half = cfg.segment.layout7_char_w * 252 / 2
        left, right = int(np.floor(center - half)), int(np.ceil(center + half))
        expected = normalize_char(plate[:, left:right])
        actual = imread(out / rows[position]["path"], cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(actual, expected)


def test_segment_source_saves_exact_segment_char_bytes_and_uses_gt_rectification(tmp_path, monkeypatch):
    import lpr_tools.train.build_chars as builder
    from lpr.config import load_config

    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "scene.png"
    _image(source, size=(300, 120), value=19)
    quad = "20 15 271 15 271 94 20 94"
    index = tmp_path / "index.csv"
    _write_index(index, [_row(source, quad=quad)])
    expected = np.stack([np.full((32, 32), 20 + i * 21, np.uint8) for i in range(7)])
    seen = []

    def fake_segment(rect, layout, cfg):
        seen.append((rect, layout, cfg))
        return SegmentResult(True, layout, [], expected.copy(), 1.0)

    monkeypatch.setattr(builder, "segment_plate", fake_segment)
    out = tmp_path / "segment-chars"
    counts = builder.build_chars(index, raw, out, source="segment")
    cfg = load_config()
    assert len(seen) == 1
    rect, layout, passed_cfg = seen[0]
    assert layout == 7 and passed_cfg.model_dump() == cfg.segment.model_dump()
    assert rect.image.shape == (cfg.rectify.size_7[1], cfg.rectify.size_7[0], 3)
    np.testing.assert_allclose(rect.candidate.corners,
                               np.asarray([[20, 15], [271, 15], [271, 94], [20, 94]], np.float32))
    assert sum(counts.values()) == 7
    rows = list(csv.DictReader((out / "chars_index.csv").open(encoding="utf-8")))
    assert [int(row["position"]) for row in rows] == list(range(7))
    for row in rows:
        saved = imread(out / row["path"], cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(saved, expected[int(row["position"])])
    stats = json.loads((out / "chars_stats.json").read_text(encoding="utf-8"))
    assert stats["build"]["source"] == "segment"
    assert stats["build"]["config_sha256"]
    assert stats["build"]["build_chars_sha256"]
    assert stats["build"]["segment_code_sha256"]
    assert "segment_git_commit" in stats["build"] and "segment_git_blob" in stats["build"]


def test_segment_failures_and_wrong_character_count_drop_whole_images(tmp_path, monkeypatch):
    import lpr_tools.train.build_chars as builder

    raw = tmp_path / "raw"
    raw.mkdir()
    first, second = raw / "first.png", raw / "second.png"
    _image(first)
    _image(second, value=1)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(first), _row(second, plate="沪B12345")])
    results = iter([
        SegmentResult(False, 7, [], np.empty((0, 32, 32), np.uint8), 0.),
        SegmentResult(True, 7, [], np.zeros((6, 32, 32), np.uint8), 1.),
    ])
    monkeypatch.setattr(builder, "segment_plate", lambda *_args: next(results))
    out = tmp_path / "out"
    counts = builder.build_chars(index, raw, out, source="segment")
    assert sum(counts.values()) == 0
    assert list((out / "train").glob("*/*.png")) == []
    assert list((out / "val").glob("*/*.png")) == []
    stats = json.loads((out / "chars_stats.json").read_text(encoding="utf-8"))
    assert stats["segment_ok_false_images"] == 1
    assert stats["segment_wrong_count_images"] == 1
    assert stats["segment_dropped_images"] == 2


def test_segment_malformed_success_outputs_are_counted_and_dropped(tmp_path, monkeypatch):
    import lpr_tools.train.build_chars as builder

    raw = tmp_path / "raw"
    raw.mkdir()
    sources = [raw / f"bad-{i}.png" for i in range(3)]
    for i, path in enumerate(sources):
        _image(path, value=10 + i)
    index = tmp_path / "index.csv"
    _write_index(index, [_row(path) for path in sources])
    malformed = iter([
        np.asarray(1, dtype=np.uint8),
        np.zeros((7, 32, 32), dtype=np.uint16),
        np.zeros((7, 31, 32), dtype=np.uint8),
    ])
    monkeypatch.setattr(
        builder, "segment_plate",
        lambda _rect, layout, _cfg: SegmentResult(True, layout, [], next(malformed), 1.0))
    out = tmp_path / "out"
    counts = builder.build_chars(index, raw, out, source="segment")
    assert sum(counts.values()) == 0
    assert not list(out.glob("train/*/*.png"))
    stats = json.loads((out / "chars_stats.json").read_text(encoding="utf-8"))
    assert stats["valid_source_images"] == 3
    assert stats["segment_attempted_images"] == 3
    assert stats["segment_success_images"] == 3
    assert stats["segment_invalid_images"] == 3
    assert stats["segment_wrong_count_images"] == 0
    assert stats["segment_dropped_images"] == 3


def test_exclude_golden_manifest_blocks_renamed_byte_identical_copy(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest_dir = tmp_path / "golden"
    manifest_dir.mkdir()
    golden_image = manifest_dir / "golden.png"
    renamed_copy = raw / "renamed-golden.png"
    _image(golden_image)
    shutil.copyfile(golden_image, renamed_copy)
    manifest = manifest_dir / "golden_manifest.csv"
    manifest.write_text(
        "path,plate_no,plate_color,category,source,quad,note\n"
        "golden.png,皖AY339S,blue,clear,fixture,0 0 251 0 251 79 0 79,golden_only\n",
        encoding="utf-8")
    index = tmp_path / "index.csv"
    _write_index(index, [_row(renamed_copy)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="image content matches exclude manifest"):
        build_chars(index, raw, out, exclude_golden=manifest, skip_invalid=True)
    assert not out.exists()
