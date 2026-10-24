import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpr.io_utils import imread, imwrite
from lpr_tools.data.ccpd import INDEX_FIELDS, parse_ccpd_name
from lpr_tools.train.build_chars import build_chars


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
        build_chars(index, raw, tmp_path / "out")


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
