import csv

import cv2
import numpy as np
import pytest
from lpr.charset import CHARSET
from lpr.io_utils import imread
from lpr_tools.train.synth_chars import DEFAULT_FONT, generate_dataset


def test_generate_all_labels_and_reproduce(default_font, tmp_path):
    first, second = tmp_path / "中文样本", tmp_path / "repeat"
    rows = generate_dataset(first, per_class=4)
    generate_dataset(second, per_class=4)
    assert len(rows) == 260
    with (first / "chars_index.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(rows)
    for split, expected in (("train", 3), ("val", 1)):
        for label, char in enumerate(CHARSET):
            subset = [row for row in rows if row["split"] == split and row["label"] == label]
            assert len(subset) == expected and all(row["char"] == char for row in subset)
    for row in rows:
        path = first / row["path"]
        image = imread(path, cv2.IMREAD_GRAYSCALE)
        assert image.shape == (32, 32) and image.dtype == np.uint8
        assert image.max() > 50 and image.mean() < 150
        assert path.read_bytes() == (second / row["path"]).read_bytes()
    assert len({row["path"] for row in rows}) == len(rows)
    with pytest.raises(ValueError, match="empty"):
        generate_dataset(first, per_class=2)


def test_generation_rejects_bad_parameters_and_missing_font(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "bad", per_class=1)
    with pytest.raises(OSError):
        generate_dataset(tmp_path / "missing", font_path=tmp_path / "missing.ttf")
    assert not (tmp_path / "missing").exists()
    if not DEFAULT_FONT.is_file():
        pytest.skip(f"missing local font: {DEFAULT_FONT}")


def test_province_train_only_and_per_label_counts(default_font, tmp_path):
    out = tmp_path / "province_train"
    rows = generate_dataset(out, per_class=2, provinces_only=True, train_only=True,
                            class_counts={"陕": 5})
    assert len(rows) == 30 * 2 + 5
    assert {row["split"] for row in rows} == {"train"}
    assert {row["label"] for row in rows} == set(range(31))
    assert sum(row["char"] == "陕" for row in rows) == 5
    assert sum(row["char"] == "皖" for row in rows) == 2
    assert not (out / "val").exists()
    with (out / "chars_index.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(rows)


def test_selected_labels_default_validation_remains_split(default_font, tmp_path):
    rows = generate_dataset(tmp_path / "selected", labels=["陕", "31"], per_class=4)
    assert {row["char"] for row in rows} == {"陕", "0"}
    assert {row["split"] for row in rows} == {"train", "val"}
    assert sum(row["split"] == "val" for row in rows) == 2
