import csv
import hashlib
import json

import cv2
from lpr.charset import PROVINCES
from lpr.plate_rules import match_layout
from lpr_tools.train.synth_plates import DEFAULT_FONT, FIELDS, _plate_number, _render, generate_dataset


def test_rare_province_generation_is_train_only_and_reproducible(tmp_path):
    source = tmp_path / "real.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for province_index, province in enumerate(PROVINCES):
            count = 999 if province_index == 0 else 1000
            for _ in range(count):
                writer.writerow({"file": "train/example.jpg", "plate_no": province + "A12345",
                                 "layout": 7, "color": "blue", "source": "ground_truth", "orig_path": ""})
    first, second = tmp_path / "first", tmp_path / "second"
    a = generate_dataset(first, real_labels=source, min_per_province=1000, seed=17,
                         font_paths=(DEFAULT_FONT,))
    b = generate_dataset(second, real_labels=source, min_per_province=1000, seed=17,
                         font_paths=(DEFAULT_FONT,))
    assert a["rows"] == 1
    assert a["synthetic_train_by_province"] == b["synthetic_train_by_province"]
    with (first / "labels.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == a["rows"]
    assert len(rows) == 1
    assert all(row["file"].startswith("train/") and row["source"] == "font_synthetic" for row in rows)
    assert {row["layout"] for row in rows} <= {"7", "8"}
    assert all((first / row["file"]).read_bytes() == (second / row["file"]).read_bytes() for row in rows)
    sample = cv2.imread(str(first / rows[0]["file"]))
    assert sample is not None and sample.shape in ((80, 252, 3), (80, 276, 3))
    manifest = json.loads((first / "synthetic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_only"] is True
    assert manifest["fonts"][0]["sha256"] == hashlib.sha256(DEFAULT_FONT.read_bytes()).hexdigest()
    assert not (first / "val").exists()
    assert all(match_layout(row["plate_no"]) == f"L{row['layout']}" if row["layout"] == "7" else
               match_layout(row["plate_no"]) in {"L8S", "L8L"} for row in rows)


def test_generated_layouts_obey_plate_rules_and_render_real_sized_glyphs():
    rng = __import__("numpy").random.default_rng(9)
    for layout in (7, 8):
        for _ in range(100):
            plate = _plate_number("陕", rng, layout)
            assert match_layout(plate) is not None
            assert len(plate) == layout
            image = _render(plate, layout, (DEFAULT_FONT,), rng)
            assert image.shape == (80, 252 if layout == 7 else 276, 3)


def test_generation_rejects_bad_schema_and_nonempty_destination(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("file,plate_no\na.jpg,皖A12345\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="schema"):
        generate_dataset(tmp_path / "out", real_labels=bad, font_paths=(DEFAULT_FONT,))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        generate_dataset(tmp_path / "out", real_labels=bad, font_paths=(DEFAULT_FONT,))
