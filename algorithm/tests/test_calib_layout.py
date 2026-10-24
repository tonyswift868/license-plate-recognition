import csv
import tomllib

import cv2
import numpy as np

from lpr.io_utils import imwrite
from lpr_tools.calib_layout import calibrate
from lpr_tools.data.ccpd import INDEX_FIELDS
from lpr_tools.synth_scenes import draw_plate


def _row(path, dataset, subset, split, layout, color, has_plate="1", skip_reason=""):
    width = 252 if layout == 7 else 276
    return {
        "path": str(path), "dataset": dataset, "subset": subset, "split": split,
        "has_plate": has_plate, "plate_no": "皖AB1234" if layout == 7 else "皖AD12345",
        "layout": str(layout), "plate_color": color,
        "quad": f"0 0 {width-1} 0 {width-1} 79 0 79",
        "bbox": f"0 0 {width} 80", "tilt_h": "0", "tilt_v": "0", "brightness": "100",
        "blurriness": "0", "area": "1", "skip_reason": skip_reason,
    }


def test_calibrate_truth_warp_val_geometry_and_all_sample_metrics(tmp_path):
    raw = tmp_path / "raw"
    blue = raw / "ccpd2019" / "ccpd_base" / "blue.png"
    green = raw / "ccpd2020" / "ccpd_green" / "val" / "green.png"
    no_plate = raw / "ccpd2019" / "ccpd_np" / "empty.png"
    imwrite(blue, cv2.resize(draw_plate("皖AB1234", "blue"), (252, 80)))
    imwrite(green, cv2.resize(draw_plate("皖AD12345", "green"), (276, 80)))
    imwrite(no_plate, np.zeros((80, 276, 3), np.uint8))

    rows = [
        _row(blue, "ccpd2019", "ccpd_base", "val", 7, "blue"),
        _row(green, "ccpd2020", "ccpd_green", "val", 8, "green"),
        _row(no_plate, "ccpd2019", "ccpd_np", "test", 7, "blue", "0", "no_plate"),
    ]
    index = tmp_path / "ccpd_index.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = calibrate(index, raw, n=3, seed=42)
    assert result["selected"] == 3
    assert result["counts"]["skipped:no_plate"] == 1
    assert result["metrics"]["overall"] == {"denominator": 2, "correct": 2}
    assert result["geometry"]["7"]["n"] == result["geometry"]["8"]["n"] == 1
    assert len(result["geometry"]["7"]["centers"]) == 7
    assert len(result["geometry"]["8"]["centers"]) == 8
    assert result["counts"]["observed"] == 2
    assert result["counts"]["observed_val"] == 2
    assert result["split_counts"] == {"test": 1, "val": 2}


def test_select_consumes_all_rows_caps_n_and_covers_layouts():
    from lpr_tools.calib_layout import _select

    rows = [{"dataset": "ccpd2019", "subset": "ccpd_base", "layout": str(layout),
             "path": f"{layout}-{i}", "plate_no": "x"}
            for layout in (7, 8) for i in range(4)]
    assert len(_select(rows, 20, 42)) == len(rows)
    capped = _select(rows, 2, 42)
    assert len(capped) == 2
    assert {row["layout"] for row in capped} == {"7", "8"}
    assert _select(rows, 20, 42) == _select(rows, 20, 42)


def test_projection_recovers_known_centers_and_split_province_strokes():
    from lpr_tools.calib_layout import _measure_slots

    binary = np.zeros((80, 252), np.uint8)
    # Two disconnected province strokes form one character; remaining bars encode
    # known centers and are intentionally independent of default.toml slot values.
    for x0, x1 in [(16, 20), (24, 30), (49, 61), (94, 106), (126, 138),
                    (158, 170), (190, 202), (222, 234)]:
        binary[20:60, x0:x1] = 255
    measured = _measure_slots(binary, 7)
    expected = np.array([23, 55, 100, 132, 164, 196, 228]) / 252
    assert np.allclose([x[0] for x in measured], expected, atol=1 / 252)
    assert np.allclose(measured[0][1], 14 / 252)


def _write_index(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_bad_quad_missing_image_and_np_are_counted_and_fail_metric(tmp_path):
    raw = tmp_path / "raw"
    valid = raw / "ccpd2019" / "ccpd_base" / "valid.png"
    imwrite(valid, cv2.resize(draw_plate("皖AB1234", "blue"), (252, 80)))
    missing = raw / "ccpd2019" / "ccpd_base" / "missing.png"
    bad_quad = _row(valid, "ccpd2019", "ccpd_base", "test", 7, "blue")
    bad_quad["quad"] = "0 0 0 0 0 0 0 0"
    rows = [
        _row(valid, "ccpd2019", "ccpd_base", "test", 7, "blue"),
        _row(missing, "ccpd2019", "ccpd_base", "test", 7, "blue"),
        bad_quad,
        _row(tmp_path / "outside.png", "ccpd2019", "ccpd_base", "test", 7, "blue"),
        _row(raw / "unused.png", "ccpd2019", "ccpd_np", "test", 7, "blue", "0", "no_plate"),
    ]
    index = tmp_path / "bad_index.csv"
    _write_index(index, rows)
    result = calibrate(index, raw, n=10)
    assert result["metrics"]["overall"] == {"denominator": 4, "correct": 1}
    assert result["counts"]["eligible_labeled"] == 4
    assert result["counts"]["readable_rectified"] == 1
    assert result["counts"]["failed_read_or_quad"] == 3
    assert result["counts"]["path_outside_raw"] == 1
    assert result["counts"]["detail:FileNotFoundError"] == 1
    assert result["counts"]["detail:ValueError"] == 1
    assert result["counts"]["skipped:no_plate"] == 1


def test_empty_and_no_validation_inputs_have_no_geometry_suggestions(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    empty = tmp_path / "empty.csv"
    _write_index(empty, [])
    result = calibrate(empty, raw, n=5)
    assert result["selected"] == 0 and result["metrics"] == {}
    assert result["geometry"]["7"]["n"] == result["geometry"]["8"]["n"] == 0

    plate_path = raw / "ccpd2019" / "ccpd_base" / "test.png"
    imwrite(plate_path, cv2.resize(draw_plate("皖AB1234", "blue"), (252, 80)))
    index = tmp_path / "test_only.csv"
    _write_index(index, [_row(plate_path, "ccpd2019", "ccpd_base", "test", 7, "blue")])
    result = calibrate(index, raw, n=1)
    assert result["counts"]["observed"] == 1
    assert result["geometry"]["7"]["n"] == 0


def test_report_names_parameters_width_std_and_creates_machine_outputs(tmp_path):
    from lpr_tools.calib_layout import _markdown, main

    result = {"selected": 0, "n_requested": 5, "seed": 42, "split_counts": {}, "counts": {},
              "metrics": {}, "geometry": {str(layout): {"n": 0, "centers": [], "center_std": [],
              "char_w": None, "char_w_std": None, "char_w_by_position": [],
              "char_w_std_by_position": []} for layout in (7, 8)}}
    report = _markdown(result)
    assert "layout7_centers" in report and "layout7_char_w" in report
    assert "layout8_centers" in report and "layout8_char_w" in report
    assert "char_w_std_by_position" in report and "仿 CCPD 数据只用于验证流程" in report

    raw = tmp_path / "raw"
    raw.mkdir()
    index = tmp_path / "empty.csv"
    _write_index(index, [])
    out = tmp_path / "out"
    assert main(["--index", str(index), "--raw", str(raw), "--n", "2", "--out", str(out)]) == 0
    assert (out / "calib_layout.md").is_file()
    assert (out / "calib_layout.json").is_file()
    suggestion = (out / "suggested_segment.toml").read_text(encoding="utf-8")
    assert "layout7_centers unavailable" in suggestion and "layout8_char_w unavailable" in suggestion
    assert tomllib.loads(suggestion) == {"segment": {}}



def test_projection_keeps_thin_horizontal_connections():
    from lpr_tools.calib_layout import _measure_slots

    binary = np.zeros((80, 252), np.uint8)
    spans = [(10, 34), (44, 66), (89, 111), (122, 143), (154, 176), (187, 209), (220, 241)]
    for left, right in spans:
        # U-like glyph: both stems are connected by only three rows of ink.
        binary[20:60, left:left+3] = 255
        binary[20:60, right-3:right] = 255
        binary[57:60, left:right] = 255
    measured = _measure_slots(binary, 7)
    assert np.allclose([v[0] for v in measured], [(a+b)/504 for a,b in spans])
    assert np.allclose([v[1] for v in measured], [(b-a)/252 for a,b in spans])
