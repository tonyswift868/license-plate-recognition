from pathlib import Path

import pytest
from lpr.io_utils import imread
from lpr_tools.data.ccpd import build_index, parse_ccpd_name
from lpr_tools.synth_ccpd import generate, main


def test_generate_mini_dataset_matches_truth_and_parser(tmp_path):
    out = tmp_path / "raw"
    truth = generate(out, per_subset=2, seed=17)

    assert len(truth) == 16
    assert {row["subset"] for row in truth if row["dataset"] == "ccpd2019"} == {
        "ccpd_base", "ccpd_db", "ccpd_tilt", "ccpd_blur", "ccpd_np"
    }
    assert {(row["split"]) for row in truth if row["dataset"] == "ccpd2020"} == {"train", "val", "test"}
    for row in truth:
        image = imread(row["path"])
        assert image.shape[:2] == (1160, 720)
        if row["has_plate"]:
            decoded = parse_ccpd_name(row["file"])
            assert decoded.plate_no == row["plate_no"]
            assert decoded.quad == row["quad"]
            assert decoded.layout == row["layout"]
            assert decoded.skip_reason == ""
            assert decoded.bbox == tuple(row["metrics"]["bbox"])
            assert int(decoded.area) == row["metrics"]["area_token"]
            assert (decoded.tilt_h, decoded.tilt_v) == (row["metrics"]["tilt_h"], row["metrics"]["tilt_v"])
            assert (decoded.brightness, decoded.blurriness) == (
                row["metrics"]["brightness"], row["metrics"]["blurriness"]
            )

    index = build_index(out, tmp_path / "index.csv")
    assert len(index) == len(truth)
    by_path = {Path(row["path"]).resolve(): row for row in index}
    for row in truth:
        indexed = by_path[Path(row["path"]).resolve()]
        assert indexed["split"] == row["split"]
        assert indexed["has_plate"] == ("1" if row["has_plate"] else "0")
        if row["has_plate"]:
            assert indexed["plate_no"] == row["plate_no"]
            assert [tuple(map(int, indexed["quad"].split()[i:i+2])) for i in range(0, 8, 2)] == row["quad"]


def test_generate_is_seeded_and_rejects_bad_scale(tmp_path):
    first = generate(tmp_path / "one", 1, 42)
    second = generate(tmp_path / "two", 1, 42)
    assert [(r["file"], r["plate_no"], r["quad"]) for r in first] == [
        (r["file"], r["plate_no"], r["quad"]) for r in second
    ]
    for a, b in zip(first, second, strict=False):
        assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()
    with pytest.raises(ValueError):
        generate(tmp_path / "bad", -1, 42)
    with pytest.raises(ValueError):
        generate(tmp_path / "bad_seed", 1, -1)

    # A managed rerun replaces only files listed by the previous truth manifest.
    rerun = tmp_path / "rerun"
    before = generate(rerun, 2, 42)
    after = generate(rerun, 1, 43)
    assert len(before) == 16 and len(after) == 8
    assert len(list(rerun.rglob("*.jpg"))) == 8
    (rerun / "keep.txt").write_text("unmanaged", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked"):
        generate(rerun, 1, 42)


def test_cli(tmp_path, capsys):
    assert main(["--out", str(tmp_path / "raw"), "--per-subset", "1", "--seed", "9"]) == 0
    assert "Generated 8 CCPD-shaped samples" in capsys.readouterr().out
