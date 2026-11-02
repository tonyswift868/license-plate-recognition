import pytest
from lpr_tools.eval.route_split import (
    assert_content_separation,
    path_key,
    split_assignments,
)


def test_split_is_order_and_truth_independent_with_exact_stratum_halves():
    records = [{"path": f"D:/img/{split}/{subset}/{i}.jpg", "split": split,
                "subset": subset, "truth": "hidden"}
               for split in ("val", "devhard") for subset in ("a", "b")
               for i in range(10)]
    expected = split_assignments(records)
    assert expected == split_assignments([{**r, "truth": "changed"} for r in reversed(records)])
    for split in ("val", "devhard"):
        for subset in ("a", "b"):
            assert sum(expected[path_key(r["path"])] == "fit" for r in records
                       if r["split"] == split and r["subset"] == subset) == 5


def test_split_duplicate_paths_and_content_leak_fail_closed():
    row = {"path": "D:/img/A.jpg", "split": "val", "subset": "a"}
    with pytest.raises(ValueError, match="duplicate"):
        split_assignments([row, {**row, "path": "d:\\img\\a.jpg"}])
    assignments = {"d:/a": "fit", "d:/b": "hold"}
    with pytest.raises(ValueError, match="same image bytes"):
        assert_content_separation(assignments, {"d:/a": "same", "d:/b": "same"})
    assert_content_separation(assignments, {"d:/a": "a", "d:/b": "b"})


def test_content_groups_stay_together_and_each_stratum_keeps_exact_halves():
    rows = [{"path": f"D:/img/{subset}/{i}", "split": "devhard", "subset": subset}
            for subset in ("a", "b") for i in range(8)]
    before = split_assignments(rows)
    left = next(r["path"] for r in rows if r["subset"] == "a" and before[path_key(r["path"])] == "fit")
    right = next(r["path"] for r in rows if r["subset"] == "b" and before[path_key(r["path"])] == "hold")
    hashes = {r["path"]: ("same" if r["path"] in (left, right) else r["path"]) for r in rows}
    actual = split_assignments(rows, image_sha_by_path=hashes)
    assert actual[path_key(left)] == actual[path_key(right)]
    for subset in ("a", "b"):
        assert sum(actual[path_key(r["path"])] == "fit" for r in rows if r["subset"] == subset) == 4
    assert actual == split_assignments(list(reversed(rows)), image_sha_by_path=hashes)
    assert_content_separation(actual, hashes)
