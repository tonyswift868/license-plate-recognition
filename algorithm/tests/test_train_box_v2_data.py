from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from lpr_tools.train.box_data_v2 import GROUPS, _candidate_green_rows, build_balanced_multiplicity


def _green_row(path: Path, content: bytes, label: str = "皖A123456") -> dict[str, str]:
    path.write_bytes(content)
    return {"path": str(path), "split": "val", "subset": "ccpd_green", "has_plate": "1",
            "skip_reason": "", "plate_no": label,
            "quad": "0 0 20 0 20 8 0 8", "brightness": "16"}


def test_balanced_plan_covers_all_unique_examples_and_equalizes_nine_groups():
    subsets = [group for group, size in zip(GROUPS, (5, 2, 1, 3, 2, 1, 4, 2, 1), strict=False)
               for _ in range(size)]
    plan = build_balanced_multiplicity(subsets, seed=11042, epoch=0)

    assert plan.unique_samples == len(subsets)
    assert len(plan.multiplicity) == len(subsets)
    assert min(plan.multiplicity) >= 1
    assert plan.group_unique == dict(zip(GROUPS, (5, 2, 1, 3, 2, 1, 4, 2, 1), strict=False))
    assert set(plan.group_exposures.values()) == {5}
    assert plan.total_exposures == 45
    assert sum(plan.multiplicity) == plan.total_exposures


def test_balanced_plan_is_reproducible_for_all_sixty_epochs():
    subsets = [group for group, size in zip(GROUPS, (17, 7, 4, 8, 13, 6, 9, 3, 5), strict=False)
               for _ in range(size)]
    for epoch in range(60):
        first = build_balanced_multiplicity(subsets, seed=7, epoch=epoch)
        second = build_balanced_multiplicity(subsets, seed=7, epoch=epoch)
        assert first.multiplicity == second.multiplicity
        assert len(set(first.group_exposures.values())) == 1
        assert all(value >= 1 for value in first.multiplicity)


def test_balanced_plan_validates_missing_unknown_and_negative_epoch():
    with pytest.raises(ValueError, match="empty training pool"):
        build_balanced_multiplicity([], seed=0, epoch=0)
    with pytest.raises(ValueError, match="requires every group"):
        build_balanced_multiplicity(["base"], seed=0, epoch=0)
    with pytest.raises(ValueError, match="unknown training subset"):
        build_balanced_multiplicity(["base", "mystery"], seed=0, epoch=0)
    with pytest.raises(ValueError, match="nonnegative"):
        build_balanced_multiplicity(list(GROUPS), seed=0, epoch=-1)


def test_green_candidate_filter_excludes_train_protected_holdout_and_duplicates(tmp_path):
    protected = _green_row(tmp_path / "protected.jpg", b"protected")
    train = _green_row(tmp_path / "train.jpg", b"trained")
    holdout = _green_row(tmp_path / "holdout.jpg", b"holdout")
    accepted = _green_row(tmp_path / "accepted.jpg", b"fresh")
    duplicate = _green_row(tmp_path / "duplicate.jpg", b"fresh")
    wrong_length = _green_row(tmp_path / "sixteen.jpg", b"sixteen", "皖A12345678")
    invalid = _green_row(tmp_path / "invalid.jpg", b"invalid")
    invalid["skip_reason"] = "invalid_format"
    rows = [protected, train, holdout, accepted, duplicate, wrong_length, invalid]
    index = tmp_path / "index.csv"
    fields = tuple(rows[0])
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    protected_sha = hashlib.sha256(b"protected").hexdigest()
    train_sha = hashlib.sha256(b"trained").hexdigest()
    holdout_sha = hashlib.sha256(b"holdout").hexdigest()
    candidates, rejected = _candidate_green_rows(
        index, train_hashes={train_sha}, tune_hashes=set(),
        protected_paths={str(Path(protected["path"]).resolve()).lower()},
        protected_hashes={protected_sha},
        holdout_paths={str(Path(holdout["path"]).resolve()).lower()},
        holdout_hashes={holdout_sha}, ledger={})

    assert [row["path"] for row in candidates] == [accepted["path"]]
    assert len({row["sha256"] for row in candidates}) == 1
    assert rejected["excluded_protected_path"] == 1
    assert rejected["excluded_training_sha"] == 1
    assert rejected["excluded_batch8_holdout_path"] == 1
    assert rejected["excluded_duplicate_sha"] == 1
    assert rejected["excluded_non_eight_character"] == 1

