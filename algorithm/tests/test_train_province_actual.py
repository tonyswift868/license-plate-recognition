"""Small pure-Python tests for the single-route actual-eval selectors."""
from types import SimpleNamespace

import pytest
from lpr_tools.train.eval_province_actual import aggregate, nonwan_membership, select


def test_select_uses_highest_legal_confidence_and_stable_first_tie():
    attempts = [
        {"plate": "皖A12345", "confidence": .8, "valid": True},
        {"plate": "皖B12345", "confidence": .8, "valid": True},
        {"plate": "not-a-plate", "confidence": .99, "valid": False},
        {"plate": "皖C12345", "confidence": float("nan"), "valid": True},
    ]
    assert select(attempts) is attempts[0]
    assert select([]) is None


def test_aggregate_keeps_no_read_in_denominator_and_groups_fixed_261():
    rows = [
        {"truth": "皖A12345", "prediction": "皖A12345", "split": "devhard",
         "subset": "ccpd_db", "province_261": False},
        {"truth": "粤A12345", "prediction": None, "split": "devhard",
         "subset": "ccpd_db", "province_261": True},
        {"truth": "粤B12345", "prediction": "皖B12345", "split": "val",
         "subset": "ccpd_green", "province_261": False},
    ]
    metrics = aggregate(rows)
    assert metrics["all"]["n"] == 3
    assert metrics["all"]["plate_correct"] == 1
    assert metrics["all"]["province_correct"] == 1
    assert metrics["nonwan_261"]["n"] == 1
    assert metrics["nonwan_261"]["plate_accuracy"] == 0
    assert metrics["devhard/ccpd_db"]["n"] == 2
    assert metrics["green"]["n"] == 1
    assert metrics["all"]["wrong_as_anwei"] == 1


def test_nonwan_union_counts_real_members_across_both_splits():
    def sample(path, plate):
        return SimpleNamespace(row=SimpleNamespace(path=path, plate_no=plate))

    val = [sample("v1", "粤A12345"), sample("v2", "皖A12345")]
    devhard = [sample("d1", "陕A12345"), sample("d2", "皖B12345")]
    union, counts = nonwan_membership(val, devhard, lambda path: path.lower())
    assert union == {"v1", "d1"}
    assert counts == {"val": 1, "devhard": 1}
    with pytest.raises(ValueError, match='overlap'):
        nonwan_membership([sample("same", "粤A12345")],
                          [sample("same", "粤B12345")], lambda path: path)
