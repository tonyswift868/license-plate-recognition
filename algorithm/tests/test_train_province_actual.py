"""Small pure-Python tests for the single-route actual-eval selectors."""
from lpr_tools.train.eval_province_actual import aggregate, select


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
