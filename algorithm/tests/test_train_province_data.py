from __future__ import annotations

import pytest
from lpr_tools.train.province_data import holdout_size, province_repeat_additions


@pytest.mark.parametrize("count,expected", [(0, 0), (1, 0), (4, 0), (5, 1), (19, 5), (67, 20), (1000, 20)])
def test_holdout_quota_is_deterministic_and_respects_small_province_rule(count, expected):
    assert holdout_size(count) == expected


def test_province_repeats_only_add_non_anwei_and_respect_both_caps():
    rows = ([{"label": "陕A12345"}] * 3 + [{"label": "苏A12345"}] * 100 +
            [{"label": "皖A12345"}] * 20 + [{"label": "invalid"}])
    plan = province_repeat_additions(rows, seed=12042, epoch=7)
    assert len(plan.additions) == len(rows)
    assert plan.unique_by_province["陕"] == 3
    assert plan.added_by_province["陕"] == 57
    assert plan.added_by_province["苏"] == 1900  # 100 sources × 19 additional visits
    assert plan.added_by_province["皖"] == 0
    assert max(plan.additions) <= 19
    assert all(plan.additions[i] == 0 for i in range(123, len(rows)))
    again = province_repeat_additions(rows, seed=12042, epoch=7)
    assert list(again.additions) == list(plan.additions)
    next_epoch = province_repeat_additions(rows, seed=12042, epoch=8)
    assert sum(next_epoch.additions) == sum(plan.additions)


def test_province_repeat_limits_reject_invalid_values():
    with pytest.raises(ValueError):
        province_repeat_additions([], seed=1, epoch=-1)
    with pytest.raises(ValueError):
        holdout_size(4, fraction=1.5)
