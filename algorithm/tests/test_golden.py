"""金样例回归（架构师）：防止合并引入退步。labels.csv 不存在时跳过。

通过门槛 MIN_CORRECT 随版本提高（开发文档 13.1）。
"""

from __future__ import annotations

import csv

import pytest

from lpr.io_utils import imread
from lpr.types import RecogStatus

from .conftest import GOLDEN_DIR

MIN_CORRECT = 0                                          # Phase 0 各阶段为桩；v0 打通后提高

LABELS = GOLDEN_DIR / "labels.csv"
pytestmark = [pytest.mark.golden,
              pytest.mark.skipif(not LABELS.exists(), reason="金样例尚未就绪（需要 CCPD 数据）")]


def _rows() -> list[dict[str, str]]:
    with open(LABELS, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_golden_end_to_end(recognizer):
    correct = 0
    for row in _rows():
        r = recognizer.recognize(imread(GOLDEN_DIR / row["file"]))
        expected_none = row["category"] == "np"
        if (expected_none and r.status is RecogStatus.NO_PLATE) or (r.plate_no == row["plate_no"]):
            correct += 1
    assert correct >= MIN_CORRECT
