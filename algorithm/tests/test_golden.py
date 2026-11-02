"""金样例回归（架构师）：防止合并引入退步。labels.csv 不存在时跳过。

通过门槛 MIN_CORRECT 随版本提高（开发文档 13.1）。
"""

from __future__ import annotations

import csv

import pytest
from lpr.errors import ModelLoadError
from lpr.io_utils import imread
from lpr.pipeline import PlateRecognizer

from .conftest import GOLDEN_DIR

MIN_CORRECT = 13                                         # 10-26 cnn v2 + corner_net v1：13/20；随版本提高

LABELS = GOLDEN_DIR / "labels.csv"
pytestmark = [pytest.mark.golden,
              pytest.mark.skipif(not LABELS.exists(), reason="金样例尚未就绪（需要 CCPD 数据）")]


def _rows() -> list[dict[str, str]]:
    with open(LABELS, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def real_recognizer() -> PlateRecognizer:
    """用 default.toml 指向的真实模型；模型不可用时跳过。"""
    try:
        return PlateRecognizer.from_config()
    except ModelLoadError as e:
        pytest.skip(f"真实模型不可用：{e}")


def test_golden_end_to_end(real_recognizer):
    correct = 0
    for row in _rows():
        r = real_recognizer.recognize(imread(GOLDEN_DIR / row["file"]))
        if row["category"] == "np":
            correct += r.plate_no is None                # 无牌图只要不报出车牌就算对（NO_PLATE 或 SEG_FAILED）
        else:
            correct += r.plate_no == row["plate_no"]
    assert correct >= MIN_CORRECT
