"""65 类字符表（冻结，见开发文档 7.2）。

列表顺序即模型输出下标。改动任何一项都会使已训练的模型失效，
CHARSET_SHA 写入 models/manifest.json，加载模型时校验。
"""

from __future__ import annotations

import hashlib

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]  # 31，沿用 CCPD 顺序
DIGITS = list("0123456789")                          # 10
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")           # 24，无 I、O
CHARSET = PROVINCES + DIGITS + LETTERS               # 65
NUM_CLASSES = len(CHARSET)
CHARSET_SHA = hashlib.sha256("".join(CHARSET).encode("utf-8")).hexdigest()[:12]

CHAR_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CHARSET)}

# 分组下标（升序元组），供位置约束解码与按组统计准确率使用
PROVINCE_IDX: tuple[int, ...] = tuple(range(0, 31))
DIGIT_IDX: tuple[int, ...] = tuple(range(31, 41))
LETTER_IDX: tuple[int, ...] = tuple(range(41, 65))


def index_of(ch: str) -> int:
    """字符 → 下标；不在字符表内时抛 KeyError。"""
    return CHAR_TO_INDEX[ch]
