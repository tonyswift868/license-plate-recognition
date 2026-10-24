"""字符图归一化（冻结，见开发文档 6.5）。

训练样本和推理样本必须经过同一个 normalize_char()，否则训练分布与推理分布不一致。
"""

from __future__ import annotations

import cv2
import numpy as np

CHAR_SIZE = 32            # 输出边长（冻结）
CHAR_BOX = 28             # 字符长边缩放到的像素数

# 这些底色的车牌是“深色字、浅色底”，需要反色成“字符亮、底暗”
DARK_TEXT_COLORS = frozenset({"yellow", "green", "white"})


def unify_polarity(gray: np.ndarray, color: str) -> np.ndarray:
    """按车牌底色统一极性：输出为字符亮、底暗的灰度图。

    蓝牌、黑牌本身就是浅色字，原样返回；黄、绿、白牌反色。
    分割阶段（segment.py）与字符样本生成（build_chars.py）都必须用这个函数。
    """
    if color in DARK_TEXT_COLORS:
        return 255 - gray
    return gray


def normalize_char(crop: np.ndarray) -> np.ndarray:
    """把单个字符的裁剪图归一化为 32×32 uint8。

    输入：字符亮、底暗的灰度图（或 BGR 图，内部转灰度），极性需由调用方先用 unify_polarity 统一。
    处理：保持宽高比把长边缩到 28 px → 在裁剪区域内线性拉伸到 0—255 → 居中贴到 32×32 黑底。
    输入为空时抛 ValueError。
    """
    if crop is None or crop.size == 0 or crop.ndim not in (2, 3):
        raise ValueError("normalize_char: 输入为空或维度不对")
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    h, w = g.shape
    s = CHAR_BOX / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    lo, hi = int(resized.min()), int(resized.max())
    if hi > lo:
        stretched = ((resized.astype(np.float32) - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)
    else:
        stretched = np.zeros_like(resized)

    out = np.zeros((CHAR_SIZE, CHAR_SIZE), np.uint8)
    y0, x0 = (CHAR_SIZE - nh) // 2, (CHAR_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = stretched
    return out
