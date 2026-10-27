"""检测框外扩裁片（架构师，10-27 新增）：不做透视校正，直接从原图裁出外扩后的轴对齐框。

“检测框直接读”路线的训练、评测与流水线都必须调用这里，保证三处裁法一致：
左右各外扩 pad_x × 框宽，上下各外扩 pad_y × 框高，再裁到图内。缩放到模型输入尺寸由读牌模型自己做。
"""

from __future__ import annotations

import numpy as np

from .types import Box


def crop_box(image: np.ndarray, bbox: Box, pad_x: float, pad_y: float) -> np.ndarray:
    """原图 BGR uint8 + 框 (x, y, w, h) → 外扩后的裁片拷贝（H×W×3）。

    框无效（宽或高 ≤0）或外扩后与图像无交集时返回形状 (0, 0, 3) 的空数组。"""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0 or pad_x < 0 or pad_y < 0:
        return np.zeros((0, 0, 3), np.uint8)
    ih, iw = image.shape[:2]
    x0 = max(int(np.floor(x - pad_x * w)), 0)
    y0 = max(int(np.floor(y - pad_y * h)), 0)
    x1 = min(int(np.ceil(x + w + pad_x * w)), iw)
    y1 = min(int(np.ceil(y + h + pad_y * h)), ih)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), np.uint8)
    return np.ascontiguousarray(image[y0:y1, x0:x1])
