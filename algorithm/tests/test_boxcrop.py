from __future__ import annotations

import numpy as np

from lpr.boxcrop import crop_box


def _img() -> np.ndarray:
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 50:150] = 255                              # 框 (50, 40, 100, 20)
    return img


def test_crop_box_pads_each_side():
    out = crop_box(_img(), (50, 40, 100, 20), 0.1, 0.25)
    assert out.shape == (30, 120, 3)                      # 左右各 10，上下各 5
    assert out[5:25, 10:110].min() == 255 and out[0].max() == 0


def test_crop_box_clips_to_image_and_copies():
    img = _img()
    out = crop_box(img, (190, 90, 30, 30), 0.5, 0.5)
    assert out.shape == (25, 25, 3)                       # 右下角被裁到图内
    out[:] = 7
    assert img[99, 199, 0] == 0


def test_crop_box_invalid_returns_empty():
    for bbox, pads in (((10, 10, 0, 5), (0.1, 0.1)), ((10, 10, 5, 5), (-0.1, 0.1)), ((500, 500, 5, 5), (0.1, 0.1))):
        assert crop_box(_img(), bbox, *pads).shape == (0, 0, 3)
