from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont

from lpr_tools.train.province_synth import normalized_first_char, replace_province


def _plate(color: str, label: str):
    image = np.zeros((180, 400, 3), dtype=np.uint8)
    if color == "green":
        background, foreground = (150, 220, 130), (16, 16, 16)
    else:
        background, foreground = (180, 65, 20), (245, 245, 245)
    image[30:110, 30:330] = background
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 52)
    draw.text((37, 34), label[0], fill=foreground, font=font)
    image = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    quad = np.asarray([[30, 30], [329, 30], [329, 109], [30, 109]], dtype=np.float32)
    return image, quad


@pytest.mark.parametrize("color,label", [("blue", "皖A12345"), ("green", "皖A123456")])
def test_online_replacement_updates_label_and_runtime_normalized_glyph(color, label):
    image, quad = _plate(color, label)
    rng = np.random.default_rng(42)
    changed, target, meta = replace_province(image, quad, label, "陕", rng)
    assert target == "陕" + label[1:]
    assert meta["source_province"] == "皖" and meta["target_province"] == "陕"
    assert changed.shape == image.shape and changed.dtype == np.uint8
    assert np.any(changed != image)
    normalized = normalized_first_char(changed, quad, color)
    assert normalized.shape == (32, 32) and normalized.dtype == np.uint8
    assert int(normalized.max()) == 255


def test_replacement_rejects_unknown_province_and_invalid_quad():
    image, quad = _plate("blue", "皖A12345")
    with pytest.raises(ValueError):
        replace_province(image, quad, "皖A12345", "警", np.random.default_rng(1))
    with pytest.raises(ValueError):
        replace_province(image, [[0, 0]], "皖A12345", "陕", np.random.default_rng(1))
