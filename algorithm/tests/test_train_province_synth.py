from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont

from lpr_tools.train.province_synth import _glyph_mask, normalized_first_char, replace_province


def _plate(color: str, label: str):
    if color == "green":
        background, foreground = (150, 220, 130), (16, 16, 16)
    else:
        background, foreground = (180, 65, 20), (245, 245, 245)
    canonical = np.full((80, 256, 3), background, dtype=np.uint8)
    # Production slot centers the province glyph at .0754 and the next glyph
    # starts at .153. Draw the source glyph and A in their own production cells.
    province = _glyph_mask(label[0], 26, 60, Path(r"C:\Windows\Fonts\simhei.ttf"))
    ink = np.asarray(foreground, dtype=np.uint8)
    canonical[10:70, 6:32] = np.where(province[..., None] > 0, ink, canonical[10:70, 6:32])
    pil = Image.fromarray(cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 50)
    draw.text((42, 10), label[1], fill=foreground, font=font)
    for i, char in enumerate(label[2:]):
        draw.text((82 + i * 25, 16), char, fill=foreground, font=ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 42))
    canonical = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    image = np.zeros((180, 400, 3), dtype=np.uint8)
    quad = np.asarray([[40, 35], [339, 40], [334, 119], [35, 114]], dtype=np.float32)
    dst = np.asarray([[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32)
    image = cv2.warpPerspective(canonical, cv2.getPerspectiveTransform(dst, quad),
                                (image.shape[1], image.shape[0]))
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
    assert meta["rendered_glyph_height"] >= .85 * meta["source_glyph_height"]
    assert meta["rendered_glyph_height"] >= 50
    # The production A cell begins at x=.153, outside the replacement window.
    canonical_before = cv2.warpPerspective(image, cv2.getPerspectiveTransform(
        quad, np.asarray([[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32)), (256, 80))
    canonical_after = cv2.warpPerspective(changed, cv2.getPerspectiveTransform(
        quad, np.asarray([[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32)), (256, 80))
    assert np.array_equal(canonical_before[:, 40:75], canonical_after[:, 40:75])
    # Every changed source pixel lies inside the first-character projection and
    # inside the GT plate quadrilateral; all exterior pixels remain bitwise equal.
    transform = cv2.getPerspectiveTransform(quad, np.asarray([[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32))
    inverse = np.linalg.inv(transform)
    # Pixel-center corners of canonical x=[3,37), y=[5,75).
    slot = np.asarray([[[3, 5], [36, 5], [36, 74], [3, 74]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(slot, inverse)[0].round().astype(np.int32)
    allowed = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(allowed, projected, 1)
    changed_pixels = np.any(changed != image, axis=2)
    assert not np.any(changed_pixels & (allowed == 0))
    plate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(plate_mask, quad.round().astype(np.int32), 1)
    assert not np.any(changed_pixels & (plate_mask == 0))
    normalized = normalized_first_char(changed, quad, color)
    assert normalized.shape == (32, 32) and normalized.dtype == np.uint8
    assert int(normalized.max()) == 255


def test_replacement_rejects_unknown_province_and_invalid_quad():
    image, quad = _plate("blue", "皖A12345")
    with pytest.raises(ValueError):
        replace_province(image, quad, "皖A12345", "警", np.random.default_rng(1))
    with pytest.raises(ValueError):
        replace_province(image, [[0, 0]], "皖A12345", "陕", np.random.default_rng(1))


def test_uniform_source_uses_visible_style_fallback_without_losing_the_row():
    label = "皖A123456"
    image = np.full((180, 400, 3), (150, 220, 130), dtype=np.uint8)
    quad = np.asarray([[40, 35], [339, 40], [334, 119], [35, 114]], dtype=np.float32)
    changed, target, meta = replace_province(image, quad, label, "陕", np.random.default_rng(9))
    assert target == "陕" + label[1:]
    assert meta["style_fallback"] is True and meta["source_glyph_height"] == 0
    assert meta["rendered_glyph_height"] >= 54
    to_plate = cv2.getPerspectiveTransform(quad, np.asarray(
        [[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32))
    before = cv2.warpPerspective(image, to_plate, (256, 80))
    after = cv2.warpPerspective(changed, to_plate, (256, 80))
    assert np.array_equal(before[:, 40:75], after[:, 40:75])
    assert np.any(after[:, 3:37] != before[:, 3:37])
    normalized = normalized_first_char(changed, quad, "green")
    assert normalized.shape == (32, 32) and normalized.dtype == np.uint8
    assert np.isfinite(normalized).all() and normalized.max() == 255
