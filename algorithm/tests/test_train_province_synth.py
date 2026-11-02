from __future__ import annotations

import numpy as np
import pytest
from lpr_tools.train.province_synth import (
    DEFAULT_FONT,
    _canonical,
    _erase_province_glyph,
    _glyph_mask,
    _row_background,
    normalized_first_char,
    replace_province,
)

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageFont = pytest.importorskip("PIL.ImageFont")


def _plate(color: str, label: str):
    if not DEFAULT_FONT.is_file():
        pytest.skip(f"missing local font: {DEFAULT_FONT}")
    if color == "green":
        background, foreground = (150, 220, 130), (16, 16, 16)
    else:
        background, foreground = (180, 65, 20), (245, 245, 245)
    canonical = np.full((80, 256, 3), background, dtype=np.uint8)
    # Production slot centers the province glyph at .0754 and the next glyph
    # starts at .153. Draw the source glyph and A in their own production cells.
    province = _glyph_mask(label[0], 26, 60, DEFAULT_FONT)
    ink = np.asarray(foreground, dtype=np.uint8)
    canonical[10:70, 6:32] = np.where(province[..., None] > 0, ink, canonical[10:70, 6:32])
    pil = Image.fromarray(cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(DEFAULT_FONT), 50)
    draw.text((42, 10), label[1], fill=foreground, font=font)
    for i, char in enumerate(label[2:]):
        draw.text((82 + i * 25, 16), char, fill=foreground, font=ImageFont.truetype(str(DEFAULT_FONT), 42))
    canonical = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    image = np.zeros((180, 400, 3), dtype=np.uint8)
    quad = np.asarray([[40, 35], [339, 40], [334, 119], [35, 114]], dtype=np.float32)
    dst = np.asarray([[0, 0], [255, 0], [255, 79], [0, 79]], dtype=np.float32)
    image = cv2.warpPerspective(canonical, cv2.getPerspectiveTransform(dst, quad),
                                (image.shape[1], image.shape[0]))
    return image, quad


def _row_background_reference(pixels, clean, side):
    left = np.empty((pixels.shape[0], 3), dtype=np.float32)
    right = np.empty_like(left)
    global_fill = np.median(pixels[clean], axis=0) if np.any(clean) else np.zeros(3, np.float32)
    for row_index in range(pixels.shape[0]):
        row = pixels[row_index]
        lv = row[:side][clean[row_index, :side]]
        rv = row[-side:][clean[row_index, -side:]]
        lc = np.median(lv, axis=0) if len(lv) else None
        rc = np.median(rv, axis=0) if len(rv) else None
        if lc is None and rc is None:
            available = row[clean[row_index]]
            fallback = np.median(available, axis=0) if len(available) else global_fill
            lc = rc = fallback
        elif lc is None:
            lc = rc
        elif rc is None:
            rc = lc
        left[row_index], right[row_index] = lc, rc
    return left, right


@pytest.mark.parametrize("mask_case", ["random", "left_covered", "right_covered", "row_fallback", "all_covered"])
def test_vectorized_row_background_matches_loop_reference(mask_case):
    rng = np.random.default_rng(127)
    pixels = rng.integers(0, 256, size=(80, 34, 3), dtype=np.uint8).astype(np.float32)
    clean = rng.random((80, 34)) > .42
    if mask_case == "left_covered":
        clean[:, :8] = False
    elif mask_case == "right_covered":
        clean[:, -8:] = False
    elif mask_case == "row_fallback":
        clean[11, :8] = False
        clean[11, -8:] = False
        clean[11, 8:-8] = True
    elif mask_case == "all_covered":
        clean[:] = False
    if mask_case != "all_covered":
        # Ensure a global clean fallback exists, while retaining row 11's case.
        clean[0, 12] = True
    expected = _row_background_reference(pixels, clean, 8)
    actual = _row_background(pixels, clean, 8)
    if mask_case == "all_covered":
        assert np.isnan(actual[0]).all() and np.isnan(actual[1]).all()
        return
    for got, want in zip(actual, expected, strict=False):
        np.testing.assert_array_equal(got, want)


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


@pytest.mark.parametrize("color,label", [("blue", "皖A12345"), ("green", "皖A123456")])
def test_erase_only_removes_old_province_ink_before_any_new_glyph(color, label):
    image, quad = _plate(color, label)
    plate, _, _ = _canonical(image, quad)
    x0, x1 = round(plate.shape[1] * .012), round(plate.shape[1] * .145)
    y0, y1 = round(plate.shape[0] * .06), round(plate.shape[0] * .94)
    cell = plate[y0:y1, x0:x1]
    cleaned, mask, background, stats = _erase_province_glyph(cell)
    source_distance = np.linalg.norm(cell.astype(np.float32) - background, axis=2)
    source_ink = source_distance > 24
    residual_distance = np.linalg.norm(cleaned.astype(np.float32) - background, axis=2)
    residual_ink = residual_distance > 24
    source_count = int(source_ink.sum())
    residual_count = int(residual_ink.sum())
    assert source_count > 0 and int((mask > 0).sum()) >= source_count
    assert residual_count / source_count <= .02, {
        "source_ink": source_count, "residual_ink": residual_count, "mask": stats,
    }
