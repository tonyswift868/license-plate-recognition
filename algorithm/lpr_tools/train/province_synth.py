"""Online, training-only province-glyph replacement using local fonts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
from typing import Any

from lpr.charset import PROVINCES
from lpr.charnorm import normalize_char, unify_polarity

DEFAULT_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")


@lru_cache(maxsize=4)
def _font(path: str, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(path, size=size)


def _canonical(image, quad, *, width: int = 256, height: int = 80):
    import cv2
    import numpy as np
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("quad must be four finite LT,RT,RB,LB points")
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(points, target)
    if not np.isfinite(transform).all() or abs(float(np.linalg.det(transform))) < 1e-12:
        raise ValueError("quad transform is singular")
    warped = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, transform, target


def normalized_first_char(image_bgr, quad, label_or_color: str):
    """Return the first Chinese glyph as the runtime's normalized 32x32 image.

    ``label_or_color`` may be the full plate label or one of blue/green/yellow.
    The CCPD green eight-character plate uses dark foreground and is inverted by
    the same frozen polarity helper as runtime segmentation.
    """
    import cv2
    import numpy as np
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 BGR")
    plate, _, _ = _canonical(image, quad)
    x0, x1 = int(round(plate.shape[1] * .012)), int(round(plate.shape[1] * .145))
    y0, y1 = int(round(plate.shape[0] * .06)), int(round(plate.shape[0] * .94))
    gray = cv2.cvtColor(plate[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        raise ValueError("empty province glyph cell")
    color = label_or_color if label_or_color in {"blue", "green", "yellow"} else (
        "green" if len(label_or_color) == 8 else "blue")
    unified = unify_polarity(gray, color)
    _, binary = cv2.threshold(unified, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 4:
        # Very dark or flat source slots still represent valid training rows.
        # Keep their normalized output shape and let the caller's target decide.
        return normalize_char(unified)
    x, y, w, h = cv2.boundingRect(points)
    if w < 2 or h < 4:
        raise ValueError("province glyph foreground is too small")
    pad = max(1, round(max(w, h) * .04))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(unified.shape[1], x + w + pad), min(unified.shape[0], y + h + pad)
    return normalize_char(unified[y0:y1, x0:x1])


@lru_cache(maxsize=256)
def _glyph_mask_cached(character: str, size: int, font_name: str):
    import numpy as np
    from PIL import Image, ImageDraw
    # Render square first. The target plate cell is narrow and tall; fitting the
    # font to that cell directly would shrink the glyph height to ~half-size.
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    font = _font(font_name, int(size * .88))
    left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
    text_w, text_h = right - left, bottom - top
    if text_w <= 0 or text_h <= 0:
        raise ValueError("font could not render province character")
    x = (size - text_w) // 2 - left
    y = (size - text_h) // 2 - top
    draw.text((x, y), character, fill=255, font=font)
    return np.asarray(mask, dtype=np.uint8).copy()


def _glyph_mask(character: str, width: int, height: int, font_path: Path):
    import cv2
    import numpy as np
    source = _glyph_mask_cached(character, max(width, height), str(font_path))
    points = cv2.findNonZero(source)
    if points is None:
        raise ValueError("font rendered an empty province glyph")
    x, y, w, h = cv2.boundingRect(points)
    glyph = source[y:y + h, x:x + w]
    return cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA if width < w else cv2.INTER_LINEAR)


def _masked_row_median(values, valid):
    """Vectorized float32 row medians, leaving rows with no valid samples as NaN."""
    import numpy as np
    counts = np.count_nonzero(valid, axis=1)
    result = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float32)
    active = counts > 0
    if np.any(active):
        masked = np.where(valid[active, :, None], values[active], np.nan)
        result[active] = np.nanmedian(masked, axis=1).astype(np.float32, copy=False)
    return result


def _row_background(pixels, clean, side):
    """Estimate left/right row colors, honoring clean-side and fallback rules."""
    import numpy as np
    height = pixels.shape[0]
    left_valid = clean[:, :side]
    right_valid = clean[:, -side:]
    left = _masked_row_median(pixels[:, :side], left_valid)
    right = _masked_row_median(pixels[:, -side:], right_valid)
    left_ok, right_ok = np.isfinite(left[:, 0]), np.isfinite(right[:, 0])

    # If both margins are covered, use other clean pixels on that row. Rows
    # with no clean pixels remain NaN and are filled from the global median.
    both_missing = ~left_ok & ~right_ok
    if np.any(both_missing):
        row_fallback = _masked_row_median(pixels, clean)
        left[both_missing] = row_fallback[both_missing]
        right[both_missing] = row_fallback[both_missing]
    only_left_missing = ~left_ok & right_ok
    left[only_left_missing] = right[only_left_missing]
    only_right_missing = left_ok & ~right_ok
    right[only_right_missing] = left[only_right_missing]
    return left, right


def _erase_province_glyph(cell, rng=None):
    """Inpaint the old province glyph using Otsu color distance, not a tail percentile."""
    import cv2
    import numpy as np
    cell = np.asarray(cell)
    if cell.dtype != np.uint8 or cell.ndim != 3 or cell.shape[2] != 3 or not cell.size:
        raise ValueError("province cell must be a nonempty uint8 BGR image")
    border = np.concatenate((cell[:3].reshape(-1, 3), cell[-3:].reshape(-1, 3),
                             cell[:, :2].reshape(-1, 3), cell[:, -2:].reshape(-1, 3)), axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background_luma = float(cv2.cvtColor(background[None, None, :].astype(np.uint8),
                                         cv2.COLOR_BGR2GRAY)[0, 0])
    dark_text = background_luma >= 145
    left_luma = np.median(gray[:, :2], axis=1)
    right_luma = np.median(gray[:, -2:], axis=1)
    # The brighter side is the robust background estimate for dark text; the
    # darker side serves the same purpose for white-on-blue glyphs.
    row_background = np.maximum(left_luma, right_luma) if dark_text else np.minimum(left_luma, right_luma)
    contrast = row_background[:, None] - gray if dark_text else gray - row_background[:, None]
    contrast = np.clip(contrast, 0, 255)
    contrast_u8 = np.rint(contrast).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(contrast_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = max(14.0, float(otsu_threshold))
    # Keep the Otsu core, then include low-contrast antialias pixels adjacent to
    # it. This avoids losing the majority of old strokes when Otsu's threshold
    # lands on a dense black/white foreground mode.
    core = (contrast >= threshold) & (contrast > 0)
    # A broad cell-local neighborhood catches soft JPEG/blurred strokes several
    # pixels beyond the high-contrast Otsu core while remaining inside the first
    # slot (the second character starts outside the .145 cell bound).
    core_neighborhood = cv2.dilate(core.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1) > 0
    ink = core | ((contrast >= 14.0) & core_neighborhood)
    ink = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((2, 2), np.uint8), iterations=1)

    if np.any(ink):
        # Telea blends the old high-contrast strokes back into dense masks. The
        # province cell has side margins in the frozen layout, so estimate its
        # smooth plate background row by row using unmasked border pixels only.
        # If one side is covered by ink, use the other; if both are covered, use
        # a same-row clean pixel or global clean-cell median. The caller clips
        # writes to this cell's inverse-projected polygon.
        side = max(1, min(8, cell.shape[1] // 4))
        pixels = cell.astype(np.float32)
        clean = ink == 0
        clean_pixels = pixels[clean]
        global_fill = (np.median(clean_pixels, axis=0) if clean_pixels.size
                       else background.astype(np.float32))
        left, right = _row_background(pixels, clean, side)
        missing = ~np.isfinite(left[:, 0])
        if np.any(missing):
            left[missing] = global_fill
            right[missing] = global_fill
        fraction = np.linspace(0.0, 1.0, cell.shape[1], dtype=np.float32)[None, :, None]
        background_rows = left[:, None, :] * (1.0 - fraction) + right[:, None, :] * fraction
        background_rows = np.broadcast_to(background_rows, pixels.shape).copy()
        clean_residual = pixels[clean] - background_rows[clean]
        noise_std = (float(np.clip(np.median(np.abs(clean_residual)), .35, 2.5))
                     if clean_residual.size else .75)
        if rng is None:
            rng = np.random.default_rng(0)
        noise = (rng.normal(0.0, noise_std, background_rows.shape).astype(np.float32)
                 if hasattr(rng, "normal") else 0.0)
        background_rows = np.clip(background_rows + noise, 0, 255)
        cleaned = cell.copy()
        cleaned[ink > 0] = np.clip(np.rint(background_rows[ink > 0]), 0, 255).astype(np.uint8)
    else:
        noise_std = 0.0
        cleaned = cell.copy()
    return cleaned, ink, background, {"otsu_threshold": float(otsu_threshold),
                                      "applied_threshold": threshold,
                                      "mask_fraction": float(np.mean(ink > 0)),
                                      "background_noise_std": noise_std,
                                      "dark_text": dark_text,
                                      "core_mask": core,
                                      "fill_method": "row-boundary-interpolation"}


def replace_province(image_bgr, quad, source_label: str, target_province: str,
                     rng: Any, *, font_path: str | Path = DEFAULT_FONT
                     ) -> tuple[Any, str, dict[str, object]]:
    """Replace the first plate glyph in-place by perspective-aware rendering.

    Args use BGR uint8 pixels and four GT points in current-image coordinates,
    ordered LT, RT, RB, LB. The image may be an original frame or any crop when
    ``quad`` has been translated/transformed into that same image. The returned
    label changes only its first character. Pillow and cv2 are imported lazily
    inside lpr_tools, keeping runtime package dependencies unchanged.
    """
    import cv2
    import numpy as np
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 BGR")
    if not source_label or source_label[0] not in PROVINCES:
        raise ValueError("source label must begin with a supported province")
    if target_province not in PROVINCES:
        raise ValueError("target province is outside the frozen 31-class charset")
    font_path = Path(font_path)
    if not font_path.is_file():
        raise FileNotFoundError(f"local synthesis font missing: {font_path}")

    plate, source_to_plate, _ = _canonical(image, quad)
    height, width = plate.shape[:2]
    # Match the production layout: first slot ends at x=.127; the next glyph
    # cell begins around x=.153. The .145 boundary preserves a safety margin.
    x0, x1 = int(round(width * .012)), int(round(width * .145))
    y0, y1 = int(round(height * .06)), int(round(height * .94))
    cell = plate[y0:y1, x0:x1].copy()
    if cell.size == 0:
        raise ValueError("empty province glyph cell")

    # Remove old strokes with a bimodal color-distance threshold and small AA
    # coverage margin, independently of the new glyph renderer.
    cleaned, glyph, background, erase_meta = _erase_province_glyph(cell, rng)

    # Preserve the original glyph's foreground color (white on blue, dark on
    # green/yellow) and softly match its blur/noise before the source perspective.
    core_mask = erase_meta.pop("core_mask")
    dark_text = bool(erase_meta["dark_text"])
    foreground = cell[core_mask]
    foreground_points = cv2.findNonZero(core_mask.astype(np.uint8))
    source_glyph_height = (cv2.boundingRect(foreground_points)[3]
                           if foreground_points is not None else 0)
    style_fallback = foreground_points is None or foreground.size < 24 or source_glyph_height < 12
    if style_fallback:
        text_color = np.asarray([16.0, 16.0, 16.0] if dark_text else
                                [240.0, 240.0, 240.0], dtype=np.float32)
    else:
        text_color = np.median(foreground.astype(np.float32), axis=0)
    if foreground_points is None:
        glyph_height = min(y1 - y0 - 4, 60)
    else:
        glyph_height = min(y1 - y0 - 4, max(54, min(64, source_glyph_height)))
    glyph_width = min(x1 - x0 - 4, max(22, round(glyph_height * .43)))
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    glyph_mask = _glyph_mask(target_province, glyph_width, glyph_height, font_path)
    gx = (mask.shape[1] - glyph_width) // 2
    gy = (mask.shape[0] - glyph_height) // 2
    mask[gy:gy + glyph_height, gx:gx + glyph_width] = glyph_mask
    angle = float(rng.uniform(-1.8, 1.8)) if hasattr(rng, "uniform") else 0.0
    scale = float(rng.uniform(.94, 1.04)) if hasattr(rng, "uniform") else 1.0
    affine = cv2.getRotationMatrix2D(((x1 - x0 - 1) / 2, (y1 - y0 - 1) / 2), angle, scale)
    affine[:, 2] += [float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))]
    mask = cv2.warpAffine(mask, affine, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    rendered = cleaned.astype(np.float32) * (1.0 - alpha) + text_color[None, None, :] * alpha
    local_blur = float(cv2.Laplacian(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
    if local_blur < 45:
        rendered = cv2.GaussianBlur(rendered, (3, 3), .55)
    noise_std = min(3.0, max(.35, float(np.std(cell.astype(np.float32) - cleaned.astype(np.float32)) * .08)))
    noise = rng.normal(0.0, noise_std, rendered.shape).astype(np.float32) if hasattr(rng, "normal") else 0.0
    rendered = np.clip(rendered + noise, 0, 255).astype(np.uint8)
    rendered_points = cv2.findNonZero(mask)
    rendered_glyph_height = cv2.boundingRect(rendered_points)[3] if rendered_points is not None else 0

    # Project only the small first-character slot back, keeping the full source
    # frame untouched elsewhere and avoiding a full-frame warp per synthetic row.
    inverse = np.linalg.inv(source_to_plate)
    local_to_plate = np.asarray([[1., 0., x0], [0., 1., y0], [0., 0., 1.]], dtype=np.float64)
    local_to_source = inverse @ local_to_plate
    corners = np.asarray([[[0., 0.], [x1 - x0 - 1., 0.],
                           [x1 - x0 - 1., y1 - y0 - 1.], [0., y1 - y0 - 1.]]], dtype=np.float32)
    source_corners = cv2.perspectiveTransform(corners, local_to_source.astype(np.float64))[0]
    ix0 = max(0, int(np.floor(source_corners[:, 0].min())) - 1)
    iy0 = max(0, int(np.floor(source_corners[:, 1].min())) - 1)
    ix1 = min(image.shape[1], int(np.ceil(source_corners[:, 0].max())) + 2)
    iy1 = min(image.shape[0], int(np.ceil(source_corners[:, 1].max())) + 2)
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError("province glyph slot falls outside source image")
    translation = np.asarray([[1., 0., ix0], [0., 1., iy0], [0., 0., 1.]], dtype=np.float64)
    local_to_crop = np.linalg.inv(translation) @ local_to_source
    patch = cv2.warpPerspective(rendered, local_to_crop, (ix1 - ix0, iy1 - iy0),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha_src = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    patch_alpha = cv2.warpPerspective(alpha_src, local_to_crop, (ix1 - ix0, iy1 - iy0),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    patch_alpha = np.where(patch_alpha >= 128, 255, 0).astype(np.uint8)
    projected_slot = np.rint(source_corners - np.asarray([ix0, iy0], dtype=np.float32)).astype(np.int32)
    polygon_mask = np.zeros_like(patch_alpha)
    cv2.fillConvexPoly(polygon_mask, projected_slot, 255)
    patch_alpha = cv2.bitwise_and(patch_alpha, polygon_mask)
    base = image[iy0:iy1, ix0:ix1].astype(np.float32)
    weight = (patch_alpha.astype(np.float32) / 255.0)[..., None]
    result = image.copy()
    result[iy0:iy1, ix0:ix1] = np.clip(base * (1.0 - weight) + patch.astype(np.float32) * weight,
                                        0, 255).astype(np.uint8)
    new_label = target_province + source_label[1:]
    return result, new_label, {"source_province": source_label[0], "target_province": target_province,
                               "slot_xyxy_canonical": [x0, y0, x1, y1],
                               "local_blur_variance": local_blur, "noise_std": noise_std,
                               "source_glyph_height": source_glyph_height,
                               "rendered_glyph_height": rendered_glyph_height,
                               "rendered_glyph_width": glyph_width,
                               "style_fallback": style_fallback,
                               "erase_mask_fraction": erase_meta["mask_fraction"],
                               "erase_threshold": erase_meta["applied_threshold"],
                               "erase_background_noise_std": erase_meta["background_noise_std"]}


__all__ = ["replace_province", "normalized_first_char"]
