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
    x0, x1 = int(round(plate.shape[1] * .035)), int(round(plate.shape[1] * .205))
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
        raise ValueError("province glyph has no foreground after polarity normalization")
    x, y, w, h = cv2.boundingRect(points)
    if w < 2 or h < 4:
        raise ValueError("province glyph foreground is too small")
    pad = max(1, round(max(w, h) * .04))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(unified.shape[1], x + w + pad), min(unified.shape[0], y + h + pad)
    return normalize_char(unified[y0:y1, x0:x1])


@lru_cache(maxsize=256)
def _glyph_mask_cached(character: str, width: int, height: int, font_name: str):
    import numpy as np
    from PIL import Image, ImageDraw
    font_size = max(8, int(height * .92))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    font = _font(font_name, font_size)
    left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
    text_w, text_h = right - left, bottom - top
    if text_w <= 0 or text_h <= 0:
        raise ValueError("font could not render province character")
    scale = min((width * .78) / text_w, (height * .84) / text_h, 1.0)
    if scale < 1:
        font = _font(font_name, max(8, int(font_size * scale)))
        left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
        text_w, text_h = right - left, bottom - top
    x = (width - text_w) // 2 - left
    y = (height - text_h) // 2 - top
    draw.text((x, y), character, fill=255, font=font)
    return np.asarray(mask, dtype=np.uint8).copy()


def _glyph_mask(character: str, width: int, height: int, font_path: Path):
    return _glyph_mask_cached(character, width, height, str(font_path))


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
    x0, x1 = int(round(width * .035)), int(round(width * .205))
    y0, y1 = int(round(height * .06)), int(round(height * .94))
    cell = plate[y0:y1, x0:x1].copy()
    if cell.size == 0:
        raise ValueError("empty province glyph cell")

    # Estimate foreground from the border/background of this slot and remove
    # only dark/bright glyph pixels before drawing the new same-color glyph.
    border = np.concatenate((cell[:3].reshape(-1, 3), cell[-3:].reshape(-1, 3),
                             cell[:, :2].reshape(-1, 3), cell[:, -2:].reshape(-1, 3)), axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(cell.astype(np.float32) - background, axis=2)
    glyph = (distance > max(24.0, float(np.percentile(distance, 70)))).astype(np.uint8) * 255
    glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if int(glyph.sum()) < 255 * 8:
        # Bright blue/green backgrounds can have gradients; fall back to the
        # upper tail of local color distance instead of erasing the full cell.
        threshold = max(14.0, float(np.percentile(distance, 82)))
        glyph = (distance > threshold).astype(np.uint8) * 255
    cleaned = cv2.inpaint(cell, glyph, 2, cv2.INPAINT_TELEA)

    # Preserve the original glyph's foreground color (white on blue, dark on
    # green/yellow) and softly match its blur/noise before the source perspective.
    foreground = cell[glyph > 0]
    if foreground.size:
        text_color = np.median(foreground.astype(np.float32), axis=0)
    else:
        text_color = np.asarray([235.0, 235.0, 235.0], dtype=np.float32)
    mask = _glyph_mask(target_province, x1 - x0, y1 - y0, font_path)
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
    base = image[iy0:iy1, ix0:ix1].astype(np.float32)
    weight = (patch_alpha.astype(np.float32) / 255.0)[..., None]
    result = image.copy()
    result[iy0:iy1, ix0:ix1] = np.clip(base * (1.0 - weight) + patch.astype(np.float32) * weight,
                                        0, 255).astype(np.uint8)
    new_label = target_province + source_label[1:]
    return result, new_label, {"source_province": source_label[0], "target_province": target_province,
                               "slot_xyxy_canonical": [x0, y0, x1, y1],
                               "local_blur_variance": local_blur, "noise_std": noise_std}


__all__ = ["replace_province", "normalized_first_char"]
