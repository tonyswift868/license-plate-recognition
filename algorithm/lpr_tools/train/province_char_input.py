"""Retain valid narrow-foreground first-character cells during offline training."""
from lpr_tools.train import province_synth


def normalized_first_char(image_bgr, quad, label_or_color):
    try:
        return province_synth.normalized_first_char(image_bgr, quad, label_or_color)
    except ValueError as error:
        if str(error) != "province glyph foreground is too small":
            raise
    # Match the existing flat-cell fallback; a tiny stroke is still a valid
    # labeled source. Preserve the entire cell instead of discarding its row.
    import cv2
    from lpr.charnorm import normalize_char, unify_polarity
    plate, _, _ = province_synth._canonical(image_bgr, quad)
    h, w = plate.shape[:2]
    cell = plate[round(h * .06):round(h * .94), round(w * .012):round(w * .145)]
    color = label_or_color if label_or_color in {"blue", "green", "yellow"} else (
        "green" if len(label_or_color) == 8 else "blue")
    return normalize_char(unify_polarity(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), color))
