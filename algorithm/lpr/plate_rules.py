"""Frozen plate layouts and character-index constraints (appendix A)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .charset import CHAR_TO_INDEX, DIGIT_IDX, LETTER_IDX, PROVINCE_IDX, PROVINCES
from .types import VehicleTypeHint

PROV = PROVINCE_IDX
LETTER = LETTER_IDX
ALNUM = tuple(sorted(DIGIT_IDX + LETTER_IDX))
DIGIT = DIGIT_IDX
DF = tuple(CHAR_TO_INDEX[c] for c in "DF")


@dataclass(frozen=True)
class PlateLayout:
    name: str
    slots: tuple[tuple[int, ...], ...]
    regex: re.Pattern[str]


_PREFIX = "[" + "".join(PROVINCES) + "][A-HJ-NP-Z]"
L7 = PlateLayout("L7", (PROV, LETTER) + (ALNUM,) * 5,
                 re.compile(_PREFIX + "[A-HJ-NP-Z0-9]{5}"))
L8S = PlateLayout("L8S", (PROV, LETTER, DF, ALNUM) + (DIGIT,) * 4,
                  re.compile(_PREFIX + "[DF][A-HJ-NP-Z0-9][0-9]{4}"))
L8L = PlateLayout("L8L", (PROV, LETTER) + (DIGIT,) * 5 + (DF,),
                  re.compile(_PREFIX + "[0-9]{5}[DF]"))
LAYOUTS = {7: (L7,), 8: (L8S, L8L)}


def match_layout(plate: str) -> str | None:
    for layout in (L7, L8S, L8L):
        if layout.regex.fullmatch(plate):
            return layout.name
    return None


def vehicle_type_hint(color: str, layout_name: str | None) -> VehicleTypeHint:
    # The special/unknown-color row in 5.8 takes precedence.
    if color not in {"blue", "yellow", "green"}:
        return "unknown"
    if layout_name == "L8S":
        return "small"
    if layout_name == "L8L":
        return "large"
    if layout_name == "L7":
        return {"blue": "small", "yellow": "large"}.get(color, "unknown")
    return "unknown"
