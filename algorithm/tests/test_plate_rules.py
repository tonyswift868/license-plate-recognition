import pytest
from lpr.charset import CHARSET
from lpr.plate_rules import ALNUM, DF, DIGIT, LETTER, PROV, match_layout, vehicle_type_hint


@pytest.mark.parametrize("plate,expected", [("皖AY339S", "L7"), ("陕AD12345", "L8S"),
    ("京A12345F", "L8L"), ("陕AI2345", None), ("陕AO2345", None),
    ("警A12345", None), ("陕A1234", None), ("陕A12345\n", None), ("陕A123456", None)])
def test_layout(plate, expected):
    assert match_layout(plate) == expected


def test_groups():
    assert [len(x) for x in (PROV, LETTER, ALNUM, DIGIT, DF)] == [31, 24, 34, 10, 2]
    assert {CHARSET[i] for i in DF} == {"D", "F"}
    assert not {"I", "O"} & {CHARSET[i] for i in ALNUM}


@pytest.mark.parametrize("color,layout,hint", [("blue", "L7", "small"),
    ("yellow", "L7", "large"), ("green", "L8S", "small"), ("yellow", "L8L", "large"),
    ("white", "L7", "unknown"), ("black", "L8L", "unknown"),
    ("unknown", "L8S", "unknown"), ("green", "L7", "unknown"), ("blue", None, "unknown")])
def test_hints(color, layout, hint):
    assert vehicle_type_hint(color, layout) == hint
