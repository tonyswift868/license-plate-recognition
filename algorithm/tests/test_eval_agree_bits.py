import itertools
import random

import pytest

from lpr_tools.eval.agree_bits import DecisionBits
from lpr_tools.eval.agree_threshold import _score


def _make_records(seed):
    rng = random.Random(seed)
    records = []
    plates = ("京A12345", "沪B23456", "粤C34567")
    for i in range(7):
        truth = plates[rng.randrange(len(plates))]
        rects = []
        for _ in range(rng.randrange(1, 5)):
            pair = []
            for side in range(2):
                if rng.random() < .15:
                    pair.append(None)
                    continue
                plate = truth if rng.random() < .55 else plates[rng.randrange(len(plates))]
                pair.append({"plate": plate, "confidence": rng.choice((.2, .4, .6, .8, 1.0)),
                             "status": rng.choice(("OK", "LOW_CONFIDENCE")), "valid": True})
            seg, reader = pair
            rects.append({"segment": seg, "reader": reader})
        records.append({"truth": truth, "split": "val" if i < 3 else "devhard",
                        "input_error": None, "lead_color": "blue", "candidate_count": 2,
                        "rects": rects})
    return records


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("prefer", ("segment", "reader"))
def test_bitset_counts_match_ordered_reference(seed, prefer):
    records = _make_records(seed)
    bits = DecisionBits(records)
    for agree_min, solo_seg, solo_reader in itertools.product((0.0, .4, .8, 1.0000000000000002), repeat=3):
        params = {"agree_min": agree_min, "solo_seg": solo_seg,
                  "solo_reader": solo_reader, "prefer": prefer}
        expected = _score(records, params)["sets"]
        actual = bits.counts(params)
        for name in ("val", "devhard"):
            assert actual[name]["accepted"] == expected[name]["accepted"]
            assert actual[name]["correct"] == expected[name]["correct"]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("prefer", ("segment", "reader"))
def test_bitset_box_bounds_cover_every_exact_point(seed, prefer):
    records = _make_records(seed + 101)
    bits = DecisionBits(records)
    values = (0.0, .4, .8, 1.0000000000000002)
    lower = {"agree_min": values[0], "solo_seg": values[0], "solo_reader": values[0],
             "prefer": prefer}
    upper = {"agree_min": values[-1], "solo_seg": values[-1], "solo_reader": values[-1],
             "prefer": prefer}
    bounds = bits.bounds(lower, upper)
    for params_values in itertools.product(values, repeat=3):
        params = {"agree_min": params_values[0], "solo_seg": params_values[1],
                  "solo_reader": params_values[2], "prefer": prefer}
        score = _score(records, params)["sets"]
        for name in ("val", "devhard"):
            assert score[name]["accepted"] <= bounds[name]["possible_accept"]
            assert score[name]["accepted"] >= bounds[name]["guaranteed_accept"]
            assert score[name]["correct"] <= bounds[name]["possible_correct"]


@pytest.mark.parametrize("prefer", ("segment", "reader"))
def test_first_guaranteed_ok_excludes_later_correct_rect_from_bound(prefer):
    row = {"truth": "T", "split": "val", "input_error": None, "lead_color": "blue",
           "candidate_count": 2, "rects": [
               {"segment": {"plate": "X", "confidence": .8, "valid": True},
                "reader": {"plate": "X", "confidence": .8, "valid": True}},
               {"segment": {"plate": "T", "confidence": .8, "valid": True},
                "reader": {"plate": "T", "confidence": .8, "valid": True}},
           ]}
    bits = DecisionBits([row])
    lower = {"agree_min": 0.0, "solo_seg": 0.0, "solo_reader": 0.0, "prefer": prefer}
    upper = {"agree_min": .8, "solo_seg": .8, "solo_reader": .8, "prefer": prefer}
    # The first rect is guaranteed OK at the high corner and always precedes
    # the later correct rect; no setting inside this box can return truth.
    assert bits.bounds(lower, upper)["val"]["possible_correct"] == 0
