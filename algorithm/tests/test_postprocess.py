import itertools
import numpy as np
import pytest
from lpr.charset import CHARSET, CHAR_TO_INDEX as IDX
from lpr.postprocess import decode
from lpr.types import RecogStatus as S
import lpr.postprocess as pp


def probabilities(plate, p=0.95):
    probs = np.full((len(plate), 65), (1-p)/64)
    for i, c in enumerate(plate):
        probs[i, IDX[c]] = p
    return probs


@pytest.mark.parametrize("plate,name", [("皖AY339S", "L7"), ("陕AD12345", "L8S"), ("京A12345F", "L8L")])
def test_decode(plate, name, cfg):
    result = decode(probabilities(plate), len(plate), "green", cfg.postprocess)
    assert result.plate_no == plate and result.layout_name == name and result.status == S.OK
    assert result.confidence == pytest.approx(.95)
    assert result.candidates[0][0] == plate


def test_raw_probability_and_threshold(cfg):
    p = probabilities("陕A12345", .80)
    assert decode(p, 7, "blue", cfg.postprocess).status == S.OK
    p[2] = 0
    p[2, IDX["京"]], p[2, IDX["1"]] = .9, .1
    r = decode(p, 7, "blue", cfg.postprocess)
    assert r.status == S.LOW_CONFIDENCE and r.char_confidences[2] == .1


def test_prior_only_selection(cfg):
    p = probabilities("皖A12345")
    p[0] = 0
    p[0, IDX["皖"]], p[0, IDX["陕"]] = .6, .4
    c = cfg.postprocess.model_copy(update={"province_prior": {"陕": 2}})
    r = decode(p, 7, "blue", c)
    assert r.plate_no == "陕A12345" and r.confidence == .4


def test_invalid_regex_path(cfg, monkeypatch):
    # Constraints normally guarantee the regex; exercise the defensive contract.
    monkeypatch.setattr(pp, "match_layout", lambda plate: None)
    assert decode(probabilities("陕A12345"), 7, "blue", cfg.postprocess).status == S.INVALID_FORMAT


def test_beam_matches_exhaustive(cfg):
    p = probabilities("陕A12345")
    for pos, weight in ((2, .6), (3, .7), (4, .8)):
        p[pos] = 0
        p[pos, IDX["1"]], p[pos, IDX["2"]] = weight, 1-weight
    r = decode(p, 7, "blue", cfg.postprocess)
    expected = sorted((("陕A" + "".join(t) + "45", .95**4 * np.prod([w if c == "1" else 1-w for c, w in zip(t, (.6, .7, .8))]))
                       for t in itertools.product("12", repeat=3)), key=lambda x: (-x[1], x[0]))[:3]
    assert [x[0] for x in r.candidates] == [x[0] for x in expected]
    assert [x[1] for x in r.candidates] == sorted([x[1] for x in r.candidates], reverse=True)


@pytest.mark.parametrize("bad", [np.zeros((6,65)), np.full((7,65), np.nan),
    np.full((7,65), -1), np.full((7,65), 2), np.zeros((7,65))])
def test_bad_probs(bad, cfg):
    with pytest.raises(ValueError):
        decode(bad, 7, "blue", cfg.postprocess)


def test_bad_config(cfg):
    for change in ({"topk": 0}, {"min_confidence": 2}, {"province_prior": {"陕": -1}}):
        with pytest.raises(ValueError):
            decode(probabilities("陕A12345"), 7, "blue", cfg.postprocess.model_copy(update=change))


def test_zero_allowed_mass(cfg):
    p = probabilities("陕A12345")
    p[0] = 0
    p[0, IDX["1"]] = 1
    r = decode(p, 7, "blue", cfg.postprocess)
    assert r.confidence == 0 and r.status == S.LOW_CONFIDENCE


def test_uniform_ties_and_prior_candidates(cfg):
    p=np.full((7,65),1/65)
    result=decode(p,7,"blue",cfg.postprocess)
    assert result.candidates[0][0]==result.plate_no
    p=probabilities("皖A12345")
    p[0]=0;p[0,IDX["皖"]]=.6;p[0,IDX["陕"]]=.4
    config=cfg.postprocess.model_copy(update={"province_prior":{"陕":2}})
    result=decode(p,7,"blue",config)
    assert result.candidates[0][0]==result.plate_no
    assert [score for _,score in result.candidates]==sorted([score for _,score in result.candidates],reverse=True)
    assert sum(score for _,score in result.candidates)==pytest.approx(1.)
