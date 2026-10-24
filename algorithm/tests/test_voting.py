from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from lpr.errors import failure_result
from lpr.testing import FakeRecognizer
from lpr.types import RecogStatus as S
from lpr.voting import vote, PlateVoteSession


def frame(plate="陕A12345", confidence=.97):
    return FakeRecognizer([(plate, "green" if len(plate)==8 else "blue")], confidence=confidence).recognize(np.zeros((120,240,3),np.uint8))


def test_corrects_one_frame():
    a, b = frame(), frame("陕A12346")
    r = vote([a,b,a])
    assert r.plate_no == a.plate_no and r.frames_used == 3
    assert r.confidence == pytest.approx(.97*2/3)
    assert r.status == S.LOW_CONFIDENCE and r.error_code == 3002
    assert r.plate_image is a.plate_image
    assert r.candidates == [(r.plate_no, r.confidence)]
    assert a.frames_used == 1


def test_modal_layout_and_fallback():
    a,b=frame(),frame("陕AD12345",.99)
    r=vote([a,b,a])
    assert r.layout == 7 and r.frames_used == 2 and r.status == S.OK
    bad=failure_result(S.NO_PLATE)
    assert vote([bad,a]).plate_no == a.plate_no
    assert vote([a,b],min_frames=3).plate_no == b.plate_no
    assert vote([bad,bad]).frames_used == 0
    assert vote([]).status == S.BAD_INPUT
    malformed=replace(a,char_confidences=[])
    assert vote([malformed,a]).frames_used == 1


def test_invalid_voted_format():
    a=frame("陕AI2345")
    assert vote([a,a]).status == S.INVALID_FORMAT


def test_stream_agreement_timeout_reset(cfg):
    s=PlateVoteSession.from_config(cfg)
    a=frame()
    assert s.feed(a,0) is None and s.feed(a,.3) is None
    assert s.feed(a,.6).frames_used == 3
    assert s.flush() is None
    assert s.feed(a,1) is None and s.feed(a,3) is None
    assert s.feed(frame("陕A12346"),3.01).frames_used == 3
    s.feed(a,4)
    assert s.flush().frames_used == 1 and s.flush() is None
    s.feed(a,5); s.reset()
    assert s.flush() is None


def test_invalid_frames_do_not_agree():
    s=PlateVoteSession(agree_frames=2)
    bad=failure_result(S.NO_PLATE)
    assert s.feed(bad,0) is None and s.feed(bad,1) is None
    assert s.feed(bad,3).status == S.NO_PLATE


def test_threaded_stream():
    s=PlateVoteSession(agree_frames=3)
    a=frame()
    with ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(lambda _:s.feed(a,0),range(30)))
    assert sum(r is not None for r in out)==10


def test_parameters_and_time():
    for kwargs in ({"agree_frames":0},{"window_s":0},{"min_frames":0},{"min_confidence":2}):
        with pytest.raises(ValueError): PlateVoteSession(**kwargs)
    for kwargs in ({"min_frames":0},{"min_confidence":2}):
        with pytest.raises(ValueError): vote([],**kwargs)
    s=PlateVoteSession();s.feed(frame(),1)
    with pytest.raises(ValueError):s.feed(frame(),0)
