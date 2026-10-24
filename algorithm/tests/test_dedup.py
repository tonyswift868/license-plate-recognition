from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from lpr.dedup import DuplicateFilter, dhash, _edit_distance_at_most_one
from lpr.errors import failure_result
from lpr.types import RecogStatus as S
from .test_voting import frame


def test_window_and_lane(cfg):
    d=DuplicateFilter.from_config(cfg);r=frame()
    assert d.check(1,r,0)==(False,"")
    assert d.check(1,r,30)==(True,"same_plate")
    assert d.check(2,r,30)==(False,"")
    assert d.check(1,r,30.01)==(False,"")
    d.reset()
    assert d.check(1,r,31)==(False,"")


def test_similar_and_image_distance():
    r=frame();near=replace(r,plate_no="陕A12346")
    d=DuplicateFilter();d.check(1,r,0)
    assert d.check(1,near,1)==(True,"similar_plate")
    assert d.check(1,replace(near,plate_image=None),2)==(False,"")
    assert d.check(1,replace(r,plate_no="京B76543"),3)==(False,"")
    up=np.tile(np.arange(9,dtype=np.uint8),(8,1))
    down=up[:,::-1].copy()
    d=DuplicateFilter(dhash_max_hamming=10)
    d.check(1,replace(r,plate_image=up),0)
    assert d.check(1,replace(near,plate_image=down),1)==(False,"")
    assert (dhash(up)^dhash(down)).bit_count()==64


@pytest.mark.parametrize("a,b,expected", [("abc","abc",True),("abc","ab",True),("ab","abc",True),
    ("abc","axbc",True),("abc","abcd",True),("abc","axc",True),("abc","axy",False),
    ("abc","axyc",False),("abc","a",False)])
def test_edit(a,b,expected):
    assert _edit_distance_at_most_one(a,b)==expected


def test_invalid_and_concurrency():
    d=DuplicateFilter()
    assert d.check(1,failure_result(S.NO_PLATE),0)==(False,"")
    r=frame()
    with ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(lambda _:d.check(1,r,1),range(20)))
    assert out.count((False,""))==1
    with pytest.raises(ValueError):d.check(1,r,0)
    for kw in ({"window_s":-1},{"dhash_max_hamming":65}):
        with pytest.raises(ValueError):DuplicateFilter(**kw)
    for image in (None,np.zeros((0,0),np.uint8),np.zeros((2,2),float),np.zeros((2,2,4),np.uint8)):
        assert dhash(image) is None
