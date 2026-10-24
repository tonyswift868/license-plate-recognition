import cv2
import numpy as np
import pytest

from lpr.errors import BadInputError
from lpr.locate import locate
from lpr.preprocess import preprocess
from lpr.rectify import classify_color, rectify
from lpr.types import PlateCandidate
from lpr_tools.synth_scenes import make_scene
from .test_locate import truth_bbox
from lpr.locate import box_iou


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5, 8])
def test_rectify_scene_original_homography(cfg, index):
    image, row = make_scene(index)
    cs = locate(preprocess(image, cfg.preprocess), cfg.locate)
    cand = max(cs, key=lambda c: box_iou(c.bbox, truth_bbox(row)))
    result = rectify(image, cand, cfg.rectify)
    assert result.layout == row['layout'] and result.color == row['color']
    width = 276 if result.layout == 8 else 252
    assert result.image.shape == (80, width, 3) and result.image.dtype == np.uint8
    assert np.array_equal(result.image, cv2.warpPerspective(image, result.homography, (width, 80)))
    mapped = cv2.perspectiveTransform(np.float32(row['quad'])[None], result.homography)[0]
    target = np.float32([[0, 0], [width-1, 0], [width-1, 79], [0, 79]])
    assert np.linalg.norm(mapped-target, axis=1).max() < 12


def test_rectify_fallback_black_unknown_and_tiny(cfg):
    image = np.zeros((120, 240, 3), np.uint8)
    cand = PlateCandidate([(20, 20), (100, 20), (100, 45), (20, 45)], (20, 20, 81, 26), 'unknown', .5, 'edge')
    result = rectify(image, cand, cfg.rectify)
    assert result.layout == 0 and result.color == 'black' and not result.image.any()
    mapped = cv2.perspectiveTransform(np.float32(cand.corners)[None], result.homography)[0]
    assert np.allclose(mapped, [[0, 0], [251, 0], [251, 79], [0, 79]], atol=1e-3)
    cand = PlateCandidate([(20, 20), (24, 20), (24, 22), (20, 22)], (20, 20, 5, 3), 'blue', .5, 'color')
    assert rectify(image, cand, cfg.rectify).image.shape == (80, 252, 3)


@pytest.mark.parametrize('image', [None, np.empty((0, 0, 3), np.uint8), np.zeros((10, 10), np.float32)])
def test_rectify_bad_input(cfg, image):
    with pytest.raises(BadInputError):
        rectify(image, None, cfg.rectify)


def test_rectify_invalid_quad_and_outside(cfg):
    image = np.zeros((120, 240, 3), np.uint8)
    for points, box in [([(1, 1)]*4, (1, 1, 1, 1)), ([(300, 0), (400, 0), (400, 30), (300, 30)], (300, 0, 100, 30))]:
        with pytest.raises(BadInputError):
            rectify(image, PlateCandidate(points, box, 'blue', .5, 'color'), cfg.rectify)


@pytest.mark.parametrize('bgr,color', [((180, 65, 15), 'blue'), ((20, 205, 235), 'yellow'),
    ((110, 210, 100), 'green'), ((230, 230, 230), 'white'), ((80, 80, 80), 'unknown')])
def test_rectify_color_vote(bgr, color):
    assert classify_color(np.full((80, 252, 3), bgr, np.uint8)) == color
