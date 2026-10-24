import cv2
import numpy as np
import pytest

from lpr.errors import BadInputError
from lpr.preprocess import preprocess


@pytest.mark.parametrize('shape', [(120, 200), (120, 200, 1), (120, 200, 3)])
def test_preprocess_channels_no_enlargement(cfg, shape):
    image = np.full(shape, 90, np.uint8)
    pre = preprocess(image, cfg.preprocess)
    assert pre.work_bgr.shape == (120, 200, 3)
    assert pre.scale == 1 and pre.offset == (0, 0)
    assert np.array_equal(pre.hsv, cv2.cvtColor(pre.work_bgr, cv2.COLOR_BGR2HSV))
    assert pre.gray.dtype == np.uint8


def test_preprocess_roi_resize_and_hsv(cfg):
    image = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), np.uint8)
    pre = preprocess(image, cfg.preprocess, (100, 150, 1000, 400))
    assert pre.offset == (100, 150) and pre.scale == .8
    expected = cv2.resize(image[150:550, 100:1100], (800, 320), interpolation=cv2.INTER_AREA)
    assert np.array_equal(pre.work_bgr, expected)
    assert np.array_equal(pre.hsv, cv2.cvtColor(expected, cv2.COLOR_BGR2HSV))
    pre = preprocess(image, cfg.preprocess, (-20, -30, 220, 230))
    assert pre.offset == (0, 0) and pre.work_bgr.shape == (200, 200, 3)


@pytest.mark.parametrize('image', [None, np.zeros((0, 0, 3), np.uint8), np.zeros((119, 400, 3), np.uint8),
    np.zeros((200, 200, 4), np.uint8), np.zeros((200, 200), np.float32)])
def test_preprocess_bad_input(cfg, image):
    with pytest.raises(BadInputError):
        preprocess(image, cfg.preprocess)


@pytest.mark.parametrize('roi', [(2000, 0, 300, 300), (0, 0, 0, 200), (0, 0, 20, 200)])
def test_preprocess_bad_roi(cfg, roi):
    with pytest.raises(BadInputError):
        preprocess(np.zeros((720, 1280, 3), np.uint8), cfg.preprocess, roi)


def test_preprocess_black_and_copy(cfg):
    image = np.zeros((120, 200, 3), np.uint8)
    pre = preprocess(image, cfg.preprocess)
    assert not pre.hsv.any() and not np.shares_memory(image, pre.work_bgr)
