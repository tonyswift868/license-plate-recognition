import cv2
import numpy as np
import pytest
from lpr.errors import BadInputError
from lpr.rectify import deskew_vertical
from lpr.types import PlateCandidate, RectifiedPlate


def _plate(shear=0., color='blue', reverse=False):
    h, w = 80, 252
    clean = np.full((h, w), 225, np.uint8)
    # Character-like glyphs with vertical stems and horizontal crossbars.
    for x in (30, 59, 88, 117, 146, 175, 204):
        cv2.rectangle(clean, (x, 18), (x+11, 63), 20, -1)
        cv2.rectangle(clean, (x, 30), (x+19, 36), 20, -1)
    if reverse:
        clean = 255-clean
    source_to_skew = np.float32([[1., shear, -shear*(h-1)/2], [0., 1., 0.]])
    image = cv2.warpAffine(clean, source_to_skew, (w, h), borderMode=cv2.BORDER_REPLICATE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cand = PlateCandidate([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)],
                          (0, 0, w, h), color, .9, 'test')
    return RectifiedPlate(image, 7, color, np.eye(3), cand)


def test_vertical_deskew_corrects_known_skew_and_composes_homography():
    rect = _plate(.20)
    rect.homography = np.array([[1., 0., 10.], [0., 1., 20.], [0., 0., 1.]])
    result = deskew_vertical(rect)
    assert result is not rect and result.image.shape == rect.image.shape
    assert -.25 < result.homography[0, 1] < -.08
    shear = np.array([[1., result.homography[0, 1],
                       -result.homography[0, 1]*39.5], [0., 1., 0.], [0., 0., 1.]])
    assert np.allclose(result.homography, shear @ rect.homography)
    def gap_ink(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean([np.mean(gray[20:62, x-2:x+3] < 100) for x in (54, 83, 112, 141, 170, 199)])
    assert gap_ink(result.image) < gap_ink(rect.image)
    assert result.candidate is rect.candidate


def test_vertical_deskew_zero_is_identity_and_does_not_mutate():
    rect = _plate(0.)
    original = rect.image.copy()
    result = deskew_vertical(rect)
    assert np.array_equal(result.image, original)
    assert np.array_equal(result.homography, np.eye(3))
    assert np.array_equal(rect.image, original)


def test_vertical_deskew_negative_shear_and_zero_search_range():
    negative = _plate(-.18)
    corrected = deskew_vertical(negative)
    assert .12 < corrected.homography[0, 1] < .24
    fixed = _plate(.2)
    identity = deskew_vertical(fixed, max_shear=0.)
    assert identity is not fixed and np.array_equal(identity.image, fixed.image)
    assert np.array_equal(identity.homography, fixed.homography)


def test_vertical_deskew_uniform_plate_is_unchanged():
    rect = _plate(0.)
    rect.image[:] = 180
    result = deskew_vertical(rect)
    assert result.image is rect.image and np.array_equal(result.homography, rect.homography)


@pytest.mark.parametrize('reverse,color', [(False, 'blue'), (True, 'white')])
def test_vertical_deskew_handles_polarity_and_ignores_frame(reverse, color):
    rect = _plate(.12, color, reverse)
    rect.image[[0, -1], :] = 0 if not reverse else 255
    rect.image[:, [0, -1]] = 0 if not reverse else 255
    out = deskew_vertical(rect)
    assert out.image.shape == rect.image.shape
    assert np.isfinite(out.homography).all()


@pytest.mark.parametrize('rect', [None, RectifiedPlate(np.empty((0, 0), np.uint8), 7,
    'blue', np.eye(3), None), RectifiedPlate(np.zeros((10, 30), np.uint8), 7,
    'blue', np.eye(3), None), RectifiedPlate(np.zeros((80, 252), np.uint8), 7,
    'blue', np.full((3, 3), np.nan), None)])
def test_vertical_deskew_rejects_invalid_rectified_images(rect):
    with pytest.raises(BadInputError):
        deskew_vertical(rect)


@pytest.mark.parametrize('max_shear,step', [(.31, .01), (.2, 0), (.2, .2), (.2, .001),
                                            (float('nan'), .01)])
def test_vertical_deskew_rejects_invalid_search(max_shear, step):
    with pytest.raises(ValueError):
        deskew_vertical(_plate(), max_shear, step)
