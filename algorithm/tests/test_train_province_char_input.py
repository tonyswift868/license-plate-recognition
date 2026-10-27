import numpy as np
import pytest

from lpr.charnorm import normalize_char, unify_polarity
from lpr_tools.train import province_synth
from lpr_tools.train.province_char_input import normalized_first_char


@pytest.mark.parametrize("orientation", ["row", "column"])
def test_tiny_foreground_keeps_full_labeled_cell(orientation):
    import cv2
    image = np.zeros((80, 256, 3), dtype=np.uint8)
    if orientation == "row":
        image[20, 10:25] = 255
    else:
        image[10:50, 20] = 255
    quad = np.array([[0, 0], [255, 0], [255, 79], [0, 79]], np.float32)
    with pytest.raises(ValueError, match="foreground is too small"):
        province_synth.normalized_first_char(image, quad, "皖A12345")
    output = normalized_first_char(image, quad, "皖A12345")
    expected = normalize_char(unify_polarity(cv2.cvtColor(image[5:75, 3:37], cv2.COLOR_BGR2GRAY), "blue"))
    assert output.shape == (32, 32) and output.dtype == np.uint8
    np.testing.assert_array_equal(output, expected)


def test_regular_foreground_is_unchanged_and_invalid_quad_still_rejected():
    image = np.zeros((80, 256, 3), dtype=np.uint8)
    image[10:50, 10:25] = 255
    quad = np.array([[0, 0], [255, 0], [255, 79], [0, 79]], np.float32)
    np.testing.assert_array_equal(normalized_first_char(image, quad, "皖A12345"),
                                  province_synth.normalized_first_char(image, quad, "皖A12345"))
    with pytest.raises(ValueError, match="quad"):
        normalized_first_char(image, np.zeros((3, 2)), "皖A12345")
