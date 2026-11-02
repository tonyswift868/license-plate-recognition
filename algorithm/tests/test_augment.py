import numpy as np
import pytest
from lpr_tools.train.augment import augment_char


def test_augment_reproducible_bounded_and_nonmutating():
    image = np.zeros((32, 32), np.uint8)
    image[4:28, 12:20] = 255
    original = image.copy()
    first, second = np.random.default_rng(42), np.random.default_rng(42)
    outputs = []
    for _ in range(100):
        result = augment_char(image, first)
        assert result.shape == (32, 32) and result.dtype == np.uint8
        np.testing.assert_array_equal(result, augment_char(image, second))
        outputs.append(result.tobytes())
    assert len(set(outputs)) == 100
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("image", [np.zeros((31, 32), np.uint8), np.zeros((32, 32), np.float32)])
def test_augment_rejects_invalid(image):
    with pytest.raises(ValueError):
        augment_char(image, np.random.default_rng(42))
