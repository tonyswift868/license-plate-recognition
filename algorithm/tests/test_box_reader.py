from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from lpr.charset import NUM_CLASSES
from lpr.recognize import load_box_reader, write_manifest_entry
from lpr.recognize.box_reader import (
    BOX_PAD,
    INPUT_SIZE,
    BoxNet,
    BoxPlateReader,
    box_tensor,
    checkpoint,
)
from lpr_tools.train.train_box_reader import cache_contains


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return BoxNet().eval()


def test_rgb_preprocessing_and_ctc_width(model):
    image = np.full((40, 180, 3), (10, 20, 30), dtype=np.uint8)
    tensor = box_tensor(image)
    assert tensor.shape == (3, 48, 160)
    assert tensor[:, 0, 0].tolist() == pytest.approx([
        (30/255-.485)/.229, (20/255-.456)/.224, (10/255-.406)/.225])
    with torch.inference_mode():
        assert model(tensor[None]).shape == (20, 1, NUM_CLASSES+1)


def test_box_pad_checkpoint_factory_and_invalid_inputs(model, tmp_path):
    path = tmp_path / "box_reader_v1.pt"
    torch.save(checkpoint(model, temperature=1.3, box_pad=(.1, .2)), path)
    write_manifest_entry(path, version="box_reader_v1", train_data="unit fixture",
                         val_char_acc=None, input_size=INPUT_SIZE)
    reader = load_box_reader(path)
    assert reader.box_pad == (.1, .2)
    assert reader.temperature == 1.3
    for value in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((30, 60), np.uint8),
                  np.zeros((30, 60, 3), np.float32)):
        assert reader.read(value) == ("", [])
    saved = checkpoint(model)
    saved["box_pad"] = [-.1, 0.]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="box_pad"):
        BoxPlateReader.from_file(path, version="unit")


def test_read_is_repeatable_and_does_not_mutate_inputs_or_model(model):
    reader = BoxPlateReader("unit", model)
    image = np.full((60, 200, 3), 80, np.uint8)
    before = image.copy()
    states = {key: value.clone() for key, value in model.state_dict().items()}
    expected = reader.read(image)
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(reader.read, [image, image]))
    assert all(result == expected for result in actual)
    assert np.array_equal(image, before)
    assert all(torch.equal(states[key], value) for key, value in model.state_dict().items())


def test_cache_boundary_falls_back_when_measured_box_needs_missing_pixels():
    row = {"x0": "20", "y0": "10", "orig_w": "200", "orig_h": "100"}
    assert cache_contains(row, (40, 25, 30, 10), BOX_PAD, (50, 100, 3))
    assert not cache_contains(row, (0, 25, 30, 10), BOX_PAD, (50, 100, 3))
    assert not cache_contains(row, (100, 25, 60, 10), BOX_PAD, (50, 100, 3))
