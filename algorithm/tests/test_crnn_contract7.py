"""PlateReader protocol tests independent of the training metrics."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from torch import nn

from lpr.charset import CHARSET
from lpr.recognize.base import PlateReader
from lpr.recognize.crnn import BLANK, CrnnNet, CrnnReader


class FixedSequence(nn.Module):
    def __init__(self, ids):
        super().__init__()
        scores = torch.full((len(ids), 1, 66), -12.0)
        for t, label in enumerate(ids):
            scores[t, 0, label] = 12.0
        self.register_buffer("scores", scores)

    def forward(self, images):
        return self.scores


@pytest.mark.parametrize("width", [252, 276])
def test_reader_contract_accepts_both_plate_widths_without_mutating_input(width):
    reader = CrnnReader("contract", FixedSequence([0, 0, BLANK, 41, BLANK, 31, 31]))
    image = np.full((80, width, 3), 57, np.uint8)
    original = image.copy()
    assert isinstance(reader, PlateReader)
    text, confs = reader.read(image)
    assert text == CHARSET[0] + CHARSET[41] + CHARSET[31]
    assert len(confs) == len(text) and all(0 <= v <= 1 for v in confs)
    assert np.array_equal(image, original)
    assert not reader.model.training


@pytest.mark.parametrize("image", [None, np.zeros((80, 252), np.uint8),
    np.zeros((80, 252, 4), np.uint8), np.zeros((80, 252, 3), np.float32),
    np.zeros((0, 252, 3), np.uint8), np.zeros((80, 251, 3), np.uint8)])
def test_invalid_images_are_unreadable(image):
    assert CrnnReader("contract", FixedSequence([BLANK])).read(image) == ("", [])


def test_blank_and_nonfinite_predictions_are_unreadable():
    model = FixedSequence([BLANK] * 5)
    reader = CrnnReader("contract", model)
    image = np.zeros((80, 252, 3), np.uint8)
    assert reader.read(image) == ("", [])
    model.scores.fill_(float("nan"))
    assert reader.read(image) == ("", [])


def test_reader_reentrant_read_has_stable_results():
    reader = CrnnReader("contract", FixedSequence([0, BLANK, 41, BLANK, 31]))
    image = np.zeros((80, 276, 3), np.uint8)
    expected = reader.read(image)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(reader.read, [image] * 9))
    assert results == [expected] * 9


@pytest.mark.parametrize("corruption", ["nan", "shape", "metadata"])
def test_state_loading_rejects_nonfinite_malformed_or_non_tensor_weights(tmp_path, corruption):
    state = CrnnNet().state_dict()
    first = next(iter(state))
    if corruption == "nan":
        state[first].fill_(float("nan"))
    elif corruption == "shape":
        state[first] = torch.zeros(1)
    else:
        state["metadata"] = "not a tensor"
    path = tmp_path / "invalid.pt"
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        CrnnReader.from_file(path, version="invalid")


def test_loading_preserves_caller_torch_thread_count(tmp_path):
    original_threads = torch.get_num_threads()
    path = tmp_path / 'valid.pt'
    torch.save(CrnnNet().state_dict(), path)
    try:
        torch.set_num_threads(1)
        reader = CrnnReader.from_file(path, version='caller-threads')
        assert torch.get_num_threads() == 1
        assert reader.version == 'caller-threads'
        assert not reader.model.training
    finally:
        torch.set_num_threads(original_threads)
