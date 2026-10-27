from types import SimpleNamespace
import os

import pytest

import numpy as np
import torch
from torch import nn

from lpr.boxcrop import crop_box
from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize.crnn import CrnnReader, plate_tensor
from lpr_tools.eval.box_direct import (PAD_GRID, _choose_best_pad, _gt_bbox_from_quad,
                                       _group_keys, _read_crnn_native, _read_outcome,
                                       _select_candidate)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-memory ABI")
def test_windows_rss_uses_native_handle_width():
    from lpr_tools.eval.box_direct import _rss_bytes

    rss = _rss_bytes()
    assert isinstance(rss, int) and rss > 0


class _FixedLogits(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.indices = [CHAR_TO_INDEX[char] for char in text]
        self.last_input = None

    def forward(self, images):
        self.last_input = images.detach().clone()
        logits = torch.full((len(self.indices), images.shape[0], NUM_CLASSES + 1), -10.0)
        for step, index in enumerate(self.indices):
            logits[step, :, index] = 10.0
        return logits


def test_pad_grid_has_fixed_x_major_then_y_order():
    assert PAD_GRID == tuple((x, y) for x in (.05, .10, .15, .25)
                             for y in (.15, .25, .40))
    assert len(PAD_GRID) == 12


def test_crop_box_is_used_at_native_resolution_without_evaluator_resize():
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    crop = crop_box(image, (5, 6, 10, 4), .1, .25)
    assert crop.shape == (6, 12, 3)
    assert np.array_equal(crop, image[5:11, 4:16])


def test_gt_bbox_uses_float_axis_aligned_extent_of_quad_not_index_bbox():
    row = SimpleNamespace(path="sample.jpg", quad=((4.25, 5.5), (14.75, 6.0),
                                                      (13.5, 10.25), (3.75, 9.5)))
    assert _gt_bbox_from_quad(row) == (3.75, 5.5, 11.0, 4.75)


def test_highest_confidence_legal_candidate_wins_and_ties_keep_top3_order():
    reads = [
        {"candidate_index": 1, "legal": True, "min_confidence": .8, "prediction": "皖A12345"},
        {"candidate_index": 2, "legal": False, "min_confidence": .99, "prediction": "not-plate"},
        {"candidate_index": 3, "legal": True, "min_confidence": .8, "prediction": "皖A54321"},
    ]
    for item in reads:
        item.update(status="OK", correct=True, matched_chars=7, char_count=7)
    assert _select_candidate(reads)["candidate_index"] == 1
    reads[2]["min_confidence"] = .9
    assert _select_candidate(reads)["candidate_index"] == 3


def test_no_legal_candidate_returns_fixed_wrong_result():
    selected = _select_candidate([{"candidate_index": 1, "legal": False,
                                   "min_confidence": .99, "char_count": 7}])
    assert selected["candidate_index"] is None
    assert selected["correct"] is False
    assert selected["status"] == "NO_LEGAL_READ"


def test_decode_status_and_whole_character_count_are_explicit():
    low = _read_outcome("皖A12345", [.7] * 7, "blue", .8, "皖A12345")
    assert low["status"] == "LOW_CONFIDENCE"
    assert low["legal"] is True
    assert low["correct"] is True
    assert low["matched_chars"] == low["char_count"] == 7
    malformed = _read_outcome("皖A?2345", [.99] * 7, "blue", .8, "皖A12345")
    assert malformed["status"] == "INVALID_FORMAT"
    assert malformed["legal"] is False


def test_best_pad_uses_actual_exact_count_and_fixed_order_for_ties():
    assert _choose_best_pad([5] * 12)["pad_order"] == 0
    best = _choose_best_pad([4, 3, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0])
    assert best["pad_order"] == 2
    assert (best["pad_x"], best["pad_y"]) == PAD_GRID[2]


def test_dark_and_tilt_rotate_groups_use_filename_brightness_and_subset():
    assert "ccpd_db_brightness_le_19" in _group_keys("devhard", "ccpd_db", 19)
    assert "ccpd_db_brightness_le_19" not in _group_keys("devhard", "ccpd_db", 20)
    assert "ccpd_tilt_plus_rotate" in _group_keys("devhard", "ccpd_tilt", None)
    assert "ccpd_tilt_plus_rotate" in _group_keys("devhard", "ccpd_rotate", None)
    assert "ccpd_tilt_plus_rotate" not in _group_keys("devhard", "ccpd_db", 10)


def test_native_adapter_matches_public_read_for_canonical_crop():
    reader = CrnnReader("test", _FixedLogits("皖A12345"), temperature=1.3)
    crop = np.zeros((80, 252, 3), np.uint8)
    assert _read_crnn_native(reader, crop, torch) == reader.read(crop)


def test_native_adapter_uses_crnn_plate_tensor_preprocessing_without_external_resize():
    reader = CrnnReader("test", _FixedLogits("皖A12345"))
    crop = np.full((43, 141, 3), (10, 80, 170), np.uint8)
    expected_tensor = plate_tensor(crop)[None]
    text, confidences = _read_crnn_native(reader, crop, torch)
    assert text == "皖A12345"
    assert len(confidences) == 7
    assert reader.model.last_input.shape == (1, 1, 40, 160)
    assert torch.equal(reader.model.last_input, expected_tensor)
