"""四角点网络模型与检测接口测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from lpr.corner_net import (
    ARCHITECTURE,
    INPUT_SIZE,
    ROI_INPUT_SIZE,
    CornerModel,
    CornerNet,
    LetterboxTransform,
    RoiConfidenceModel,
    RoiCornerRefiner,
    _decode_candidates,
    letterbox_image,
    pack_corner_roi_checkpoint,
)
from lpr.errors import ModelLoadError
from torch import nn


class _PeakModel(nn.Module):
    def __init__(self, corners: np.ndarray, score_logit: float = 12.):
        super().__init__()
        self.register_buffer("corners", torch.as_tensor(corners, dtype=torch.float32))
        self.grad_enabled: bool | None = None
        self.score_logit = score_logit

    def forward(self, image):
        self.grad_enabled = torch.is_grad_enabled()
        batch = image.shape[0]
        heatmap = torch.full((batch, 1, 80, 128), -20., dtype=image.dtype)
        heatmap[:, 0, 40, 64] = self.score_logit
        corners = self.corners.expand(batch, -1, -1, -1)
        return heatmap, corners


def test_corner_model_shapes_and_bounded_outputs():
    torch.set_num_threads(2)
    model = CornerModel()
    heatmap, corners = model(torch.zeros((2, 3, INPUT_SIZE[1], INPUT_SIZE[0])))
    assert heatmap.shape == (2, 1, 80, 128)
    assert corners.shape == (2, 8, 80, 128)
    assert torch.isfinite(heatmap).all()
    grid = model.coordinate_grid.expand(2, -1, -1, -1)
    base = torch.cat([grid[:, 0:1], grid[:, 1:2]]*4, dim=1)
    offsets = corners-base
    assert torch.all((offsets >= -.5) & (offsets <= .5))
    assert sum(parameter.numel() for parameter in model.parameters())*4 < 5*1024*1024


def test_relative_target_encoding_round_trips_a_noncenter_quad():
    model = CornerModel()
    xs = torch.tensor([47, 91], dtype=torch.long)
    ys = torch.tensor([19, 57], dtype=torch.long)
    cells = model.coordinate_grid[0, :, ys, xs].T
    target = torch.stack((cells[:, 0]-.1, cells[:, 1]-.05,
                          cells[:, 0]+.1, cells[:, 1]-.05,
                          cells[:, 0]+.1, cells[:, 1]+.05,
                          cells[:, 0]-.1, cells[:, 1]+.05), dim=1)
    relative_prediction = model.encode_relative_targets(target, ys, xs)
    relative_target = model.encode_relative_targets(target, ys, xs)
    assert torch.allclose(torch.nn.functional.l1_loss(relative_prediction, relative_target),
                          torch.tensor(0.))
    assert torch.allclose(relative_prediction-.5, target-torch.cat(
        (cells[:, 0:1], cells[:, 1:2])*4, dim=1))


def test_decode_accepts_a_valid_slanted_quad_with_narrow_axis_aligned_bbox():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transform = LetterboxTransform(1., 1., 0, 0, 100, 100, 100, 100)
    points = np.asarray([[40., 10.], [54., 8.], [52., 70.], [38., 68.]], dtype=np.float32)
    corner_map = np.broadcast_to((points / 100.).reshape(8, 1, 1), (8, 3, 3)).copy()
    heatmap = np.full((3, 3), -20., dtype=np.float32)
    heatmap[1, 1] = 12.

    candidates = _decode_candidates(image, heatmap, corner_map, transform, max_candidates=2)

    assert len(candidates) == 1
    assert candidates[0].source == "net"
    assert np.allclose(np.asarray(candidates[0].corners), points)
    assert candidates[0].bbox[2] < candidates[0].bbox[3]


@pytest.mark.parametrize("shape", [(1160, 720), (720, 1280), (900, 900)])
def test_letterbox_preserves_aspect_and_maps_points_back(shape):
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    canvas, transform = letterbox_image(image)
    assert canvas.shape == (320, 512, 3)
    assert canvas.dtype == np.uint8
    points = np.asarray([[0., 0.], [width-1., height-1.],
                         [width*.3, height*.6]], dtype=np.float32)
    forward = points.copy()
    forward[:, 0] = forward[:, 0]*transform.scale_x+transform.pad_left
    forward[:, 1] = forward[:, 1]*transform.scale_y+transform.pad_top
    assert np.allclose(transform.to_source(forward), points, atol=1e-4)


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), np.uint8),
                                    np.zeros((20, 30), np.uint8),
                                    np.zeros((20, 30, 3), np.float32)])
def test_letterbox_rejects_bad_input(image):
    with pytest.raises(ValueError):
        letterbox_image(image)


def test_decode_candidate_uses_original_coordinates_and_order():
    image = np.zeros((720, 1160, 3), dtype=np.uint8)
    transform = LetterboxTransform(.4, .4, 24, 16, 1160, 720, 512, 320)
    original = np.asarray([[100, 150], [420, 150], [420, 240], [100, 240]], np.float32)
    canvas = original.copy()
    canvas[:, 0] = canvas[:, 0]*transform.scale_x+transform.pad_left
    canvas[:, 1] = canvas[:, 1]*transform.scale_y+transform.pad_top
    normalized = canvas.copy()
    normalized[:, 0] /= transform.target_width
    normalized[:, 1] /= transform.target_height
    corner_map = np.zeros((8, 80, 128), np.float32)
    corner_map[:, 40, 64] = normalized.reshape(-1)
    heatmap = np.full((1, 80, 128), -20., np.float32)
    heatmap[0, 40, 64] = 8.
    candidates = _decode_candidates(image, heatmap, corner_map, transform, 3)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "net" and candidate.color == "unknown"
    assert candidate.bbox == (100, 150, 321, 91)
    assert np.allclose(candidate.corners, original, atol=1e-3)
    assert _decode_candidates(image, heatmap, corner_map, transform, 0) == []


def test_detect_runs_in_inference_mode_and_respects_confidence():
    image = np.zeros((720, 1160, 3), dtype=np.uint8)
    _transform = LetterboxTransform(.4, .4, 24, 16, 1160, 720, 512, 320)
    corners = np.zeros((8, 80, 128), np.float32)
    raw = np.asarray([[100, 150], [420, 150], [420, 240], [100, 240]], np.float32)
    raw[:, 0] = (raw[:, 0]*.4+24)/512
    raw[:, 1] = (raw[:, 1]*.4+16)/320
    corners[:, 40, 64] = raw.reshape(-1)
    model = _PeakModel(corners)
    detector = CornerNet("test", model)
    assert len(detector.detect(image, 1)) == 1
    assert model.grad_enabled is False
    assert detector.detect(image.astype(np.float32), 1) == []
    assert detector.detect(image, 0) == []


def test_from_file_loads_only_matching_checkpoint(tmp_path):
    path = tmp_path / "corner.pt"
    model = CornerModel()
    torch.save({"state_dict": model.state_dict(), "input_size": list(INPUT_SIZE),
                "architecture": ARCHITECTURE}, path)
    detector = CornerNet.from_file(path, version="corner_net_v1")
    assert detector.version == "corner_net_v1"
    assert detector.score_threshold == pytest.approx(.25)
    assert not detector.model.training
    assert all(torch.equal(model.state_dict()[key], detector.model.state_dict()[key])
               for key in model.state_dict())


def test_from_file_rejects_wrong_architecture(tmp_path):
    path = tmp_path / "corner.pt"
    torch.save({"state_dict": CornerModel().state_dict(), "input_size": list(INPUT_SIZE),
                "architecture": "untrusted"}, path)
    with pytest.raises(ModelLoadError):
        CornerNet.from_file(path, version="corner_net_v1")


def test_from_file_uses_score_threshold_metadata(tmp_path):
    image = np.zeros((720, 1160, 3), dtype=np.uint8)
    path = tmp_path / "corner.pt"
    torch.save({"state_dict": CornerModel().state_dict(), "input_size": list(INPUT_SIZE),
                "architecture": ARCHITECTURE, "score_threshold": .75}, path)
    detector = CornerNet.from_file(path, version="corner_net_v2")
    corners = np.zeros((8, 80, 128), np.float32)
    raw = np.asarray([[100, 150], [420, 150], [420, 240], [100, 240]], np.float32)
    raw[:, 0] = (raw[:, 0]*.4+24)/512
    raw[:, 1] = (raw[:, 1]*.4+16)/320
    corners[:, 40, 64] = raw.reshape(-1)
    detector.model = _PeakModel(corners, score_logit=0.)

    assert detector.score_threshold == pytest.approx(.75)
    assert detector.detect(image, 1) == []
    assert len(CornerNet("default", _PeakModel(corners, score_logit=0.)).detect(image, 1)) == 1


@pytest.mark.parametrize("threshold", [True, False, float("nan"), float("inf"), 0., -0.1, 1., 1.1, "0.5"])
def test_from_file_rejects_invalid_score_threshold_metadata(tmp_path, threshold):
    path = tmp_path / "corner.pt"
    torch.save({"state_dict": CornerModel().state_dict(), "input_size": list(INPUT_SIZE),
                "architecture": ARCHITECTURE, "score_threshold": threshold}, path)
    with pytest.raises(ModelLoadError, match="score_threshold"):
        CornerNet.from_file(path, version="corner_net_v2")


def test_from_file_keeps_pure_state_dict_compatibility(tmp_path):
    path = tmp_path / "corner-state-dict.pt"
    torch.save(CornerModel().state_dict(), path)
    detector = CornerNet.from_file(path, version="legacy")
    assert detector.score_threshold == pytest.approx(.25)


class _RoiPeakModel(nn.Module):
    def __init__(self, normalized, confidence_logit=12.):
        super().__init__()
        self.register_buffer("normalized", torch.as_tensor(normalized, dtype=torch.float32))
        self.confidence_logit = float(confidence_logit)

    def forward(self, images):
        batch = images.shape[0]
        return (self.normalized.expand(batch, -1),
                torch.full((batch,), self.confidence_logit, dtype=images.dtype))


def test_pack_and_load_combined_checkpoint_preserves_old_coarse_and_roi_weights(tmp_path):
    coarse_path, roi_path, packed = (tmp_path / "coarse.pt", tmp_path / "roi.pt", tmp_path / "combined.pt")
    coarse, roi = CornerModel(), RoiConfidenceModel()
    torch.save(coarse.state_dict(), coarse_path)
    torch.save(roi.state_dict(), roi_path)
    pack_corner_roi_checkpoint(coarse_path, roi_path, packed, proposal_threshold=.05,
                               final_confidence_threshold=.8, roi_padding_per_side=.25)

    detector = CornerNet.from_file(packed, version="combined")

    assert packed.stat().st_size < 5 * 1024 * 1024
    assert detector.proposal_threshold == pytest.approx(.05)
    assert detector.score_threshold == pytest.approx(.8)
    assert detector.roi_padding_per_side == pytest.approx(.25)
    assert detector.roi_model is not None and not detector.roi_model.training
    assert all(torch.equal(roi.state_dict()[k], detector.roi_model.state_dict()[k])
               for k in roi.state_dict())


def test_roi_refiner_forward_shape_matches_training_contract():
    model = RoiCornerRefiner()
    outputs = model(torch.zeros((2, 3, ROI_INPUT_SIZE[1], ROI_INPUT_SIZE[0])))
    assert outputs.shape == (2, 8)
    assert torch.all((outputs >= 0.) & (outputs <= 1.))


@pytest.mark.parametrize("field,value", [
    ("proposal_threshold", float("nan")),
    ("proposal_threshold", True),
    ("final_confidence_threshold", 1.0),
    ("roi_padding_per_side", float("inf")),
])
def test_combined_checkpoint_rejects_invalid_decode_metadata(tmp_path, field, value):
    from lpr.corner_net import COMBINED_ARCHITECTURE

    path = tmp_path / "combined-invalid.pt"
    package = {"architecture": COMBINED_ARCHITECTURE, "input_size": list(INPUT_SIZE),
               "roi_input_size": list(ROI_INPUT_SIZE),
               "coarse_state_dict": CornerModel().state_dict(),
               "roi_state_dict": RoiConfidenceModel().state_dict(),
               "proposal_threshold": .05, "final_confidence_threshold": .8,
               "roi_padding_per_side": .25}
    package[field] = value
    torch.save(package, path)
    with pytest.raises(ModelLoadError, match="metadata"):
        CornerNet.from_file(path, version="invalid")


def test_combined_detect_maps_roi_corners_filters_confidence_and_returns_roi_score():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    source = np.asarray([[50., 35.], [110., 35.], [110., 76.], [50., 76.]], np.float32)
    # The decoded coarse bbox rounds to (40,30,81,31); 25% per edge yields (19,22)-(142,69).
    roi_x0, roi_y0, roi_x1, roi_y1 = 19, 22, 142, 69
    roi_canvas, roi_transform = letterbox_image(image[roi_y0:roi_y1, roi_x0:roi_x1], ROI_INPUT_SIZE)
    roi_local = source - np.asarray([roi_x0, roi_y0], np.float32)
    roi_local[:, 0] = roi_local[:, 0] * roi_transform.scale_x + roi_transform.pad_left
    roi_local[:, 1] = roi_local[:, 1] * roi_transform.scale_y + roi_transform.pad_top
    roi_normalized = roi_local / np.asarray(ROI_INPUT_SIZE, np.float32)
    roi_model = _RoiPeakModel(roi_normalized.reshape(8), confidence_logit=0.)
    detector = CornerNet("combined", _PeakModel(_corners_for_source_box(image, (40, 30, 80, 30))),
                         score_threshold=.6, roi_model=roi_model,
                         proposal_threshold=.05, roi_padding_per_side=.25)

    # The candidate score is the ROI sigmoid (.5), so the final confidence threshold rejects it.
    assert detector.detect(image, 2) == []
    detector.score_threshold = .4
    candidates = detector.detect(image, 2)
    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(.5)
    assert np.allclose(candidates[0].corners, source, atol=1.1)
    assert np.allclose(candidates[0].bbox, (50, 35, 61, 42), atol=2.)


def _corners_for_source_box(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, box_w, box_h = bbox
    points = np.asarray([[x, y], [x + box_w, y], [x + box_w, y + box_h],
                         [x, y + box_h]], dtype=np.float32)
    # letterbox_image(image) scales to 512x256 and pads 32px vertically.
    points[:, 0] = points[:, 0] * 2.56 / 512.
    points[:, 1] = (points[:, 1] * 2.56 + 32.) / 320.
    corners = np.zeros((8, 80, 128), dtype=np.float32)
    corners[:, 40, 64] = points.reshape(8)
    return corners
