"""Standalone ROI corner refinement geometry and safe checkpoint contract."""
from pathlib import Path

import numpy as np
import pytest
import torch
from lpr.corner_net import (
    REFINE_ARCHITECTURE,
    REFINE_INPUT_SIZE,
    CornerRefiner,
    RefineModel,
    _roi_bounds,
    letterbox_image,
    refine_coordinates,
)
from lpr.errors import ModelLoadError
from lpr.types import PlateCandidate
from torch import nn


def candidate(source="color"):
    return PlateCandidate([(180., 160.), (419., 160.), (419., 239.), (180., 239.)],
                          (180, 160, 240, 80), "green", .73, source)


class Peaks(nn.Module):
    def __init__(self, *, reversed_order=False, nan=False):
        super().__init__()
        points = [(20, 20), (90, 20), (90, 44), (20, 44)]
        if reversed_order:
            points = points[::-1]
        logits = torch.full((1, 4, 64, 128), -100.)
        for k, (x, y) in enumerate(points):
            logits[0, k, y, x] = 100.
        if nan:
            logits[0, 0, 0, 0] = float("nan")
        self.register_buffer("logits", logits)

    def forward(self, images):
        return self.logits.expand(images.shape[0], -1, -1, -1)


def test_refine_model_spatial_shape_and_gradient():
    torch.set_num_threads(2)
    model = RefineModel()
    logits = model(torch.rand(2, 3, 128, 256))
    assert logits.shape == (2, 4, 64, 128)
    assert sum(p.numel() for p in model.parameters()) * 4 < 5_000_000
    refine_coordinates(logits).square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 96, 256))


def test_refine_exact_inverse_letterbox_and_contract():
    image = np.full((400, 800, 3), 127, np.uint8)
    before = image.copy()
    original = candidate()
    result = CornerRefiner(Peaks()).refine(image, original)
    x0, y0, x1, y1 = _roi_bounds(image.shape, original.bbox, .25)
    _, transform = letterbox_image(image[y0:y1, x0:x1], REFINE_INPUT_SIZE)
    local = np.asarray([(20.5/128, 20.5/64), (90.5/128, 20.5/64),
                        (90.5/128, 44.5/64), (20.5/128, 44.5/64)], np.float32)
    expected = transform.to_source(local * REFINE_INPUT_SIZE) + [x0, y0]
    np.testing.assert_allclose(result.corners, expected, atol=1e-4)
    assert (result.color, result.source, result.score) == (original.color, original.source, original.score)
    assert result is not original
    np.testing.assert_array_equal(image, before)
    assert original.bbox == (180, 160, 240, 80)


@pytest.mark.parametrize("source", ["net", "color", "edge", "merged"])
def test_refine_accepts_any_candidate_source(source):
    c = candidate(source)
    assert CornerRefiner(Peaks()).refine(np.zeros((400, 800, 3), np.uint8), c).source == source


def test_refine_invalid_geometry_and_input_falls_back():
    c = candidate()
    image = np.zeros((400, 800, 3), np.uint8)
    assert CornerRefiner(Peaks(reversed_order=True)).refine(image, c) is c
    assert CornerRefiner(Peaks(nan=True)).refine(image, c) is c
    assert CornerRefiner(Peaks()).refine(image.astype(float), c) is c
    c.bbox = (float("nan"), 1, 2, 3)
    assert CornerRefiner(Peaks()).refine(image, c) is c
    c.bbox = (-1000, -1000, 3, 3)
    assert CornerRefiner(Peaks()).refine(image, c) is c


def test_refine_checkpoint_safe_strict_and_finite(tmp_path: Path):
    path = tmp_path / "refine.pt"
    model = RefineModel()
    payload = {"architecture": REFINE_ARCHITECTURE, "input_size": REFINE_INPUT_SIZE,
               "padding": .25, "state_dict": model.state_dict()}
    torch.save(payload, path)
    loaded = CornerRefiner.from_file(path)
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in model.state_dict().items())
    for key, bad in [("architecture", "unknown"), ("input_size", [256, 96]), ("padding", float("nan"))]:
        changed = {**payload, key: bad}
        torch.save(changed, path)
        with pytest.raises(ModelLoadError):
            CornerRefiner.from_file(path)
    payload["state_dict"]["head.1.bias"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ModelLoadError):
        CornerRefiner.from_file(path)
