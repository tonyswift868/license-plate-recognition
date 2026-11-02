from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from lpr.corner_net import REFINE_INPUT_SIZE
from lpr_tools import train_refine as train


def test_empirical_proposal_bbox_uses_center_and_log_size_residuals() -> None:
    gt = (100., 50., 80., 20.)
    proposal = train.sample_proposal_bbox(gt, np.asarray([.1, -.2, np.log(1.25), np.log(.5)]))
    assert proposal == pytest.approx((98., 50.7795422, 100., 10.4409155))


def test_proposal_residuals_are_finite_and_clamped() -> None:
    with pytest.raises(ValueError, match="finite"):
        train.sample_proposal_bbox((0., 0., 10., 4.), np.asarray([0., 0., np.nan, 0.]))
    clipped = train.sample_proposal_bbox((0., 0., 10., 4.), np.asarray([100., -100., 10., -10.]))
    assert clipped[2] == pytest.approx(10 * np.exp(.65))
    assert clipped[3] == pytest.approx(4 * np.exp(-.65))
    assert abs((clipped[0] + clipped[2] / 2) / 10 - .95) < 1e-7


def test_parse_gt_rejects_source_quad_outside_decoded_image() -> None:
    row = {"path": "fixture.jpg", "quad": "0 0 9 0 9 5 0 5", "bbox": "0 0 10 6"}
    quad, bbox = train.parse_gt(row, 10, 6)
    assert quad.shape == (4, 2) and bbox == (0., 0., 10., 6.)
    row["quad"] = "0 0 10 0 9 5 0 5"
    with pytest.raises(ValueError, match="outside 10x6"):
        train.parse_gt(row, 10, 6)


def test_expanded_proposal_roi_maps_ordered_gt_corners_inside_letterbox() -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    quad = np.asarray([[42, 37], [158, 37], [160, 63], [40, 63]], np.float32)
    tensor, target, before = train.crop_target(image, (40., 35., 120., 30.), quad)
    assert tensor.shape == (3, REFINE_INPUT_SIZE[1], REFINE_INPUT_SIZE[0])
    assert target.shape == (4, 2) and torch.all((target >= 0) & (target <= 1))
    assert before.ndim == 0 and before > 0
    assert target[0, 0] < target[1, 0] and target[2, 0] > target[3, 0]
    assert target[0, 1] < target[3, 1] and target[1, 1] < target[2, 1]


def test_crop_rejects_proposal_that_cannot_contain_gt_corners() -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    quad = np.asarray([[20, 20], [80, 20], [80, 40], [20, 40]], np.float32)
    with pytest.raises(ValueError, match="does not contain"):
        train.crop_target(image, (100., 50., 10., 10.), quad)


def test_gaussian_corner_targets_are_normalized_and_coordinate_loss_has_gradient() -> None:
    target = torch.tensor([[[.2, .3], [.8, .3], [.8, .7], [.2, .7]]])
    maps = train.make_corner_heatmaps(target)
    assert maps.shape == (1, 4, 64, 128)
    torch.testing.assert_close(maps.sum((-1, -2)), torch.ones((1, 4)))
    logits = torch.zeros((1, 4, 64, 128), requires_grad=True)
    loss, parts = train.refine_loss(logits, target)
    assert torch.isfinite(loss) and set(parts) == {"heatmap", "coordinates"}
    assert parts["heatmap"].item() == pytest.approx(np.log(64 * 128), rel=1e-6)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_heatmaps_reject_nonfinite_or_outside_targets() -> None:
    with pytest.raises(ValueError, match="finite"):
        train.make_corner_heatmaps(torch.full((1, 4, 2), float("nan")))
    with pytest.raises(ValueError, match="range"):
        train.make_corner_heatmaps(torch.full((1, 4, 2), 1.1))



def test_dataset_records_and_boundedly_resamples_out_of_bounds_source_quad(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
    row = {"path": str(image_path), "quad": "0 0 10 0 9 5 0 5", "bbox": "0 0 10 6",
           "subset": "ccpd_base"}
    dataset = train.RefineDataset([row], np.zeros((120, 4), np.float32))
    with pytest.raises(RuntimeError, match="after 20 draws"):
        dataset[0]
    assert dataset.counters["source_annotation_invalid_paths"] == {train.source._path_key(str(image_path))}
    assert dataset.counters["source_annotation_errors"]
    assert dataset.counters["retry_exhaustions"] == 1


def test_proposal_distribution_requires_the_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "dist.json"
    path.write_text(json.dumps({"schema": "unexpected", "samples": [[0, 0, 0, 0]] * 120}))
    with pytest.raises(ValueError, match="schema"):
        train.read_proposal_distribution(path)
    trace = tmp_path / "trace.jsonl"
    trace.write_text("independent holdout trace\n", encoding="utf-8")
    import hashlib
    payload = {"schema": "normalized_center_log_size_v1", "source_scope": "independent holdout only",
               "source_traces": [{"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}],
               "samples": [[0, 0, 0, 0]] * 120}
    path.write_text(json.dumps(payload))
    samples, digest = train.read_proposal_distribution(path)
    assert samples.shape == (120, 4) and len(digest) == 64


def test_training_budget_rejects_a_third_round_before_data_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rounds 1/2"):
        train.train(data_dir=tmp_path / "missing", distribution_path=tmp_path / "missing.json",
                    out_dir=tmp_path / "run", round_number=3, steps=1)
