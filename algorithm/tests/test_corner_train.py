from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lpr.corner_net import INPUT_SIZE, OUTPUT_STRIDE, CornerModel, letterbox_image
from lpr_tools import train_corner as train_corner_module
from lpr_tools.train_corner import (
    _focal_heatmap_loss,
    _checkpoint,
    _load_checkpoint,
    _safe_negative_crop,
    calibrate_threshold,
    center_cell,
    encode_relative_target,
    gather_quad_loss,
    make_heatmap,
    quad_cell_targets,
    read_rows,
    relative_prediction,
    _valid_quad,
    _official_val_path_set,
    _select_training_positives,
    _split_np_rows,
    _path_key,
    _input_csv_provenance,
    _verify_index_split_membership,
    _run_step_limit_reached,
    prepare_corner_data,
)


def test_center_relative_encoding_matches_forward_absolute_coordinates() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    ix, iy = 40, 20
    absolute = np.array([[.31, .24], [.57, .25], [.59, .36], [.30, .35]], np.float32)
    rel = encode_relative_target(absolute, ix, iy, gw, gh)
    center = np.array([(ix + .5) / gw, (iy + .5) / gh], np.float32)
    decoded = (rel.reshape(4, 2) + center - .5)
    np.testing.assert_allclose(decoded, absolute, atol=1e-7)

    # A forward output at the selected cell is absolute; the helper converts it
    # to the same relative coordinate system as the target, so exact predictions
    # produce zero regression loss.
    corner_map = torch.zeros((1, 8, gh, gw), dtype=torch.float32)
    corner_map[0, :, iy, ix] = torch.from_numpy(absolute.reshape(8))
    predicted_rel = relative_prediction(corner_map, ix, iy, gw, gh)[0]
    assert torch.nn.functional.smooth_l1_loss(predicted_rel, torch.from_numpy(rel)).item() == 0.


def test_center_cell_and_heatmap_use_xy_grid_order() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    quad = np.array([[.31, .24], [.57, .25], [.59, .36], [.30, .35]], np.float32)
    ix, iy = center_cell(quad, gw, gh)
    heatmap, hx, hy = make_heatmap(quad, gw, gh)
    assert (ix, iy) == (hx, hy)
    assert heatmap[hy, hx] == 1.
    assert 0. < heatmap[hy, hx - 1] < 1.
    assert 0. < heatmap[hy - 1, hx - 1] < heatmap[hy, hx - 1]
    assert heatmap[hy + 2, hx] == 0.


def test_neighbor_quad_targets_decode_same_absolute_quad_with_zero_loss() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    quad = np.array([[.31, .24], [.57, .25], [.59, .36], [.30, .35]], np.float32)
    ix, iy = center_cell(quad, gw, gh)
    targets = quad_cell_targets(quad, ix, iy, gw, gh, radius=1)
    assert len(targets) == 9
    assert all(abs(x-ix) <= 1 and abs(y-iy) <= 1 for x, y, _ in targets)
    assert quad_cell_targets(quad, ix, iy, gw, gh, has_plate=False, radius=1) == []
    corner_map = torch.zeros((1, 8, gh, gw), dtype=torch.float32)
    for x, y, _ in targets:
        corner_map[0, :, y, x] = torch.from_numpy(quad.reshape(8))
    losses = []
    for x, y, target_rel in targets:
        pred_rel = relative_prediction(corner_map, x, y, gw, gh)[0]
        losses.append(torch.nn.functional.smooth_l1_loss(pred_rel, torch.from_numpy(target_rel)))
    assert len(losses) == 9
    assert sum(losses).item() == 0.
    gathered = gather_quad_loss(corner_map, [0] * 9,
                               [x for x, _, _ in targets], [y for _, y, _ in targets],
                               np.tile(quad.reshape(8), (9, 1)), gw, gh)
    assert gathered is not None and gathered.item() == 0.


def test_gather_quad_loss_matches_loop_value_and_gradient_with_mixed_labels() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    torch.manual_seed(91)
    base = torch.rand((3, 8, gh, gw))
    labels = torch.tensor([1., 0., 1.])
    entries = [(0, 40, 20), (0, 41, 20), (2, 54, 37), (2, 53, 38)]
    entries = [entry for entry in entries if labels[entry[0]] > .5]
    targets = np.random.default_rng(91).uniform(.2, .8, (len(entries), 8)).astype(np.float32)

    loop_map = base.clone().requires_grad_(True)
    losses = []
    for (batch, x, y), target in zip(entries, targets):
        prediction = relative_prediction(loop_map[batch:batch+1], x, y, gw, gh)[0]
        target_rel = encode_relative_target(target.reshape(4, 2), x, y, gw, gh)
        losses.append(torch.nn.functional.smooth_l1_loss(prediction, torch.from_numpy(target_rel), beta=.02))
    loop_loss = torch.stack(losses).mean()
    loop_loss.backward()

    gathered_map = base.clone().requires_grad_(True)
    gathered_loss = gather_quad_loss(gathered_map,
                                    [batch for batch, _, _ in entries],
                                    [x for _, x, _ in entries],
                                    [y for _, _, y in entries], targets, gw, gh)
    assert gathered_loss is not None
    gathered_loss.backward()
    torch.testing.assert_close(gathered_loss, loop_loss)
    torch.testing.assert_close(gathered_map.grad, loop_map.grad, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(gathered_map.grad[1]) == 0  # the negative example has no regression gradient


def test_gather_quad_loss_has_no_coordinate_gradient_for_all_negative_batch() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    assert gather_quad_loss(torch.zeros((2, 8, gh, gw)), [], [], [], np.empty((0, 8)), gw, gh) is None
    model = CornerModel()
    heat, _corners = model(torch.zeros((2, 3, INPUT_SIZE[1], INPUT_SIZE[0])))
    loss = _focal_heatmap_loss(heat, torch.zeros_like(heat)) + heat.sum() * 0.
    loss.backward()
    assert model.corner_head.weight.grad is None


def test_neighbor_quad_targets_clip_at_canvas_edges() -> None:
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    quad = np.array([[.001, .001], [.007, .001], [.007, .012], [.001, .012]], np.float32)
    ix, iy = center_cell(quad, gw, gh)
    assert (ix, iy) == (0, 0)
    targets = quad_cell_targets(quad, ix, iy, gw, gh, radius=1)
    assert len(targets) == 4
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y, _ in targets)


def test_letterbox_quad_roundtrip_preserves_source_coordinates() -> None:
    image = np.zeros((720, 1160, 3), np.uint8)
    _, transform = letterbox_image(image)
    source = np.array([[120., 200.], [600., 210.], [590., 300.], [110., 290.]], np.float32)
    canvas = source.copy()
    canvas[:, 0] = canvas[:, 0] * transform.scale_x + transform.pad_left
    canvas[:, 1] = canvas[:, 1] * transform.scale_y + transform.pad_top
    restored = transform.to_source(canvas)
    np.testing.assert_allclose(restored, source, atol=1e-4)


def test_read_rows_rejects_test_or_malformed_samples(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    row = {"path": "train.jpg", "dataset": "ccpd", "subset": "ccpd_base", "split": "test",
           "has_plate": "1", "plate_no": "", "layout": "7", "plate_color": "blue",
           "quad": "1 1 10 1 10 5 1 5", "bbox": "1 1 9 4", "tilt_h": "0", "tilt_v": "0",
           "brightness": "1", "blurriness": "0", "area": "1", "skip_reason": ""}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="unsafe split"):
        read_rows(path, "train")


def test_safe_negative_crop_does_not_intersect_annotated_bbox() -> None:
    image = np.arange(180 * 240 * 3, dtype=np.uint8).reshape(180, 240, 3)
    quad = np.array([[90, 70], [150, 72], [148, 105], [88, 103]], np.float32)
    crop = _safe_negative_crop(image, quad, np.random.default_rng(7))
    assert crop.ndim == 3 and crop.shape[2] == 3
    # For this plate location each permissible side region is large enough for
    # a patch; the generated negative is a strict crop outside the padded bbox.
    assert crop.size > 0


def test_heatmap_loss_backpropagates_and_negative_is_finite() -> None:
    logits = torch.zeros((2, 1, 8, 12), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 3, 4] = 1.
    loss = _focal_heatmap_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert _focal_heatmap_loss(logits.detach()[1:], torch.zeros_like(logits[1:])).item() > 0


def test_corner_model_output_can_be_decoded_at_selected_cell() -> None:
    model = CornerModel().eval()
    gh, gw = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    with torch.inference_mode():
        heat, corners = model(torch.zeros((1, 3, INPUT_SIZE[1], INPUT_SIZE[0])))
    assert heat.shape == (1, 1, gh, gw)
    assert corners.shape == (1, 8, gh, gw)
    assert torch.isfinite(corners).all()
    # Every returned quad value is absolute normalized canvas coordinates.
    assert float(corners.min()) >= -.5 and float(corners.max()) <= 1.5


def test_validity_gate_keeps_diagonal_plate_with_square_axis_aligned_bbox() -> None:
    # Long edges run diagonally; its axis-aligned bbox is close to square.
    points_px = np.array([[218, 122], [294, 222], [318, 198], [242, 98]], np.float32)
    points = points_px / np.array(INPUT_SIZE, np.float32)
    assert _valid_quad(points)


def test_safe_checkpoint_can_resume_model_optimizer_and_step(tmp_path: Path) -> None:
    model = CornerModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "resume.pt"
    _checkpoint(path, model, optimizer, epoch=2, step=17,
                config={"seed": 42}, history=[{"step": 17}])
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    restored = CornerModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    epoch, step, history = _load_checkpoint(path, restored, restored_optimizer)
    assert (epoch, step, history) == (2, 17, [{"step": 17}])
    for key, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_validation_threshold_uses_negative_false_positive_rate() -> None:
    metrics = calibrate_threshold([.8, .7, .2], [.15, .12, .05])
    threshold = metrics["calibrated_score_threshold"]
    assert threshold is not None
    assert metrics["val_positive_recall_at_threshold"] == 1.
    assert metrics["val_negative_false_positive_rate_at_threshold"] == 0.


def _metadata_row(name: str, subset: str, split: str, *, no_plate: bool = False,
                  skip_reason: str = "") -> dict[str, str]:
    return {"path": f"D:/fixture/{name}.jpg",
            "dataset": "ccpd2020" if subset == "ccpd_green" else "ccpd2019",
            "subset": subset, "split": split,
            "has_plate": "0" if no_plate else "1",
            "skip_reason": skip_reason or ("no_plate" if no_plate else ""),
            "plate_no": "" if no_plate else "ABC1234", "layout": "7",
            "plate_color": "blue", "quad": "" if no_plate else "1 1 2 1 2 2 1 2",
            "bbox": "" if no_plate else "1 1 1 1", "tilt_h": "0", "tilt_v": "0",
            "brightness": "1", "blurriness": "1", "area": "1"}


def test_preparation_positive_sampler_is_seeded_green_complete_and_excludes_paths() -> None:
    rows = ([_metadata_row(f"base{i}", "ccpd_base", "train") for i in range(6)]
            + [_metadata_row(f"green{i}", "ccpd_green", "train") for i in range(2)])
    excluded = {_path_key("D:/fixture/base0.jpg")}
    a = _select_training_positives(rows, excluded, train_n=4, seed=42)
    b = _select_training_positives(rows, excluded, train_n=4, seed=42)
    assert a == b and len(a) == 4
    assert sum(row["subset"] == "ccpd_green" for row in a) == 2
    assert not ({_path_key(row["path"]) for row in a} & excluded)


def test_preparation_validation_sampler_is_seeded_stratified_and_excludes_paths() -> None:
    rows = ([_metadata_row(f"valbase{i}", "ccpd_base", "val") for i in range(3)]
            + [_metadata_row(f"valgreen{i}", "ccpd_green", "val") for i in range(3)])
    excluded = {_path_key("D:/fixture/valbase0.jpg"), _path_key("D:/fixture/valgreen0.jpg")}
    a, meta_a = _official_val_path_set(rows, excluded, n=4, seed=42)
    b, meta_b = _official_val_path_set(rows, excluded, n=4, seed=42)
    assert a == b and meta_a["paths"] == meta_b["paths"] and len(a) == 4
    assert meta_a["selected_by_subset"] == {
        "ccpd2019/ccpd_base": 2, "ccpd2020/ccpd_green": 2}
    assert not (a & excluded)


def test_preparation_np_split_uses_floor_80_20_and_excludes_paths() -> None:
    rows = [_metadata_row(f"np{i}", "ccpd_np", "test", no_plate=True) for i in range(7)]
    excluded = {_path_key("D:/fixture/np2.jpg")}
    train_a, val_a = _split_np_rows(rows, excluded, seed=42)
    train_b, val_b = _split_np_rows(rows, excluded, seed=42)
    train_keys = {_path_key(row["path"]) for row in train_a}
    val_keys = {_path_key(row["path"]) for row in val_a}
    assert train_a == train_b and val_a == val_b
    assert len(train_a) == int(.8 * 6) == 4 and len(val_a) == 2
    assert train_keys.isdisjoint(val_keys) and not ((train_keys | val_keys) & excluded)


def test_custom_csv_provenance_records_actual_hash_without_canonical_claim(tmp_path: Path) -> None:
    train_csv, val_csv = tmp_path / "custom-train.csv", tmp_path / "custom-val.csv"
    train_csv.write_text("path,split\\ntrain.jpg,train\\n", encoding="utf-8")
    val_csv.write_text("path,split\\nval.jpg,val\\n", encoding="utf-8")
    provenance = _input_csv_provenance(train_csv, val_csv, {
        "train_sample": {"csv_sha256": "0" * 64},
        "validation": {"csv_sha256": "1" * 64},
    })
    import hashlib
    assert provenance["train_sha256"] == hashlib.sha256(train_csv.read_bytes()).hexdigest()
    assert provenance["val_sha256"] == hashlib.sha256(val_csv.read_bytes()).hexdigest()
    assert provenance["train_csv_is_canonical"] is False
    assert provenance["val_csv_is_canonical"] is False
    assert provenance["train_sampling_audit_hash_match"] is None
    assert provenance["val_sampling_audit_hash_match"] is None


def test_run_step_limit_counts_from_resumed_checkpoint_step() -> None:
    assert not _run_step_limit_reached(step=1401, run_start_step=1400, limit=10)
    assert not _run_step_limit_reached(step=1409, run_start_step=1400, limit=10)
    assert _run_step_limit_reached(step=1410, run_start_step=1400, limit=10)
    assert not _run_step_limit_reached(step=5800, run_start_step=3600, limit=0)


def test_index_membership_rejects_spoofed_declared_split(tmp_path: Path) -> None:
    index_path = tmp_path / "ccpd_index.csv"
    fields = ["path", "split", "has_plate", "subset"]
    index_rows = [
        {"path": "D:/fixture/train.jpg", "split": "train", "has_plate": "1", "subset": "ccpd_base"},
        {"path": "D:/fixture/val.jpg", "split": "val", "has_plate": "1", "subset": "ccpd_green"},
        {"path": "D:/fixture/np_train.jpg", "split": "test", "has_plate": "0", "subset": "ccpd_np"},
        {"path": "D:/fixture/np_val.jpg", "split": "test", "has_plate": "0", "subset": "ccpd_np"},
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    train = [{**index_rows[0], "quad": "0 0 1 0 1 1 0 1"}]
    val = [{**index_rows[1], "split": "val", "quad": "0 0 1 0 1 1 0 1"}]
    np_train = [{**index_rows[2], "split": "train", "source_split": "test"}]
    np_val = [{**index_rows[3], "split": "val", "source_split": "test"}]
    authorized_train = {_path_key(index_rows[2]["path"])}
    authorized_val = {_path_key(index_rows[3]["path"])}
    checked = _verify_index_split_membership(
        train, val, np_train, np_val, index_path=index_path,
        authorized_np_train=authorized_train, authorized_np_val=authorized_val)
    assert checked["source_membership_checked"] and checked["np_safe_partition_checked"]
    with pytest.raises(ValueError, match="split mismatch"):
        _verify_index_split_membership(
            [{**train[0], "split": "train"}],
            [{**val[0], "split": "train"}], np_train, np_val,
            index_path=index_path, authorized_np_train=authorized_train,
            authorized_np_val=authorized_val)


def _prepare_inputs_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fields = ["path", "dataset", "subset", "split", "has_plate", "plate_no", "layout",
              "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
              "blurriness", "area", "skip_reason"]
    rows = [
        _metadata_row("train-base", "ccpd_base", "train"),
        _metadata_row("train-green", "ccpd_green", "train"),
        _metadata_row("val-base", "ccpd_base", "val"),
        _metadata_row("val-green", "ccpd_green", "val"),
        *[_metadata_row(f"np{i}", "ccpd_np", "test", no_plate=True) for i in range(5)],
    ]
    index = tmp_path / "ccpd_index.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    exclusions = []
    for name in ("frozen.csv", "golden.csv", "undecodable.csv"):
        path = tmp_path / name
        path.write_text("path\n", encoding="utf-8")
        exclusions.append(path)
    monkeypatch.setattr(train_corner_module, "INDEX_CSV", index)
    monkeypatch.setattr(train_corner_module, "FROZEN_MANIFEST", exclusions[0])
    monkeypatch.setattr(train_corner_module, "GOLDEN_MANIFEST", exclusions[1])
    monkeypatch.setattr(train_corner_module, "UNDECODABLE_MANIFEST", exclusions[2])


def test_prepare_corner_data_verifies_np_pixels_and_never_writes_valid_on_failure(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_inputs_fixture(monkeypatch, tmp_path)
    calls: list[str] = []

    def decode_ok(path: str) -> np.ndarray:
        calls.append(path)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(train_corner_module, "imread", decode_ok)
    good_out = tmp_path / "prepared-good"
    result = prepare_corner_data(output_dir=good_out, train_n=2, val_n=2, verify_np_images=True)
    assert len(calls) == result["counts"]["np_train"] + result["counts"]["np_val"] == 5
    assert json.loads((good_out / "np-split-audit.json").read_text(encoding="utf-8"))[
        "all_selected_decoded_valid"] is True

    calls.clear()

    def decode_failure(path: str) -> np.ndarray:
        calls.append(path)
        if len(calls) == 3:
            raise ValueError("synthetic decode failure")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(train_corner_module, "imread", decode_failure)
    bad_out = tmp_path / "prepared-bad"
    with pytest.raises(ValueError, match="NP image decode validation failed"):
        prepare_corner_data(output_dir=bad_out, train_n=2, val_n=2, verify_np_images=True)
    assert len(calls) == 5
    assert not (bad_out / "np-split-audit.json").exists()
    assert not bad_out.exists()


def test_prepare_corner_data_refuses_different_existing_output_without_overwrite(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_inputs_fixture(monkeypatch, tmp_path)
    out = tmp_path / "prepared"
    out.mkdir()
    sentinel = out / "train_sampled_2.csv"
    sentinel.write_bytes(b"preserve these bytes")
    with pytest.raises(FileExistsError, match="refusing to overwrite different prepared data"):
        prepare_corner_data(output_dir=out, train_n=2, val_n=2)
    assert sentinel.read_bytes() == b"preserve these bytes"
    assert list(out.iterdir()) == [sentinel]


def test_roi_train_pool_rejects_validation_overlap() -> None:
    row = {"path": "C:/data/a.jpg", "split": "train", "has_plate": "1",
           "subset": "ccpd_base", "quad": "10 10 30 10 30 20 10 20",
           "bbox": "10 10 20 10"}
    with pytest.raises(ValueError, match="validation"):
        train_corner_module._validate_rows(
            [row], positive=True, forbidden=set(),
            val_keys={_path_key(row["path"])})


def test_roi_letterbox_and_geometric_augmentation_are_seed_deterministic() -> None:
    image = np.arange(120 * 240 * 3, dtype=np.uint8).reshape(120, 240, 3)
    row = {"path": "fixture.jpg", "quad": "75 48 160 48 160 74 75 74",
           "bbox": "75 48 85 26"}
    first = train_corner_module._positive_roi(
        image, row, np.random.default_rng(18), geometric=True)
    second = train_corner_module._positive_roi(
        image, row, np.random.default_rng(18), geometric=True)
    assert np.array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert first[0].shape == (96, 256, 3)
    assert first[1].shape == (8,)
    assert np.isfinite(first[1]).all() and np.all((first[1] >= 0) & (first[1] <= 1))


def test_roi_negative_placeholder_quad_has_no_regression_gradient() -> None:
    predicted = torch.tensor([[.1] * 8, [.9] * 8], requires_grad=True)
    targets = torch.tensor([[.2] * 8, [0.] * 8])
    labels = torch.tensor([1., 0.])
    positive_mask = labels > .5
    loss = torch.nn.functional.smooth_l1_loss(
        predicted[positive_mask], targets[positive_mask], beta=.015)
    loss.backward()
    assert torch.all(predicted.grad[1] == 0)
    assert torch.any(predicted.grad[0] != 0)


def test_roi_source_quad_outside_image_is_classified_and_bounded_resampled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"path": "bad.jpg", "quad": "10 10 742 10 742 30 10 30",
           "bbox": "10 10 732 20"}
    good = {"path": "good.jpg", "quad": "10 10 50 10 50 30 10 30",
            "bbox": "10 10 40 20"}
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(train_corner_module.SourceQuadOutOfBounds, match="exceeds 80x80"):
        train_corner_module._positive_roi(image, bad, np.random.default_rng(7), geometric=False)
    choices = iter([bad, good])
    monkeypatch.setattr(train_corner_module.random, "choice", lambda _rows: next(choices))
    monkeypatch.setattr(train_corner_module, "imread", lambda _path: image)
    invalid: set[str] = set()
    counters = {"source_quad_resample_draws": 0, "source_quad_retry_exhaustions": 0}
    tensor, target = train_corner_module._sample_positive_roi(
        [bad, good], np.random.default_rng(8), geometric=False,
        invalid_source_paths=invalid, counters=counters)
    assert _path_key("bad.jpg") in invalid
    assert counters == {"source_quad_resample_draws": 1, "source_quad_retry_exhaustions": 0}
    assert tensor.shape == (3, 96, 256)
    assert torch.isfinite(target).all()


def test_strong_roi_geometry_transforms_image_markers_with_ordered_corners() -> None:
    width, height = train_corner_module.ROI_WH
    points = np.asarray([[42, 28], [205, 29], [200, 68], [45, 70]], dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    marker_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for index, (x, y) in enumerate(points):
        image[int(y)-1:int(y)+2, int(x)-1:int(x)+2, :] = marker_colors[index]
    target = (points / np.asarray([width, height], dtype=np.float32)).reshape(8)
    seed = next(seed for seed in range(1000)
                if np.random.default_rng(seed).random() >= .90)
    warped_a, corners_a, mode_a = train_corner_module._strong_roi_geometry(
        image, target, np.random.default_rng(seed))
    warped_b, corners_b, mode_b = train_corner_module._strong_roi_geometry(
        image, target, np.random.default_rng(seed))
    assert mode_a == mode_b == "strong"
    assert np.array_equal(warped_a, warped_b)
    np.testing.assert_allclose(corners_a, corners_b)
    transformed = corners_a.reshape(4, 2) * np.asarray([width, height], dtype=np.float32)
    for index, (x, y) in enumerate(transformed):
        channels = warped_a.astype(np.int16)
        if index == 0:
            marker = (channels[:, :, 0] > 220) & (channels[:, :, 1] < 20) & (channels[:, :, 2] < 20)
        elif index == 1:
            marker = (channels[:, :, 1] > 220) & (channels[:, :, 0] < 20) & (channels[:, :, 2] < 20)
        elif index == 2:
            marker = (channels[:, :, 2] > 220) & (channels[:, :, 0] < 20) & (channels[:, :, 1] < 20)
        else:
            marker = (channels[:, :, 0] > 220) & (channels[:, :, 1] > 220) & (channels[:, :, 2] < 20)
        yy, xx = np.where(marker)
        assert len(xx) > 0
        assert abs(float(xx.mean()) - float(x)) < 2.5
        assert abs(float(yy.mean()) - float(y)) < 2.5


def test_roi_confidence_resume_restores_full_state_dict(tmp_path: Path) -> None:
    initial = train_corner_module.RoiConfidenceModel()
    checkpoint = tmp_path / "full-roi-state.pt"
    torch.save(initial.state_dict(), checkpoint)
    restored = train_corner_module.RoiConfidenceModel()
    digest = train_corner_module._load_roi_initial_state(
        restored, checkpoint, full_model=True)
    assert len(digest) == 64
    for name, value in initial.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_average_coarse_state_dicts_uses_equal_arithmetic_mean() -> None:
    first = {"weight": torch.tensor([1., 5.]), "bias": torch.tensor([[2.]])}
    second = {"weight": torch.tensor([3., 1.]), "bias": torch.tensor([[4.]])}
    third = {"weight": torch.tensor([5., 3.]), "bias": torch.tensor([[6.]])}
    averaged = train_corner_module._average_state_dicts([first, second, third])
    assert torch.equal(averaged["weight"], torch.tensor([3., 3.]))
    assert torch.equal(averaged["bias"], torch.tensor([[4.]]))


def test_average_coarse_rejects_incompatible_tensor_metadata() -> None:
    base = {"weight": torch.ones(2, dtype=torch.float32)}
    incompatible_states = [
        {"other": torch.ones(2, dtype=torch.float32)},
        {"weight": torch.ones(3, dtype=torch.float32)},
        {"weight": torch.ones(2, dtype=torch.float64)},
    ]
    for incompatible in incompatible_states:
        with pytest.raises(ValueError):
            train_corner_module._average_state_dicts([base, incompatible])
