"""Independent-keypoint model contracts, separate from legacy v1/v2."""
import numpy as np
import pytest
import torch
from torch import nn

from lpr.corner_net import (CornerNet, KeypointModel, KeypointNet, INPUT_SIZE,
                            keypoint_coordinates, letterbox_image)
from lpr.errors import ModelLoadError


class FixedHeatmaps(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.full((1, 5, 80, 128), -20.)
        for ch, (x, y) in enumerate(((64, 40), (30, 25), (90, 25), (90, 55), (30, 55))):
            logits[0, ch, y, x] = 12.
        self.register_buffer("logits", logits)
        self.grad_enabled = None

    def forward(self, image):
        self.grad_enabled = torch.is_grad_enabled()
        return self.logits.expand(image.shape[0], -1, -1, -1)


def test_keypoint_all_pyramid_levels_receive_finite_gradients():
    torch.set_num_threads(2)
    model = KeypointModel()
    output = model(torch.rand(1, 3, 320, 512))
    assert output.shape == (1, 5, 80, 128)
    output.square().mean().backward()
    assert sum(p.numel() for p in model.parameters()) * 4 < 5_000_000
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for level in (model.s4, model.s8, model.s16, model.s32):
        assert any(torch.count_nonzero(p.grad) for p in level.parameters())


@pytest.mark.parametrize("shape", [(720, 1280), (1160, 720), (900, 900)])
def test_keypoint_original_coordinates_order_and_inference_mode(shape):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    model = FixedHeatmaps()
    detector = KeypointNet("fixture", model, score_threshold=.05)
    before = image.copy()
    result = detector.detect(image, 3)
    assert len(result) == 1 and result[0].source == "net"
    assert model.grad_enabled is False and not model.training
    _, transform = letterbox_image(image)
    canvas_xy = np.array([[30.5*4, 25.5*4], [90.5*4, 25.5*4],
                          [90.5*4, 55.5*4], [30.5*4, 55.5*4]], np.float32)
    expected = transform.to_source(canvas_xy)
    expected[:, 0] = expected[:, 0].clip(0, shape[1]-1)
    expected[:, 1] = expected[:, 1].clip(0, shape[0]-1)
    assert np.allclose(result[0].corners, expected, atol=1e-3)
    assert np.array_equal(image, before)
    assert detector.detect(image, 0) == []
    assert detector.detect(image, True) == []


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), np.uint8),
                                   np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3), np.float32)])
def test_keypoint_invalid_input(image):
    assert KeypointNet("fixture", FixedHeatmaps()).detect(image, 2) == []


def test_keypoint_empty_nonfinite_and_nonconvex_predictions_rejected():
    model = FixedHeatmaps()
    detector = KeypointNet("fixture", model)
    image = np.zeros((320, 512, 3), np.uint8)
    model.logits[:, 0] = -20.
    assert detector.detect(image, 2) == []
    model.logits[:] = float("nan")
    assert detector.detect(image, 2) == []
    model.logits[:] = 12.  # All corners collapse onto the same center.
    assert detector.detect(image, 2) == []


def test_keypoint_checkpoint_strict_roundtrip_and_wrong_canvas(tmp_path):
    model = KeypointModel()
    payload = {"architecture": model.architecture, "input_size": list(INPUT_SIZE),
               "state_dict": model.state_dict(), "score_threshold": .05}
    file = tmp_path / "keypoints.pt"
    torch.save(payload, file)
    loaded = CornerNet.from_file(file, version="test")
    assert isinstance(loaded, KeypointNet) and loaded.version == "test"
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in model.state_dict().items())
    payload["input_size"] = [384, 256]
    torch.save(payload, file)
    with pytest.raises(ModelLoadError, match="input_size"):
        CornerNet.from_file(file, version="test")


@pytest.mark.parametrize("threshold", [True, 0, 1, float("nan")])
def test_keypoint_rejects_invalid_metadata_threshold(tmp_path, threshold):
    model = KeypointModel()
    file = tmp_path / "bad.pt"
    torch.save({"architecture": model.architecture, "input_size": list(INPUT_SIZE),
                "state_dict": model.state_dict(), "score_threshold": threshold}, file)
    with pytest.raises(ModelLoadError):
        CornerNet.from_file(file, version="bad")


def test_keypoint_targets_exact_grid_and_negative_loss():
    from lpr_tools.train_corner import make_keypoint_targets, keypoint_loss
    quads = torch.tensor([[[.2, .3], [.8, .3], [.8, .7], [.2, .7]], [[0., 0.]]*4])
    labels = torch.tensor([1., 0.])
    target = make_keypoint_targets(quads, labels)
    assert target.shape == (2, 5, 80, 128)
    assert not target[1].any()
    assert target[0, 1, 24, 25] == 1  # nearest cell center: floor(.2*128)
    assert torch.all(target[0].amax((-2, -1)) == 1)
    logits = torch.zeros_like(target, requires_grad=True)
    loss, parts = keypoint_loss(logits, target, quads, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    negative_logits = torch.zeros((1, 5, 80, 128), requires_grad=True)
    _, parts = keypoint_loss(negative_logits, torch.zeros_like(negative_logits),
                            torch.zeros(1, 4, 2), torch.zeros(1))
    assert parts["softargmax_xy"] == 0


def test_keypoint_training_one_step_records_negatives_and_safe_deploy(tmp_path, monkeypatch):
    import json
    from lpr_tools import train_corner as trainer

    class TinyDataset:
        def __init__(self, *args, **kwargs):
            assert kwargs["negatives"] is True
        def __len__(self):
            return 8
        def set_epoch(self, epoch):
            pass
        def __getitem__(self, i):
            positive = i % 2 == 0
            target = torch.tensor([.2, .3, .8, .3, .8, .7, .2, .7]) if positive else torch.zeros(8)
            return (torch.zeros(3, 320, 512), target, torch.tensor(float(positive)),
                    {"subset": "fixture", "synthetic_background": not positive})

    monkeypatch.setattr(trainer, "CornerDataset", TinyDataset)
    monkeypatch.setattr(trainer, "_keypoint_safe_inputs", lambda path: ([{}], [], [], {"fixture": True}))
    trainer.train_keypoint_model(data_dir=tmp_path, run_dir=tmp_path, steps=1)
    deployed = CornerNet.from_file(tmp_path / "keypoint-deploy-step-1.pt", version="fixture")
    assert isinstance(deployed, KeypointNet)
    checkpoint = torch.load(tmp_path / "checkpoint-step-1.pt", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["exposure_counters"]["total_image_exposures"] == 8
    assert checkpoint["exposure_counters"]["positive_exposures"] == 4
    assert checkpoint["exposure_counters"]["synthetic_background_exposures"] == 4
    assert all(torch.equal(v, deployed.model.state_dict()[k]) for k, v in checkpoint["state_dict"].items())
    with pytest.raises(ValueError, match="fresh keypoint run already exists"):
        trainer.train_keypoint_model(data_dir=tmp_path, run_dir=tmp_path, steps=1)
    with pytest.raises(ValueError, match="step 2000"):
        trainer.train_keypoint_model(data_dir=tmp_path, run_dir=tmp_path, steps=1,
                                    round_number=2, resume=tmp_path / "checkpoint-step-1.pt")


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": 2001}, {"round_number": 3},
                                    {"round_number": 2}])
def test_keypoint_training_rejects_unbounded_or_invalid_round_before_data(tmp_path, kwargs):
    from lpr_tools.train_corner import train_keypoint_model
    with pytest.raises(ValueError):
        train_keypoint_model(data_dir=tmp_path / "absent", run_dir=tmp_path, **kwargs)


def test_keypoint_cli_requires_data_directory(monkeypatch):
    import sys
    from lpr_tools.train_corner import main
    monkeypatch.setattr(sys, "argv", ["train_corner", "--stage", "keypoint"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_keypoint_safe_inputs_binds_expected_csvs_before_audit(tmp_path, monkeypatch):
    from lpr_tools import train_corner as trainer
    for name in ("safe_train.csv", "safe_val.csv", "np_train_safe.csv", "np_val_safe.csv",
                 "hard_pool_train.csv", "hard_tuning_holdout.csv", "hard_pool_audit.json"):
        (tmp_path / name).touch()
    for name in ("FROZEN_MANIFEST", "GOLDEN_MANIFEST", "UNDECODABLE_MANIFEST"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(trainer, name, path)
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "devhard_v1.csv").touch()
    monkeypatch.setattr(trainer, "DATA_ROOT", tmp_path)
    calls = []
    def audit(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("audit reached")
    monkeypatch.setattr(trainer, "_read_safe_pools", audit)
    with pytest.raises(RuntimeError, match="audit reached"):
        trainer._keypoint_safe_inputs(tmp_path)
    assert calls[0][0] == (tmp_path / "safe_train.csv", tmp_path / "np_train_safe.csv",
                           tmp_path / "safe_val.csv", tmp_path / "np_val_safe.csv")
    assert calls[0][1]["hard_train_csv"] == tmp_path / "hard_pool_train.csv"
    assert calls[0][1]["hard_val_csv"] == tmp_path / "hard_tuning_holdout.csv"
