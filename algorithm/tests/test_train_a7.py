"""A-7 multi-source, resumable CNN and calibrated SVM checks."""

import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from lpr.recognize.svm import SvmClassifier
from lpr_tools.train import data as train_data
from lpr_tools.train.synth_chars import generate_dataset
from lpr_tools.train.train_cnn import _load_sources
from lpr_tools.train.train_cnn import train as train_cnn
from lpr_tools.train.train_svm import train as train_svm


def test_mixed_data_directory_sampling_and_partial_classes(default_font, tmp_path):
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    generate_dataset(full, per_class=2, seed=1)
    generate_dataset(partial, per_class=2, seed=2)
    # remove a class in both splits; the aggregate still covers all 65 classes
    for split in ("train", "val"):
        for path in (partial / split / "0").glob("*.png"):
            path.unlink()
    images, labels, weights = _load_sources([partial, full], [3, 1], "train", seed=5,
                                             require_all=True)
    assert len(images) == len(labels) and len(weights) == len(labels)
    assert len(np.unique(labels)) == 65
    partial_count = sum(len(list((partial / "train" / str(c)).glob("*.png"))) for c in range(65))
    assert weights[0] == pytest.approx(3 / partial_count)
    first_source_count = partial_count
    assert weights[:first_source_count].sum() / weights.sum() == pytest.approx(0.75)


def test_class_sampling_boost_preserves_source_mix(default_font, tmp_path):
    from lpr.charset import index_of

    primary, supplement = tmp_path / "primary", tmp_path / "supplement"
    generate_dataset(primary, per_class=2, seed=31)
    generate_dataset(supplement, per_class=2, seed=32)
    images, labels, weights = _load_sources(
        [primary, supplement], [3, 1], "train", seed=5, require_all=True,
        class_boosts={index_of("0"): 2.0},
    )
    assert len(images) == len(labels) == len(weights)
    primary_count = len(labels) // 2
    assert weights[:primary_count].sum() == pytest.approx(3.0)
    assert weights[primary_count:].sum() == pytest.approx(1.0)
    # Dataset order is charset order, with one training sample per class here.
    digit_zero, digit_one = 31, 32
    assert weights[digit_zero] / weights[digit_one] == pytest.approx(2.0)


def test_resume_rejects_changed_class_sampling_boost(default_font, tmp_path):
    from lpr.charset import index_of

    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=41)
    checkpoint = tmp_path / "model.checkpoint.pt"
    out = tmp_path / "model.pt"
    train_cnn(data, out, version="v", epochs=1, batch_size=130,
              checkpoint_path=checkpoint, class_boosts={index_of("0"): 2.0})
    with pytest.raises(ValueError, match="hyperparameters"):
        train_cnn(data, out, version="v", epochs=1, batch_size=130,
                  checkpoint_path=checkpoint, resume=checkpoint,
                  class_boosts={index_of("0"): 3.0})


def test_resume_accepts_pre_boost_checkpoint_without_boosts(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=42)
    checkpoint = tmp_path / "model.checkpoint.pt"
    out = tmp_path / "model.pt"
    train_cnn(data, out, version="v", epochs=2, batch_size=130,
              checkpoint_path=checkpoint, max_epochs_this_run=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["config"].pop("class_boosts")
    state["config"].pop("init_weights")
    state["config"].pop("samples_per_epoch")
    torch.save(state, checkpoint)
    history = train_cnn(data, out, version="v", epochs=2, batch_size=130,
                        checkpoint_path=checkpoint, resume=checkpoint)
    assert [row["epoch"] for row in history] == [2]


def test_init_weights_warm_starts_with_fresh_optimizer_and_records_provenance(default_font,
        tmp_path, monkeypatch):
    from lpr.recognize.cnn_model import CharCNN
    from lpr_tools.train import train_cnn as module

    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=401)
    init_path = tmp_path / "initial.pt"
    out = tmp_path / "warm.pt"
    checkpoint = tmp_path / "warm.checkpoint.pt"
    torch.manual_seed(91)
    initial = CharCNN().state_dict()
    torch.save(initial, init_path)
    source_hash = hashlib.sha256(init_path.read_bytes()).hexdigest()

    captured = {}
    original_adamw = module.torch.optim.AdamW
    original_load = module.torch.load

    def capture_adamw(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured["parameters"] = [parameter.detach().clone() for parameter in parameters]
        optimizer = original_adamw(parameters, *args, **kwargs)
        captured["optimizer_initial_state_size"] = len(optimizer.state)
        return optimizer

    def safe_load(*args, **kwargs):
        captured["weights_only"] = kwargs.get("weights_only")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(module.torch.optim, "AdamW", capture_adamw)
    monkeypatch.setattr(module.torch, "load", safe_load)
    train_cnn(data, out, version="warm", epochs=2, batch_size=130,
              checkpoint_path=checkpoint, init_weights=init_path,
              max_epochs_this_run=1)

    named_parameters = list(CharCNN().named_parameters())
    assert captured["weights_only"] is True
    assert captured["optimizer_initial_state_size"] == 0
    for (name, _), actual in zip(named_parameters, captured["parameters"], strict=False):
        torch.testing.assert_close(actual, initial[name], rtol=0, atol=0)
    assert hashlib.sha256(init_path.read_bytes()).hexdigest() == source_hash
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_origin = {"path": str(init_path.resolve()), "sha256": source_hash}
    assert state["config"]["init_weights"] == expected_origin

    # Resume inherits the initial weights' provenance but does not reload them.
    monkeypatch.setattr(module.torch.optim, "AdamW", original_adamw)
    history = train_cnn(data, out, version="warm", epochs=2, batch_size=130,
                        checkpoint_path=checkpoint, resume=checkpoint)
    assert [row["epoch"] for row in history] == [2]
    sidecar = json.loads(out.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert sidecar["init_weights"] == expected_origin


def test_init_weights_rejects_incompatible_state_dict_keys(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=402)
    invalid = tmp_path / "invalid.pt"
    torch.save({"not_a_charcnn_key": torch.zeros(1)}, invalid)
    with pytest.raises(ValueError, match="incompatible with CharCNN"):
        train_cnn(data, tmp_path / "out.pt", version="bad", epochs=1,
                  init_weights=invalid)


@pytest.mark.parametrize("state", [{"epoch": 1}, {"weight": torch.tensor(float("nan"))}])
def test_init_weights_rejects_non_tensor_or_non_finite_source(tmp_path, state):
    from lpr_tools.train.train_cnn import _state_dict_source

    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="only"):
        _state_dict_source(invalid)


def test_init_weights_rejects_resume_combination_and_output_collision(default_font, tmp_path):
    from lpr.recognize.cnn_model import CharCNN

    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=403)
    checkpoint = tmp_path / "model.checkpoint.pt"
    out = tmp_path / "model.pt"
    train_cnn(data, out, version="v", epochs=1, batch_size=130,
              checkpoint_path=checkpoint)
    init_path = tmp_path / "init.pt"
    torch.save(CharCNN().state_dict(), init_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_cnn(data, out, version="v", epochs=1, checkpoint_path=checkpoint,
                  resume=checkpoint, init_weights=init_path)
    with pytest.raises(ValueError, match="distinct paths"):
        train_cnn(data, init_path, version="v", epochs=1, init_weights=init_path)


def test_init_weights_cli_rejects_resume_combination(monkeypatch):
    from lpr_tools.train import train_cnn as module

    monkeypatch.setattr("sys.argv", ["train_cnn", "--data", "unused", "--out", "unused.pt",
                                      "--version", "v", "--resume", "checkpoint.pt",
                                      "--init-weights", "model.pt"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_samples_per_epoch_controls_sampler_and_loss_normalization(default_font, tmp_path, monkeypatch):
    from lpr_tools.train import train_cnn as module

    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=404)
    captured = {}
    original_sampler = module.WeightedRandomSampler

    def recording_sampler(weights, num_samples, *args, **kwargs):
        captured["num_samples"] = num_samples
        return original_sampler(weights, num_samples, *args, **kwargs)

    class ConstantLoss:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _logits, _labels):
            return torch.tensor(2.5, requires_grad=True)

    monkeypatch.setattr(module, "WeightedRandomSampler", recording_sampler)
    monkeypatch.setattr(module.nn, "CrossEntropyLoss", ConstantLoss)
    out = tmp_path / "sampled.pt"
    checkpoint = tmp_path / "sampled.checkpoint.pt"
    train_cnn(data, out, version="sampled", epochs=1, batch_size=8,
              samples_per_epoch=17, checkpoint_path=checkpoint)
    assert captured["num_samples"] == 17
    with out.with_suffix(".train.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["loss"]) == pytest.approx(2.5)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["config"]["samples_per_epoch"] == 17
    sidecar = json.loads(out.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert sidecar["actual_draws_per_epoch"] == 17


def test_resume_rejects_changed_samples_per_epoch(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=405)
    out = tmp_path / "sampled.pt"
    checkpoint = tmp_path / "sampled.checkpoint.pt"
    train_cnn(data, out, version="sampled", epochs=2, batch_size=130,
              samples_per_epoch=17, checkpoint_path=checkpoint, max_epochs_this_run=1)
    with pytest.raises(ValueError, match="hyperparameters"):
        train_cnn(data, out, version="sampled", epochs=2, batch_size=130,
                  samples_per_epoch=18, checkpoint_path=checkpoint, resume=checkpoint)


@pytest.mark.parametrize("value", [0, -1, 1.5, "17", True])
def test_samples_per_epoch_api_rejects_nonpositive_or_noninteger(value, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        train_cnn([], tmp_path / "bad.pt", version="bad", samples_per_epoch=value)


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "invalid"])
def test_samples_per_epoch_cli_rejects_nonpositive_or_noninteger(monkeypatch, value):
    from lpr_tools.train import train_cnn as module

    monkeypatch.setattr("sys.argv", ["train_cnn", "--data", "unused", "--out", "unused.pt",
                                      "--version", "v", "--samples-per-epoch", value])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_temperature_fit_recovers_well_specified_scalar():
    from lpr_tools.train.calibrate import fit_temperature

    scores = np.tile(np.log([[0.75, 0.25]]), (4, 1))
    result = fit_temperature(scores, np.array([0, 0, 0, 1], dtype=np.int64))
    assert result["temperature"] == pytest.approx(1.0, abs=1e-5)
    assert result["calibrated_nll"] == pytest.approx(result["raw_nll"], abs=1e-10)
    assert result["samples"] == 4 and result["iterations"] == 40


def test_apply_temperature_scales_only_final_linear_layer_without_mutation():
    from lpr.recognize.cnn_model import CharCNN
    from lpr_tools.train.calibrate import apply_temperature

    source = CharCNN().state_dict()
    original = {key: value.clone() for key, value in source.items()}
    result = apply_temperature(source, 2.0)
    assert set(result) == set(source)
    for key, value in source.items():
        assert torch.equal(value, original[key])
        if key in {"classifier.5.weight", "classifier.5.bias"}:
            torch.testing.assert_close(result[key], value / 2.0, rtol=0, atol=0)
        else:
            torch.testing.assert_close(result[key], value, rtol=0, atol=0)
        assert result[key].data_ptr() != value.data_ptr()


@pytest.mark.parametrize("scores,labels", [
    (np.array([[0.0, np.nan]]), np.array([0])),
    (np.array([[0.0, 1.0]]), np.array([0.0])),
    (np.array([[0.0, 1.0]]), np.array([2])),
    (np.array([0.0, 1.0]), np.array([0])),
])
def test_temperature_fit_rejects_invalid_scores_or_labels(scores, labels):
    from lpr_tools.train.calibrate import fit_temperature

    with pytest.raises(ValueError):
        fit_temperature(scores, labels)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), True])
def test_apply_temperature_rejects_invalid_temperature(temperature):
    from lpr.recognize.cnn_model import CharCNN
    from lpr_tools.train.calibrate import apply_temperature

    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(CharCNN().state_dict(), temperature)


def test_apply_temperature_rejects_bad_final_layer_shapes_and_nonfinite_weights():
    from lpr.charset import NUM_CLASSES
    from lpr.recognize.cnn_model import CharCNN
    from lpr_tools.train.calibrate import apply_temperature

    state = CharCNN().state_dict()
    bad_shape = dict(state)
    bad_shape["classifier.5.weight"] = torch.zeros(NUM_CLASSES, 255)
    with pytest.raises(ValueError, match="incompatible shapes"):
        apply_temperature(bad_shape, 1.5)
    nonfinite = dict(state)
    nonfinite["classifier.5.bias"] = torch.full_like(state["classifier.5.bias"], float("nan"))
    with pytest.raises(ValueError, match="finite"):
        apply_temperature(nonfinite, 1.5)


@pytest.mark.parametrize("boosts", [["0=0"], ["皖=nan"], ["?=2"], ["0=2", "0=3"]])
def test_class_boost_cli_rejects_invalid_or_duplicate_values(monkeypatch, boosts):
    from lpr_tools.train import train_cnn as module

    args = ["train_cnn", "--data", "unused", "--out", "unused.pt", "--version", "v"]
    for value in boosts:
        args.extend(["--class-boost", value])
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_parallel_reader_keeps_order_across_bounded_batches(tmp_path, monkeypatch):
    paths = [tmp_path / f"{index:04d}.png" for index in range(1031)]

    def fake_imread(path, _flags):
        value = int(Path(path).stem)
        # Vary completion order so test covers ordered map results rather than timing.
        time.sleep((value % 5) * 0.0001)
        return np.full((32, 32), value % 256, dtype=np.uint8)

    monkeypatch.setattr(train_data, "imread", fake_imread)
    with ThreadPoolExecutor(max_workers=train_data.READ_WORKERS) as executor:
        images = list(train_data._read_batched(paths, executor))
    assert len(images) == len(paths)
    assert [int(image[0, 0]) for image in images] == [index % 256 for index in range(len(paths))]


def test_resume_matches_uninterrupted_and_keeps_inference_state_dict(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=4, seed=42)
    full_path = tmp_path / "full.pt"
    split_path = tmp_path / "split.pt"
    full_history = train_cnn(data, full_path, version="full", epochs=2, batch_size=130)
    checkpoint = tmp_path / "split.checkpoint.pt"
    train_cnn(data, split_path, version="split", epochs=2, batch_size=130,
              checkpoint_path=checkpoint, max_epochs_this_run=1)
    resumed_log = tmp_path / "resumed-new.csv"
    resumed_history = train_cnn(data, split_path, version="split", epochs=2, batch_size=130,
                                checkpoint_path=checkpoint, resume=checkpoint, log_path=resumed_log)
    assert resumed_log.read_text(encoding="utf-8").splitlines()[0].startswith("epoch,")
    for key in ("epoch", "loss", "val_char_acc", "lr"):
        assert [row[key] for row in resumed_history] == [row[key] for row in full_history[1:]]
    expected = torch.load(full_path, weights_only=True)
    actual = torch.load(split_path, weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in actual.values())
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    metadata = json.loads(split_path.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert metadata["train_seconds"] > 0


def test_resume_rejects_incompatible_data(default_font, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a, per_class=2, seed=1)
    generate_dataset(b, per_class=2, seed=2)
    checkpoint = tmp_path / "model.checkpoint.pt"
    out = tmp_path / "model.pt"
    train_cnn(a, out, version="v", epochs=1, batch_size=130, checkpoint_path=checkpoint)
    with pytest.raises(ValueError, match="data directories"):
        train_cnn(b, out, version="v", epochs=2, batch_size=130,
                  checkpoint_path=checkpoint, resume=checkpoint)


def test_train_only_supplement_is_valid_and_checkpoint_is_safe(default_font, tmp_path):
    primary, supplement = tmp_path / "primary", tmp_path / "supplement"
    generate_dataset(primary, per_class=2, seed=11)
    generate_dataset(supplement, per_class=2, seed=12)
    for path in (supplement / "val").glob("*/*.png"):
        path.unlink()
    for directory in (supplement / "val").iterdir():
        directory.rmdir()
    out = tmp_path / "mixed.pt"
    checkpoint = tmp_path / "nested" / "mixed.ckpt"
    history = train_cnn([primary, supplement], out, version="mixed", epochs=1, batch_size=130,
                        data_weights=[1, 2], checkpoint_path=checkpoint)
    assert len(history) == 1 and checkpoint.exists()
    state = torch.load(checkpoint, weights_only=True)
    assert "numpy_rng" in state and isinstance(state["numpy_rng"]["keys"], torch.Tensor)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[out.name]
    source_rows = json.loads(manifest["train_data"])
    assert [row["weight"] for row in source_rows] == [1, 2]


@pytest.mark.parametrize("lr,weight_decay", [(float("nan"), 1e-4), (1e-3, float("inf"))])
def test_rejects_nonfinite_hyperparameters(default_font, tmp_path, lr, weight_decay):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=3)
    with pytest.raises(ValueError, match="positive"):
        train_cnn(data, tmp_path / "bad.pt", version="bad", epochs=1,
                  lr=lr, weight_decay=weight_decay)


def test_rejects_checkpoint_output_path_collision(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=4)
    out = tmp_path / "same.pt"
    with pytest.raises(ValueError, match="distinct paths"):
        train_cnn(data, out, version="bad", epochs=1, checkpoint_path=out)


def test_validation_group_metrics_and_best_publish_artifact(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=51)
    out = tmp_path / "publish.pt"
    history = train_cnn(data, out, version="publish", epochs=1, batch_size=130)
    with out.with_suffix(".train.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"val_province_acc", "val_alnum_acc"}.issubset(rows[0])
    best_path = out.with_name(f"{out.stem}.best{out.suffix}")
    for key, value in torch.load(best_path, weights_only=True).items():
        torch.testing.assert_close(torch.load(out, weights_only=True)[key], value, rtol=0, atol=0)
    sidecar = json.loads(out.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert sidecar["best_epoch"] == 1 and sidecar["val_province_acc"] is not None
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[out.name]
    assert manifest["val_char_acc"] == sidecar["val_char_acc"] == history[-1]["val_char_acc"]


def test_calibrated_svm_roundtrip_and_excludes_uncalibratable_class(default_font, tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=4, seed=42)
    out = tmp_path / "calibrated.joblib"
    accuracy = train_svm(data, out, version="cal", calibrated=True, max_per_class=2)
    assert 0 <= accuracy <= 1
    loaded = SvmClassifier.from_file(out, version="cal")
    assert loaded.predict_proba(np.zeros((2, 32, 32), dtype=np.uint8)).shape == (2, 65)
    train_one = data / "train" / "0"
    for sample in list(train_one.glob("*.png"))[:2]:
        sample.unlink()
    sparse_out = tmp_path / "sparse.joblib"
    train_svm(data, sparse_out, version="sparse", calibrated=True, max_per_class=2)
    metadata = json.loads(sparse_out.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert metadata["excluded_classes"] == [{"label": 0, "samples": 1}]
    sparse = SvmClassifier.from_file(sparse_out, version="sparse")
    probabilities = sparse.predict_proba(np.zeros((2, 32, 32), dtype=np.uint8))
    assert (probabilities[:, 0] == 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
