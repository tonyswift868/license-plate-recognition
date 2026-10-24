"""A-7 multi-source, resumable CNN and calibrated SVM checks."""

import json
import csv

import numpy as np
import pytest
import torch

from lpr.recognize.svm import SvmClassifier
from lpr_tools.train.synth_chars import generate_dataset
from lpr_tools.train.train_cnn import _load_sources, train as train_cnn
from lpr_tools.train.train_svm import train as train_svm


def test_mixed_data_directory_sampling_and_partial_classes(tmp_path):
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


def test_resume_matches_uninterrupted_and_keeps_inference_state_dict(tmp_path):
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


def test_resume_rejects_incompatible_data(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a, per_class=2, seed=1); generate_dataset(b, per_class=2, seed=2)
    checkpoint = tmp_path / "model.checkpoint.pt"
    out = tmp_path / "model.pt"
    train_cnn(a, out, version="v", epochs=1, batch_size=130, checkpoint_path=checkpoint)
    with pytest.raises(ValueError, match="data directories"):
        train_cnn(b, out, version="v", epochs=2, batch_size=130,
                  checkpoint_path=checkpoint, resume=checkpoint)


def test_train_only_supplement_is_valid_and_checkpoint_is_safe(tmp_path):
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
def test_rejects_nonfinite_hyperparameters(tmp_path, lr, weight_decay):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=3)
    with pytest.raises(ValueError, match="positive"):
        train_cnn(data, tmp_path / "bad.pt", version="bad", epochs=1,
                  lr=lr, weight_decay=weight_decay)


def test_rejects_checkpoint_output_path_collision(tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=2, seed=4)
    out = tmp_path / "same.pt"
    with pytest.raises(ValueError, match="distinct paths"):
        train_cnn(data, out, version="bad", epochs=1, checkpoint_path=out)


def test_validation_group_metrics_and_best_publish_artifact(tmp_path):
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


def test_calibrated_svm_roundtrip_and_excludes_uncalibratable_class(tmp_path):
    data = tmp_path / "synth"
    generate_dataset(data, per_class=4, seed=42)
    out = tmp_path / "calibrated.joblib"
    accuracy = train_svm(data, out, version="cal", calibrated=True, max_per_class=2)
    assert 0 <= accuracy <= 1
    from lpr.recognize.svm import SvmClassifier
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
