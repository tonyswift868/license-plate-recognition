"""Train tiny font baselines and exercise their manifest-backed runtime contract."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lpr.recognize import load_classifier, read_manifest
from lpr.recognize.svm import hog_features
from lpr_tools.train.build_templates import build
from lpr_tools.train.data import load_split
from lpr_tools.train.synth_chars import generate_dataset
from lpr_tools.train.train_svm import train


@pytest.fixture(scope="module")
def chars_data(tmp_path_factory):
    data = tmp_path_factory.mktemp("baseline") / "chars"
    generate_dataset(data, per_class=8)
    return data


@pytest.mark.parametrize("kind,extension,builder", [("template", "npz", build), ("svm", "joblib", train)])
def test_baseline_factory_round_trip(chars_data, tmp_path, kind, extension, builder):
    path = tmp_path / f"model.{extension}"
    accuracy = builder(chars_data, path, version=f"{kind}_test")
    model = load_classifier(kind, path)
    images, labels = load_split(chars_data, "val")
    original = images.copy()
    probs = model.predict_proba(images)
    assert probs.shape == (len(images), 65) and probs.dtype == np.float32
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert accuracy == float(np.mean(probs.argmax(axis=1) == labels))
    assert accuracy > 0.2  # detect wrong class ordering or constant predictions
    assert model.predict_proba(images[:0]).shape == (0, 65)
    assert read_manifest(tmp_path)[path.name]["val_char_acc"] == accuracy
    assert model.version == f"{kind}_test" and model.name == kind
    np.testing.assert_array_equal(images, original)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(model.predict_proba, [images[:8]] * 3):
            np.testing.assert_array_equal(result, probs[:8])


def test_hog_and_deterministic_sample_cap(chars_data):
    images, labels = load_split(chars_data, "train", max_per_class=3, seed=42)
    again, again_labels = load_split(chars_data, "train", max_per_class=3, seed=42)
    np.testing.assert_array_equal(images, again)
    np.testing.assert_array_equal(labels, again_labels)
    assert (np.bincount(labels, minlength=65) == 3).all()
    features = hog_features(images)
    assert features.shape == (195, 324) and features.dtype == np.float32
    assert np.isfinite(features).all()
    assert hog_features(images[:0]).shape == (0, 324)


def test_reader_rejects_incomplete_dataset(tmp_path):
    with pytest.raises(ValueError, match="class 0"):
        load_split(tmp_path, "train")


@pytest.mark.parametrize("cap", [1, 501])
def test_svm_rejects_invalid_cap(tmp_path, cap):
    with pytest.raises(ValueError, match="2..500"):
        train(tmp_path, tmp_path / "unused.joblib", version="invalid", max_per_class=cap)
