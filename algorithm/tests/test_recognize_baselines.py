"""Train tiny font baselines and exercise their manifest-backed runtime contract."""

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from lpr.io_utils import imwrite
from lpr.recognize import load_classifier, read_manifest
from lpr.recognize.template import TemplateClassifier
from lpr.recognize.svm import hog_features
from lpr_tools.train.build_templates import build
from lpr_tools.train.data import load_split
from lpr_tools.train.synth_chars import generate_dataset
from lpr_tools.train.train_svm import train, validation_accuracy


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


def _write_sparse_chars(root, split, counts):
    for label, count in counts.items():
        for sample in range(count):
            rng = np.random.default_rng(label * 100 + sample)
            image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
            # Give each class a stable spatial signal while retaining sample variation.
            image[:, label % 24:(label % 24) + 4] = 255
            imwrite(root / split / str(label) / f"{sample}.png", image)


def test_sparse_template_train_mask_and_legacy_npz(tmp_path):
    data = tmp_path / "chars"
    _write_sparse_chars(data, "train", {0: 2, 31: 3})
    _write_sparse_chars(data, "val", {0: 1, 31: 1})
    path = tmp_path / "sparse.npz"
    build(data, path, version="sparse-template")
    classifier = load_classifier("template", path)
    images, _ = load_split(data, "val", require_all=False)
    probs = classifier.predict_proba(images)
    assert probs.shape == (2, 65) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.all(probs[:, 0] > 0) and np.all(probs[:, 31] > 0)
    assert np.all(probs[:, 1:31] == 0) and np.all(probs[:, 32:] == 0)
    assert classifier.predict_proba(images[:0]).shape == (0, 65)

    # Old archives have no class mask and therefore retain the historic all-class behavior.
    legacy = tmp_path / "legacy.npz"
    with legacy.open("wb") as handle:
        np.savez_compressed(handle, templates=np.zeros((65, 32, 32), np.float32),
                            temperature=np.float32(0.1))
    old = TemplateClassifier.from_file(legacy, version="legacy")
    assert old.present_classes.all()


def test_sparse_calibrated_svm_maps_classes_and_records_exclusions(tmp_path):
    data = tmp_path / "chars"
    _write_sparse_chars(data, "train", {0: 3, 31: 1, 41: 3})
    _write_sparse_chars(data, "val", {0: 1, 31: 1, 41: 1})
    path = tmp_path / "sparse.joblib"
    accuracy = train(data, path, version="sparse-svm", calibrated=True)
    assert 0 <= accuracy <= 1
    details = json.loads(path.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert details["trained_classes"] == [0, 41]
    assert details["excluded_classes"] == [{"label": 31, "samples": 1}]
    classifier = load_classifier("svm", path)
    images, _ = load_split(data, "val", require_all=False)
    probs = classifier.predict_proba(images)
    assert probs.shape == (3, 65) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.all(probs[:, 1:41] == 0) and np.all(probs[:, 42:] == 0)
    assert np.all(probs[:, [0, 41]] >= 0)
    assert classifier.predict_proba(images[:0]).shape == (0, 65)


def test_sparse_svm_requires_two_trainable_classes(tmp_path):
    data = tmp_path / "chars"
    _write_sparse_chars(data, "train", {0: 2, 31: 1})
    _write_sparse_chars(data, "val", {0: 1})
    with pytest.raises(ValueError, match="at least two classes"):
        train(data, tmp_path / "unused.joblib", version="too-small", calibrated=True)


def test_svm_validation_accuracy_uses_bounded_batches_and_global_count():
    class PredictByBatch:
        def __init__(self):
            self.batch_sizes = []
            self.offset = 0

        def predict_proba(self, images):
            self.batch_sizes.append(len(images))
            probs = np.zeros((len(images), 65), dtype=np.float32)
            # Only the final two samples are wrong; this catches unweighted means
            # of batch accuracies when the last batch is much smaller.
            predictions = np.zeros(len(images), dtype=np.int64)
            global_indexes = np.arange(self.offset, self.offset + len(images))
            predictions[global_indexes >= 2048] = 1
            probs[np.arange(len(images)), predictions] = 1
            self.offset += len(images)
            return probs

    classifier = PredictByBatch()
    images = np.zeros((2050, 32, 32), dtype=np.uint8)
    labels = np.zeros(2050, dtype=np.int64)
    assert validation_accuracy(classifier, images, labels) == pytest.approx(2048 / 2050)
    assert classifier.batch_sizes == [1024, 1024, 2]
    assert max(classifier.batch_sizes) <= 1024
