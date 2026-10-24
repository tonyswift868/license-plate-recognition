"""CharCNN architecture, factory round trip, concurrency and CPU latency."""

from concurrent.futures import ThreadPoolExecutor
from statistics import median
from time import perf_counter

import numpy as np
import pytest
import torch

from lpr.recognize import load_classifier, write_manifest_entry
from lpr.recognize.cnn import CnnClassifier
from lpr.recognize.cnn_model import CharCNN


@pytest.fixture
def classifier():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield CnnClassifier("random_test", CharCNN())
    torch.set_num_threads(previous)


def test_architecture():
    model = CharCNN().eval()
    assert 600_000 <= sum(p.numel() for p in model.parameters()) <= 760_000
    with torch.inference_mode():
        assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 65)


def test_probabilities_and_empty(classifier):
    chars = np.random.default_rng(42).integers(0, 256, (8, 32, 32), dtype=np.uint8)
    before = chars.copy()
    probs = classifier.predict_proba(chars)
    assert probs.shape == (8, 65) and probs.dtype == np.float32
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(chars, before)
    assert classifier.predict_proba(chars[:0]).shape == (0, 65)
    assert not classifier.model.training
    with ThreadPoolExecutor(max_workers=3) as pool:
        for output in pool.map(classifier.predict_proba, [chars] * 3):
            np.testing.assert_array_equal(output, probs)


def test_factory_round_trip(tmp_path, classifier):
    path = tmp_path / "random.pt"
    torch.save(classifier.model.state_dict(), path)
    write_manifest_entry(path, version="test_v1", train_data="random", val_char_acc=None)
    restored = load_classifier("cnn", path)
    assert restored.version == "test_v1" and restored.name == "cnn"
    chars = np.zeros((2, 32, 32), np.uint8)
    np.testing.assert_array_equal(restored.predict_proba(chars), classifier.predict_proba(chars))


def test_batch_latency(classifier):
    chars = np.zeros((8, 32, 32), np.uint8)
    for _ in range(10):
        classifier.predict_proba(chars)
    timings = []
    for _ in range(40):
        start = perf_counter()
        classifier.predict_proba(chars)
        timings.append((perf_counter() - start) * 1000)
    elapsed = median(timings)
    print(f"8-character inference median (2 torch threads): {elapsed:.3f} ms")
    assert elapsed <= 10, f"median {elapsed:.3f} ms > 10 ms"


@pytest.mark.parametrize("chars", [np.zeros((32, 32), np.uint8), np.zeros((1, 32, 32), np.float32)])
def test_invalid_input(classifier, chars):
    with pytest.raises(ValueError):
        classifier.predict_proba(chars)


def test_training_logs_state_dict_and_manifest(tmp_path, classifier):
    import csv
    import json

    from lpr_tools.train.synth_chars import generate_dataset
    from lpr_tools.train.train_cnn import train

    data = tmp_path / "chars"
    generate_dataset(data, per_class=2)
    out = tmp_path / "trained.pt"
    history = train(data, out, version="smoke_test", epochs=2, batch_size=65)
    assert len(history) == 2 and all(np.isfinite(row["loss"]) for row in history)
    assert history[1]["lr"] < history[0]["lr"]
    with out.with_suffix(".train.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    entry = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[out.name]
    assert entry["val_char_acc"] == history[-1]["val_char_acc"]
    state = torch.load(out, weights_only=True)
    assert state and all(isinstance(value, torch.Tensor) for value in state.values())
    assert load_classifier("cnn", out).predict_proba(np.zeros((1, 32, 32), np.uint8)).shape == (1, 65)
    repeated = tmp_path / "repeat.pt"
    assert train(data, repeated, version="repeat", epochs=2, batch_size=65) == history
    for key, value in torch.load(repeated, weights_only=True).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
