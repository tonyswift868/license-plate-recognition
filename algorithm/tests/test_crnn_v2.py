import cv2
import numpy as np
import pytest
import torch
from lpr.charset import CHARSET
from lpr.recognize.crnn import BLANK, INPUT_SIZE, CrnnNet, CrnnReader, _greedy_decode, plate_tensor
from torch import nn


@pytest.fixture(scope="module")
def base_state():
    torch.manual_seed(7)
    return {key: tensor.detach().clone() for key, tensor in CrnnNet().state_dict().items()}


def test_legacy_state_defaults_to_temperature_one_and_preserves_preprocessing(tmp_path, base_state):
    path = tmp_path / "legacy.pt"
    torch.save(base_state, path)

    reader = CrnnReader.from_file(path, version="legacy-v1")
    image = np.random.default_rng(4).integers(0, 256, size=(80, 252, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    expected_tensor = torch.from_numpy(np.ascontiguousarray(resized)).float().div_(255).sub_(0.5).div_(0.5)
    expected_tensor = expected_tensor.unsqueeze(0)
    actual_tensor = plate_tensor(image)
    assert tuple(actual_tensor.shape) == (1, 40, 160)
    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    direct_model = CrnnNet().eval()
    direct_model.load_state_dict(base_state, strict=True)
    with torch.inference_mode():
        logits = direct_model(expected_tensor.unsqueeze(0))
    assert reader.temperature == 1.0
    assert reader.read(image) == _greedy_decode(logits[:, 0])


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.full((5, 1, BLANK + 1), -3.0)
        for step, label in enumerate((2, BLANK, 33, BLANK, 34)):
            logits[step, 0, label] = 3.0
        self.register_buffer("logits", logits)

    def forward(self, images):
        return self.logits


def test_positive_temperature_preserves_argmax_text_and_reduces_confidence():
    image = np.zeros((80, 252, 3), dtype=np.uint8)
    raw_reader = CrnnReader("raw", FixedLogits(), temperature=1.0)
    warm_reader = CrnnReader("warm", FixedLogits(), temperature=3.0)

    raw_text, raw_conf = raw_reader.read(image)
    warm_text, warm_conf = warm_reader.read(image)

    assert raw_text == warm_text == CHARSET[2] + CHARSET[33] + CHARSET[34]
    assert len(raw_conf) == len(warm_conf) == 3
    assert all(after < before for before, after in zip(raw_conf, warm_conf, strict=False))


def test_pure_tensor_state_with_temperature_loads_and_keeps_strict_model_keys(tmp_path, base_state):
    path = tmp_path / "calibrated.pt"
    state = {key: value.clone() for key, value in base_state.items()}
    state["ctc_temperature"] = torch.tensor(2.5)
    torch.save(state, path)

    reader = CrnnReader.from_file(path, version="calibrated")

    assert reader.temperature == pytest.approx(2.5)
    assert set(reader.model.state_dict()) == set(base_state)
    image = np.zeros((80, 252, 3), dtype=np.uint8)
    assert isinstance(reader.read(image), tuple)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_loader_rejects_nonpositive_or_nonfinite_temperature(tmp_path, base_state, temperature):
    state = {key: value.clone() for key, value in base_state.items()}
    state["ctc_temperature"] = torch.tensor(temperature)
    path = tmp_path / "bad-temperature.pt"
    torch.save(state, path)

    with pytest.raises(ValueError):
        CrnnReader.from_file(path, version="invalid")


def test_loader_rejects_vector_temperature(tmp_path, base_state):
    state = {key: value.clone() for key, value in base_state.items()}
    state["ctc_temperature"] = torch.tensor([1.0, 2.0])
    path = tmp_path / "vector-temperature.pt"
    torch.save(state, path)

    with pytest.raises(ValueError, match="scalar"):
        CrnnReader.from_file(path, version="invalid")


@pytest.mark.parametrize("mutation", ["unexpected", "missing"])
def test_temperature_metadata_does_not_relax_strict_network_state(tmp_path, base_state, mutation):
    state = {key: value.clone() for key, value in base_state.items()}
    state["ctc_temperature"] = torch.tensor(1.7)
    if mutation == "unexpected":
        state["not_a_crnn_parameter"] = torch.tensor(1.0)
    else:
        state.pop(next(iter(base_state)))
    path = tmp_path / f"strict-{mutation}.pt"
    torch.save(state, path)

    with pytest.raises(RuntimeError):
        CrnnReader.from_file(path, version="invalid")
