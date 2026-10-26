import numpy as np
import torch

from lpr.charset import CHARSET
from lpr.recognize.crnn import BLANK, INPUT_SIZE, CrnnNet, CrnnReader, _greedy_decode


def test_compact_net_outputs_ctc_sequence():
    model = CrnnNet().eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 1, INPUT_SIZE[1], INPUT_SIZE[0]))
    assert logits.shape == (40, 1, 66)
    assert sum(p.numel() for p in model.parameters()) * 4 < 5_000_000


def test_greedy_decode_collapses_repeats_and_blank():
    ids = [4, 4, BLANK, 4, 5, 5, BLANK]
    scores = torch.full((len(ids), 66), -10.0)
    for t, idx in enumerate(ids):
        scores[t, idx] = 10.0
    text, confs = _greedy_decode(scores)
    assert text == CHARSET[4] + CHARSET[4] + CHARSET[5]
    assert len(confs) == 3 and all(0.99 < c <= 1 for c in confs)


def test_reader_returns_empty_for_malformed_input():
    reader = CrnnReader("test", CrnnNet())
    assert reader.read(np.zeros((80, 252), np.uint8)) == ("", [])


def test_reader_resizes_plate_to_shared_input_size():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shape = None

        def forward(self, images):
            self.seen_shape = tuple(images.shape)
            return torch.zeros(INPUT_SIZE[0] // 4, images.shape[0], 66)

    model = RecordingModel()
    reader = CrnnReader("test", model)
    reader.read(np.zeros((80, 276, 3), np.uint8))
    assert model.seen_shape == (1, 1, INPUT_SIZE[1], INPUT_SIZE[0])

