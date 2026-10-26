from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..charset import CHARSET, NUM_CLASSES
from . import read_manifest

BLANK = NUM_CLASSES
INPUT_SIZE = (160, 40)  # width, height; tuned continuation input for small candidate text


class CrnnNet(nn.Module):
    """Compact CRNN: convolutional width sequence, bidirectional GRU, CTC logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.GRU(96, 96, num_layers=1, bidirectional=True, batch_first=False)
        self.classifier = nn.Linear(192, NUM_CLASSES + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # N,1,40,W → N,T,192 → T,N,66
        features = self.features(images).squeeze(2).transpose(1, 2)
        sequence, _ = self.sequence(features.transpose(0, 1))
        return self.classifier(sequence)


def _greedy_decode(logits: torch.Tensor) -> tuple[str, list[float]]:
    probs = logits.softmax(dim=-1)
    values, indices = probs.max(dim=-1)
    chars: list[str] = []
    confs: list[float] = []
    previous = BLANK
    for index, confidence in zip(indices.tolist(), values.tolist()):
        if index != BLANK and index != previous:
            chars.append(CHARSET[index])
            confs.append(float(confidence))
        previous = index
    return "".join(chars), confs


class CrnnReader:
    """CPU inference for a rectified whole plate; the model is trained from scratch."""

    name = "crnn"

    def __init__(self, version: str, model: CrnnNet) -> None:
        self.version = version
        self.model = model.cpu().eval()

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CrnnReader":
        entry = read_manifest(path.parent).get(path.name)
        if entry is not None and entry.get("input_size") != list(INPUT_SIZE):
            raise ValueError(f"CRNN manifest input_size must match {INPUT_SIZE}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or any(
                not isinstance(k, str) or not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                for k, v in state.items()):
            raise ValueError("CRNN model must be a finite tensor state_dict")
        model = CrnnNet()
        model.load_state_dict(state, strict=True)
        return cls(version, model)

    def read(self, plate_image: np.ndarray) -> tuple[str, list[float]]:
        """BGR plate at 252/276 × 80; reject malformed inputs with the empty result."""
        if (not isinstance(plate_image, np.ndarray) or plate_image.dtype != np.uint8
                or plate_image.ndim != 3 or plate_image.shape[2] != 3
                or plate_image.shape[0] != 80 or plate_image.shape[1] not in (252, 276)):
            return "", []
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255).sub_(0.5).div_(0.5)
        with torch.inference_mode():
            logits = self.model(tensor[None, None])[..., :NUM_CLASSES + 1]
        if not torch.isfinite(logits).all():
            return "", []
        return _greedy_decode(logits[:, 0])

