"""Direct detection-box recognition: RGB MobileNet features, height attention and CTC.

Only the training constructor may request generic torchvision ImageNet weights.
Loading a released reader never accesses the network or a pretrained-weight cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..charset import CHARSET_SHA, NUM_CLASSES
from .crnn import _greedy_decode

INPUT_SIZE = (160, 48)
BOX_PAD = (0.08, 0.12)
ARCH = "mobilenet_v3_small_stem_hattn_bigru96_ctc_v1"


class BoxNet(nn.Module):
    """Keep 20 width positions and learn which feature rows contain each character."""

    def __init__(self, *, imagenet: bool = False) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet else None
        # Keep the first three ImageNet residual blocks (width stride 8).
        # CPU probe rejected the full backbone before any release training.
        self.features = mobilenet_v3_small(weights=weights).features[:4]
        self.project = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.height_score = nn.Conv2d(96, 1, 1)
        self.sequence = nn.GRU(96, 96, bidirectional=True)
        self.classifier = nn.Linear(192, NUM_CLASSES + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.project(self.features(images))
        attention = self.height_score(features).softmax(dim=2)
        columns = (features * attention).sum(dim=2).permute(2, 0, 1)
        sequence, _ = self.sequence(columns)
        return self.classifier(sequence)


def box_tensor(image: np.ndarray) -> torch.Tensor:
    """The same BGR->RGB resize and ImageNet normalization in training and inference."""
    rgb = cv2.cvtColor(cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.)
    return tensor.sub_(torch.tensor((.485, .456, .406))[:, None, None]).div_(
        torch.tensor((.229, .224, .225))[:, None, None])


def checkpoint(model: BoxNet, *, temperature: float = 1.,
               box_pad: tuple[float, float] = BOX_PAD) -> dict:
    return {"architecture": ARCH, "charset_sha": CHARSET_SHA,
            "input_size": list(INPUT_SIZE), "box_pad": list(box_pad),
            "temperature": float(temperature),
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


class BoxPlateReader:
    """Thread-safe CPU BoxReader; expects an unrectified crop_box BGR uint8 result."""

    name = "box_reader"

    def __init__(self, version: str, model: BoxNet, *, temperature: float = 1.,
                 box_pad: tuple[float, float] = BOX_PAD) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if len(box_pad) != 2 or not all(np.isfinite(v) and 0 <= v <= 1 for v in box_pad):
            raise ValueError("box_pad must contain two finite ratios in [0, 1]")
        self.version = version
        self.model = model.cpu().eval()
        self.temperature = float(temperature)
        self.box_pad = tuple(float(v) for v in box_pad)

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "BoxPlateReader":
        from . import read_manifest

        entry = read_manifest(Path(path).parent).get(Path(path).name)
        if entry is not None and entry.get("input_size") != list(INPUT_SIZE):
            raise ValueError("box reader manifest input_size mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(saved, dict) or saved.get("architecture") != ARCH
                or saved.get("charset_sha") != CHARSET_SHA
                or saved.get("input_size") != list(INPUT_SIZE)):
            raise ValueError("unsupported box reader checkpoint")
        state = saved.get("state_dict")
        if not isinstance(state, dict) or not state or any(
                not isinstance(k, str) or not isinstance(v, torch.Tensor)
                or not torch.isfinite(v).all() for k, v in state.items()):
            raise ValueError("box reader state_dict must contain finite tensors")
        model = BoxNet()
        model.load_state_dict(state, strict=True)
        return cls(version, model, temperature=saved["temperature"], box_pad=saved["box_pad"])

    def read(self, plate_image: np.ndarray) -> tuple[str, list[float]]:
        if (not isinstance(plate_image, np.ndarray) or plate_image.dtype != np.uint8
                or plate_image.ndim != 3 or plate_image.shape[2] != 3
                or min(plate_image.shape[:2]) <= 0):
            return "", []
        tensor = box_tensor(plate_image)
        with torch.inference_mode():
            logits = self.model(tensor[None]) / self.temperature
            if not torch.isfinite(logits).all():
                return "", []
            return _greedy_decode(logits[:, 0])
