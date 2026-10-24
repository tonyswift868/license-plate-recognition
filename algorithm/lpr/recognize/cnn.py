"""CNN 推理封装（员工 A，任务 A2）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..charset import NUM_CLASSES
from .cnn_model import CharCNN


class CnnClassifier:
    """实现 recognize.base.CharClassifier。"""

    name = "cnn"

    def __init__(self, version: str, model: CharCNN) -> None:
        self.version = version
        self.model = model.cpu().eval()

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CnnClassifier":
        """加载 state_dict（torch.load(..., weights_only=True)），模型置为 eval 模式。"""
        model = CharCNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(version, model)

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        """N×32×32 uint8 → N×65 float32（softmax），推理包在 torch.inference_mode() 中。"""
        if chars.ndim != 3 or chars.shape[1:] != (32, 32) or chars.dtype != np.uint8:
            raise ValueError("chars must be N x 32 x 32 uint8")
        if len(chars) == 0:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        with torch.inference_mode():
            inputs = torch.from_numpy(np.ascontiguousarray(chars)).unsqueeze(1).float() / 255.0
            return self.model(inputs).softmax(dim=1).numpy()
