"""CharCNN 网络结构，训练与推理共用（员工 A，任务 A2；结构见开发文档 8.1）。"""

from __future__ import annotations

import torch
from torch import nn

from ..charset import NUM_CLASSES


class CharCNN(nn.Module):
    """输入 N×1×32×32 float32（像素 / 255），输出 N×65 logits。参数量约 68 万。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(1, 32), block(32, 32), nn.MaxPool2d(2),
            block(32, 64), block(64, 64), nn.MaxPool2d(2),
            block(64, 128), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(2048, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
