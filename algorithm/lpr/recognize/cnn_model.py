"""CharCNN 网络结构，训练与推理共用（员工 A，任务 A2；结构见开发文档 8.1）。"""

from __future__ import annotations

import torch
from torch import nn

from ..charset import NUM_CLASSES


class CharCNN(nn.Module):
    """输入 N×1×32×32 float32（像素 / 255），输出 N×65 logits。参数量约 68 万。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        raise NotImplementedError("CharCNN 尚未实现（员工 A，任务 A2）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CharCNN 尚未实现（员工 A，任务 A2）")
