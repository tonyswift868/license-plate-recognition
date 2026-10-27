"""Learned affine STN and wider ImageNet features for unrectified plate boxes.

The STN receives only pixels at inference. Training may supervise its affine
parameters from the training annotation; there is no external corner input.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..charset import NUM_CLASSES

INPUT_SIZE_V2 = (192, 64)
ARCH_V2 = "mobilenet_v3_small_stem_stn_gn_bigru160_ctc_v2"


class AffineLocalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 6)))
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(384, 96),
                                       nn.ReLU(inplace=True), nn.Linear(96, 6))
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
        self.register_buffer("identity", torch.tensor([[1., 0., 0.], [0., 1., 0.]]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        thumbnail = F.interpolate(images, size=(32, 96), mode="bilinear", align_corners=False)
        # Bounded deltas retain a well-defined image grid while permitting 45deg
        # rotation, shear, scale and translation. Identity at initialization.
        delta = self.regressor(self.features(thumbnail)).reshape(-1, 2, 3)
        return self.identity[None] + delta.tanh() * .9


class BoxNetV2(nn.Module):
    architecture = ARCH_V2
    input_size = INPUT_SIZE_V2

    def __init__(self, *, imagenet: bool = False) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet else None
        self.stn = AffineLocalizer()
        self.features = mobilenet_v3_small(weights=weights).features[:4]
        # The deeper ImageNet backbone failed the measured CPU budget. Keep the
        # first three pretrained residual blocks and spend capacity on the wider
        # high-resolution recognition head and a trainable geometric front end.
        self.project = nn.Sequential(
            nn.Conv2d(24, 96, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(8, 96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 160, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(8, 160), nn.ReLU(inplace=True))
        self.height_score = nn.Conv2d(160, 1, 1)
        self.sequence = nn.GRU(160, 160, bidirectional=True)
        self.classifier = nn.Linear(320, NUM_CLASSES + 1)

    def train(self, mode: bool = True):
        super().train(mode)
        # Repeated draws are identical within one epoch. Frozen pretrained
        # backbone BN statistics plus per-image head GroupNorm make source
        # deduplication exactly preserve the expanded batch computation.
        for layer in self.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, images: torch.Tensor, *, return_theta: bool = False):
        theta = self.stn(images)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        aligned = F.grid_sample(images, grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
        features = self.project(self.features(aligned))
        attention = self.height_score(features).softmax(dim=2)
        columns = (features * attention).sum(dim=2).permute(2, 0, 1)
        sequence, _ = self.sequence(columns)
        logits = self.classifier(sequence)
        return (logits, theta) if return_theta else logits
