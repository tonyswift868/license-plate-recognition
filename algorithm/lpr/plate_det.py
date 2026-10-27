"""Own box/keypoint detector with a torchvision ImageNet classification backbone.

RGB ImageNet normalization is shared by training and inference. Detection boxes
are regressed independently of the auxiliary corners; bad corners never remove
an otherwise valid detection. Loading a deployment file never downloads weights.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .corner_net import letterbox_image
from .errors import ModelLoadError
from .types import PlateCandidate

ARCHITECTURE = "plate_det_v1_resnet18_stage3_fpn_box_keypoints"
DEFAULT_SIZE = 288
OUTPUT_STRIDE = 8
IMAGENET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def _spatial_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.Hardswish(),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.BatchNorm2d(channels), nn.Hardswish(),
    )


class PlateDetModel(nn.Module):
    """Stride-8 center, independent offset/log-size box and corner heads."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        backbone = resnet18(weights=IMAGENET_WEIGHTS if pretrained else None, progress=False)
        self.features = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                      backbone.maxpool, backbone.layer1,
                                      backbone.layer2, backbone.layer3)
        self.lateral = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in (128, 256, 256))
        self.neck = _spatial_block(64)
        self.center = nn.Conv2d(64, 1, 1)
        self.box = nn.Conv2d(64, 4, 1)
        self.corners = nn.Conv2d(64, 8, 1)
        nn.init.constant_(self.center.bias, -2.19)
        nn.init.zeros_(self.box.bias)
        nn.init.constant_(self.box.bias[2:], math.log(4.))
        with torch.no_grad():
            self.corners.bias.copy_(torch.tensor(
                [-.5, -.5, .5, -.5, .5, .5, -.5, .5]).atanh())
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 128
                or rgb.shape[-2] % 32 or rgb.shape[-1] % 32):
            raise ValueError("PlateDetModel expects RGB [B,3,H,W], H/W>=128 and divisible by32")
        x = (rgb - self.mean) / self.std
        levels = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (5, 6):
                levels.append(x)
        levels.append(F.max_pool2d(levels[-1], 2, stride=2))
        top = self.lateral[2](levels[2])
        mid = self.lateral[1](levels[1]) + F.interpolate(
            top, size=levels[1].shape[-2:], mode="nearest")
        fine = self.lateral[0](levels[0]) + F.interpolate(
            mid, size=levels[0].shape[-2:], mode="nearest")
        fused = self.neck(fine)
        return {"heatmap": self.center(fused), "box": self.box(fused),
                "corners": self.corners(fused)}


def _iou(a, b) -> float:
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    intersection = x*y
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union > 0 else 0.


def decode_predictions(outputs: dict[str, torch.Tensor], image_shape: tuple[int, ...],
                       transform, *, max_candidates: int, score_threshold: float = .25,
                       nms_iou: float = .5) -> list[PlateCandidate]:
    """Decode in production score order; box validity is independent of corners."""
    if max_candidates <= 0:
        return []
    heatmap, boxes, quads = (outputs[k] for k in ("heatmap", "box", "corners"))
    if (heatmap.ndim != 4 or heatmap.shape[:2] != (1, 1)
            or boxes.shape != (1, 4, *heatmap.shape[-2:])
            or quads.shape != (1, 8, *heatmap.shape[-2:])):
        raise ValueError("invalid plate detector output shapes")
    if not all(torch.isfinite(t).all() for t in (heatmap, boxes)):
        return []
    scores = heatmap.sigmoid()
    peaks = scores * (scores >= F.max_pool2d(scores, 3, stride=1, padding=1))
    values, indices = peaks.flatten().topk(min(40, peaks.numel()))
    values, indices = values.cpu().numpy(), indices.cpu().numpy()
    box_values, quad_values = boxes[0].cpu().numpy(), quads[0].cpu().numpy()
    grid_width = heatmap.shape[-1]
    height, width = image_shape[:2]
    found: list[PlateCandidate] = []
    for score, flat_index in zip(values, indices):
        if float(score) < score_threshold:
            continue
        y, x = divmod(int(flat_index), grid_width)
        raw = box_values[:, y, x]
        offset = 1. / (1. + np.exp(-np.clip(raw[:2], -30., 30.)))
        center = (np.asarray([x, y], np.float32) + offset) * OUTPUT_STRIDE
        wh = np.exp(np.clip(raw[2:], -6., 7.)) * OUTPUT_STRIDE
        edge = transform.to_source(np.stack((center-wh/2, center+wh/2)))
        x0 = max(0, min(width, math.floor(float(edge[0, 0]))))
        y0 = max(0, min(height, math.floor(float(edge[0, 1]))))
        x1 = max(0, min(width, math.ceil(float(edge[1, 0]))))
        y1 = max(0, min(height, math.ceil(float(edge[1, 1]))))
        bbox = (x0, y0, x1-x0, y1-y0)
        if min(bbox[2:]) < 2 or any(_iou(bbox, c.bbox) >= nms_iou for c in found):
            continue
        relative = np.tanh(quad_values[:, y, x]).reshape(4, 2)
        points = transform.to_source(center + relative * wh)
        points[:, 0] = np.clip(points[:, 0], 0, width-1)
        points[:, 1] = np.clip(points[:, 1], 0, height-1)
        if (not np.isfinite(points).all()
                or not cv2.isContourConvex(points.astype(np.float32))
                or cv2.contourArea(points.astype(np.float32), oriented=True) < 1.):
            points = np.asarray([(x0, y0), (x1-1, y0),
                                 (x1-1, y1-1), (x0, y1-1)], np.float32)
        found.append(PlateCandidate([tuple(map(float, p)) for p in points], bbox,
                                    "unknown", float(score), "net"))
        if len(found) >= min(max_candidates, 3):
            break
    return found


class PlateDetector:
    name = "plate_det"

    def __init__(self, model: PlateDetModel, *, version: str = "plate_det_v1",
                 image_size: int = DEFAULT_SIZE, score_threshold: float = .25) -> None:
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 128 <= image_size <= 1024 or image_size % 32:
            raise ValueError("detector image_size must be a multiple of32 in [128,1024]")
        if (isinstance(score_threshold, bool) or not isinstance(score_threshold, (float, int))
                or not math.isfinite(score_threshold) or not 0 < score_threshold < 1):
            raise ValueError("detector threshold must be finite and in (0,1)")
        self.version, self.image_size = version, image_size
        self.score_threshold = float(score_threshold)
        self.model = model.cpu().eval()
        # Build CPU convolution primitives during loading, before frame latency
        # is measured; this also rejects finite but numerically invalid buffers.
        with torch.inference_mode():
            warm = self.model(torch.zeros((1, 3, image_size, image_size)))
        if not all(torch.isfinite(value).all() for value in warm.values()):
            raise ValueError("detector produces nonfinite outputs during CPU initialization")

    @classmethod
    def from_checkpoint(cls, payload: dict, *, version: str,
                        score_threshold: float | None = None) -> "PlateDetector":
        if payload.get("architecture") != ARCHITECTURE:
            raise ModelLoadError("plate detector architecture mismatch")
        size = payload.get("input_size")
        if not isinstance(size, (tuple, list)) or len(size) != 2 or size[0] != size[1]:
            raise ModelLoadError("plate detector must have a square input_size")
        try:
            model = PlateDetModel(pretrained=False)
            model.load_state_dict(payload["state_dict"], strict=True)
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError("nonfinite detector weights")
            return cls(model, version=version, image_size=size[0],
                       score_threshold=(payload.get("score_threshold", .25)
                                        if score_threshold is None else score_threshold))
        except Exception as exc:
            raise ModelLoadError(f"plate detector checkpoint invalid: {exc}") from exc

    @classmethod
    def from_file(cls, path: Path, *, version: str = "plate_det_v1",
                  score_threshold: float | None = None) -> "PlateDetector":
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint must be a dictionary")
            return cls.from_checkpoint(payload, version=version, score_threshold=score_threshold)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"plate detector load failed: {path}: {exc}") from exc

    def detect(self, image: np.ndarray, max_candidates: int) -> list[PlateCandidate]:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
                or image.ndim != 3 or image.shape[2] != 3
                or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
                or max_candidates <= 0):
            return []
        canvas, transform = letterbox_image(image, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).float().div_(255.)
        with torch.inference_mode():
            outputs = self.model(tensor)
            return decode_predictions(outputs, image.shape, transform,
                                      max_candidates=max_candidates,
                                      score_threshold=self.score_threshold)


def load_plate_detector(model_path: Path, *, score_threshold: float | None = None) -> PlateDetector:
    """Manifest-verified factory; an optional threshold never rewrites weights.

    Without an override the checkpoint's threshold (v1:0.25) is retained.
    The signature also supports the integration tool's factory(Path) contract.
    """
    from .corner_net import CornerNet
    from .recognize import verify_model

    path = Path(model_path)
    entry = verify_model(path)
    detector = CornerNet.from_file(path, version=entry["version"],
                                   score_threshold_override=score_threshold)
    if not isinstance(detector, PlateDetector):
        raise ModelLoadError("expected a plate detector checkpoint")
    return detector
