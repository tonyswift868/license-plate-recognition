"""从零训练的轻量车牌中心热图与四角点回归网络（K1 Plan B-1）。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .errors import ModelLoadError
from .types import PlateCandidate

INPUT_SIZE = (512, 320)  # width, height；训练、验证和推理共用。
OUTPUT_STRIDE = 4
MAX_DETECTIONS = 2
SCORE_THRESHOLD = 0.25
ARCHITECTURE = "corner_net_v2_center_relative_pyramid_rf111"
ROI_ARCHITECTURE = "roi_refiner_with_binary_plate_confidence_v0"
ROI_INPUT_SIZE = (256, 96)  # width, height
COMBINED_ARCHITECTURE = f"{ARCHITECTURE}+{ROI_ARCHITECTURE}"


def _group_count(channels: int) -> int:
    return next(groups for groups in range(min(8, channels), 0, -1) if channels % groups == 0)


def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


def _dilated_context(channels: int, dilation: int) -> nn.Sequential:
    """深度可分离的空洞卷积，扩上下文而不显著增加CPU FLOPs。"""
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                  groups=channels, bias=False),
        nn.GroupNorm(_group_count(channels), channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.GroupNorm(_group_count(channels), channels),
        nn.SiLU(inplace=True),
    )


class CornerModel(nn.Module):
    """stride-4 FCN：中心热图 logits 与该中心格对应的归一化全图四角。

    输入为 BGR float32 `[B, 3, 320, 512]`，范围 `[0, 1]`。输出
    `(heatmap_logits, corners)`，形状分别为 `[B,1,80,128]`、`[B,8,80,128]`。
    输出角点按 LT, RT, RB, LB 排序。角点头先预测以格点中心为原点、范围
    `[-.5,.5]` 的归一化偏移，forward再加上格点在完整画布中的归一化中心，
    因而返回的仍是全图绝对归一化坐标。训练时只在中心正样本格点上回归，
    中心峰附近的3×3有效格共同接受角点监督，目标编码为 `GT_corner - cell_xy + .5`。
    网络不加载任何预训练参数。
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_block(3, 12, 2),
            _conv_block(12, 16, 2),
            _conv_block(16, 24, 2),
        )
        self.shallow_projection = nn.Conv2d(16, 16, 1, bias=False)
        self.register_buffer("coordinate_grid", self._make_coordinate_grid(), persistent=False)
        # Context runs on stride 8 (40x64), while stride-4 shallow features
        # preserve localization. The two dilated depthwise blocks give roughly
        # 111 input-pixel effective receptive field with <100M MAC/image.
        self.context = nn.Sequential(
            _dilated_context(24, dilation=2),
            _dilated_context(24, dilation=4),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(42, 24, 1, bias=False),
            nn.GroupNorm(_group_count(24), 24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1, groups=24, bias=False),
            nn.GroupNorm(_group_count(24), 24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 24, 1, bias=False),
            nn.GroupNorm(_group_count(24), 24),
            nn.SiLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(24, 1, 1)
        self.corner_head = nn.Conv2d(24, 8, 1)

    @staticmethod
    def _make_coordinate_grid() -> torch.Tensor:
        height, width = INPUT_SIZE[1]//OUTPUT_STRIDE, INPUT_SIZE[0]//OUTPUT_STRIDE
        y = (torch.arange(height, dtype=torch.float32)+.5) / height
        x = (torch.arange(width, dtype=torch.float32)+.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("CornerModel expects [B,3,H,W] BGR tensors")
        if image.shape[-2:] != (INPUT_SIZE[1], INPUT_SIZE[0]):
            raise ValueError(f"CornerModel expects HxW={INPUT_SIZE[1]}x{INPUT_SIZE[0]}")
        shallow = self.stem[:2](image)
        context = self.context(self.stem[2](shallow))
        features = torch.nn.functional.interpolate(context, size=shallow.shape[-2:],
                                                   mode="bilinear", align_corners=False)
        coordinates = self.coordinate_grid.to(dtype=features.dtype).expand(image.shape[0], -1, -1, -1)
        features = self.fusion(torch.cat((features, self.shallow_projection(shallow), coordinates), dim=1))
        heatmap = self.heatmap_head(features)
        relative_offsets = torch.sigmoid(self.corner_head(features))-.5
        x_centers = coordinates[:, 0:1]
        y_centers = coordinates[:, 1:2]
        centers = torch.stack((x_centers, y_centers)*4, dim=2).reshape(image.shape[0], 8,
                                                                         features.shape[2], features.shape[3])
        corners = centers + relative_offsets
        return heatmap, corners

    def encode_relative_targets(self, absolute_corners: torch.Tensor,
                                center_y: torch.Tensor,
                                center_x: torch.Tensor) -> torch.Tensor:
        """把绝对归一化角点转换为峰值格中心相对目标，范围约 `[0,1]`。

        训练时对 `forward` 返回的绝对预测及GT都调用同一编码，再计算回归
        loss；这与绝对坐标误差等价，避免训练目标与检测输出混用两种原点。
        """
        if absolute_corners.ndim != 2 or absolute_corners.shape[1] != 8:
            raise ValueError("absolute_corners must have shape [B,8]")
        batch = absolute_corners.shape[0]
        if center_x.numel() != batch or center_y.numel() != batch:
            raise ValueError("center cell index count must match batch size")
        center_y = center_y.to(device=absolute_corners.device, dtype=torch.long).reshape(-1)
        center_x = center_x.to(device=absolute_corners.device, dtype=torch.long).reshape(-1)
        height, width = self.coordinate_grid.shape[-2:]
        if (torch.any(center_x < 0) or torch.any(center_x >= width)
                or torch.any(center_y < 0) or torch.any(center_y >= height)):
            raise ValueError("center cell index is out of bounds")
        grid = self.coordinate_grid.to(device=absolute_corners.device,
                                       dtype=absolute_corners.dtype)
        batch_index = torch.arange(batch, device=absolute_corners.device)
        center_xy = grid.expand(batch, -1, -1, -1)[batch_index, :, center_y, center_x]
        center_corners = torch.cat((center_xy[:, 0:1], center_xy[:, 1:2])*4, dim=1)
        return absolute_corners-center_corners+.5


# Independent keypoint heatmaps; original implementation inspired by dense pose supervision.
class _KeypointStem(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )


class _KeypointDSBlock(nn.Module):
    """Depthwise spatial conv followed by pointwise channel mixing."""
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.GroupNorm(_group_count(in_ch), in_ch), nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class KeypointModel(nn.Module):
    """Multi-scale one-stage keypoint model; returns [B,5,80,128] logits."""
    architecture = "keypoint_v3_center_four_independent_heatmaps_pyramid"

    def __init__(self) -> None:
        super().__init__()
        self.stem = _KeypointStem(3, 32, 2)            # stride 2
        self.s4 = _KeypointDSBlock(32, 48, 2)                  # stride 4, precise edges
        self.s8 = _KeypointDSBlock(48, 64, 2)
        self.s16 = _KeypointDSBlock(64, 96, 2)
        self.s32 = _KeypointDSBlock(96, 96, 2)
        self.lateral = nn.ModuleList(nn.Conv2d(ch, 32, 1, bias=False)
                                     for ch in (48, 64, 96, 96))
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, 1, bias=False),
            nn.GroupNorm(8, 48), nn.SiLU(inplace=True),
        )
        self.heatmaps = nn.Conv2d(48, 5, 1)
        # A mild prior improves early sparse-positive stability.
        nn.init.constant_(self.heatmaps.bias, -4.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, INPUT_SIZE[1], INPUT_SIZE[0]):
            raise ValueError("expected BGR float tensor [B,3,320,512]")
        x2 = self.stem(image)
        x4 = self.s4(x2)
        x8 = self.s8(x4)
        x16 = self.s16(x8)
        x32 = self.s32(x16)
        levels = (x4, x8, x16, x32)
        target_size = x4.shape[-2:]
        # Each deeper context level contributes to the stride-4 representation.
        fused = sum(torch.nn.functional.interpolate(proj(feature), size=target_size, mode="bilinear",
                                  align_corners=False)
                    for proj, feature in zip(self.lateral, levels))
        return self.heatmaps(self.fuse(fused))


def keypoint_coordinates(logits: torch.Tensor, *, temperature: float = .05) -> torch.Tensor:
    """Spatial expectation for each channel, output normalized canvas xy [B,C,2]."""
    if logits.ndim != 4 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("logits must be [B,C,H,W] and temperature positive")
    b, c, h, w = logits.shape
    xs = (torch.arange(w, device=logits.device, dtype=logits.dtype) + .5) / w
    ys = (torch.arange(h, device=logits.device, dtype=logits.dtype) + .5) / h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    probs = torch.softmax(logits.flatten(2) / temperature, dim=-1).reshape(b, c, h, w)
    return torch.stack(((probs * xx).sum(dim=(-2, -1)),
                        (probs * yy).sum(dim=(-2, -1))), dim=-1)



@dataclass(frozen=True)
class LetterboxTransform:
    """Rounding-aware map from source image pixels to a letterbox canvas."""

    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int

    def to_source(self, points: np.ndarray) -> np.ndarray:
        mapped = np.asarray(points, dtype=np.float32).copy()
        mapped[..., 0] = (mapped[..., 0] - self.pad_left) / self.scale_x
        mapped[..., 1] = (mapped[..., 1] - self.pad_top) / self.scale_y
        return mapped


def letterbox_image(image: np.ndarray,
                    size: tuple[int, int] = INPUT_SIZE
                    ) -> tuple[np.ndarray, LetterboxTransform]:
    """等比例补边缩放 uint8 BGR 图；返回 uint8 canvas 和精确反变换信息。"""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("letterbox_image expects a nonempty uint8 BGR image")
    target_width, target_height = size
    if target_width < 1 or target_height < 1:
        raise ValueError("letterbox size must be positive")
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[pad_top:pad_top+resized_height, pad_left:pad_left+resized_width] = resized
    transform = LetterboxTransform(
        resized_width / width, resized_height / height, pad_left, pad_top,
        width, height, target_width, target_height,
    )
    return canvas, transform


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax+aw, bx+bw)-max(ax, bx))
    ih = max(0, min(ay+ah, by+bh)-max(ay, by))
    intersection = iw*ih
    union = aw*ah+bw*bh-intersection
    return intersection/union if union > 0 else 0.


def _candidate_from_peak(image: np.ndarray, score: float, normalized_corners: np.ndarray,
                         transform: LetterboxTransform) -> PlateCandidate | None:
    height, width = image.shape[:2]
    points = np.asarray(normalized_corners, dtype=np.float32).reshape(4, 2).copy()
    points[:, 0] *= transform.target_width
    points[:, 1] *= transform.target_height
    points = transform.to_source(points)
    points[:, 0] = np.clip(points[:, 0], 0, width-1)
    points[:, 1] = np.clip(points[:, 1], 0, height-1)
    return _candidate_from_source_points(image, score, points)


def _candidate_from_source_points(image: np.ndarray, score: float,
                                  points: np.ndarray) -> PlateCandidate | None:
    height, width = image.shape[:2]
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or not cv2.isContourConvex(points):
        return None
    area = abs(float(cv2.contourArea(points)))
    if not 0.00002*width*height <= area <= .5*width*height:
        return None
    side_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(side_lengths < 2.):
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    bbox = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
            min(width, int(np.ceil(x1))+1)-max(0, int(np.floor(x0))),
            min(height, int(np.ceil(y1))+1)-max(0, int(np.floor(y0))))
    if min(bbox[2:]) <= 0:
        return None
    return PlateCandidate(
        corners=[(float(x), float(y)) for x, y in points], bbox=bbox,
        color="unknown", score=float(score), source="net",
    )


def _conv_bn_act(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.SiLU(inplace=True))


class RoiCornerRefiner(nn.Module):
    """Spatially pooled ROI regressor trained from random initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_act(3, 16, 2), _conv_bn_act(16, 24, 2), _conv_bn_act(24, 32, 2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, 1, bias=False), nn.GroupNorm(8, 48), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 10)),
        )
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(48 * 4 * 10, 96), nn.SiLU(),
                                       nn.Linear(96, 8), nn.Sigmoid())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, ROI_INPUT_SIZE[1], ROI_INPUT_SIZE[0]):
            raise ValueError(f"RoiCornerRefiner expects [B,3,{ROI_INPUT_SIZE[1]},{ROI_INPUT_SIZE[0]}]")
        return self.regressor(self.features(images))


class RoiConfidenceModel(nn.Module):
    """Refinement and confidence heads; state keys match the out-only pilot."""

    def __init__(self) -> None:
        super().__init__()
        self.refiner = RoiCornerRefiner()
        self.confidence = nn.Sequential(
            nn.Flatten(), nn.Linear(48 * 4 * 10, 96), nn.SiLU(), nn.Linear(96, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.refiner.features(images)
        return self.refiner.regressor(features), self.confidence(features).squeeze(1)


def _valid_fraction(value: object, name: str, *, allow_zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (float(value) < 0. if allow_zero else float(value) <= 0.)
            or float(value) >= 1.):
        raise ValueError(f"{name} must be a finite number in {'[0,1)' if allow_zero else '(0,1)'}")
    return float(value)


def _roi_bounds(image_shape: tuple[int, ...], bbox: tuple[int, int, int, int],
                padding: float) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x, y, box_w, box_h = bbox
    return (max(0, int(math.floor(x - padding * box_w))),
            max(0, int(math.floor(y - padding * box_h))),
            min(width, int(math.ceil(x + box_w + padding * box_w))),
            min(height, int(math.ceil(y + box_h + padding * box_h))))


def _refine_candidate(image: np.ndarray, coarse: PlateCandidate, roi_model: RoiConfidenceModel,
                      padding: float) -> tuple[PlateCandidate | None, float]:
    x0, y0, x1, y1 = _roi_bounds(image.shape, coarse.bbox, padding)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, 0.
    roi, transform = letterbox_image(image[y0:y1, x0:x1], ROI_INPUT_SIZE)
    tensor = torch.from_numpy(roi.transpose(2, 0, 1).copy()).unsqueeze(0).float().div_(255.)
    normalized, logits = roi_model(tensor)
    confidence = float(torch.sigmoid(logits[0]).item())
    local = normalized[0].detach().cpu().numpy().reshape(4, 2) * np.asarray(ROI_INPUT_SIZE, np.float32)
    points = transform.to_source(local) + np.asarray([x0, y0], dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
    return _candidate_from_source_points(image, confidence, points), confidence


def pack_corner_roi_checkpoint(coarse_path: Path, roi_path: Path, output_path: Path, *,
                              proposal_threshold: float = .05,
                              final_confidence_threshold: float = .8,
                              roi_padding_per_side: float = .25) -> Path:
    """Pack compatible weights-only pilot states into a single deployable file."""
    proposal_threshold = _valid_fraction(proposal_threshold, "proposal_threshold")
    final_confidence_threshold = _valid_fraction(final_confidence_threshold,
                                                  "final_confidence_threshold")
    if (isinstance(roi_padding_per_side, bool) or not isinstance(roi_padding_per_side, (int, float))
            or not math.isfinite(float(roi_padding_per_side))
            or not 0. <= float(roi_padding_per_side) <= 1.):
        raise ValueError("roi_padding_per_side must be finite and between 0 and 1")
    coarse = torch.load(Path(coarse_path), map_location="cpu", weights_only=True)
    coarse_state = coarse.get("state_dict", coarse) if isinstance(coarse, dict) else coarse
    if isinstance(coarse, dict) and "state_dict" in coarse:
        if coarse.get("architecture") != ARCHITECTURE or tuple(coarse.get("input_size", ())) != INPUT_SIZE:
            raise ModelLoadError("coarse checkpoint architecture or input_size mismatch")
    roi_state = torch.load(Path(roi_path), map_location="cpu", weights_only=True)
    if not isinstance(coarse_state, dict) or not isinstance(roi_state, dict):
        raise ModelLoadError("checkpoint inputs must contain state_dicts")
    # Validate exact tensor names/shapes before writing the package.
    CornerModel().load_state_dict(coarse_state, strict=True)
    RoiConfidenceModel().load_state_dict(roi_state, strict=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": COMBINED_ARCHITECTURE, "input_size": list(INPUT_SIZE),
                "roi_input_size": list(ROI_INPUT_SIZE), "coarse_state_dict": coarse_state,
                "roi_state_dict": roi_state, "proposal_threshold": proposal_threshold,
                "final_confidence_threshold": final_confidence_threshold,
                "roi_padding_per_side": float(roi_padding_per_side)}, output_path)
    return output_path


def _decode_candidates(image: np.ndarray, heatmap_logits: np.ndarray,
                       corner_map: np.ndarray, transform: LetterboxTransform,
                       max_candidates: int, score_threshold: float = SCORE_THRESHOLD
                       ) -> list[PlateCandidate]:
    """解码置信峰并回到原图坐标；同一网格热图只保留局部峰。"""
    if max_candidates <= 0:
        return []
    heatmap = np.asarray(heatmap_logits, dtype=np.float32).squeeze()
    corners = np.asarray(corner_map, dtype=np.float32)
    if heatmap.ndim != 2 or corners.shape != (8, *heatmap.shape):
        return []
    scores = 1. / (1. + np.exp(-np.clip(heatmap, -40, 40)))
    local_max = cv2.dilate(scores, np.ones((3, 3), np.uint8))
    ys, xs = np.where((scores >= score_threshold) & (scores >= local_max-1e-7))
    order = sorted(zip(ys.tolist(), xs.tolist()), key=lambda pos: (-float(scores[pos]), pos[0], pos[1]))
    found: list[PlateCandidate] = []
    for y, x in order:
        candidate = _candidate_from_peak(image, float(scores[y, x]), corners[:, y, x], transform)
        if candidate is None:
            continue
        if any(_bbox_iou(candidate.bbox, old.bbox) >= .5 for old in found):
            continue
        found.append(candidate)
        if len(found) >= min(max_candidates, MAX_DETECTIONS):
            break
    return found


class CornerNet:
    """实现 `CornerDetector`：letterbox 推理，返回原图像素坐标候选。"""

    name = "corner_net"

    def __init__(self, version: str, model: CornerModel,
                 score_threshold: float = SCORE_THRESHOLD,
                 roi_model: RoiConfidenceModel | None = None,
                 proposal_threshold: float = SCORE_THRESHOLD,
                 roi_padding_per_side: float = .25) -> None:
        if (isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float))
                or not math.isfinite(float(score_threshold))
                or not 0. < float(score_threshold) < 1.):
            raise ValueError("score_threshold must be a finite number strictly between 0 and 1")
        self.version = version
        self.model = model.to("cpu").eval()
        self.score_threshold = float(score_threshold)
        self.roi_model = roi_model.to("cpu").eval() if roi_model is not None else None
        self.proposal_threshold = _valid_fraction(proposal_threshold, "proposal_threshold")
        if (isinstance(roi_padding_per_side, bool) or not isinstance(roi_padding_per_side, (int, float))
                or not math.isfinite(float(roi_padding_per_side))
                or not 0. <= float(roi_padding_per_side) <= 1.):
            raise ValueError("roi_padding_per_side must be finite and between 0 and 1")
        self.roi_padding_per_side = float(roi_padding_per_side)

    @classmethod
    def from_file(cls, path: Path, *, version: str) -> "CornerNet":
        """只加载安全 state_dict；兼容纯state_dict和本项目版本化checkpoint。"""
        try:
            checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
            score_threshold = SCORE_THRESHOLD
            if isinstance(checkpoint, dict) and checkpoint.get("architecture") == KeypointModel.architecture:
                if tuple(checkpoint.get("input_size", ())) != INPUT_SIZE:
                    raise ModelLoadError("keypoint checkpoint input_size mismatch")
                model = KeypointModel()
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                return KeypointNet(version, model,
                                   score_threshold=checkpoint.get("score_threshold", .05))
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                if checkpoint.get("architecture") != ARCHITECTURE:
                    raise ModelLoadError("corner checkpoint architecture mismatch")
                if tuple(checkpoint.get("input_size", ())) != INPUT_SIZE:
                    raise ModelLoadError("corner checkpoint input_size mismatch")
                score_threshold = checkpoint.get("score_threshold", SCORE_THRESHOLD)
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and checkpoint.get("architecture") == COMBINED_ARCHITECTURE:
                if tuple(checkpoint.get("input_size", ())) != INPUT_SIZE:
                    raise ModelLoadError("combined corner checkpoint input_size mismatch")
                if tuple(checkpoint.get("roi_input_size", ())) != ROI_INPUT_SIZE:
                    raise ModelLoadError("combined corner checkpoint roi_input_size mismatch")
                model, roi_model = CornerModel(), RoiConfidenceModel()
                model.load_state_dict(checkpoint["coarse_state_dict"], strict=True)
                roi_model.load_state_dict(checkpoint["roi_state_dict"], strict=True)
                try:
                    return cls(version, model,
                               score_threshold=checkpoint.get("final_confidence_threshold", .8),
                               roi_model=roi_model,
                               proposal_threshold=checkpoint.get("proposal_threshold", .05),
                               roi_padding_per_side=checkpoint.get("roi_padding_per_side", .25))
                except ValueError as exc:
                    raise ModelLoadError(f"invalid combined corner checkpoint metadata: {exc}") from exc
            elif isinstance(checkpoint, dict) and {"coarse_state_dict", "roi_state_dict"} <= checkpoint.keys():
                raise ModelLoadError("combined corner checkpoint architecture mismatch")
            else:
                state_dict = checkpoint
            if not isinstance(state_dict, dict) or not state_dict:
                raise ModelLoadError("corner checkpoint has no state_dict")
            model = CornerModel()
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            try:
                return cls(version, model, score_threshold=score_threshold)
            except ValueError as exc:
                raise ModelLoadError(f"invalid corner checkpoint score_threshold: {score_threshold!r}") from exc
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"四角点网络加载失败：{path}：{exc}") from exc

    def detect(self, image: np.ndarray, max_candidates: int) -> list[PlateCandidate]:
        """对uint8 BGR输入推理；候选按置信度降序并使用原图坐标。"""
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
                or image.ndim != 3 or image.shape[2] != 3
                or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)):
            return []
        if max_candidates <= 0:
            return []
        canvas, transform = letterbox_image(image)
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).unsqueeze(0).float().div_(255.)
        with torch.inference_mode():
            heatmap, corners = self.model(tensor)
            coarse_candidates = _decode_candidates(
                image, heatmap[0, 0].cpu().numpy(), corners[0].cpu().numpy(),
                transform, min(max_candidates, MAX_DETECTIONS),
                score_threshold=(self.proposal_threshold if self.roi_model is not None else self.score_threshold),
            )
            if self.roi_model is None:
                return coarse_candidates
            refined: list[PlateCandidate] = []
            for candidate in coarse_candidates:
                result, confidence = _refine_candidate(
                    image, candidate, self.roi_model, self.roi_padding_per_side)
                if result is not None and confidence >= self.score_threshold:
                    refined.append(result)
        refined.sort(key=lambda item: -item.score)
        selected: list[PlateCandidate] = []
        for candidate in refined:
            if any(_bbox_iou(candidate.bbox, old.bbox) >= .5 for old in selected):
                continue
            selected.append(candidate)
            if len(selected) >= min(max_candidates, MAX_DETECTIONS):
                break
        return selected


class KeypointNet(CornerNet):
    """Single-plate center/four-corner heatmaps, with the same detector protocol.

    Independent corner association supports the single-plate CCPD task only.
    No shared mutable inference state or input enhancement is introduced.
    """

    def detect(self, image: np.ndarray, max_candidates: int) -> list[PlateCandidate]:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
                or image.ndim != 3 or image.shape[2] != 3
                or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
                or max_candidates <= 0):
            return []
        canvas, transform = letterbox_image(image)
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).unsqueeze(0).float().div_(255.)
        with torch.inference_mode():
            logits = self.model(tensor)
            if not torch.isfinite(logits).all():
                return []
            scores = torch.sigmoid(logits[0]).flatten(1).max(dim=1).values
            center_score = float(scores[0])
            if center_score < self.score_threshold:
                return []
            normalized = keypoint_coordinates(logits[:, 1:5])[0].cpu().numpy()
            score = center_score * float(scores[1:5].prod().pow(.25))
        # Coordinates are in the full letterbox canvas, never in a predicted bbox.
        candidate = _candidate_from_peak(image, score, normalized, transform)
        return [candidate] if candidate is not None else []
