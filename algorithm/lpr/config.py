"""配置加载：TOML → Pydantic v2 模型（见开发文档 5.7）。

加载顺序：config/default.toml 打底 → 指定的配置文件（只需写要改的键）→ overrides。
所有默认值只在 default.toml 里写一份，这里的模型不设默认值。
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .charset import PROVINCES
from .errors import ConfigError
from .types import Box

ALGORITHM_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ALGORITHM_DIR / "config" / "default.toml"
ENV_CONFIG = "LPR_CONFIG"

HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(_Section):
    torch_threads: int
    model_path: str
    classifier: Literal["cnn", "template", "svm"]
    max_candidates: int
    corner_model_path: str                               # 空字符串 = 不启用四角点网络
    plate_reader_model_path: str                         # 空字符串 = 不启用整牌识别


class PreprocessConfig(_Section):
    min_side: int
    work_width: int
    clahe_clip: float
    clahe_grid: int
    blur_ksize: int


class LocateConfig(_Section):
    aspect_min: float
    aspect_max: float
    area_ratio_min: float
    area_ratio_max: float
    max_tilt_deg: float
    close_kernel_rel: tuple[float, float]
    min_row_transitions: int
    nms_iou: float
    blue_hsv: HsvRange
    yellow_hsv: HsvRange
    green_hsv: HsvRange


class RectifyConfig(_Section):
    size_7: tuple[int, int]
    size_8: tuple[int, int]
    expand_ratio: float
    deskew_max_deg: float

    @field_validator("size_7")
    @classmethod
    def _frozen_size_7(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v != (252, 80):
            raise ValueError("rectify.size_7 冻结为 [252, 80]")
        return v

    @field_validator("size_8")
    @classmethod
    def _frozen_size_8(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v != (276, 80):
            raise ValueError("rectify.size_8 冻结为 [276, 80]")
        return v


class SegmentConfig(_Section):
    binarize: Literal["otsu", "adaptive"]
    layout7_centers: list[float]
    layout7_char_w: float
    layout8_centers: list[float]
    layout8_char_w: float
    char_size: int

    @field_validator("layout7_centers")
    @classmethod
    def _seven(cls, v: list[float]) -> list[float]:
        if len(v) != 7:
            raise ValueError("segment.layout7_centers 必须有 7 个值")
        return v

    @field_validator("layout8_centers")
    @classmethod
    def _eight(cls, v: list[float]) -> list[float]:
        if len(v) not in (0, 8):
            raise ValueError("segment.layout8_centers 必须为空或有 8 个值")
        return v

    @field_validator("char_size")
    @classmethod
    def _frozen_char_size(cls, v: int) -> int:
        if v != 32:
            raise ValueError("segment.char_size 冻结为 32")
        return v


class PostprocessConfig(_Section):
    min_confidence: float
    topk: int
    province_prior: dict[str, float]
    arbitration: Literal["first_ok", "agree"]            # agree：分割路线与整牌识别读法一致才自动通过（须启用 CRNN）
    agree_min_confidence: float                          # agree：两边置信度的较小值 ≥ 此值才算一致通过
    solo_min_confidence_segment: float                   # agree：读法不一致或缺一边时，单边自动通过门槛；>1 = 关闭
    solo_min_confidence_reader: float
    disagree_prefer: Literal["segment", "reader"]        # agree：不一致且都未单边通过时返回哪一边的读法
    nonok_fallback: Literal["rank", "agreement_then_reader"]  # agree：没有任何结果自动通过时返回哪个读法

    @field_validator("province_prior")
    @classmethod
    def _known_provinces(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = [k for k in v if k not in PROVINCES]
        if unknown:
            raise ValueError(f"postprocess.province_prior 含未知省份字：{unknown}")
        return v


class VotingConfig(_Section):
    min_frames: int
    agree_frames: int
    window_s: float


class DedupConfig(_Section):
    window_s: float
    dhash_max_hamming: int


class LprConfig(_Section):
    runtime: RuntimeConfig
    preprocess: PreprocessConfig
    locate: LocateConfig
    rectify: RectifyConfig
    segment: SegmentConfig
    postprocess: PostprocessConfig
    voting: VotingConfig
    dedup: DedupConfig
    lanes: dict[str, tuple[int, int, int, int]]

    def lane_roi(self, lane_id: int | None) -> Box | None:
        """车道 ROI（原图坐标 x, y, w, h）；没配置该车道时返回 None，表示用全图。"""
        if lane_id is None:
            return None
        return self.lanes.get(str(lane_id))

    def resolve_path(self, p: str) -> Path:
        """配置中的相对路径一律相对 algorithm/ 目录。"""
        path = Path(p)
        return path if path.is_absolute() else ALGORITHM_DIR / path


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"配置文件不存在：{path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"配置文件格式错误：{path}：{e}") from e


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("lanes", "province_prior"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """overrides 两种写法：section__key=值（如 runtime__classifier="svm"），或 section={...}。"""
    out = dict(raw)
    for key, value in overrides.items():
        if "__" in key:
            section, field = key.split("__", 1)
            if not isinstance(out.get(section), dict):
                raise ConfigError(f"未知配置段：{section}")
            out[section] = {**out[section], field: value}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            raise ConfigError(f"无法识别的覆盖项：{key}（应写成 section__key=值）")
    return out


def load_config(path: str | os.PathLike[str] | None = None, **overrides: Any) -> LprConfig:
    """加载配置。path 为空时先读环境变量 LPR_CONFIG，再缺省为 config/default.toml。

    键名错误、取值非法、文件缺失时抛 ConfigError。
    """
    raw = _read_toml(DEFAULT_CONFIG_PATH)
    chosen = path or os.environ.get(ENV_CONFIG)
    if chosen and Path(chosen).resolve() != DEFAULT_CONFIG_PATH:
        raw = _deep_merge(raw, _read_toml(Path(chosen)))
    raw = _apply_overrides(raw, overrides)
    try:
        return LprConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"配置校验失败：\n{e}") from e
