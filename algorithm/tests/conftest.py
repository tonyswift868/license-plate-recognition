"""公共夹具（架构师）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from lpr.config import LprConfig, load_config
from lpr.pipeline import PlateRecognizer
from lpr.testing import UniformClassifier

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "data" / "golden"


@pytest.fixture(scope="session")
def cfg() -> LprConfig:
    return load_config()


@pytest.fixture
def recognizer(cfg: LprConfig) -> PlateRecognizer:
    """不需要模型文件的真实流水线（分类器输出均匀分布）。"""
    return PlateRecognizer(cfg, UniformClassifier())


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((720, 1280, 3), np.uint8)


def draw_synthetic_plate(text: str = "A12345", color: str = "blue",
                         size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """在灰色背景上画一块示意车牌（cv2 不能写中文，只画字母数字），供各阶段单测使用。"""
    w, h = size
    img = np.full((h, w, 3), 110, np.uint8)
    bg = {"blue": (170, 70, 10), "yellow": (20, 190, 235), "green": (130, 215, 120)}[color]
    fg = (255, 255, 255) if color == "blue" else (0, 0, 0)
    pw, ph = 252 * 2 // 3 * 2, 80 * 2 // 3 * 2
    x, y = (w - pw) // 2, int(h * 0.65)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), bg, -1)
    cv2.putText(img, text, (x + 20, y + ph - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0, fg, 5)
    return img


@pytest.fixture
def synthetic_plate_image() -> np.ndarray:
    return draw_synthetic_plate()


@pytest.fixture(scope="session")
def default_font() -> Path:
    """Skip only tests which need the local Chinese font."""
    from lpr_tools.train.synth_chars import DEFAULT_FONT

    if not DEFAULT_FONT.is_file():
        pytest.skip(f"missing local font: {DEFAULT_FONT}")
    return DEFAULT_FONT
