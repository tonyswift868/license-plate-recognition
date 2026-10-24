"""图像读写工具（架构师）：中文路径安全，字节流解码。

cv2.imread / cv2.imwrite 遇到中文路径会失败（开发文档第 15 章），读写图片一律走这里。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def imread(path: str | os.PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读图（支持中文路径）。文件不存在抛 FileNotFoundError，无法解码抛 ValueError。"""
    data = np.fromfile(os.fspath(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags) if data.size else None
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    return image


def imwrite(path: str | os.PathLike[str], image: np.ndarray, params: list[int] | None = None) -> None:
    """写图（支持中文路径），按扩展名编码，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image, params or [])
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    buf.tofile(os.fspath(path))


def decode_image(data: bytes | bytearray | memoryview) -> np.ndarray | None:
    """jpg/png 字节流 → BGR uint8；空数据或无法解码时返回 None。"""
    if not data:
        return None
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    try:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def encode_jpg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("无法编码为 jpg")
    return buf.tobytes()


def list_images(root: str | os.PathLike[str], recursive: bool = False) -> list[Path]:
    """列出目录下的图片（按路径排序）；root 本身是图片文件时返回它自己。"""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
