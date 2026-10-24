"""Shared offline recognizer selection and benchmark utilities."""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
import cv2
import numpy as np

from lpr.config import load_config
from lpr.pipeline import PlateRecognizer
from lpr.testing import FakeRecognizer, UniformClassifier


def add_backend_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fake", action="store_true", help="integration stub, not model performance")
    group.add_argument("--uniform", action="store_true", help="real pipeline with uniform classifier (no weights)")
    parser.add_argument("--classifier", choices=("cnn", "template", "svm"), default="cnn")


def make_recognizer(args, *, samples_dir=None):
    if args.fake:
        return FakeRecognizer(samples_dir=samples_dir)
    cfg = load_config(runtime__classifier=args.classifier)
    if args.uniform:
        return PlateRecognizer(cfg, UniformClassifier())
    return PlateRecognizer.from_config(runtime__classifier=args.classifier)


def mode_name(args):
    return "fake" if args.fake else "uniform" if args.uniform else args.classifier


def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = map(int, value.lower().split("x"))
        if w < 1 or h < 1:
            raise ValueError
        return w, h
    except ValueError as exc:
        raise ValueError("size must be positive WIDTHxHEIGHT") from exc


def letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    scale = min(w/image.shape[1], h/image.shape[0])
    nw, nh = min(w, max(1, round(image.shape[1]*scale))), min(h, max(1, round(image.shape[0]*scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), np.uint8)
    left, top = (w-nw)//2, (h-nh)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas


def runtime_info() -> dict:
    import torch
    cfg = load_config()
    torch.set_num_threads(cfg.runtime.torch_threads)
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0 and result.stdout.strip():
                cpu = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"cpu": cpu, "torch_threads": torch.get_num_threads()}


def latency_stats(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"p50_ms": float(np.percentile(array, 50)), "p95_ms": float(np.percentile(array, 95)),
            "max_ms": float(array.max())}
