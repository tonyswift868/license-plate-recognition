"""Shared offline recognizer selection and benchmark utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import cv2
import numpy as np

from lpr.config import ALGORITHM_DIR, load_config
from lpr.pipeline import PlateRecognizer
from lpr.testing import FakeRecognizer, UniformClassifier


class AttributedPlateRecognizer(PlateRecognizer):
    """Observe actual final pipeline selection without changing its arbitration."""
    def __init__(self, recognizer):
        super().__init__(recognizer.config, recognizer.classifier,
                         corner_detector=recognizer.corner_detector,
                         plate_reader=recognizer.plate_reader,
                         box_reader=recognizer.box_reader)
        self.last_selection = None
        self.reader_invoked_calls = 0
        self.corner_invoked_calls = 0
        self.box_reader_invoked_calls = 0
        if self.plate_reader is not None:
            self.plate_reader = _CountingModel(self.plate_reader, "read", self, "reader_invoked_calls")
        if self.corner_detector is not None:
            self.corner_detector = _CountingModel(self.corner_detector, "detect", self, "corner_invoked_calls")
        if self.box_reader is not None:
            self.box_reader = _CountingModel(self.box_reader, "read", self, "box_reader_invoked_calls")

    def _reset_observation(self):
        self.last_selection = None
        self.reader_invoked_calls = 0
        self.corner_invoked_calls = 0
        self.box_reader_invoked_calls = 0

    def _recognize_safe(self, image, lane_id, debug):
        self._reset_observation()
        return super()._recognize_safe(image, lane_id, debug)

    def _recognize_bytes_safe(self, data, lane_id):
        self._reset_observation()
        return super()._recognize_bytes_safe(data, lane_id)

    def _build(self, attempt, dbg):
        result = super()._build(attempt, dbg)
        self.last_selection = {
            "source": attempt.rect.candidate.source,
            # B keeps the original crop on the attempt and has no segmented chars;
            # preserve the historical fallback for attempts predating plate_image.
            "crnn_selected": attempt.seg is None and getattr(attempt, "plate_image", None) is None,
        }
        return result


class _CountingModel:
    def __init__(self, model, method, owner, counter):
        self._model, self._method, self._owner, self._counter = model, method, owner, counter

    @property
    def version(self):
        return self._model.version

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __getattribute__(self, name):
        if name in {"read", "detect"}:
            method = object.__getattribute__(self, "_method")
            if name == method:
                model = object.__getattribute__(self, "_model")
                owner = object.__getattribute__(self, "_owner")
                counter = object.__getattribute__(self, "_counter")
                def counted(*args, **kwargs):
                    setattr(owner, counter, getattr(owner, counter) + 1)
                    return getattr(model, method)(*args, **kwargs)
                return counted
        return object.__getattribute__(self, name)


def add_backend_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fake", action="store_true", help="integration stub, not model performance")
    group.add_argument("--uniform", action="store_true", help="real pipeline with uniform classifier (no weights)")
    parser.add_argument("--config", default=None, help="runtime TOML config (defaults to LPR_CONFIG/default)")
    parser.add_argument("--classifier", choices=("cnn", "template", "svm"), default=None)


def config_for_args(args):
    """Resolve the effective config once; old Namespace callers remain supported."""
    classifier = getattr(args, "classifier", None)
    overrides = {"runtime__classifier": classifier} if classifier else {}
    return load_config(getattr(args, "config", None), **overrides)


def model_identity(cfg, field):
    configured = getattr(cfg.runtime, field)
    path = cfg.resolve_path(configured) if configured else None
    exists = bool(path and path.is_file())
    digest = None
    version = None
    if exists:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        # Manifests are optional; do not claim a version when unavailable.
        manifest = path.parent / "manifest.json"
        if manifest.is_file():
            try:
                version = json.loads(manifest.read_text(encoding="utf-8")).get(path.name, {}).get("version")
            except (OSError, ValueError, AttributeError):
                pass
    return {"configured": bool(configured), "configured_path": configured or None,
            "path": str(path) if path else None, "exists": exists,
            "sha256": digest, "version": version}


def _selected_config_path(config_arg=None):
    chosen = config_arg or os.environ.get("LPR_CONFIG")
    if not chosen:
        return (ALGORITHM_DIR / "config" / "default.toml").resolve()
    path = Path(chosen)
    # Mirror load_config's special fallback for algorithm-relative config/ paths.
    if not path.is_absolute() and path.parts[:1] == ("config",) and not path.exists():
        path = ALGORITHM_DIR / path
    return path.resolve()


def effective_config_info(cfg, config_arg=None):
    config_path = str(_selected_config_path(config_arg))
    dumped = cfg.model_dump(mode="json")
    canonical = json.dumps(dumped, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"config_path": config_path,
            "effective_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "effective_config": dumped,
            "models": {"classifier": model_identity(cfg, "model_path"),
                       "corner": model_identity(cfg, "corner_model_path"),
                       "plate_reader": model_identity(cfg, "plate_reader_model_path"),
                       "box_reader": model_identity(cfg, "box_reader_model_path")}}


def make_recognizer(args, *, samples_dir=None, cfg=None):
    if getattr(args, "fake", False):
        return FakeRecognizer(samples_dir=samples_dir)
    cfg = cfg or config_for_args(args)
    recognizer = (PlateRecognizer(cfg, UniformClassifier()) if getattr(args, "uniform", False)
                  else PlateRecognizer.from_config(getattr(args, "config", None),
                       **({"runtime__classifier": getattr(args, "classifier")} if getattr(args, "classifier", None) else {})))
    return AttributedPlateRecognizer(recognizer)


def mode_name(args):
    if getattr(args, "fake", False):
        return "fake"
    if getattr(args, "uniform", False):
        return "uniform"
    return getattr(args, "classifier", None) or config_for_args(args).runtime.classifier


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


def runtime_info(cfg=None) -> dict:
    import torch
    cfg = cfg or load_config()
    torch.set_num_threads(cfg.runtime.torch_threads)
    cv2.setNumThreads(1)
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
