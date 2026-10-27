"""分类器工厂与 models/manifest.json 校验（架构师）。

manifest.json 以模型文件名为键，每个模型一条记录（见开发文档 7.5）：
    {"char_cnn_v1.pt": {"name": "char_cnn_v1.pt", "version": "char_cnn_v1", "charset_sha": "...",
                        "input_size": [32, 32], "train_data": "...", "val_char_acc": 0.98,
                        "sha256": "...", "created_at": "2026-10-28T12:00:00"}}
训练脚本保存模型后必须调用 write_manifest_entry() 登记；加载时字符表哈希或文件哈希对不上就拒绝加载。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..charset import CHARSET_SHA
from ..errors import ModelLoadError
from .base import BoxReader, CharClassifier, CornerDetector, PlateReader

MANIFEST_NAME = "manifest.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(models_dir: Path) -> dict[str, dict[str, Any]]:
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_manifest_entry(model_path: Path, *, version: str, train_data: str,
                         val_char_acc: float | None, input_size: tuple[int, int] = (32, 32)) -> dict[str, Any]:
    """登记（或覆盖）一个模型文件的 manifest 记录，返回该记录。"""
    model_path = Path(model_path)
    entries = read_manifest(model_path.parent)
    entry = {
        "name": model_path.name,
        "version": version,
        "charset_sha": CHARSET_SHA,
        "input_size": list(input_size),
        "train_data": train_data,
        "val_char_acc": val_char_acc,
        "sha256": file_sha256(model_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    entries[model_path.name] = entry
    with open(model_path.parent / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return entry


def verify_model(model_path: Path) -> dict[str, Any]:
    """检查模型文件存在、已在 manifest 登记、字符表与文件哈希一致；返回 manifest 记录。"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelLoadError(f"模型文件不存在：{model_path}")
    entry = read_manifest(model_path.parent).get(model_path.name)
    if entry is None:
        raise ModelLoadError(f"模型未在 {model_path.parent / MANIFEST_NAME} 登记：{model_path.name}")
    if entry.get("charset_sha") != CHARSET_SHA:
        raise ModelLoadError(
            f"字符表哈希不匹配：模型 {entry.get('charset_sha')}，当前 {CHARSET_SHA}（{model_path.name}）")
    if entry.get("sha256") and entry["sha256"] != file_sha256(model_path):
        raise ModelLoadError(f"模型文件与 manifest 记录的 sha256 不一致：{model_path.name}")
    return entry


def load_classifier(kind: str, model_path: Path) -> CharClassifier:
    """按配置 runtime.classifier 加载分类器。任何失败都转为 ModelLoadError。"""
    entry = verify_model(model_path)
    try:
        if kind == "cnn":
            from .cnn import CnnClassifier as cls
        elif kind == "template":
            from .template import TemplateClassifier as cls
        elif kind == "svm":
            from .svm import SvmClassifier as cls
        else:
            raise ModelLoadError(f"未知分类器类型：{kind}")
        return cls.from_file(Path(model_path), version=entry["version"])
    except ModelLoadError:
        raise
    except NotImplementedError as e:
        raise ModelLoadError(f"分类器 {kind} 尚未实现：{e}") from e
    except Exception as e:
        raise ModelLoadError(f"分类器 {kind} 加载失败：{model_path}：{e}") from e


def _load_network(label: str, import_cls, model_path: Path):
    """四角点网络与整牌识别共用的加载流程：manifest 校验 → from_file；任何失败转为 ModelLoadError。"""
    entry = verify_model(model_path)
    try:
        return import_cls().from_file(Path(model_path), version=entry["version"])
    except ModelLoadError:
        raise
    except NotImplementedError as e:
        raise ModelLoadError(f"{label}尚未实现：{e}") from e
    except Exception as e:
        raise ModelLoadError(f"{label}加载失败：{model_path}：{e}") from e


def load_corner_detector(model_path: Path) -> CornerDetector:
    """加载四角点网络（runtime.corner_model_path）。"""
    def cls():
        from ..corner_net import CornerNet
        return CornerNet
    return _load_network("四角点网络", cls, model_path)


def load_plate_reader(model_path: Path) -> PlateReader:
    """加载整牌识别 CRNN（runtime.plate_reader_model_path）。"""
    def cls():
        from .crnn import CrnnReader
        return CrnnReader
    return _load_network("整牌识别网络", cls, model_path)


def load_box_reader(model_path: Path) -> BoxReader:
    """Load a detection-box reader with manifest and frozen-charset verification."""
    def cls():
        from .box_reader import BoxPlateReader
        return BoxPlateReader
    return _load_network("检测框识别网络", cls, model_path)
