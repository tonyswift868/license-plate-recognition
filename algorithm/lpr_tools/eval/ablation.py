"""Compare character classifiers on one normalized character directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from lpr.charset import CHARSET, PROVINCE_IDX
from lpr.io_utils import imread
from lpr.recognize import load_classifier

WARMUP_CALLS = 5
CHAR_READ_WORKERS = 4
CHAR_READ_WINDOW = 512


def load_chars(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read sparse <class-index>/*.png input; labels stay in frozen charset order."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"character directory does not exist: {root}")
    def samples():
        for class_dir in sorted(root.iterdir(), key=lambda p: p.name):
            if not class_dir.is_dir():
                continue
            try:
                label = int(class_dir.name)
            except ValueError:
                continue
            if not 0 <= label < len(CHARSET):
                raise ValueError(f"class index out of range: {class_dir}")
            for path in sorted(class_dir.glob("*.png")):
                yield label, path

    images: list[np.ndarray] = []
    labels: list[int] = []
    sample_iter = iter(samples())
    with ThreadPoolExecutor(max_workers=CHAR_READ_WORKERS) as pool:
        pending = deque()
        for _ in range(CHAR_READ_WINDOW):
            try:
                label, path = next(sample_iter)
            except StopIteration:
                break
            pending.append((label, path, pool.submit(imread, path, 0)))
        while pending:
            label, path, future = pending.popleft()
            image = future.result()
            if image is None or image.shape != (32, 32) or image.dtype != np.uint8:
                raise ValueError(f"expected 32 x 32 uint8 character: {path}")
            images.append(image)
            labels.append(label)
            try:
                next_label, next_path = next(sample_iter)
            except StopIteration:
                continue
            pending.append((next_label, next_path, pool.submit(imread, next_path, 0)))
    if not images:
        raise ValueError(f"no character PNG samples found in {root}")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _training_seconds(model_path: Path, entry: dict) -> float | None:
    """Read explicit metadata only; never infer training time from file timestamps."""
    for key in ("train_seconds", "training_seconds"):
        value = entry.get(key)
        if value is not None:
            try:
                seconds = float(value)
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
            except (TypeError, ValueError):
                pass
    sidecar = model_path.with_suffix(".training.json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            seconds = float(data["train_seconds"])
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
    log_path = model_path.with_suffix(".train.csv")
    if log_path.is_file():
        try:
            with log_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows and all(row.get("epoch_seconds", "").strip() for row in rows):
                values = [float(row["epoch_seconds"]) for row in rows]
                if all(math.isfinite(v) and v >= 0 for v in values):
                    return sum(values)
        except (OSError, ValueError, TypeError):
            pass
    return None


def _manifest_entry(model_path: Path) -> dict:
    manifest = model_path.parent / "manifest.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        entry = value.get(model_path.name, {})
        return entry if isinstance(entry, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def evaluate_model(kind: str, model_path: Path, images: np.ndarray,
                   labels: np.ndarray) -> dict:
    classifier = load_classifier(kind, model_path)
    # Five single-character warmups; then each sample is timed as a one-item call.
    for i in range(WARMUP_CALLS):
        classifier.predict_proba(images[i % len(images):i % len(images) + 1])
    predicted = np.empty(len(labels), dtype=np.int64)
    elapsed_ns = 0
    for i, image in enumerate(images):
        started = time.perf_counter_ns()
        probs = np.asarray(classifier.predict_proba(image[None, ...]))
        elapsed_ns += time.perf_counter_ns() - started
        if probs.shape != (1, len(CHARSET)) or not np.isfinite(probs).all():
            raise ValueError(f"{kind} returned invalid probability shape or values: {probs.shape}")
        if np.any(probs < -1e-6) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{kind} returned rows that are not probability distributions")
        predicted[i] = int(probs[0].argmax())
    correct = predicted == labels

    def accuracy(mask: np.ndarray) -> tuple[float | None, int]:
        count = int(mask.sum())
        return (float(correct[mask].mean()) if count else None, count)

    province_acc, province_n = accuracy(np.isin(labels, PROVINCE_IDX))
    alnum_acc, alnum_n = accuracy((labels >= 31) & (labels < len(CHARSET)))
    entry = _manifest_entry(model_path)
    seconds = _training_seconds(model_path, entry)
    return {
        "kind": kind,
        "overall_accuracy": float(correct.mean()),
        "overall_n": len(labels),
        "province_accuracy": province_acc,
        "province_n": province_n,
        "alnum_accuracy": alnum_acc,
        "alnum_n": alnum_n,
        "mean_ms": elapsed_ns / len(labels) / 1_000_000,
        "model_bytes": model_path.stat().st_size,
        "train_seconds": seconds,
    }


def evaluate(chars: Path, models: dict[str, Path]) -> dict:
    if set(models) != {"cnn", "template", "svm"}:
        raise ValueError("--models must specify exactly cnn, template, and svm")
    torch.set_num_threads(2)
    images, labels = load_chars(chars)
    metrics = [evaluate_model(kind, Path(models[kind]), images, labels)
               for kind in ("template", "svm", "cnn")]
    return {"chars": str(chars), "sample_count": len(labels), "warmup_calls": WARMUP_CALLS,
            "timing_scope": "one predict_proba call per character; image read/model load excluded; five single-character warmups per model",
            "models": metrics, "plate_accuracy": None}


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_markdown(report: dict) -> str:
    lines = ["# 字符分类器对比", "",
             f"- 测试字符数：{report['sample_count']}",
             f"- 计时口径：{report['timing_scope']}",
             "- 测试集只需包含实际存在的数字类别目录；缺样本分组准确率显示 N/A。", "",
             "| 方法 | 总体准确率 | 省份字准确率 | 字母数字准确率 | 单字符平均耗时 (ms) | 模型大小 (MiB) | 训练耗时 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["models"]:
        train = "N/A" if m["train_seconds"] is None else f"{m['train_seconds']:.3f}"
        lines.append(f"| {m['kind']} | {_pct(m['overall_accuracy'])} (n={m['overall_n']}) | "
                     f"{_pct(m['province_accuracy'])} (n={m['province_n']}) | "
                     f"{_pct(m['alnum_accuracy'])} (n={m['alnum_n']}) | {m['mean_ms']:.4f} | "
                     f"{m['model_bytes'] / (1024 * 1024):.3f} | {train} |")
    lines.extend(["", "## 整牌准确率", "", "等待 `eval_accuracy` 的整牌结果填入；字符级结果不能替代整牌评测。", ""])
    plate = report.get("plate_accuracy")
    if plate:
        lines[lines.index("等待 `eval_accuracy` 的整牌结果填入；字符级结果不能替代整牌评测。"):] = [
            "由 `eval_accuracy` 填入：", ""] + [f"- {name}: {_pct(value)}" for name, value in plate.items()] + [""]
    return "\n".join(lines)


def fill_plate_accuracy(report_path: Path, results: dict[str, float]) -> None:
    """Integration point for eval_accuracy: replace the report's plate accuracy section."""
    for name, value in results.items():
        try:
            accuracy = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid plate accuracy for {name}: {value!r}") from exc
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"plate accuracy for {name} must be finite and between 0 and 1")
    path = Path(report_path)
    text = path.read_text(encoding="utf-8")
    marker = "## 整牌准确率\n"
    if marker not in text:
        raise ValueError("ablation report is missing the 整牌准确率 section")
    section = marker + "\n由 `eval_accuracy` 填入：\n\n" + "\n".join(
        f"- {name}: {_pct(float(value))}" for name, value in results.items()) + "\n"
    path.write_text(text.split(marker, 1)[0] + section, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chars", type=Path, required=True,
                        help="normalized 32x32 PNGs under numeric class-index directories")
    parser.add_argument("--models", nargs=3, required=True, metavar=("cnn=PATH", "template=PATH", "svm=PATH"))
    parser.add_argument("--out", type=Path, required=True, help="output directory for ablation.md")
    args = parser.parse_args(argv)
    try:
        models = {}
        for item in args.models:
            if "=" not in item:
                raise ValueError(f"invalid model specification: {item}")
            key, value = item.split("=", 1)
            if key in models or key not in {"cnn", "template", "svm"} or not value:
                raise ValueError(f"invalid model specification: {item}")
            models[key] = Path(value)
        report = evaluate(args.chars, models)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ablation.md").write_text(render_markdown(report), encoding="utf-8")
        print(render_markdown(report), end="")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
