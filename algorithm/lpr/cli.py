"""命令行（架构师）：单图 / 批量识别、多帧识别、导出调试图。

    python -m lpr.cli 图片或目录 [...] [--lane 1] [--many] [--fake] [--json] [--debug-dir out/dbg]
    python -m lpr.cli algorithm/data/raw/local --classifier svm --config my.toml

--many：把所有输入当作同一辆车的多帧，调用 recognize_many（联调前模拟多图上传，见 17.2 W2）。
--fake：使用 FakeRecognizer，不需要模型文件。
--uniform：真实流水线 + UniformClassifier（各类等概率），不需要模型文件，用于检查阶段一到四。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from .config import load_config
from .errors import ConfigError, ModelLoadError
from .io_utils import imread, imwrite, list_images
from .pipeline import PlateRecognizer
from .testing import FakeRecognizer, UniformClassifier
from .types import RecognitionResult


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m lpr.cli", description="M2 车牌识别命令行")
    p.add_argument("inputs", nargs="+", help="图片文件或目录")
    p.add_argument("--lane", type=int, default=None, help="车道号（使用配置中的车道 ROI）")
    p.add_argument("--many", action="store_true", help="所有输入作为同一车辆的多帧，调用 recognize_many")
    p.add_argument("--fake", action="store_true", help="使用 FakeRecognizer")
    p.add_argument("--uniform", action="store_true", help="真实流水线 + UniformClassifier，不需要模型文件")
    p.add_argument("--config", default=None, help="配置文件路径（缺省读 LPR_CONFIG，再缺省 default.toml）")
    p.add_argument("--classifier", choices=["cnn", "template", "svm"], default=None, help="覆盖 runtime.classifier")
    p.add_argument("--model", default=None, help="覆盖 runtime.model_path")
    p.add_argument("--recursive", action="store_true", help="递归查找目录中的图片")
    p.add_argument("--json", action="store_true", help="每个结果输出一行 JSON")
    p.add_argument("--debug-dir", default=None, help="导出校正图与各阶段中间图的目录")
    p.add_argument("-v", "--verbose", action="store_true", help="输出 loguru 日志")
    return p.parse_args(argv)


def _build_recognizer(args: argparse.Namespace) -> PlateRecognizer | FakeRecognizer:
    if args.fake:
        return FakeRecognizer()
    overrides: dict[str, object] = {}
    if args.classifier:
        overrides["runtime__classifier"] = args.classifier
    if args.model:
        overrides["runtime__model_path"] = args.model
    if args.uniform:
        return PlateRecognizer(load_config(args.config, **overrides), UniformClassifier())
    return PlateRecognizer.from_config(args.config, **overrides)


def _print(result: RecognitionResult, name: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"path": name, **result.to_dict()}, ensure_ascii=False))
    else:
        print(f"{name}\t{result.status.value}\t{result.plate_no or '-'}\t"
              f"{result.confidence:.3f}\t{result.elapsed_ms:.1f}ms\t{result.message}")


def _dump_debug(result: RecognitionResult, out_dir: Path, stem: str) -> None:
    if result.plate_image is not None:
        imwrite(out_dir / f"{stem}_plate.png", result.plate_image)
    for key, img in (result.debug or {}).items():
        imwrite(out_dir / f"{stem}_{key}.png", img)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "WARNING")

    paths = [p for inp in args.inputs for p in list_images(inp, recursive=args.recursive)]
    if not paths:
        print("没有找到图片（支持 jpg、jpeg、png、bmp）", file=sys.stderr)
        return 1
    try:
        recognizer = _build_recognizer(args)
    except (ConfigError, ModelLoadError) as e:
        print(f"无法创建识别器：{e}\n模型就绪前可加 --fake（联调桩）或 --uniform（真实流水线、无模型）。",
              file=sys.stderr)
        return 2
    recognizer.warmup()
    out_dir = Path(args.debug_dir) if args.debug_dir else None

    if args.many:
        result = recognizer.recognize_many([p.read_bytes() for p in paths], lane_id=args.lane)
        _print(result, f"{len(paths)} 帧", args.json)
        if out_dir:
            _dump_debug(result, out_dir, "many")
        return 0

    for path in paths:
        try:
            image = imread(path)
        except (OSError, ValueError) as e:
            print(f"{path}\t读取失败：{e}", file=sys.stderr)
            continue
        result = recognizer.recognize(image, lane_id=args.lane, debug=out_dir is not None)
        _print(result, str(path), args.json)
        if out_dir:
            _dump_debug(result, out_dir, path.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
