"""Calibrate plate character slots from CCPD ground-truth quadrilaterals.

The calibration path deliberately rectifies each labeled image directly.  It
does not use candidate refinement or the segmenter's slot configuration when
measuring character geometry.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from lpr.config import load_config
from lpr.io_utils import imread
from lpr.segment import _fallback_binary_from_gray, fallback_binary_plate, segment
from lpr.types import PlateCandidate, RectifiedPlate
from lpr_tools.data.ccpd import read_index


def _quad(value: str) -> np.ndarray:
    nums = [float(v) for v in value.split()]
    if len(nums) != 8 or not np.isfinite(nums).all():
        raise ValueError("quad must contain four finite x y points")
    quad = np.asarray(nums, np.float32).reshape(4, 2)
    if abs(cv2.contourArea(quad)) < 1 or not cv2.isContourConvex(quad):
        raise ValueError("quad must be convex and non-degenerate")
    return quad


def _validate_quad_bounds(quad: np.ndarray, image: np.ndarray) -> None:
    h, w = image.shape[:2]
    if (np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= w)
            or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= h)):
        raise ValueError("quad lies outside image bounds")


def _measure_slots(binary: np.ndarray, layout: int) -> list[tuple[float, float]]:
    """Estimate character spans from ink runs, without consulting segment cfg."""
    h, w = binary.shape
    projection = (binary > 0).sum(axis=0)
    # binary_plate already removes speckles by connected-component area. Keep
    # every remaining ink column: a height threshold splits U/V/7 into strokes
    # and biases both the centers and the observed character widths.
    active = projection > 0
    # Fill tiny holes within a glyph (e.g. separated strokes) before extracting runs.
    for i in range(1, w - 1):
        if not active[i] and active[i - 1] and active[i + 1]:
            active[i] = True
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    # Merge the closest neighboring runs until each remaining span represents one
    # character. The enlarged inter-character gap remains the last gap to preserve.
    while len(runs) > layout:
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        i = int(np.argmin(gaps))
        runs[i:i + 2] = [(runs[i][0], runs[i + 1][1])]
    if len(runs) != layout:
        raise ValueError("no_observation")
    return [((left + right) / (2 * w), (right - left) / w) for left, right in runs]


def _select(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Round-robin all split rows by dataset, subset, and layout deterministically."""
    pool = rows
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        key = (row.get("dataset", ""), row.get("subset", ""), row.get("layout", "unknown"))
        strata[key].append(row)
    rng = random.Random(seed)
    keys = sorted(strata)
    for key in keys:
        strata[key].sort(key=lambda r: (r.get("path", ""), r.get("plate_no", "")))
        rng.shuffle(strata[key])
    selected = []
    target = min(n, len(pool))
    while len(selected) < target:
        advanced = False
        for key in keys:
            if strata[key] and len(selected) < n:
                selected.append(strata[key].pop())
                advanced = True
        if not advanced:
            break
    return selected


def calibrate(index: str | Path, raw: str | Path, n: int = 2000, seed: int = 42) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    rows = _select(read_index(index), n, seed)
    raw_root = Path(raw)
    cfg = load_config()
    counts: Counter[str] = Counter()
    metrics: dict[str, Counter[str]] = defaultdict(Counter)
    measurements: dict[int, list[list[tuple[float, float]]]] = {7: [], 8: []}
    split_counts = Counter(r.get("split", "") for r in rows)

    for row in rows:
        subset_key = f"{row.get('dataset', '')}/{row.get('subset', '')}"
        if row.get("has_plate") != "1" or row.get("skip_reason"):
            counts[f"skipped:{row.get('skip_reason') or 'no_plate'}"] += 1
            continue
        try:
            layout = int(row["layout"])
            if layout not in (7, 8):
                raise ValueError("unsupported_layout")
            color = row.get("plate_color") or ("green" if layout == 8 else "blue")
        except (ValueError, KeyError, TypeError):
            counts["invalid_label_or_layout"] += 1
            continue
        subset_key = f"{row.get('dataset', '')}/{row.get('subset', '')}"
        key = f"{subset_key}/layout{layout}"
        overall_key = f"overall/layout{layout}"
        counts["eligible_labeled"] += 1
        for group in (key, overall_key, "overall"):
            metrics[group]["denominator"] += 1
            metrics[group]["correct"] += 0
        try:
            root_resolved = raw_root.resolve()
            path = Path(row["path"])
            path = path.resolve() if path.is_absolute() else (root_resolved / path).resolve()
            if not path.is_relative_to(root_resolved):
                counts["path_outside_raw"] += 1
                counts["failed_read_or_quad"] += 1
                continue
            source = imread(path)
            quad = _quad(row["quad"])
            _validate_quad_bounds(quad, source)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            counts["unreadable_or_invalid"] += 1
            counts["failed_read_or_quad"] += 1
            counts[f"detail:{type(exc).__name__}"] += 1
            continue

        width = 252 if layout == 7 else 276
        target = np.float32([[0, 0], [width - 1, 0], [width - 1, 79], [0, 79]])
        try:
            matrix = cv2.getPerspectiveTransform(quad, target)
            plate = cv2.warpPerspective(source, matrix, (width, 80), flags=cv2.INTER_LINEAR)
        except cv2.error:
            counts["rectification_error"] += 1
            counts["failed_read_or_quad"] += 1
            continue
        x0, y0 = np.floor(quad.min(axis=0)).astype(int)
        x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
        candidate = PlateCandidate([tuple(map(float, point)) for point in quad],
                                   (int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)),
                                   color, 1., "ground_truth")
        counts["readable_rectified"] += 1
        rect = RectifiedPlate(plate, layout, color, matrix, candidate)
        # Re-threshold unified gray independently of segment's success path;
        # calibration must not inherit its CC deletions or sample selection.
        gray, binary = fallback_binary_plate(rect, cfg.segment)
        result = segment(rect, layout, cfg.segment)
        correct = bool(result.ok and len(result.boxes) == layout)
        for group in (key, overall_key, "overall"):
            metrics[group]["correct"] += int(correct)
        projection = None
        method = "otsu"
        if np.count_nonzero(binary):
            try:
                projection = _measure_slots(binary, layout)
            except ValueError:
                pass
        if projection is None:
            adaptive_cfg = cfg.segment.model_copy(update={"binarize": "adaptive"})
            adaptive = _fallback_binary_from_gray(gray, adaptive_cfg)
            if np.count_nonzero(adaptive):
                try:
                    projection = _measure_slots(adaptive, layout)
                    method = "adaptive"
                except ValueError:
                    pass
        if projection is None:
            counts["no_observation"] += 1
            if row.get("split") == "val":
                counts["no_observation_val"] += 1
        else:
            counts["observed"] += 1
            counts[f"observed_{method}"] += 1
            if row.get("split") == "val":
                measurements[layout].append(projection)
                counts["observed_val"] += 1

    stats = {}
    for layout, samples in measurements.items():
        if not samples:
            stats[str(layout)] = {"n": 0, "centers": [], "center_std": [], "char_w": None, "char_w_std": None}
            continue
        arr = np.asarray(samples, dtype=np.float64)  # sample × character × (center,width)
        centers, widths = arr[:, :, 0], arr[:, :, 1]
        stats[str(layout)] = {
            "n": len(samples), "centers": centers.mean(axis=0).round(6).tolist(),
            "center_median": np.median(centers, axis=0).round(6).tolist(),
            "center_std": centers.std(axis=0).round(6).tolist(),
            "char_w": round(float(widths.mean()), 6), "char_w_std": round(float(widths.std()), 6),
            "char_w_median": round(float(np.median(widths)), 6),
            "char_w_by_position": widths.mean(axis=0).round(6).tolist(),
            "char_w_std_by_position": widths.std(axis=0).round(6).tolist(),
        }
    return {"selected": len(rows), "split_counts": dict(split_counts), "counts": dict(counts),
            "metrics": {k: dict(v) for k, v in sorted(metrics.items())}, "geometry": stats,
            "seed": seed, "n_requested": n}


def _markdown(result: dict) -> str:
    lines = ["# CCPD 版式校准报告", "", f"抽样：{result['selected']} / requested {result['n_requested']}，seed={result['seed']}；split={result['split_counts']}",
             "", "几何只统计 split=val 的真值四角点透视校正图；对统一灰度的独立 raw mask 做投影，不读取 segment 槽位参数，也不按 segment 是否成功筛样。先尝试 Otsu；投影不能形成版式对应的字符组时，再尝试 adaptive。观测计数按阈值来源拆分。相对板宽，标准差为总体标准差。",
             "每个位数的 char_w 是逐位宽度均值的总均值；char_w_by_position 提供各位结果。报告同时给出均值和中位数候选；建议值仅供评审，不自动修改配置。", "",
             "仿 CCPD 数据只用于验证流程，不能作为正式指标；正式指标需在 CCPD 原始数据上复算。", "",
             "## 建议参数", ""]
    for layout in (7, 8):
        g = result["geometry"][str(layout)]
        lines += [f"### Layout {layout}（val n={g['n']}）", "", f"- `layout{layout}_centers`: `{g['centers']}`",
                  f"- `layout{layout}_centers_median`: `{g.get('center_median', [])}`",
                  f"- `layout{layout}_centers_std`: `{g['center_std']}`",
                  f"- `layout{layout}_char_w`: `{g['char_w']}`（std `{g['char_w_std']}`）",
                  f"- `layout{layout}_char_w_median`: `{g.get('char_w_median')}`",
                  f"- `layout{layout}_char_w_by_position`: `{g.get('char_w_by_position', [])}`",
                  f"- `layout{layout}_char_w_std_by_position`: `{g.get('char_w_std_by_position', [])}`", "",
                  f"建议 TOML 键：`layout{layout}_centers`、`layout{layout}_char_w`（无 val 观测时不输出建议值）", ""]
    lines += ["## 真值定位下 segment 位数正确率", "", "正确定义为 segment().ok 且输出框数等于真值 layout；无观测仍计入分母，正确性按 segment() 实际结果计算。", "",
              "| 子集 / 版式 | 正确 | 分母 | 正确率 |", "|---|---:|---:|---:|"]
    for key, value in result["metrics"].items():
        den, num = value.get("denominator", 0), value.get("correct", 0)
        lines.append(f"| {key} | {num} | {den} | {num / den:.4f} |" if den else f"| {key} | 0 | 0 | N/A |")
    lines += ["", "## 跳过与异常计数", "", "```json", json.dumps(result["counts"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = calibrate(args.index, args.raw, args.n)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "calib_layout.md"
    report.write_text(_markdown(result), encoding="utf-8")
    (out / "calib_layout.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    suggestion = ["# Median candidates from independent validation projections; review against end-to-end accuracy before use.", "[segment]", ""]
    for layout in (7, 8):
        g = result["geometry"][str(layout)]
        if g["n"]:
            suggestion += [f"layout{layout}_centers = {json.dumps(g['center_median'])}",
                           f"layout{layout}_char_w = {json.dumps(g['char_w_median'])}", ""]
        else:
            suggestion += [f"# layout{layout}_centers unavailable: no validation observations",
                           f"# layout{layout}_char_w unavailable: no validation observations", ""]
    (out / "suggested_segment.toml").write_text("\n".join(suggestion), encoding="utf-8")
    print(_markdown(result))
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
