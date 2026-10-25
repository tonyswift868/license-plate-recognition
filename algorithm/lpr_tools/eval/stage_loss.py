"""Paired operational loss analysis for normal and oracle eval traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path


def _read_trace(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item["path"] in rows:
                raise ValueError(f"duplicate trace path at line {line_no}: {item['path']}")
            rows[item["path"]] = item
    return rows


def analyze(normal_path, oracle_path):
    normal, oracle = _read_trace(normal_path), _read_trace(oracle_path)
    if list(normal) != list(oracle):
        raise ValueError("normal and oracle traces must contain the same paths in the same order")
    pairs = [(normal[path], oracle[path]) for path in normal]
    if any(n.get("truth") != o.get("truth") or n.get("category") != o.get("category")
           for n, o in pairs):
        raise ValueError("normal and oracle traces must agree on truth and category for each path")
    decoded = [(n, o) for n, o in pairs if not n.get("decode_error") and not o.get("decode_error")]
    plated = [(n, o) for n, o in decoded if n["category"] != "np"]
    normal_errors = [(n, o) for n, o in plated if n["prediction"] != n["truth"]]
    loss = Counter()
    for n, o in normal_errors:
        if n.get("top3_hit") is False:
            bucket = "localization_top3_miss"
        elif n.get("top3_hit") is True and n.get("status") in {"SEG_FAILED", "NO_PLATE"}:
            bucket = "rectification_segmentation_operational"
        elif n.get("prediction"):
            bucket = "recognition_segmentation_residual"
        else:
            bucket = "unclassified_no_output"
        loss[bucket] += 1

    quadrants = Counter()
    oracle_seg_failed = 0
    oracle_residual_wrong = 0
    character_errors = {}
    for name, trace_index in (("normal", 0), ("oracle", 1)):
        substitutions, positions = Counter(), Counter()
        extras, missing = Counter(), Counter()
        for pair in plated:
            result = pair[trace_index]
            if result.get("prediction"):
                for position, (truth_char, pred_char) in enumerate(
                        zip_longest(result["truth"], result["prediction"], fillvalue=None), 1):
                    if truth_char is None:
                        extras[position] += 1
                    elif pred_char is None:
                        missing[position] += 1
                    elif truth_char != pred_char:
                        substitutions[(truth_char, pred_char)] += 1
                        positions[position] += 1
        character_errors[name] = {
            "no_prediction_rows": sum(not bool(pair[trace_index].get("prediction")) for pair in plated),
            "substitutions_top20": [{"truth": a, "prediction": b, "count": count}
                                    for (a, b), count in substitutions.most_common(20)],
            "substitution_position_counts": {str(k): v for k, v in sorted(positions.items())},
            "extra_character_positions": {str(k): v for k, v in sorted(extras.items())},
            "missing_character_positions": {str(k): v for k, v in sorted(missing.items())},
        }
    for n, o in plated:
        normal_ok = n["prediction"] == n["truth"]
        oracle_ok = o["prediction"] == o["truth"]
        quadrants["both_correct" if normal_ok and oracle_ok else
                  "normal_only_correct" if normal_ok else
                  "oracle_only_correct" if oracle_ok else "both_wrong"] += 1
        if not oracle_ok and o.get("segmentation_failed"):
            oracle_seg_failed += 1
        if not oracle_ok and not o.get("segmentation_failed"):
            oracle_residual_wrong += 1
    return {
        "scope": "paired operational description; not a causal additive decomposition",
        "trace_rows": len(pairs), "decoded_pairs": len(decoded), "plated_decoded_pairs": len(plated),
        "decode_mismatch_or_skip": sum(bool(n.get("decode_error") or o.get("decode_error")) for n, o in pairs),
        "normal_errors": len(normal_errors),
        "mutually_exclusive_normal_error_buckets": dict(loss),
        "normal_oracle_exact_match_quadrants": dict(quadrants),
        "oracle_segmentation_failures_among_wrong": oracle_seg_failed,
        "oracle_wrong_without_segmentation_failure": oracle_residual_wrong,
        "character_errors": character_errors,
    }


def render_markdown(result):
    buckets = result["mutually_exclusive_normal_error_buckets"]
    lines = ["# Paired stage loss analysis", "",
             "范围：同一批验证图的 normal/oracle 配对操作性归因；不是可加的因果分解。Oracle 用标注四角做 GT warp 替代原定位与校正，仍运行现有 segment、classifier 和 decode。", "",
             f"可配对识别样本：{result['plated_decoded_pairs']}；normal 整牌错误：{result['normal_errors']}。", "",
             "## Normal 错误互斥归因", "", "| 操作性桶 | 张数 | 占 normal 错误 |", "|---|---:|---:|"]
    for key, value in buckets.items():
        share = value / result["normal_errors"] if result["normal_errors"] else 0
        lines.append(f"| {key} | {value} | {share:.1%} |")
    lines.extend(["", "Top-3 miss 表示定位候选未覆盖真值框（IoU ≥ 0.5）；Top-3 hit 后 SEG_FAILED/NO_PLATE 记作校正+分割操作性损失；有输出但整牌错记作识别+分割残差。其余无输出记为未分类。", "",
                  "## Normal / oracle 配对", "", "| 配对结果 | 张数 |", "|---|---:|"])
    for key, value in result["normal_oracle_exact_match_quadrants"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"Oracle 错误中 segmentation failed：{result['oracle_segmentation_failures_among_wrong']}；未发生分割失败但仍错误：{result['oracle_wrong_without_segmentation_failure']}。后者包含字符分类、分割边界误差、后处理等，不能称为纯 CNN 因果误差。"]
    for name in ("normal", "oracle"):
        stats = result["character_errors"][name]
        lines += ["", f"## {name.title()} 字符替换混淆 Top 20", "", f"无字符输出样本：{stats['no_prediction_rows']}（不计入字符混淆）", "", "| 真值 | 识别 | 次数 |", "|---|---|---:|"]
        for item in stats["substitutions_top20"]:
            lines.append(f"| {item['truth']} | {item['prediction']} | {item['count']} |")
        lines += ["", f"### {name.title()} 替换错误位次", "", "| 位次（从 1 开始） | 替换数 |", "|---:|---:|"]
        for position, count in stats["substitution_position_counts"].items():
            lines.append(f"| {position} | {count} |")
        lines.append(f"长度错位另计：多余字符 {sum(stats['extra_character_positions'].values())}，缺失字符 {sum(stats['missing_character_positions'].values())}。")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--out", required=True, help="analysis.json path; adjacent analysis.md is written too")
    args = parser.parse_args(argv)
    result = analyze(args.normal, args.oracle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_name("analysis.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"analysis={out}")
    print(f"markdown={out.with_name('analysis.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
