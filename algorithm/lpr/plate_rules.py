"""车牌版式、位置约束集合、正则、车型映射（员工 C，任务 C3；规则见开发文档附录 A、5.8）。

建议内容（具体结构由员工 C 设计，只要 postprocess.decode 与 voting 用得上即可）：
- 位置约束集合 PROV / LETTER / ALNUM / DIGIT / DF（字符下标集合，基于 charset 的分组）；
- 版式 L7、L8S、L8L 各位的约束与正则；
- match_layout(plate) → "L7" | "L8S" | "L8L" | None；
- vehicle_type_hint(color, layout_name) → "small" | "large" | "unknown"。
"""

from __future__ import annotations


def match_layout(plate: str) -> str | None:
    """车牌号符合哪个版式的正则，都不符合返回 None。"""
    raise NotImplementedError("plate_rules 尚未实现（员工 C，任务 C3）")


def vehicle_type_hint(color: str, layout_name: str | None) -> str:
    """按 5.8 的表由底色与版式给出车型提示。"""
    raise NotImplementedError("plate_rules 尚未实现（员工 C，任务 C3）")
