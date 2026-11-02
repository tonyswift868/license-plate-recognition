"""Compare trace-only text fallback policies without changing auto-pass decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from lpr.config import ALGORITHM_DIR, load_config

from lpr_tools.eval import agree_threshold
from lpr_tools.eval.common import effective_config_info
from lpr_tools.eval.index_samples import (
    FIXED_HARD_MANIFEST,
    FIXED_VAL_MANIFEST,
    fixed_manifest_samples,
    path_key,
)

ROOT = Path(__file__).resolve().parents[3]
INDEX_DEFAULT = ALGORITHM_DIR / "data/index/ccpd_index.csv"
RULES = ("baseline", "first_candidate_reader", "max_reader", "agreement_then_reader")
RULE_TEXT = {
    "baseline": "现有 agree_threshold.replay_record 完整回放。",
    "first_candidate_reader": "仅候选 index 0；按 rect 遍历顺序选首个 valid reader。",
    "max_reader": "全候选/rect 的 valid reader 中选 confidence 最大者；同分保留 trace 顺序。",
    "agreement_then_reader": "先在同 rect 双路线均 valid 且 plate 相同的项中选 min(conf) 最大者；同分保留 trace 顺序；没有同文 pair 时退到 max_reader。",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity_verified(identity):
    return (identity.get("identity_migration", {}).get("verified") is True
            or identity.get("verified") is True
            or identity.get("audit", {}).get("verified") is True)


def _route(item):
    return bool(item) and item.get("valid", True) and item.get("plate") is not None


def _reader_choices(record):
    for rect_pos, rect in enumerate(record.get("rects", [])):
        reader = rect.get("reader")
        if _route(reader):
            yield rect_pos, rect, reader


def select_reader(record, rule):
    """Return selection descriptor; this function deliberately never reads truth."""
    choices = list(_reader_choices(record))
    if rule == "first_candidate_reader":
        return next(({"plate": r["plate"], "confidence": r["confidence"],
                      "selection_source": f"reader:candidate0:rect{pos}"}
                     for pos, rect, r in choices if rect.get("candidate_index") == 0), None)
    if rule == "max_reader":
        if not choices:
            return None
        pos, rect, reader = max(choices, key=lambda x: float(x[2]["confidence"]))
        return {"plate": reader["plate"], "confidence": reader["confidence"],
                "selection_source": f"reader:candidate{rect.get('candidate_index')}:rect{pos}"}
    if rule == "agreement_then_reader":
        pairs = []
        for pos, rect, reader in choices:
            segment = rect.get("segment")
            if _route(segment) and segment["plate"] == reader["plate"]:
                pairs.append((pos, rect, reader, min(float(segment["confidence"]),
                                                       float(reader["confidence"]))))
        if pairs:
            pos, rect, reader, pair_conf = max(pairs, key=lambda x: x[3])
            return {"plate": reader["plate"], "confidence": pair_conf,
                    "selection_source": f"same_rect_agreement:candidate{rect.get('candidate_index')}:rect{pos}",
                    "confidence_note": "min(segment,reader); agreement score, not calibrated probability"}
        return select_reader(record, "max_reader")
    raise ValueError(f"unknown rule: {rule}")


def apply_rule(record, baseline, rule):
    if rule == "baseline":
        return {**baseline, "selection_source": "baseline", "changed": False}
    if baseline.get("status") == "OK" or baseline.get("status") in {
            "BAD_INPUT", "NO_PLATE", "UNSUPPORTED"}:
        return {**baseline, "selection_source": "baseline_preserved", "changed": False}
    selected = select_reader(record, rule)
    if selected is None:
        return {**baseline, "selection_source": "baseline_no_legal_reader", "changed": False}
    changed = selected["plate"] != baseline.get("plate")
    result = {**baseline, "plate": selected["plate"], "confidence": selected["confidence"],
              "selection_source": selected["selection_source"], "changed": changed}
    if "confidence_note" in selected:
        result["confidence_note"] = selected["confidence_note"]
    return result


def compare_record(record, params):
    baseline = agree_threshold.replay_record(record, **params)
    return {rule: apply_rule(record, baseline, rule) for rule in RULES}


def _groups(item, meta):
    keys = ["all", item["subset"]]
    if item["split"] == "val":
        keys.append("val")
        if item["subset"] == "ccpd_green":
            keys.append("valgreen")
    else:
        keys.append("hard")
    if item["subset"] == "ccpd_db" and meta.get("brightness") is not None:
        try:
            if float(meta["brightness"]) <= 19:
                keys.append("db_brightness_le_19")
        except (TypeError, ValueError):
            pass
    if item["subset"] in {"ccpd_tilt", "ccpd_rotate"}:
        keys.append("tilt_plus_rotate")
    return keys


def score(rows, rule):
    grouped = {}
    for item in rows:
        baseline = item["results"]["baseline"]
        out = item["results"][rule]
        truth = item["truth"]
        for key in _groups(item, item["metadata"]):
            v = grouped.setdefault(key, {"n": 0, "whole_plate_correct": 0, "auto_pass": 0,
                "auto_pass_correct": 0, "rescued_non_ok": 0, "regressed_non_ok": 0,
                "non_ok_text_changed": 0, "cannot_pure_fallback_rescue": 0})
            v["n"] += 1
            correct = out.get("plate") == truth
            base_correct = baseline.get("plate") == truth
            v["whole_plate_correct"] += int(correct)
            v["auto_pass"] += int(out.get("status") == "OK")
            v["auto_pass_correct"] += int(out.get("status") == "OK" and correct)
            v["cannot_pure_fallback_rescue"] += int(
                baseline.get("status") in {"BAD_INPUT", "NO_PLATE", "UNSUPPORTED"})
            if rule != "baseline" and baseline.get("status") != "OK":
                v["non_ok_text_changed"] += int(out.get("changed", False))
                v["rescued_non_ok"] += int(not base_correct and correct)
                v["regressed_non_ok"] += int(base_correct and not correct)
    for v in grouped.values():
        n, passed = v["n"], v["auto_pass"]
        v["whole_plate_accuracy"] = v["whole_plate_correct"] / n if n else None
        v["auto_pass_rate"] = passed / n if n else None
        v["auto_pass_precision"] = v["auto_pass_correct"] / passed if passed else None
    return grouped


def _validate(trace: Path, identity_path: Path, index_path: Path):
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("trace_sha256") != _sha(trace):
        raise ValueError("trace content SHA differs from identity")
    if identity.get("trace_rows") != 3100:
        raise ValueError("trace identity must certify exactly 3100 rows")
    pipeline_path = ROOT / "algorithm/lpr/pipeline.py"
    if identity.get("pipeline_sha256") != _sha(pipeline_path):
        raise ValueError("pipeline SHA differs from trace identity")
    if not _identity_verified(identity):
        raise ValueError("trace identity is not verified")
    models = identity.get("models")
    if not isinstance(models, dict) or len(models) != 3:
        raise ValueError("trace identity lacks the three model hashes")
    for model in models.values():
        path = Path(model["path"])
        if not path.is_file() or _sha(path) != model.get("sha256"):
            raise ValueError(f"model identity mismatch: {path}")
    config_path = Path(identity.get("config", ""))
    if not config_path.is_file():
        raise ValueError("trace identity config file missing")
    cfg = load_config(config_path)
    if effective_config_info(cfg, config_path)["effective_config_sha256"] != identity.get("effective_config_sha256"):
        raise ValueError("effective config SHA differs from trace identity")
    for field, model in models.items():
        configured = Path(cfg.resolve_path(getattr(cfg.runtime, field)))
        recorded = Path(model["path"])
        if configured.resolve() != recorded.resolve():
            raise ValueError(f"configured model path differs from trace identity: {field}")
    model_manifest = ROOT / "algorithm/models/manifest.json"
    if identity.get("model_manifest_sha256") != _sha(model_manifest):
        raise ValueError("model manifest SHA differs from trace identity")
    manifests = {str(p): _sha(p) for p in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST)}
    if identity.get("manifests") != manifests:
        raise ValueError("fixed manifest SHA differs from trace identity")
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 3100 or [r.get("split") for r in records] != ["val"] * 1000 + ["test"] * 2100:
        raise ValueError("trace must contain exactly ordered val1000 then devhard2100 rows")
    expected = []
    for manifest, purpose in ((FIXED_VAL_MANIFEST, "val"), (FIXED_HARD_MANIFEST, "dev-hard")):
        samples, _ = fixed_manifest_samples(index_path, manifest, purpose=purpose)
        expected.extend(samples)
    if len(expected) != len(records):
        raise ValueError("manifest/trace row count mismatch")
    for i, (record, sample) in enumerate(zip(records, expected, strict=False)):
        if record.get("input_error"):
            raise ValueError(f"trace contains failed input at row {i}: {record.get('input_error')}")
        if (path_key(record.get("path", "")) != path_key(sample.row.path)
                or record.get("truth") != sample.row.plate_no
                or record.get("subset") != sample.subset):
            raise ValueError(f"ordered path/label/subset mismatch at row {i}")
    return records, identity, expected


def validate_params(params):
    required = {"agree_min", "solo_seg", "solo_reader", "prefer"}
    if not isinstance(params, dict) or set(params) != required:
        raise ValueError(f"params JSON must contain exactly {sorted(required)}")
    for key in ("agree_min", "solo_seg", "solo_reader"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if params["prefer"] not in {"segment", "reader"}:
        raise ValueError("prefer must be 'segment' or 'reader'")
    return params


def run(trace: Path, params_path: Path, index_path: Path, out: Path):
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params = validate_params(params)
    records, identity, samples = _validate(trace, Path(str(trace) + ".identity.json")
                                           if Path(str(trace) + ".identity.json").is_file()
                                           else trace.with_name("trace_identity.json"), index_path)
    from lpr_tools.eval.ceiling import _load_index_fields
    needed = {path_key(sample.row.path) for sample in samples}
    index_fields = _load_index_fields(index_path, needed)
    out.mkdir(parents=True)
    detail = []
    for record, sample in zip(records, samples, strict=False):
        meta = index_fields[path_key(sample.row.path)]
        results = compare_record(record, params)
        detail.append({"path": record["path"], "truth": record["truth"], "split": "val" if record["split"] == "val" else "hard",
            "subset": record["subset"], "metadata": {"brightness": meta.get("brightness")}, "results": results})
    with (out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in detail:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    reports = {rule: score(detail, rule) for rule in RULES}
    ranking = sorted(RULES, key=lambda rule: (-reports[rule]["all"]["whole_plate_correct"], RULES.index(rule)))
    report = {"input_identity": identity, "params": params, "rules": RULE_TEXT,
              "scope": {"no_auto_pass_change": True,
                        "selection_uses_ground_truth": False,
                        "bad_input_no_plate_unsupported_never_replaced": True},
              "rule_order_tie_break": list(RULES), "selected_best": ranking[0],
              "scores": reports}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Trace-only fallback rule comparison", "", f"Selected by total whole-plate correct: **{ranking[0]}**.",
             "", "Only non-OK rows are eligible for text replacement; status is preserved, so auto-pass membership is unchanged.",
             "BAD_INPUT, NO_PLATE and UNSUPPORTED are never replaced. Ground truth is used for scoring only.", "",
             "## Rules", ""]
    lines.extend(f"- **{key}**: {RULE_TEXT[key]}" for key in RULES)
    lines += ["", "## Summary", "", "| rule | group | n | whole correct | pass correct | pass rate (%) | precision (%) | rescued non-OK | regressed non-OK | cannot rescue |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rule in RULES:
        for group in ("all", "val", "hard", "db_brightness_le_19", "tilt_plus_rotate", "valgreen"):
            if group not in reports[rule]:
                continue
            v = reports[rule][group]
            def pct(x):
                return "—" if x is None else f"{100*x:.2f}"
            lines.append(f"| {rule} | {group} | {v['n']} | {v['whole_plate_correct']} | {v['auto_pass_correct']} | {pct(v['auto_pass_rate'])} | {pct(v['auto_pass_precision'])} | {v['rescued_non_ok']} | {v['regressed_non_ok']} | {v['cannot_pure_fallback_rescue']} |")
    lines += ["", "Per-subset scores and full input identity are in `report.json`.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.trace, args.params, args.index, args.out)
    print(json.dumps({"selected_best": report["selected_best"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
