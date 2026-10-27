"""Offline evaluation for direct detection-box plate recognition.

This tool uses fixed tune/validation manifests and the production candidate
ordering, while reading each candidate's axis-aligned bbox without rectifying
the image. It never changes the runtime pipeline or model settings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lpr_tools.train.box_data import bbox_from_quad
from lpr_tools.train.box_data import jitter_bbox as apply_box_jitter
from lpr_tools.train.box_data import read_jitter_samples as measured_box_deltas
from lpr_tools.train.box_data import sample_jitter

ROOT = Path(__file__).resolve().parents[3]
ALGORITHM = ROOT / "algorithm"
INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
VAL_MANIFEST = Path(r"D:\keshe2\algorithm\data\index\val1000_v1.csv")
DEV_MANIFEST = Path(r"D:\keshe2\algorithm\data\index\devhard_v1.csv")
CEILING_REPORT = Path(r"D:\keshe2\algorithm\out\tmp_arch\ceiling_v2\report.json")
MODEL = ALGORITHM / "models" / "box_reader_v1.pt"
JITTER_SOURCE = ROOT / "algorithm" / "out" / "receipts" / "batch9" / "geometry" / "train_per_image.jsonl"
TUNE_SUBSETS = {"db", "blur", "tilt", "rotate", "fn", "challenge", "weather"}


def legal_prediction(text: str, confidences: Iterable[float]) -> tuple[str, float] | None:
    """Keep only frozen plate layouts; rank legal reads by minimum char confidence."""
    from lpr.plate_rules import match_layout

    confs = tuple(float(v) for v in confidences)
    if len(text) not in (7, 8) or len(confs) != len(text):
        return None
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in confs):
        return None
    if match_layout(text) is None:
        return None
    return text, min(confs)


def choose_highest_legal(reads: Iterable[tuple[str, Iterable[float]]]
                         ) -> tuple[str, float] | None:
    """Choose the first candidate among ties at the maximum legal confidence."""
    best = None
    for text, confs in reads:
        item = legal_prediction(text, confs)
        if item is not None and (best is None or item[1] > best[1]):
            best = item
    return best


def _resolve_image(path: str, source: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else source.resolve().parent / candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        rows = list(reader)
    if len(rows) != 2100:
        raise ValueError(f"safe tune index must contain 2,100 origins; got {len(rows)}")
    short_names = TUNE_SUBSETS
    folds, paths, hashes, subsets = defaultdict(int), set(), set(), defaultdict(int)
    from lpr.plate_rules import match_layout

    for row in rows:
        if row["subset"] in short_names:
            row["subset"] = "ccpd_" + row["subset"]
        subset = row["subset"].removeprefix("ccpd_")
        if subset not in TUNE_SUBSETS:
            raise ValueError(f"unexpected safe tune subset: {row['subset']!r}")
        fold = row["calibration_fold"]
        if fold not in {"fit", "report"}:
            raise ValueError(f"unexpected safe tune calibration fold: {fold!r}")
        resolved = _resolve_image(row["path"], path)
        path_key = os.path.normcase(os.path.normpath(str(resolved)))
        digest = row["sha256"].lower()
        if path_key in paths or digest in hashes:
            raise ValueError("safe tune must contain unique source paths and content hashes")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("safe tune rows require SHA-256 source identities")
        if row["calibration_fold"] not in {"fit", "report"}:
            raise ValueError("safe tune calibration fold must be fit or report")
        if match_layout(row["label"]) is None:
            raise ValueError(f"invalid safe tune plate label: {row['label']!r}")
        bbox_from_quad(row["quad"])
        try:
            int(row["brightness"])
        except ValueError as exc:
            raise ValueError("safe tune brightness must be an integer") from exc
        paths.add(path_key)
        hashes.add(digest)
        folds[fold] += 1
        subsets[subset] += 1
    if folds != {"fit": 1050, "report": 1050}:
        raise ValueError(f"safe tune must contain 1,050 origins per fold; got {dict(folds)}")
    if subsets != {name: 300 for name in TUNE_SUBSETS}:
        raise ValueError(f"safe tune must contain 300 origins per hard subset; got {dict(subsets)}")
    return rows


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[row["subset"]].append(row)
        if row["subset"] == "ccpd_db" and row["brightness"] is not None and row["brightness"] <= 19:
            grouped["ccpd_db_brightness_le_19"].append(row)
        if row["subset"] in {"ccpd_tilt", "ccpd_rotate"}:
            grouped["ccpd_tilt_plus_rotate"].append(row)
    return grouped


def _pack(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    return {"correct": correct, "n": len(rows),
            "accuracy": correct / len(rows) if rows else None}


def _jitter_delta(subset: str, rng: random.Random,
                  samples: dict[str, list[tuple[float, float, float, float]]]
                  ) -> tuple[float, float, float, float]:
    return sample_jitter(rng, subset.removeprefix("ccpd_"), samples)


def _read_tune(reader, rows: list[dict[str, str]], source: Path,
               pad: tuple[float, float], seed: int,
               jitter_samples: dict[str, list[tuple[float, float, float, float]]],
               timing: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from lpr.boxcrop import crop_box
    from lpr.io_utils import imread

    rng = random.Random(seed)
    gt_records, jitter_records = [], []
    for raw in rows:
        image_path = _resolve_image(raw["path"], source)
        image = imread(image_path)
        gt = bbox_from_quad(raw["quad"])
        for mode, bbox, output in (
                ("gt_bbox", gt, gt_records),
                ("measured_jitter", apply_box_jitter(gt, _jitter_delta(raw["subset"], rng, jitter_samples)), jitter_records)):
            crop = crop_box(image, bbox, *pad)
            if crop.size == 0:
                prediction = None
            else:
                started = time.perf_counter_ns()
                text, confs = reader.read(crop)
                timing.append((time.perf_counter_ns() - started) / 1e6)
                prediction = legal_prediction(text, confs)
            output.append({"path": raw["path"], "subset": raw["subset"],
                           "brightness": int(raw["brightness"]) if raw["brightness"].isdigit() else None,
                           "truth": raw["label"], "prediction": prediction[0] if prediction else "",
                           "confidence": prediction[1] if prediction else None,
                           "correct": bool(prediction and prediction[0] == raw["label"])})
    return gt_records, jitter_records


def _fixed_rows(index_path: Path, manifest_path: Path, purpose: str) -> list[dict[str, Any]]:
    from lpr_tools.data.ccpd import parse_ccpd_name
    from lpr_tools.eval.index_samples import fixed_manifest_samples
    from lpr_tools.data.manifest import image_path

    samples, _ = fixed_manifest_samples(index_path, manifest_path, purpose=purpose)
    result = []
    for sample in samples:
        result.append({"path": str(image_path(sample.row, manifest_path).resolve()),
                       "label": sample.row.plate_no, "subset": sample.subset,
                       "brightness": parse_ccpd_name(sample.row.path).brightness,
                       "quad": " ".join(str(v) for point in sample.row.quad for v in point)})
    return result


def _read_actual(reader, image_rows: list[dict[str, Any]], cfg, corner_detector,
                 pad: tuple[float, float], timing: list[float]) -> list[dict[str, Any]]:
    from lpr.boxcrop import crop_box
    from lpr.io_utils import imread
    from lpr.locate import locate
    from lpr.pipeline import _in_roi, _merge_candidates
    from lpr.preprocess import preprocess

    records = []
    for row in image_rows:
        image = imread(Path(row["path"]))
        pre = preprocess(image, cfg.preprocess)
        classic = locate(pre, cfg.locate, 3)
        net = [candidate for candidate in corner_detector.detect(image, 3)
               if _in_roi(candidate, cfg.lane_roi(None))]
        candidates = _merge_candidates(net, classic, 3)[:3]
        reads = []
        for rank, candidate in enumerate(candidates):
            crop = crop_box(image, candidate.bbox, *pad)
            if crop.size == 0:
                continue
            started = time.perf_counter_ns()
            text, confs = reader.read(crop)
            timing.append((time.perf_counter_ns() - started) / 1e6)
            legal = legal_prediction(text, confs)
            if legal is not None:
                reads.append({"rank": rank, "text": legal[0], "confidence": legal[1]})
        selected = max(reads, key=lambda item: item["confidence"], default=None)
        records.append({"path": row["path"], "subset": row["subset"], "truth": row["label"],
                        "brightness": row["brightness"], "candidate_count": len(candidates),
                        "legal_read_count": len(reads),
                        "prediction": selected["text"] if selected else "",
                        "selected_rank": selected["rank"] if selected else None,
                        "confidence": selected["confidence"] if selected else None,
                        "correct": bool(selected and selected["text"] == row["label"])})
    return records


def _compare_ceiling(path: Path, split: str, group: str,
                     route: str = "actual/crnn") -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        baseline = data["groups"][split][group][route]
        return {"accuracy": baseline["accuracy"], "correct": baseline["correct"], "n": baseline["n"]}
    except (KeyError, TypeError):
        return None


def run(tune_csv: Path, output: Path, *, config_path: Path | None = None,
        model_path: Path = MODEL, index_path: Path = INDEX,
        val_manifest: Path = VAL_MANIFEST, dev_manifest: Path = DEV_MANIFEST,
        ceiling_path: Path = CEILING_REPORT, jitter_path: Path = JITTER_SOURCE, seed: int = 7319,
        include_actual: bool = True) -> dict[str, Any]:
    import torch
    import cv2
    from lpr.config import load_config
    from lpr.recognize import load_box_reader, load_corner_detector

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg = load_config(config_path)
    corner_path = cfg.resolve_path(cfg.runtime.corner_model_path)
    corner_detector = load_corner_detector(corner_path)
    reader = load_box_reader(model_path)
    pad = tuple(reader.box_pad)
    if len(pad) != 2 or min(pad) < 0:
        raise ValueError("model box_pad must be a pair of nonnegative values")

    timings: list[float] = []
    tune = _read_csv(tune_csv)
    jitter_samples = measured_box_deltas(jitter_path)
    tune_gt, tune_jitter = _read_tune(reader, tune, tune_csv, pad, seed, jitter_samples, timings)
    result: dict[str, Any] = {
        "schema": "box-reader-eval-v1", "model": str(model_path.resolve()),
        "box_pad": list(pad), "seed": seed,
        "jitter_source": {"path": str(jitter_path.resolve()),
                          "sha256": hashlib.sha256(jitter_path.read_bytes()).hexdigest(),
                          "matched_train_counts": {k: len(v) for k, v in sorted(jitter_samples.items())}},
        "tune": {mode: {group: _pack(rows) for group, rows in _groups(records).items()}
                 for mode, records in (("gt_bbox", tune_gt), ("measured_jitter", tune_jitter))},
        "actual": {}, "timing": {},
        "scope": "fixed manifests; exact full-plate text; legal read ranked by minimum character confidence; no rectification"}
    actual_traces = []
    if include_actual:
        for split, manifest, purpose in (("val", val_manifest, "val"),
                                         ("devhard", dev_manifest, "dev-hard")):
            rows = _fixed_rows(index_path, manifest, purpose)
            records = _read_actual(reader, rows, cfg, corner_detector, pad, timings)
            actual_traces.extend((split, record) for record in records)
            result["actual"][split] = {
                group: _actual_group_summary(items, ceiling_path, split, group)
                for group, items in _groups(records).items()}
    if timings:
        ordered = sorted(timings)
        result["timing"] = {"read_calls": len(ordered), "mean_ms": statistics.fmean(ordered),
                            "p95_ms": ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)],
                            "max_ms": ordered[-1], "over_10ms": sum(value > 10 for value in ordered),
                            "torch_threads": torch.get_num_threads()}
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix(".jsonl")
    records = [("tune/gt_bbox", tune_gt), ("tune/measured_jitter", tune_jitter)]
    with trace.open("w", encoding="utf-8") as stream:
        for mode, values in records:
            for item in values:
                stream.write(json.dumps({"evaluation": mode, **item}, ensure_ascii=False) + "\n")
        for split, item in actual_traces:
            stream.write(json.dumps({"evaluation": f"actual/{split}", **item}, ensure_ascii=False) + "\n")
    result["records_jsonl"] = str(trace.resolve())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _actual_group_summary(items: list[dict[str, Any]], ceiling_path: Path,
                          split: str, group: str) -> dict[str, Any]:
    summary = _pack(items)
    baseline = _compare_ceiling(ceiling_path, split, group)
    summary["ceiling_crnn_actual"] = baseline
    summary["ceiling_crnn_gt"] = _compare_ceiling(ceiling_path, split, group, "ground_truth/crnn")
    summary["delta_vs_ceiling"] = (summary["accuracy"] - baseline["accuracy"]
                                   if baseline is not None and summary["accuracy"] is not None else None)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--val", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--dev-hard", type=Path, default=DEV_MANIFEST)
    parser.add_argument("--ceiling", type=Path, default=CEILING_REPORT)
    parser.add_argument("--jitter-jsonl", type=Path, default=JITTER_SOURCE)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--tune-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.tune_csv, args.out, config_path=args.config, model_path=args.model,
                 index_path=args.index, val_manifest=args.val, dev_manifest=args.dev_hard,
                 ceiling_path=args.ceiling, jitter_path=args.jitter_jsonl,
                 seed=args.seed, include_actual=not args.tune_only)
    print(f"report={args.out.resolve()} tune_gt={report['tune']['gt_bbox']['all']['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
