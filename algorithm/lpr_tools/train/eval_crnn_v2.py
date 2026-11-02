"""Evaluate CRNN v2 plate crops via the production CrnnReader.read() path."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lpr.recognize.crnn import CrnnNet, CrnnReader

CONFIDENCE_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
SUBSETS = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")


def _resolve_crop(index_path: Path, crop: str) -> Path:
    path = Path(crop)
    if path.is_absolute():
        return path
    candidate = index_path.parent / path
    return candidate if candidate.exists() else Path.cwd() / path


def load_index(index_path: Path, fold: str | None = None,
               gt_only: bool = False) -> list[dict[str, str]]:
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"file", "orig_path", "source", "layout", "plate_no", "crop_path",
                    "subset", "brightness", "partition", "calibration_fold"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"index must include columns: {sorted(required)}")
        rows = list(reader)
    if fold is not None:
        rows = [row for row in rows if row["calibration_fold"] == fold]
    if gt_only:
        rows = [row for row in rows if _is_gt(row["source"])]
    if not rows:
        raise ValueError("index selection contains no rows")
    for row in rows:
        if not row["plate_no"] or row["layout"] not in ("7", "8"):
            raise ValueError(f"invalid plate label/layout for {row['file']}")
        if len(row["plate_no"]) != int(row["layout"]):
            raise ValueError(f"label length disagrees with layout for {row['file']}")
    return rows


def _is_gt(source: str) -> bool:
    return source == "ground_truth" or source.endswith(":ground_truth")


def _groups(row: dict[str, str]) -> list[str]:
    gt = _is_gt(row["source"])
    groups = ["overall", "GT" if gt else "pipeline"]
    subset = row["subset"].lower()
    if gt and subset in SUBSETS:
        groups.append(f"{subset}_GT")
    if gt and subset == "db" and row["brightness"].strip().isdigit() and int(row["brightness"]) <= 19:
        groups.append("db_dark_GT")
    return groups


def _make_accumulator() -> dict[str, Any]:
    return {"n": 0, "plate_correct": 0, "char_correct": 0, "char_total": 0,
            "conf_bins": [{"lower": lo, "upper": hi, "n": 0, "correct": 0,
                           "confidence_sum": 0.0}
                          for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:], strict=False)]}


def _confidence_bin(confidence: float) -> int:
    # Right-closed final bin includes confidence==1 exactly.
    return min(len(CONFIDENCE_EDGES) - 2,
               max(0, next((i for i, edge in enumerate(CONFIDENCE_EDGES[1:])
                           if confidence < edge), len(CONFIDENCE_EDGES) - 2)))


def _add(acc: dict[str, Any], truth: str, prediction: str, confidence: float) -> None:
    correct = prediction == truth
    acc["n"] += 1
    acc["plate_correct"] += int(correct)
    acc["char_correct"] += sum(a == b for a, b in zip(truth, prediction, strict=False))
    acc["char_total"] += max(len(truth), len(prediction))
    bucket = acc["conf_bins"][_confidence_bin(confidence)]
    bucket["n"] += 1
    bucket["correct"] += int(correct)
    bucket["confidence_sum"] += confidence


def _finish(acc: dict[str, Any]) -> dict[str, Any]:
    n = acc["n"]
    bins = []
    for item in acc["conf_bins"]:
        count = item["n"]
        bins.append({"range": [item["lower"], item["upper"]], "n": count,
                     "plate_accuracy": item["correct"] / count if count else None,
                     "mean_confidence": item["confidence_sum"] / count if count else None})
    return {"n": n, "plate_accuracy": acc["plate_correct"] / n if n else None,
            "char_accuracy": acc["char_correct"] / acc["char_total"] if acc["char_total"] else None,
            "confidence_bins": bins}


def evaluate(index_path: Path, model_path: Path, out_dir: Path, *,
             fold: str | None = None, gt_only: bool = False) -> dict[str, Any]:
    rows = load_index(index_path, fold=fold, gt_only=gt_only)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if "ctc_temperature" in state:
        temperature = float(state["ctc_temperature"].item())
        state = {key: value for key, value in state.items() if key != "ctc_temperature"}
    else:
        temperature = 1.0
    model = CrnnNet().cpu().eval()
    model.load_state_dict(state, strict=True)
    reader = CrnnReader("crnn_v2_eval", model, temperature=temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, dict[str, Any]] = defaultdict(_make_accumulator)
    for name in ("overall", "GT", "pipeline", *(f"{subset}_GT" for subset in SUBSETS),
                 "db_dark_GT"):
        totals[name]
    latencies: list[float] = []
    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "orig_path", "source", "subset",
                                                    "brightness", "partition", "calibration_fold",
                                                    "truth", "prediction", "confidence",
                                                    "char_confidences", "read_ms"])
        writer.writeheader()
        warmed = False
        for row in rows:
            crop_path = _resolve_crop(index_path, row["crop_path"])
            image = cv2.imdecode(np.fromfile(crop_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable crop: {crop_path}")
            if not warmed:
                for _ in range(5):
                    reader.read(image)
                warmed = True
            began = time.perf_counter()
            prediction, confidences = reader.read(image)
            read_ms = (time.perf_counter() - began) * 1000.0
            confidence = min(confidences) if confidences else 0.0
            truth = row["plate_no"]
            latencies.append(read_ms)
            for group in _groups(row):
                _add(totals[group], truth, prediction, confidence)
            writer.writerow({**{key: row.get(key, "") for key in
                                ("file", "orig_path", "source", "subset", "brightness",
                                 "partition", "calibration_fold")},
                             "truth": truth, "prediction": prediction, "confidence": confidence,
                             "char_confidences": json.dumps(confidences, ensure_ascii=False),
                             "read_ms": read_ms})
    groups = {name: _finish(value) for name, value in sorted(totals.items())}
    report = {"model": str(model_path.resolve()), "index": str(index_path.resolve()),
              "fold": fold, "gt_only": gt_only, "groups": groups,
              "read_latency_ms": {"n": len(latencies), "warmup_reads": 5,
                                  "mean": float(np.mean(latencies)),
                                  "p95": float(np.percentile(latencies, 95)),
                                  "max": max(latencies)},
              "timing_scope": "two-thread CPU CrnnReader.read(), includes plate_tensor and CTC decode, excludes file I/O",
              "confidence_definition": "minimum per-character confidence from CrnnReader.read(); empty prediction is 0",
              "confidence_edges": list(CONFIDENCE_EDGES),
              "temperature": temperature}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="tune or val index CSV")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold")
    parser.add_argument("--gt-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.index, args.model, args.out, fold=args.fold,
                              gt_only=args.gt_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
