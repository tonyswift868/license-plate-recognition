"""Measure whole-plate CRNN on untouched real validation crops through read()."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lpr.plate_rules import match_layout
from lpr.recognize import file_sha256, load_plate_reader



def evaluate(data: Path, model: Path, out: Path) -> dict:
    with (data / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if Path(r["file"]).parts[0] == "val"]
    if not rows or any(r["source"] == "font_synthetic" for r in rows):
        raise ValueError("real validation rows are required; synthetic validation is forbidden")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reader = load_plate_reader(model)
    groups = defaultdict(lambda: {"total": 0, "correct": 0, "province_correct": 0,
                                  "char_correct": 0, "char_total": 0,
                                  "auto_pass": 0, "auto_correct": 0})
    latencies = []
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "source", "truth", "prediction",
                                                   "confidence", "char_confidences", "read_ms"])
        writer.writeheader()
        for index, row in enumerate(rows):
            image = cv2.imdecode(np.fromfile(data / row["file"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable validation crop: {row['file']}")
            if index == 0:
                for _ in range(5):
                    reader.read(image)
            began = time.perf_counter()
            text, confs = reader.read(image)
            ms = (time.perf_counter() - began) * 1000
            latencies.append(ms)
            truth = row["plate_no"]
            confidence = min(confs) if confs else 0.0
            auto = bool(match_layout(text)) and confidence >= .80
            source = "ground_truth" if row["source"] == "ground_truth" else "pipeline_candidate"
            for group in ("all", source, f"layout_{len(truth)}", f"province_{truth[0]}"):
                values = groups[group]
                values["total"] += 1
                values["correct"] += text == truth
                values["province_correct"] += bool(text) and text[0] == truth[0]
                values["char_correct"] += sum(a == b for a, b in zip(text, truth))
                values["char_total"] += max(len(text), len(truth))
                values["auto_pass"] += auto
                values["auto_correct"] += auto and text == truth
            writer.writerow({"file": row["file"], "source": row["source"], "truth": truth,
                             "prediction": text, "confidence": confidence,
                             "char_confidences": json.dumps(confs), "read_ms": ms})
    for values in groups.values():
        values.update(plate_accuracy=values["correct"] / values["total"],
                      province_accuracy=values["province_correct"] / values["total"],
                      char_accuracy=values["char_correct"] / values["char_total"],
                      auto_pass_rate=values["auto_pass"] / values["total"],
                      auto_pass_precision=values["auto_correct"] / values["auto_pass"] if values["auto_pass"] else None)
    result = {"model_path": str(model.resolve()), "model_sha256": file_sha256(model),
              "labels_sha256": file_sha256(data / "labels.csv"), "split": "val", "groups": dict(groups),
              "threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
              "read_latency_ms": {"n": len(latencies), "warmup": 5, "mean": float(np.mean(latencies)),
                                  "p50": float(np.percentile(latencies, 50)),
                                  "p95": float(np.percentile(latencies, 95)), "max": max(latencies)},
              "timing_scope": "single read() including image preprocessing and CTC; excludes image I/O",
              "confidence_scope": "crop-level valid format and minimum character confidence >= 0.80; not full-pipeline auto-pass",
              "char_metric": "position matches / sum(max(truth length, prediction length))"}
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data, args.model, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
