"""CPU-only safe-set scorer for the batch-12 province readers.

Scores the existing safe tune rows, the per-province holdout, and deterministic
31x30 province-swap rows. It never reads the fixed val/devhard manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_TMP = _ROOT / "algorithm" / "out" / "tmp" / "batch12-runtime"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_RUNTIME_TMP)
os.environ["TMP"] = str(_RUNTIME_TMP)


def _rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _quad(text: str):
    values = [float(value) for value in text.replace(",", " ").split()]
    if len(values) != 8:
        raise ValueError("quad must contain eight ordered coordinates")
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


def _image(path: Path, cv2, np):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError(f"cannot decode source image: {path}")
    return image


def _rectified(image, quad, width, cv2):
    import numpy as np
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, 79], [0, 79]])
    matrix = cv2.getPerspectiveTransform(np.float32(quad), target)
    result = cv2.warpPerspective(image, matrix, (width, 80))
    if result is None or result.size == 0:
        raise ValueError("empty GT rectification")
    return result


def _first_char_patch(normalizer, image, quad, full_label: str):
    """Preserve plate length/color context while preparing the province glyph."""
    return normalizer(image, quad, full_label)


def _metrics(rows):
    province_correct = sum(row["province_correct"] for row in rows)
    plate_correct = sum(row["plate_correct"] for row in rows if row["plate_correct"] is not None)
    plate_n = sum(row["plate_correct"] is not None for row in rows)
    count = len(rows)
    return {
        "n": count,
        "province_correct": province_correct,
        "province_accuracy": province_correct / count if count else None,
        "correct": plate_correct if plate_n else province_correct,
        "plate_n": plate_n,
        "plate_correct": plate_correct if plate_n else None,
        "plate_accuracy": plate_correct / plate_n if plate_n else None,
        "predicted_皖": sum(row["predicted_province"] == "皖" for row in rows),
        "read_as_皖_wrong": sum(row["predicted_province"] == "皖" and not row["province_correct"] for row in rows),
    }


def score(kind: str, model_path: Path, data_dir: Path, output: Path | None = None):
    import cv2
    import numpy as np
    import torch
    from lpr.charset import CHARSET, CHAR_TO_INDEX, PROVINCES
    from lpr.boxcrop import crop_box
    from lpr.recognize import load_box_reader, load_classifier, load_plate_reader
    from lpr.recognize.crnn import CrnnNet, CrnnReader
    from lpr.recognize.cnn_model import CharCNN
    from lpr.recognize.box_reader import BoxPlateReader, model_from_checkpoint
    from lpr_tools.train.province_synth import normalized_first_char, replace_province

    if kind not in {"box", "crnn", "char"}:
        raise ValueError("kind must be box, crnn, or char")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and saved.get("format") == "province-reader-training-v1":
        if saved.get("kind") != kind:
            raise ValueError("training checkpoint kind does not match --kind")
        weights = saved.get("model_state")
    else:
        weights = saved
    if kind == "box":
        if weights is saved:
            reader = load_box_reader(model_path)
        else:
            model = model_from_checkpoint(weights)
            reader = BoxPlateReader("batch12-selection", model,
                                    temperature=weights["temperature"],
                                    box_pad=tuple(weights["box_pad"]))
        pad_x, pad_y = reader.box_pad
        province_reader = None
    elif kind == "crnn":
        if weights is saved:
            reader = load_plate_reader(model_path)
        else:
            state = dict(weights)
            temperature = state.pop("ctc_temperature", torch.tensor(1.0))
            if temperature.ndim != 0 or not torch.isfinite(temperature):
                raise ValueError("CRNN checkpoint temperature must be a finite scalar")
            model = CrnnNet()
            model.load_state_dict(state, strict=True)
            reader = CrnnReader("batch12-selection", model, temperature=float(temperature.item()))
        province_reader = None
    else:
        if weights is saved:
            province_reader = load_classifier("cnn", model_path)
        else:
            model = CharCNN()
            model.load_state_dict(weights, strict=True)
            from lpr.recognize.cnn import CnnClassifier
            province_reader = CnnClassifier("batch12-selection", model)
        reader = None

    root = Path(data_dir)
    tune_path = root / "tune.csv"
    if not tune_path.exists():
        tune_path = root.parent / "batch11" / "tune.csv"
    sources = []
    for filename, mode in ((tune_path, "real"), (root / "province_holdout.csv", "real"),
                           (root / "synth_eval.csv", "synthetic")):
        if filename.exists():
            for raw in _rows(filename):
                row = dict(raw)
                row["_mode"] = mode
                row["_panel"] = ("synthetic" if filename.name == "synth_eval.csv" else
                                 "province_holdout" if filename.name == "province_holdout.csv" else "tune")
                row["_source_file"] = str(filename.resolve())
                row["target_province"] = row.get("target_province") or row.get("province") or row["label"][0]
                sources.append(row)
    if not sources:
        raise FileNotFoundError(f"no tune/holdout/synth evaluation rows in {root}")

    grouped = defaultdict(list)
    all_rows = []
    for row in sources:
        image = _image(Path(row["path"]), cv2, np)
        quad = _quad(row["quad"])
        truth = row["label"]
        if row["_mode"] == "synthetic":
            seed = int(row.get("seed") or 0)
            image, truth, _meta = replace_province(
                image, quad, truth, row["target_province"], np.random.default_rng(seed))
        if not truth or truth[0] not in CHAR_TO_INDEX:
            raise ValueError(f"illegal province label: {truth!r}")
        if kind == "char":
            patch = _first_char_patch(normalized_first_char, image, quad, truth)
            if not isinstance(patch, np.ndarray) or patch.dtype != np.uint8 or patch.shape != (32, 32):
                raise ValueError("normalized_first_char must return uint8 32 x 32")
            probabilities = province_reader.predict_proba(patch[None, ...])[0]
            province_indices = [CHAR_TO_INDEX[c] for c in PROVINCES]
            province_pred = CHARSET[province_indices[int(np.argmax(probabilities[province_indices]))]]
            plate_pred = None
        else:
            if kind == "box":
                xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                bbox = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                crop = crop_box(image, bbox, pad_x, pad_y)
                plate_pred, confidences = reader.read(crop)
            else:
                width = 276 if len(truth) == 8 else 252
                crop = _rectified(image, quad, width, cv2)
                plate_pred, confidences = reader.read(crop)
            province_pred = plate_pred[0] if plate_pred else ""
        record = {
            "path": row["path"], "source": row["_source_file"], "mode": row["_mode"],
            "panel": row["_panel"],
            "province": truth[0], "predicted_province": province_pred,
            "province_correct": province_pred == truth[0],
            "plate_correct": None if plate_pred is None else plate_pred == truth,
            "prediction": plate_pred, "subset": row.get("subset"),
            "brightness": row.get("brightness"), "synthetic_target": row.get("target_province") if row["_mode"] == "synthetic" else None,
        }
        all_rows.append(record)
        grouped[row["_mode"]].append(record)
        grouped["panel/" + row["_panel"]].append(record)
        grouped["panel_province/" + row["_panel"] + "/" + truth[0]].append(record)
        grouped["province/" + truth[0]].append(record)
        if record["subset"]:
            grouped["subset/" + record["subset"]].append(record)

    overall = _metrics(all_rows)
    metrics = {"overall": overall, **{key: _metrics(rows) for key, rows in sorted(grouped.items())}}
    plate_accuracy = overall["plate_accuracy"]
    selection_score = overall["province_accuracy"] if plate_accuracy is None else (
        overall["province_accuracy"] + plate_accuracy) / 2
    result = {
        "schema": "batch12-province-reader-score-v1", "kind": kind,
        "model": str(model_path.resolve()), "data_dir": str(root.resolve()),
        "selection_score": selection_score,
        "province_accuracy": overall["province_accuracy"],
        "plate_accuracy": plate_accuracy,
        "correct": overall["correct"], "n": overall["n"],
        "metrics": metrics,
        "read_as_皖_wrong": overall["read_as_皖_wrong"],
        "selection_rule": "safe tune + province holdout + deterministic synthetic panel; plate readers average province and full-plate accuracy; char CNN uses province accuracy restricted to PROVINCE_IDX",
        "sources": sorted({row["source"] for row in all_rows}),
        "excluded_fixed_val_dev": True,
        "records": all_rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("box", "crnn", "char"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("algorithm/out/data/batch12"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = score(args.kind, args.model, args.data_dir, args.out)
    print(json.dumps({key: result[key] for key in (
        "selection_score", "province_accuracy", "plate_accuracy", "correct", "n",
        "kind", "read_as_皖_wrong")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
