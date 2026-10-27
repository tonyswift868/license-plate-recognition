"""Evaluate safe batch-12 CharCNN panels as GT first-glyph and full segment route."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_TMP = _ROOT / "algorithm" / "out" / "tmp" / "batch12-runtime"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_RUNTIME_TMP)
os.environ["TMP"] = str(_RUNTIME_TMP)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _segment_decode(rect, classifier, cfg, *, segmenter=None, decoder=None,
                    layout_getter=None, ranker=None):
    """Production segment/decode selection with injectable seams for contract tests."""
    if segmenter is None or decoder is None or layout_getter is None or ranker is None:
        from lpr import pipeline
        from lpr.segment import segment
        segmenter = segment
        decoder = pipeline.decode
        layout_getter = pipeline._layouts_to_try
        ranker = pipeline._rank
    best = None
    attempted = []
    for layout in layout_getter(rect):
        attempted.append(layout)
        try:
            segmented = segmenter(rect, layout, cfg.segment)
            if not segmented.ok:
                continue
            probabilities = classifier.predict_proba(segmented.chars)
            decision = decoder(probabilities, segmented.layout, rect.color, cfg.postprocess)
        except Exception as exc:
            return {"prediction": None, "status": "SEG_FAILED", "attempted_layouts": attempted,
                    "error": f"{type(exc).__name__}:{exc}"}
        if best is None or ranker(decision) > best[0]:
            best = (ranker(decision), decision)
        if getattr(decision.status, "value", decision.status) == "OK":
            break
    if best is None:
        return {"prediction": None, "status": "SEG_FAILED", "attempted_layouts": attempted,
                "error": None}
    decision = best[1]
    return {"prediction": decision.plate_no,
            "status": getattr(decision.status, "value", decision.status),
            "confidence": float(decision.confidence),
            "attempted_layouts": attempted, "error": None}


def _gt_rectified(image, quad, label: str, color: str, cv2, np):
    from lpr.types import PlateCandidate, RectifiedPlate

    width = 276 if len(label) == 8 else 252
    src = np.asarray(quad, dtype=np.float32)
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, 79], [0, 79]])
    homography = cv2.getPerspectiveTransform(src, target)
    warped = cv2.warpPerspective(image, homography, (width, 80))
    if warped is None or warped.size == 0:
        raise ValueError("empty ground-truth quad warp")
    xs, ys = src[:, 0], src[:, 1]
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    candidate = PlateCandidate([tuple(map(float, point)) for point in src], bbox, color, 1.0, "gt")
    return RectifiedPlate(warped, len(label), color, homography, candidate)


def _segment_record(base_record, truth: str, segment_read: dict):
    plate_pred = segment_read["prediction"]
    segment_province = plate_pred[0] if plate_pred else None
    return {
        **base_record, **segment_read,
        "predicted_province": segment_province,
        "province_correct": segment_province == truth[0],
        # A segmentation failure remains in the full panel denominator.
        "plate_correct": plate_pred == truth,
    }


def _sources(data_dir: Path):
    from lpr_tools.train.score_province_readers import _rows

    tune_path = data_dir / "tune.csv"
    if not tune_path.exists():
        tune_path = data_dir.parent / "batch11" / "tune.csv"
    for filename, panel in ((tune_path, "tune"),
                            (data_dir / "province_holdout.csv", "province_holdout"),
                            (data_dir / "synth_eval.csv", "synthetic")):
        if not filename.is_file():
            continue
        for raw in _rows(filename):
            row = dict(raw)
            row["_panel"] = panel
            row["_source_file"] = str(filename.resolve())
            row["target_province"] = row.get("target_province") or row.get("province") or row["label"][0]
            yield row


def _summarize(records):
    from lpr_tools.train.score_province_readers import _metrics

    groups = defaultdict(list)
    for row in records:
        groups["overall"].append(row)
        panel, province, subset = row["panel"], row["province"], row.get("subset")
        groups[f"panel/{panel}"].append(row)
        groups[f"province/{province}"].append(row)
        groups[f"panel_province/{panel}/{province}"].append(row)
        if subset:
            groups[f"subset/{subset}"].append(row)
            groups[f"panel_subset/{panel}/{subset}"].append(row)
    return {name: _metrics(rows) for name, rows in sorted(groups.items())}


def evaluate(model_path: Path, data_dir: Path, config_path: Path, output: Path | None = None):
    import cv2
    import numpy as np
    import torch
    from lpr.charset import CHARSET, CHAR_TO_INDEX, PROVINCES, PROVINCE_IDX
    from lpr.config import load_config
    from lpr.recognize import load_classifier
    from lpr.recognize.cnn_model import CharCNN
    from lpr.recognize.cnn import CnnClassifier
    from lpr_tools.train.province_synth import normalized_first_char, replace_province
    from lpr_tools.train.score_province_readers import _image, _metrics, _quad, _first_char_patch

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model_path, data_dir, config_path = (Path(model_path).resolve(), Path(data_dir).resolve(),
                                         Path(config_path).resolve())
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and saved.get("format") == "province-reader-training-v1":
        if saved.get("kind") != "char":
            raise ValueError("training checkpoint kind does not match CharCNN evaluator")
        model = CharCNN()
        model.load_state_dict(saved["model_state"], strict=True)
        classifier = CnnClassifier("batch12-char-eval", model)
    else:
        classifier = load_classifier("cnn", model_path)
    cfg = load_config(config_path)
    province_indices = np.asarray(PROVINCE_IDX, dtype=np.int64)
    first_records, segment_records = [], []
    for row in _sources(data_dir):
        image_path = Path(row["path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if row.get("sha256") and _sha256(image_path).lower() != row["sha256"].lower():
            raise ValueError(f"safe panel source SHA mismatch: {image_path}")
        image = _image(image_path, cv2, np)
        quad = _quad(row["quad"])
        truth = row["label"]
        if row["_panel"] == "synthetic":
            image, truth, _meta = replace_province(
                image, quad, truth, row["target_province"],
                np.random.default_rng(int(row.get("seed") or 0)))
        if not truth or truth[0] not in CHAR_TO_INDEX:
            raise ValueError(f"illegal province label: {truth!r}")
        color = "green" if len(truth) == 8 else "blue"

        char_patch = _first_char_patch(normalized_first_char, image, quad, truth)
        probabilities = classifier.predict_proba(char_patch[None, ...])[0]
        gt_first = CHARSET[int(province_indices[np.argmax(probabilities[province_indices])])]
        base_record = {"path": row["path"], "source": row["_source_file"],
                       "panel": row["_panel"], "subset": row.get("subset"),
                       "province": truth[0], "truth": truth}
        first_records.append({**base_record, "predicted_province": gt_first,
                              "province_correct": gt_first == truth[0],
                              "plate_correct": None, "prediction": None})

        rect = _gt_rectified(image, quad, truth, color, cv2, np)
        segment_read = _segment_decode(rect, classifier, cfg)
        segment_records.append(_segment_record(base_record, truth, segment_read))
        del image, rect, char_patch, probabilities

    result = {
        "schema": "batch12-char-province-eval-v1", "kind": "char",
        "model": str(model_path), "data_dir": str(data_dir), "config": str(config_path),
        "scope": "safe tune + province holdout + deterministic synthetic panel; no fixed val/dev or test data",
        "gt_first_character_classifier": {"metrics": _summarize(first_records),
                                           "records": first_records},
        "segment_first_character_and_full_plate": {
            "metrics": _summarize(segment_records), "records": segment_records,
            "plate_denominator": "all panel rows; segmentation failures count incorrect"},
        "overall": {"gt_first_province_accuracy": _metrics(first_records)["province_accuracy"],
                    "segment_province_accuracy": _metrics(segment_records)["province_accuracy"],
                    "segment_full_plate_accuracy": _metrics(segment_records)["plate_accuracy"]},
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("algorithm/out/data/batch12"))
    parser.add_argument("--config", type=Path, default=Path("algorithm/config/default.toml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.model, args.data_dir, args.config, args.out)
    print(json.dumps({"model": result["model"], "out": str(args.out.resolve()),
                      "panels": ["tune", "province_holdout", "synthetic"],
                      "segment_full_plate_accuracy": result["overall"]["segment_full_plate_accuracy"]},
                     ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
