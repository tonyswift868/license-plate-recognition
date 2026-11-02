"""Run the full three-route pipeline on the fixed province holdout and synth panel."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HOLDOUT_N = 486
SYNTHETIC_N = 930


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"unexpected panel columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV row in {path}")
    return rows


def _quad(value: str) -> list[list[float]]:
    numbers = [float(item) for item in value.replace(",", " ").split()]
    if len(numbers) != 8:
        raise ValueError("panel quad must contain four ordered x,y pairs")
    points = [[numbers[i], numbers[i + 1]] for i in range(0, 8, 2)]
    if not all(abs(item) < 1e8 for point in points for item in point):
        raise ValueError("panel quad contains invalid coordinates")
    return points


def _validate(rows: list[dict[str, str]], panel: str, provinces: tuple[str, ...],
              expected_n: int) -> list[dict]:
    if len(rows) != expected_n:
        raise ValueError(f"{panel} panel must have {expected_n} rows; got {len(rows)}")
    prepared = []
    for index, row in enumerate(rows):
        label = row["label"]
        if not label or len(label) not in (7, 8) or label[0] not in provinces:
            raise ValueError(f"invalid province label in {panel} row {index}")
        item = {**row, "_row": index, "_truth": label, "_source_path": Path(row["path"]),
                "_quad": _quad(row["quad"])}
        if panel == "province_holdout":
            if row["province"] != label[0]:
                raise ValueError(f"province column disagrees with label in row {index}")
        else:
            if row["target_province"] not in provinces:
                raise ValueError(f"invalid synthetic target province in row {index}")
            if not row["seed"].isdigit():
                raise ValueError(f"invalid synthesis seed in row {index}")
            item["_truth"] = row["target_province"] + label[1:]
        prepared.append(item)
    if panel == "synth_eval":
        counts = Counter(row["target_province"] for row in prepared)
        expected = {province: 30 for province in provinces}
        if counts != expected:
            raise ValueError(f"synthetic panel must contain exactly 30 rows per province: {counts}")
    return prepared


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    accepted = [row for row in rows if row["status"] == "OK"]
    correct = sum(row["plate_prediction"] == row["truth"] for row in rows)
    province_correct = sum(row["predicted_province"] == row["truth"][0] for row in rows)
    accepted_correct = sum(row["plate_prediction"] == row["truth"] for row in accepted)
    return {
        "n": n,
        "plate_correct": correct,
        "plate_accuracy": correct / n if n else None,
        "province_correct": province_correct,
        "province_accuracy": province_correct / n if n else None,
        "predicted_皖": sum(row["predicted_province"] == "皖" for row in rows),
        "wrong_as_皖": sum(row["predicted_province"] == "皖" and row["truth"][0] != "皖" for row in rows),
        "auto_accepted": len(accepted),
        "auto_correct": accepted_correct,
        "auto_precision": accepted_correct / len(accepted) if accepted else None,
        "input_error": sum(bool(row["input_error"]) for row in rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


def summarize(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        groups[f"panel/{record['panel']}"].append(record)
        groups[f"province/{record['truth'][0]}"].append(record)
        groups[f"panel_province/{record['panel']}/{record['truth'][0]}"].append(record)
        if record.get("subset"):
            groups[f"subset/{record['subset']}"].append(record)
            groups[f"panel_subset/{record['panel']}/{record['subset']}"].append(record)
    return {name: _metrics(rows) for name, rows in sorted(groups.items())}


def _model_identity(recognizer, config_path: Path) -> dict:
    from lpr.config import ALGORITHM_DIR

    cfg = recognizer.config
    models = {}
    for field in ("model_path", "plate_reader_model_path", "box_reader_model_path", "corner_model_path"):
        value = getattr(cfg.runtime, field)
        path = cfg.resolve_path(value) if value else None
        models[field] = None if path is None else {
            "path": str(path.resolve()), "sha256": _sha256(path),
        }
    manifest = ALGORITHM_DIR / "models" / "manifest.json"
    return {
        "version": recognizer.version,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "effective_config": json.loads(cfg.model_dump_json()),
        "models": models,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "detector_input_size": cfg.runtime.detector_input_size,
        "torch_threads": cfg.runtime.torch_threads,
    }


def _run_panel(recognizer, panel: str, rows: list[dict], records_stream,
               source_hashes: dict[str, str], progress) -> list[dict]:
    import numpy as np
    from lpr.errors import failure_result
    from lpr.io_utils import imread
    from lpr.types import RecogStatus
    from lpr_tools.train.province_synth import replace_province

    records = []
    for item in rows:
        path = item["_source_path"].resolve()
        key = str(path).casefold()
        source_hash = source_hashes.get(key)
        input_error = None
        image = None
        generated = None
        synth_meta = None
        frame_shape = None
        synth_seed = None
        try:
            if source_hash is None:
                input_error = "source file is missing"
            else:
                image = imread(path)
                if panel == "synth_eval":
                    synth_seed = int(item["seed"])
                    image, generated, synth_meta = replace_province(
                        image, item["_quad"], item["label"], item["target_province"],
                        np.random.default_rng(synth_seed))
                    if generated != item["_truth"]:
                        raise ValueError("synthesizer output label differs from frozen target")
                frame_shape = list(image.shape)
                result = recognizer.recognize(image)
        except (OSError, ValueError) as exc:
            input_error = f"{type(exc).__name__}: {exc}"
        if input_error:
            result = failure_result(RecogStatus.BAD_INPUT)
        prediction = result.plate_no
        province_prediction = prediction[0] if prediction else None
        truth = item["_truth"]
        record = {
            "panel": panel,
            "row": item["_row"],
            "path": str(path),
            "source_sha256": source_hash,
            "subset": item.get("subset"),
            "split": item.get("split"),
            "brightness": item.get("brightness"),
            "source_label": item["label"],
            "truth": truth,
            "synth_target_province": item.get("target_province") if panel == "synth_eval" else None,
            "synth_seed": synth_seed if synth_seed is not None else (item.get("seed") if panel == "synth_eval" else None),
            "synth_metadata": synth_meta if panel == "synth_eval" and not input_error else None,
            "panel_row_metadata": {key: value for key, value in item.items() if not key.startswith("_")},
            "frame_shape": frame_shape,
            "frame_resized": False,
            "status": result.status.value,
            "plate_prediction": prediction,
            "predicted_province": province_prediction,
            "plate_correct": prediction == truth,
            "province_correct": province_prediction == truth[0],
            "confidence": float(result.confidence),
            "auto_accepted": result.status == RecogStatus.OK,
            "input_error": input_error,
            "elapsed_ms": float(result.elapsed_ms),
        }
        records.append(record)
        records_stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if len(records) % 50 == 0:
            progress(panel, item["_row"] + 1, len(rows))
        del result, image
    return records


def run(*, config_path: Path | None, holdout_path: Path, synthetic_path: Path,
        out_dir: Path) -> dict:
    import cv2
    import torch
    from loguru import logger
    from lpr.charset import PROVINCES
    from lpr.config import DEFAULT_CONFIG_PATH
    from lpr.pipeline import PlateRecognizer
    from lpr_tools.train.province_synth import DEFAULT_FONT

    config_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    holdout_path, synthetic_path, out_dir = (Path(path).resolve() for path in
                                             (holdout_path, synthetic_path, out_dir))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    holdout = _validate(_rows(holdout_path, ("path", "province", "label", "subset", "brightness", "quad", "sha256", "split")),
                        "province_holdout", tuple(PROVINCES), HOLDOUT_N)
    synthetic = _validate(_rows(synthetic_path, ("path", "label", "subset", "brightness", "quad", "sha256", "target_province", "seed")),
                          "synth_eval", tuple(PROVINCES), SYNTHETIC_N)
    source_rows = [*holdout, *synthetic]
    row_seeds = [int(item["seed"]) for item in synthetic]
    seed_counts = Counter(row_seeds)
    source_hashes = {}
    source_errors = {}
    for item in source_rows:
        path = item["_source_path"].resolve()
        key = str(path).casefold()
        if key not in source_hashes and key not in source_errors:
            if not path.is_file():
                source_errors[key] = "source file is missing"
            else:
                actual_hash = _sha256(path)
                if actual_hash != item["sha256"].lower():
                    raise ValueError(f"source file sha256 differs from panel: {path}")
                source_hashes[key] = actual_hash
    for item in source_rows:
        if str(item["_source_path"].resolve()).casefold() not in source_hashes:
            continue
        if item["sha256"].lower() != source_hashes[str(item["_source_path"].resolve()).casefold()]:
            raise ValueError(f"same source path has inconsistent SHA in panel rows: {item['_source_path']}")

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    logger.remove()
    recognizer = (PlateRecognizer.from_config() if config_path == DEFAULT_CONFIG_PATH.resolve()
                  else PlateRecognizer.from_config(str(config_path)))
    if recognizer.config.runtime.torch_threads > 4:
        raise ValueError("province pipeline evaluation is limited to four Torch threads")
    identity = _model_identity(recognizer, config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "batch17-province-pipeline-v1",
        "model_identity": identity,
        "holdout": {"path": str(holdout_path), "sha256": _sha256(holdout_path), "rows": len(holdout)},
        "synthetic": {"path": str(synthetic_path), "sha256": _sha256(synthetic_path), "rows": len(synthetic),
                      "seed_policy": "numpy.random.default_rng(int(row.seed)); row seed is authoritative",
                      "seed_n": len(row_seeds), "seed_unique": len(seed_counts),
                      "seed_min": min(seed_counts), "seed_max": max(seed_counts),
                      "font_path": str(DEFAULT_FONT.resolve()), "font_sha256": _sha256(DEFAULT_FONT)},
        "input_errors_preflight": source_errors,
        "input_method": {
            "holdout": "existing full-resolution source frame; no crop/resize; pipeline detects and locates on its ordinary inputs",
            "synthetic": "existing replace_province on the full-resolution original frame with source-image quad and row RNG; generated frame is passed unchanged to the pipeline",
            "synthetic_crop_difference": "batch12 single-reader panel synthesizes on the full source frame, then uses GT box/quad crops for each isolated reader; this evaluation uses the same synthesized source-frame pixels but the full pipeline detector/locate/candidate crops",
            "image_decode": "lpr.io_utils.imread, same byte-array cv2.imdecode path as batch12 panel reader",
        },
        "records": str((out_dir / "records.jsonl.gz").resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_records = []
    with gzip.open(out_dir / "records.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
        def progress(panel, done, total):
            print(f"{panel} {done}/{total}", flush=True)
        for panel, rows in (("province_holdout", holdout), ("synth_eval", synthetic)):
            all_records.extend(_run_panel(recognizer, panel, rows, stream, source_hashes, progress))
    if len(all_records) != HOLDOUT_N + SYNTHETIC_N:
        raise AssertionError("fixed province panel denominator changed")
    report = {
        "schema": "batch17-province-pipeline-report-v1",
        "model_identity": identity,
        "counts": {"province_holdout": HOLDOUT_N, "synth_eval": SYNTHETIC_N,
                   "total": len(all_records)},
        "metrics": summarize(all_records),
        "inputs": {"holdout_sha256": meta["holdout"]["sha256"],
                   "synth_sha256": meta["synthetic"]["sha256"]},
        "pixel_method": meta["input_method"],
        "input_errors": {"total": sum(bool(r["input_error"]) for r in all_records),
                          "by_panel": {p: sum(bool(r["input_error"]) for r in all_records if r["panel"] == p)
                                       for p in ("province_holdout", "synth_eval")}},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="model/config file; omitted uses the default config")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(config_path=args.config, holdout_path=args.holdout,
                 synthetic_path=args.synthetic, out_dir=args.out)
    print(json.dumps({"counts": report["counts"], "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
