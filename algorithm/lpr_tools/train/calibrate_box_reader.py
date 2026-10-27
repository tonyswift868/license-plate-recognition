"""Fit and audit a bounded CTC temperature for box_reader_v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from lpr.charset import CHAR_TO_INDEX
from lpr_tools.train.box_data import bbox_from_quad
from lpr.plate_rules import match_layout

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = ROOT / "algorithm" / "out" / "data" / "batch10" / "tune.csv"
DEFAULT_TRAIN_DATA = ROOT / "algorithm" / "out" / "data" / "batch10" / "pool.csv"
DEFAULT_POOL_AUDIT = ROOT / "algorithm" / "out" / "data" / "batch10" / "audit.json"
DEFAULT_BEST = ROOT / "algorithm" / "out" / "train" / "box_reader_v1" / "best.pt"
DEFAULT_MODEL = ROOT / "algorithm" / "models" / "box_reader_v1.pt"
DEFAULT_REPORT = ROOT / "algorithm" / "out" / "train" / "box_reader_v1" / "calibration_report.json"
DEFAULT_CACHE_INDEX = Path(r"D:\keshe2\algorithm\data\boxcrop_cache\box_reader_v1\index.csv")
DEFAULT_CACHE_AUDIT = DEFAULT_CACHE_INDEX.parent / "audit.json"
EXPECTED_PER_FOLD = 1050
MIN_RELEASE_EPOCHS = 30
EXPECTED_POOL_COUNT = 210_616


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_pool_provenance(path: Path) -> dict[str, str]:
    """Return the actual training-pool identity used for model registration."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training pool index is required for model provenance: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def read_calibration_rows(index_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read exactly the two safe tune folds and prove path/content isolation."""
    required = {"path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold"}
    rows: dict[str, list[dict[str, str]]] = {"fit": [], "report": []}
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"calibration index must contain {sorted(required)}")
        for row in reader:
            fold = row["calibration_fold"]
            if fold not in rows:
                raise ValueError(f"unexpected calibration fold: {fold!r}")
            label = row["label"]
            if match_layout(label) is None or any(char not in CHAR_TO_INDEX for char in label):
                raise ValueError(f"invalid calibration label: {label!r}")
            if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"].lower()):
                raise ValueError("calibration rows require a SHA-256 source identity")
            bbox_from_quad(row["quad"])
            rows[fold].append(row)
    if any(len(rows[fold]) != EXPECTED_PER_FOLD for fold in rows):
        raise ValueError(f"expected {EXPECTED_PER_FOLD} rows per fit/report fold; got "
                         f"{ {fold: len(values) for fold, values in rows.items()} }")
    for fold, selected in rows.items():
        paths = [os.path.normcase(os.path.normpath(row["path"])) for row in selected]
        hashes = [row["sha256"].lower() for row in selected]
        if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
            raise ValueError(f"duplicate source path or content hash in {fold} fold")
    fit_paths = {os.path.normcase(os.path.normpath(row["path"])) for row in rows["fit"]}
    report_paths = {os.path.normcase(os.path.normpath(row["path"])) for row in rows["report"]}
    fit_hashes = {row["sha256"].lower() for row in rows["fit"]}
    report_hashes = {row["sha256"].lower() for row in rows["report"]}
    if fit_paths & report_paths or fit_hashes & report_hashes:
        raise ValueError("fit/report calibration folds share paths or source bytes")
    return rows["fit"], rows["report"]


def validate_training_history(history_path: Path, best_path: Path, train_data_path: Path,
                              tune_path: Path, *, cache_index_path: Path = DEFAULT_CACHE_INDEX,
                              cache_audit_path: Path = DEFAULT_CACHE_AUDIT,
                              pool_audit_path: Path = DEFAULT_POOL_AUDIT,
                              expected_pool_count: int = EXPECTED_POOL_COUNT) -> dict[str, Any]:
    """Prove the selected checkpoint came from at least 30 complete GPU epochs."""
    history_path = Path(history_path).resolve()
    best_path = Path(best_path).resolve()
    train_data_path = Path(train_data_path).resolve()
    tune_path = Path(tune_path).resolve()
    cache_index_path = Path(cache_index_path).resolve()
    cache_audit_path = Path(cache_audit_path).resolve()
    pool_audit_path = Path(pool_audit_path).resolve()
    if not best_path.is_file():
        raise FileNotFoundError(f"best release checkpoint is missing: {best_path}")
    data = json.loads(history_path.read_text(encoding="utf-8"))
    recipe = data.get("recipe", {})
    epochs = data.get("history", [])
    if recipe.get("smoke") is not False:
        raise ValueError("release calibration refuses a smoke checkpoint")
    if not str(recipe.get("device", "")).startswith("cuda"):
        raise ValueError("release calibration requires a CUDA training history")
    if int(recipe.get("epochs", 0)) < MIN_RELEASE_EPOCHS or len(epochs) < MIN_RELEASE_EPOCHS:
        raise ValueError(f"release checkpoint requires at least {MIN_RELEASE_EPOCHS} complete epochs")
    recipe_train = Path(recipe.get("train", "")).resolve()
    if Path(recipe.get("tune", "")).resolve() != tune_path:
        raise ValueError("training history points at a different tune index")
    if Path(recipe.get("out", "")).resolve() != best_path.parent:
        raise ValueError("best checkpoint is outside the output directory recorded by training history")
    train_sha = _sha256_file(train_data_path)
    tune_sha = _sha256_file(tune_path)
    pool_audit = json.loads(pool_audit_path.read_text(encoding="utf-8-sig"))
    expected_pool_sha = pool_audit.get("output_hashes", {}).get("pool.csv")
    if expected_pool_sha != train_sha:
        raise ValueError("calibration pool SHA-256 does not match the batch10 isolation audit")
    with train_data_path.open(encoding="utf-8-sig", newline="") as stream:
        pool_count = sum(1 for _ in csv.DictReader(stream))
    if pool_count != expected_pool_count:
        raise ValueError(f"audited original pool must contain {expected_pool_count} rows; got {pool_count}")
    if data.get("tune_sha256") != tune_sha:
        raise ValueError("training history tune SHA-256 does not match the safe tune index")
    packed_cache = recipe_train == cache_index_path
    if packed_cache:
        if recipe.get("packed_cache") is not True or recipe.get("decode_jpeg") is True:
            raise ValueError("packed cache training history must use the packed_cache loader")
        if not cache_index_path.is_file() or not cache_audit_path.is_file():
            raise FileNotFoundError("packed cache index and audit are required to validate training history")
        cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8-sig"))
        cache_sha = _sha256_file(cache_index_path)
        if cache_audit.get("pool_sha256") != train_sha:
            raise ValueError("packed cache audit does not identify the original audited pool")
        if (cache_audit.get("cache_index_sha256") != cache_sha
                or data.get("train_sha256") != cache_sha):
            raise ValueError("packed cache index SHA-256 does not match cache audit and training history")
        cache_count = int(cache_audit.get("pool_count", -1))
        with cache_index_path.open(encoding="utf-8-sig", newline="") as stream:
            cache_reader = csv.DictReader(stream)
            required_cache_fields = {"path", "label", "subset", "brightness", "sha256", "quad",
                                     "cache_pack", "pack_offset", "pack_length", "x0", "y0",
                                     "orig_w", "orig_h"}
            if not cache_reader.fieldnames or not required_cache_fields.issubset(cache_reader.fieldnames):
                raise ValueError("packed cache index is missing required image and tar-offset columns")
            index_count = sum(1 for _ in cache_reader)
        if cache_count != pool_count or index_count != pool_count:
            raise ValueError("packed cache count, cache index rows, and original pool count differ")
        if ("samples" in cache_audit
                and int(cache_audit["samples"]) != pool_count):
            raise ValueError("packed cache audit sample count differs from the original pool")
        if int(cache_audit.get("quality", -1)) != 65:
            raise ValueError("packed cache must use the audited JPEG quality 65")
        tar_path = Path(cache_audit.get("cache_tar_path", "")).resolve()
        if tar_path != cache_index_path.parent / "crops.tar" or not tar_path.is_file():
            raise ValueError("packed cache audit does not point at the approved crops.tar")
        if int(cache_audit.get("tar_bytes", -1)) != tar_path.stat().st_size:
            raise ValueError("packed cache tar size does not match its audit")
        if int(cache_audit.get("index_bytes", -1)) != cache_index_path.stat().st_size:
            raise ValueError("packed cache index size does not match its audit")
        if int(cache_audit.get("total_cache_bytes", -1)) > 3_000_000_000:
            raise ValueError("packed cache exceeds the 3 GB disk limit")
        tar_sha = str(cache_audit.get("cache_tar_sha256", "")).lower()
        if len(tar_sha) != 64 or any(c not in "0123456789abcdef" for c in tar_sha):
            raise ValueError("packed cache audit has no valid tar SHA-256")
        train_count = cache_count
        train_index_sha = cache_sha
    else:
        if recipe_train != train_data_path:
            raise ValueError("training history points at neither audited pool nor approved packed cache")
        if recipe.get("packed_cache") is True:
            raise ValueError("packed cache run must use the approved cache index path")
        if data.get("train_sha256") != train_sha:
            raise ValueError("training history pool SHA-256 does not match the audited pool")
        train_count = pool_count
        train_index_sha = train_sha
    epoch_ids = [int(item.get("epoch", -1)) for item in epochs]
    if epoch_ids != list(range(1, len(epochs) + 1)):
        raise ValueError("training history epochs are not consecutive from epoch 1")
    for item in epochs:
        if int(item.get("samples", -1)) != train_count or int(item.get("unique_samples", -1)) != train_count:
            raise ValueError(f"epoch {item.get('epoch')} did not consume the complete training pool")
    best_epoch = int(data.get("best_epoch", -1))
    if best_epoch not in epoch_ids:
        raise ValueError("best_epoch is missing from completed training history")
    expected_best = max(epochs, key=lambda item: float(item["tune"]["accuracy"]))
    if best_epoch != int(expected_best["epoch"]):
        raise ValueError("best checkpoint epoch disagrees with tune-selected history")
    if int(data.get("total_exposures", -1)) != train_count * len(epochs):
        raise ValueError("total exposure count does not match complete epoch history")
    return {"path": str(history_path), "sha256": _sha256_file(history_path),
            "epochs": len(epochs), "best_epoch": best_epoch,
            "training_samples_per_epoch": train_count, "total_exposures": train_count * len(epochs),
            "device": recipe["device"], "smoke": False,
            "packed_cache": packed_cache, "training_index_sha256": train_index_sha,
            "source_pool_count": pool_count,
            "pool_audit": {"path": str(pool_audit_path), "sha256": _sha256_file(pool_audit_path),
                           "pool_sha256": expected_pool_sha},
            "cache_audit": ({"path": str(cache_audit_path), "sha256": _sha256_file(cache_audit_path),
                             "cache_index_sha256": train_index_sha, "pool_count": pool_count,
                             "quality": 65, "tar_path": str(tar_path), "tar_sha256": tar_sha,
                             "tar_bytes": tar_path.stat().st_size,
                             "index_bytes": cache_index_path.stat().st_size,
                             "total_cache_bytes": int(cache_audit["total_cache_bytes"])}
                            if packed_cache else None)}


def _collect_logits(model: BoxNet, rows: list[dict[str, str]], box_pad: tuple[float, float],
                    index_path: Path
                    ) -> list[torch.Tensor]:
    import torch

    from lpr.boxcrop import crop_box
    from lpr.io_utils import imread
    from lpr.recognize.box_reader import BoxNet, box_tensor

    logits_list = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = index_path.resolve().parent / path
        digest = _sha256_file(path)
        if digest.lower() != row["sha256"].lower():
            raise ValueError(f"calibration source SHA-256 changed: {path}")
        image = imread(path)
        crop = crop_box(image, bbox_from_quad(row["quad"]), *box_pad)
        if crop.size == 0:
            raise ValueError(f"empty GT bbox crop: {path}")
        with torch.inference_mode():
            logits = model(box_tensor(crop)[None]).detach().cpu().float()
        if logits.ndim != 3 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
            raise ValueError(f"invalid model logits for {path}")
        logits_list.append(logits)
    return logits_list


def _report_fold(rows: list[dict[str, str]], logits_list: list[torch.Tensor],
                 temperature: float) -> dict[str, Any]:
    import torch

    from lpr.recognize.crnn import _greedy_decode
    from lpr_tools.train.calibrate_crnn_v2 import _ctc_loss, _ece_summary

    before_nll = after_nll = 0.0
    correct, confidence_before, confidence_after = [], [], []
    predictions = []
    for row, logits in zip(rows, logits_list):
        truth = row["label"]
        if not torch.equal(logits.argmax(-1), (logits / temperature).argmax(-1)):
            raise AssertionError("positive scalar temperature changed CTC frame argmax")
        before_nll += float(_ctc_loss(logits.double(), [truth]))
        after_nll += float(_ctc_loss(logits.double(), [truth], temperature))
        raw, conf_raw = _greedy_decode(logits[:, 0])
        calibrated, conf_calibrated = _greedy_decode(logits[:, 0] / temperature)
        if raw != calibrated:
            raise AssertionError("positive scalar temperature changed decoded plate string")
        predictions.append(raw)
        correct.append(raw == truth)
        confidence_before.append(min(conf_raw) if conf_raw else 0.0)
        confidence_after.append(min(conf_calibrated) if conf_calibrated else 0.0)
    n = len(rows)
    return {"n": n, "temperature": temperature, "ctc_nll_before": before_nll / n,
            "ctc_nll_after": after_nll / n, "argmax_string_invariant": True,
            "accuracy": sum(correct) / n,
            "confidence_before": _ece_summary(correct, confidence_before),
            "confidence_after": _ece_summary(correct, confidence_after),
            "predictions": predictions}


def calibrate(index_path: Path = DEFAULT_INDEX, best_path: Path = DEFAULT_BEST,
              model_path: Path = DEFAULT_MODEL, report_path: Path = DEFAULT_REPORT,
              train_data_path: Path = DEFAULT_TRAIN_DATA,
              history_path: Path | None = None) -> dict[str, Any]:
    import cv2
    import torch

    from lpr.recognize import read_manifest, write_manifest_entry
    from lpr.recognize.box_reader import ARCH, BoxNet, INPUT_SIZE, checkpoint
    from lpr_tools.train.calibrate_crnn_v2 import BOUNDS, fit_temperature

    if model_path.exists():
        raise FileExistsError(f"refusing to replace existing released model: {model_path}")
    old_manifest = read_manifest(model_path.parent)
    if model_path.name in old_manifest:
        raise ValueError(f"manifest already contains {model_path.name}; refusing to replace")
    train_pool = training_pool_provenance(train_data_path)
    history_path = history_path or best_path.parent / "history.json"
    history = validate_training_history(history_path, best_path, train_data_path, index_path)
    fit_rows, report_rows = read_calibration_rows(index_path)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("architecture") != ARCH:
        raise ValueError("best checkpoint must contain BoxReader architecture metadata")
    model = BoxNet().cpu().eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    input_size = saved.get("input_size")
    box_pad = tuple(float(v) for v in saved.get("box_pad", ()))
    if input_size != list(INPUT_SIZE) or len(box_pad) != 2 or any(not 0 <= v <= 1 for v in box_pad):
        raise ValueError("best checkpoint input_size or box_pad is invalid")

    fit_logits = _collect_logits(model, fit_rows, box_pad, index_path)
    fitted = fit_temperature(fit_logits, [row["label"] for row in fit_rows], BOUNDS)
    report_logits = _collect_logits(model, report_rows, box_pad, index_path)
    report = _report_fold(report_rows, report_logits, fitted["temperature"])
    original_report = _report_fold(report_rows, report_logits, 1.0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    exported = checkpoint(model, temperature=fitted["temperature"], box_pad=box_pad)
    torch.save(exported, model_path)
    if model_path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("calibrated model exceeds the 20 MB release limit")
    write_manifest_entry(model_path, version="box_reader_v1", train_data=train_pool["path"],
                         val_char_acc=None, input_size=INPUT_SIZE)
    manifest_after = read_manifest(model_path.parent)
    if any(manifest_after.get(name) != entry for name, entry in old_manifest.items()):
        raise AssertionError("manifest registration changed a pre-existing entry")
    digest = _sha256_file(model_path)
    result = {"schema": "box-reader-temperature-v1", "best_checkpoint": str(best_path.resolve()),
              "best_checkpoint_sha256": _sha256_file(best_path),
              "tune_index": str(index_path.resolve()), "tune_sha256": _sha256_file(index_path),
              "training_pool": train_pool,
              "training_history": history,
              "folds": {"fit": {"n": len(fit_rows), "origins": len({r['sha256'] for r in fit_rows})},
                        "report": {"n": len(report_rows), "origins": len({r['sha256'] for r in report_rows}),
                                   "path_overlap": 0, "sha_overlap": 0}},
              "fit": fitted,
              "report": {key: value for key, value in report.items() if key != "predictions"},
              "report_before_calibration": {key: value for key, value in original_report.items() if key != "predictions"},
              "report_predictions_unchanged": report["predictions"] == original_report["predictions"],
              "temperature_bounds": list(BOUNDS), "export": {"path": str(model_path.resolve()),
                  "sha256": digest, "bytes": model_path.stat().st_size,
                  "manifest_entry": manifest_after[model_path.name]},
              "scope_note": "fit/report are safe tuning origins, not a final blind evaluation; calibration uses GT axis-aligned bbox without jitter"}
    if not result["report_predictions_unchanged"]:
        raise AssertionError("positive scalar temperature changed report strings")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--best", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    result = calibrate(args.index, args.best, args.model_out, args.report,
                       args.train_data, args.history)
    print(json.dumps({"temperature": result["fit"]["temperature"],
                      "report_nll_before": result["report_before_calibration"]["ctc_nll_before"],
                      "report_nll_after": result["report"]["ctc_nll_after"],
                      "report_ece_before": result["report_before_calibration"]["confidence_before"]["ece"],
                      "report_ece_after": result["report"]["confidence_after"]["ece"],
                      "report_brier_before": result["report_before_calibration"]["confidence_before"]["brier"],
                      "report_brier_after": result["report"]["confidence_after"]["brier"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
