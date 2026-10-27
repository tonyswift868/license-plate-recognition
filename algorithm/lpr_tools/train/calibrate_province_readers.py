"""Calibrate and safely export the three batch-12 province readers."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_TMP = _ROOT / "algorithm" / "out" / "tmp" / "batch12-runtime"
_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_RUNTIME_TMP)
os.environ["TMP"] = str(_RUNTIME_TMP)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

DEFAULT_TUNE = _ROOT / "algorithm" / "out" / "data" / "batch11" / "tune.csv"
DEFAULT_MODELS = _ROOT / "algorithm" / "models"
DEFAULT_CHAR_BASE = DEFAULT_MODELS / "char_cnn_v2.pt"
MODEL_NAMES = {"box": "box_reader_v3.pt", "crnn": "crnn_v4.pt", "char": "char_cnn_v3.pt"}
MODEL_VERSIONS = {"box": "box_reader_v3", "crnn": "crnn_v4", "char": "char_cnn_v3"}
MIN_EPOCHS = {"box": 20, "crnn": 15, "char": 10}
EXPECTED_TRAIN_ROWS = 210564


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unwrap_model(payload: Any, kind: str):
    if (isinstance(payload, dict) and payload.get("format") ==
            "province-reader-training-v1"):
        if payload.get("kind") != kind:
            raise ValueError(f"training checkpoint kind mismatch: expected {kind!r}")
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("training checkpoint lacks model_state")
        return state, payload
    if not isinstance(payload, dict):
        raise ValueError("reader checkpoint must be a dictionary")
    return payload, {}


def _char_province_only(old_state: dict[str, Any], new_state: dict[str, Any], *,
                        province_count: int = 31) -> dict[str, Any]:
    """Require every CharCNN tensor outside the province output rows to be equal."""
    import torch
    weight_key, bias_key = "classifier.5.weight", "classifier.5.bias"
    if set(old_state) != set(new_state):
        raise ValueError("CharCNN checkpoint tensor keys differ from char_cnn_v2")
    if weight_key not in old_state or bias_key not in old_state:
        raise ValueError("CharCNN checkpoint lacks the registered final classifier layer")
    old_weight, new_weight = old_state[weight_key], new_state[weight_key]
    old_bias, new_bias = old_state[bias_key], new_state[bias_key]
    if (old_weight.shape != new_weight.shape or old_bias.shape != new_bias.shape
            or old_weight.shape[0] <= province_count or old_bias.shape[0] != old_weight.shape[0]):
        raise ValueError("CharCNN final classifier shape is incompatible")
    weight_changed = (old_weight[:province_count] != new_weight[:province_count]).any(dim=1)
    bias_changed = old_bias[:province_count] != new_bias[:province_count]
    changed_province_rows = int((weight_changed | bias_changed).sum())
    if not torch.equal(old_weight[province_count:], new_weight[province_count:]):
        raise ValueError("CharCNN non-province classifier weight rows changed")
    if not torch.equal(old_bias[province_count:], new_bias[province_count:]):
        raise ValueError("CharCNN non-province classifier bias rows changed")
    unchanged = []
    for key in old_state:
        if key in {weight_key, bias_key}:
            continue
        if not torch.equal(old_state[key], new_state[key]):
            raise ValueError(f"CharCNN non-head tensor changed: {key}")
        unchanged.append(key)
    return {"all_non_head_tensors_identical": True,
            "non_province_classifier_rows_identical": True,
            "verified_non_head_tensor_count": len(unchanged),
            "province_rows_with_changes": changed_province_rows,
            "province_class_count": province_count}


def _temperature_export(kind: str, model_state: dict[str, Any], temperature: float):
    import torch
    if kind == "box":
        if model_state.get("architecture") not in {
                "mobilenet_v3_small_stem_hattn_bigru96_ctc_v1",
                "mobilenet_v3_small_stem_stn_gn_bigru160_ctc_v2"}:
            raise ValueError("unsupported box-reader architecture")
        if not (temperature > 0 and float(temperature) < float("inf")):
            raise ValueError("temperature must be positive and finite")
        exported = dict(model_state)
        exported["temperature"] = float(temperature)
        return exported
    if kind == "crnn":
        if not (temperature > 0 and float(temperature) < float("inf")):
            raise ValueError("temperature must be positive and finite")
        exported = {key: value.detach().cpu().clone() for key, value in model_state.items()
                    if key != "ctc_temperature"}
        exported["ctc_temperature"] = torch.tensor(float(temperature), dtype=torch.float32)
        return exported
    if kind == "char":
        return {key: value.detach().cpu().clone() for key, value in model_state.items()}
    raise ValueError("kind must be box, crnn, or char")


def _quad(text: str):
    import numpy as np
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("tune quad must contain eight finite coordinates")
    return np.asarray(values, dtype=np.float32).reshape(4, 2)


def _read_source(row: dict[str, str], cv2, np):
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual.lower() != row["sha256"].lower():
        raise ValueError(f"tune source SHA mismatch: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode calibration source: {path}")
    return path, image


def _collect_logits(kind: str, model, rows, tune_path: Path, box_pad, input_size):
    import cv2
    import numpy as np
    import torch
    from lpr.boxcrop import crop_box
    from lpr.recognize.box_reader import box_tensor
    from lpr.recognize.crnn import plate_tensor
    from lpr_tools.train.box_data import bbox_from_quad

    logits_list = []
    for row in rows:
        path, image = _read_source(row, cv2, np)
        quad = _quad(row["quad"])
        if kind == "box":
            crop = crop_box(image, bbox_from_quad(row["quad"]), *box_pad)
            if crop.size == 0:
                raise ValueError(f"empty GT box crop: {path}")
            tensor = box_tensor(crop, input_size)[None]
        else:
            width = 276 if len(row["label"]) == 8 else 252
            target = np.float32([[0, 0], [width - 1, 0],
                                 [width - 1, 79], [0, 79]])
            transform = cv2.getPerspectiveTransform(quad, target)
            crop = cv2.warpPerspective(image, transform, (width, 80))
            if crop.size == 0:
                raise ValueError(f"empty GT quad warp: {path}")
            tensor = plate_tensor(crop)[None]
        with torch.inference_mode():
            logits = model(tensor).detach().cpu().float()
        if logits.ndim != 3 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
            raise ValueError(f"invalid CTC logits for {path}")
        logits_list.append(logits)
        del image, crop, tensor, logits
    return logits_list


def _fit_and_report(kind: str, state: dict[str, Any], rows_fit, rows_report,
                    tune_path: Path, input_size, box_pad):
    import cv2
    import torch
    from lpr.recognize.box_reader import model_from_checkpoint
    from lpr.recognize.crnn import CrnnNet
    from lpr_tools.train.calibrate_box_reader import _report_fold
    from lpr_tools.train.calibrate_crnn_v2 import BOUNDS, fit_temperature

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if kind == "box":
        model = model_from_checkpoint(state).cpu().eval()
    else:
        weights = {key: value for key, value in state.items() if key != "ctc_temperature"}
        model = CrnnNet().cpu().eval()
        model.load_state_dict(weights, strict=True)
    fit_logits = _collect_logits(kind, model, rows_fit, tune_path, box_pad, input_size)
    fit = fit_temperature(fit_logits, [row["label"] for row in rows_fit], BOUNDS)
    report_logits = _collect_logits(kind, model, rows_report, tune_path, box_pad, input_size)
    report = _report_fold(rows_report, report_logits, fit["temperature"])
    report_before = _report_fold(rows_report, report_logits, 1.0)
    if report["predictions"] != report_before["predictions"]:
        raise AssertionError("positive temperature changed report strings")
    return fit, report, report_before


def calibrate(kind: str, model_path: Path, tune_path: Path, *,
              model_out: Path | None = None, report_path: Path | None = None,
              char_base_path: Path = DEFAULT_CHAR_BASE,
              history_path: Path | None = None,
              data_dir: Path = _ROOT / "algorithm" / "out" / "data" / "batch12") -> dict[str, Any]:
    import torch
    from lpr.recognize import read_manifest, write_manifest_entry, verify_model
    from lpr_tools.train.calibrate_box_reader import read_calibration_rows

    if kind not in {"box", "crnn", "char"}:
        raise ValueError("kind must be box, crnn, or char")
    model_path, tune_path = Path(model_path).resolve(), Path(tune_path).resolve()
    history_path = Path(history_path or (model_path.parent / "history.json")).resolve()
    model_out = Path(model_out or (DEFAULT_MODELS / MODEL_NAMES[kind])).resolve()
    report_path = Path(report_path or (model_out.parent.parent / "out" / "train" / model_out.stem / "calibration_report.json")).resolve()
    if not model_path.is_file() or not tune_path.is_file():
        raise FileNotFoundError("model checkpoint and tune CSV must exist")
    if model_path == model_out:
        raise ValueError("source checkpoint and output model must differ")
    if model_out.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a model or calibration report")
    old_manifest = copy.deepcopy(read_manifest(model_out.parent))
    if model_out.name in old_manifest:
        raise ValueError(f"manifest already contains {model_out.name}")
    fit_rows, report_rows = read_calibration_rows(tune_path)
    if len(fit_rows) != 1200 or len(report_rows) != 1200:
        raise ValueError("batch12 temperature calibration requires the original 1,200/1,200 tune folds")
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    state, wrapper = _unwrap_model(saved, kind)
    if not wrapper:
        raise ValueError("calibration requires a full-epoch training wrapper checkpoint")
    training_proof = _validate_training_history(kind, wrapper, data_dir, history_path, tune_path)
    model_sha = _sha256(model_path)
    state_sha = _state_sha(kind, state)

    fit = report = report_before = None
    char_proof = None
    temperature = 1.0
    if kind in {"box", "crnn"}:
        fit, report, report_before = _fit_and_report(
            kind, state, fit_rows, report_rows, tune_path,
            tuple(state["input_size"]) if kind == "box" else (160, 40),
            tuple(state["box_pad"]) if kind == "box" else None)
        temperature = float(torch.tensor(fit["temperature"], dtype=torch.float32).item())
        exported = _temperature_export(kind, state, temperature)
    else:
        if not char_base_path.is_file():
            raise FileNotFoundError(char_base_path)
        old_base = verify_model(char_base_path)
        base_state = torch.load(char_base_path, map_location="cpu", weights_only=True)
        char_proof = _char_province_only(base_state, state)
        exported = _temperature_export(kind, state, 1.0)
        char_proof.update(base_model=str(char_base_path.resolve()),
                          base_model_sha256=old_base.get("sha256"))

    input_size = tuple(state["input_size"]) if kind == "box" else ((160, 40) if kind == "crnn" else (32, 32))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_out.with_name(model_out.name + f".{os.getpid()}.tmp")
    if temp_path.exists():
        raise FileExistsError(temp_path)
    try:
        torch.save(exported, temp_path)
        if temp_path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("released model exceeds the 20 MiB file limit")
        if model_out.exists():
            raise FileExistsError(model_out)
        temp_path.rename(model_out)
        train_data = json.dumps({"selected_checkpoint": str(model_path),
                                 "selected_checkpoint_sha256": model_sha,
                                 "selected_state_sha256": state_sha,
                                 "wrapper_epoch": wrapper.get("epoch"),
                                 "best_epoch": wrapper.get("best_epoch"),
                                 "training_proof": training_proof,
                                 "tune_path": str(tune_path), "tune_sha256": _sha256(tune_path),
                                 "fit_rows": len(fit_rows), "report_rows": len(report_rows),
                                 "calibration_temperature": temperature,
                                 "scope": "safe tuning folds only; not a blind evaluation"},
                                ensure_ascii=True)
        entry = write_manifest_entry(model_out, version=MODEL_VERSIONS[kind],
                                     train_data=train_data, val_char_acc=None,
                                     input_size=input_size)
        manifest_after = read_manifest(model_out.parent)
        if any(manifest_after.get(name) != value for name, value in old_manifest.items()):
            raise AssertionError("calibration modified a pre-existing manifest entry")
        if manifest_after.get(model_out.name) != entry:
            raise AssertionError("new model manifest entry failed round-trip verification")
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    output_sha = _sha256(model_out)
    if kind == "box":
        reloaded = torch.load(model_out, map_location="cpu", weights_only=True)
        if reloaded.get("temperature") != temperature:
            raise AssertionError("box temperature metadata did not round-trip")
        if _state_sha(kind, reloaded) != state_sha:
            raise AssertionError("box export changed tensors or architecture metadata")
    elif kind == "crnn":
        reloaded = torch.load(model_out, map_location="cpu", weights_only=True)
        if not torch.equal(reloaded["ctc_temperature"], torch.tensor(temperature)):
            raise AssertionError("CRNN temperature tensor did not round-trip")
        if _state_sha(kind, reloaded) != state_sha:
            raise AssertionError("CRNN export changed the selected raw state")
    else:
        reloaded = torch.load(model_out, map_location="cpu", weights_only=True)
        char_proof = _char_province_only(base_state, reloaded)
        if _state_sha(kind, reloaded) != state_sha:
            raise AssertionError("CharCNN export changed the selected raw state")

    result = {"schema": "batch12-province-calibration-v1", "kind": kind,
              "source_checkpoint": str(model_path), "source_checkpoint_sha256": model_sha,
              "source_state_sha256": state_sha, "wrapper_epoch": wrapper.get("epoch"),
              "best_epoch": wrapper.get("best_epoch"), "tune_path": str(tune_path),
              "tune_sha256": _sha256(tune_path),
              "folds": {"fit": {"rows": len(fit_rows), "groups": _groups(fit_rows)},
                        "report": {"rows": len(report_rows), "groups": _groups(report_rows),
                                   "path_sha_overlap": 0}},
              "temperature": temperature, "fit": fit, "report": report,
              "report_before_calibration": report_before,
              "report_predictions_unchanged": (report["predictions"] == report_before["predictions"])
                  if report is not None else None,
              "training_proof": training_proof,
              "char_province_only_proof": char_proof,
              "export": {"path": str(model_out), "sha256": output_sha,
                         "bytes": model_out.stat().st_size, "manifest_entry": entry},
              "preexisting_manifest_entries_unchanged": True}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def _state_sha(kind: str, state: dict[str, Any]) -> str:
    import torch
    digest = hashlib.sha256()
    if kind == "box":
        inner = state.get("state_dict")
        if not isinstance(inner, dict):
            raise ValueError("box checkpoint lacks inference state_dict")
        for key in sorted(inner):
            value = inner[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"box state entry is not a tensor: {key}")
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes())
    else:
        for key in sorted(state):
            value = state[key]
            if key == "ctc_temperature":
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"state entry is not a tensor: {key}")
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_training_history(kind: str, wrapper: dict[str, Any], data_dir: Path,
                              history_path: Path, tune_path: Path):
    """Verify full-epoch selection and the exact audited inputs before release."""
    report = json.loads(history_path.read_text(encoding="utf-8-sig"))
    history = report.get("history")
    if not isinstance(history, list) or len(history) < MIN_EPOCHS[kind]:
        raise ValueError(f"training history has fewer than {MIN_EPOCHS[kind]} full epochs")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance != wrapper.get("provenance"):
        raise ValueError("checkpoint and history provenance differ")
    if wrapper.get("kind") != kind or wrapper.get("cursor") != 0:
        raise ValueError("selected checkpoint kind/cursor is invalid")
    if report.get("kind") != kind or report.get("epochs_completed") != len(history):
        raise ValueError("history kind or completed epoch count is inconsistent")
    if [item.get("epoch") for item in history] != list(range(1, len(history) + 1)):
        raise ValueError("history epochs are not consecutive full epochs")
    scores = []
    for item in history:
        if item.get("real_unique") != EXPECTED_TRAIN_ROWS or item.get("real_physical") != EXPECTED_TRAIN_ROWS:
            raise ValueError("history contains an epoch without full audited real-source coverage")
        synthetic_count = item.get("synthetic_physical")
        synth_by_province = item.get("synthetic_provinces")
        if (not isinstance(synthetic_count, int) or synthetic_count < 0
                or not isinstance(synth_by_province, dict)
                or sum(int(value) for value in synth_by_province.values()) != synthetic_count):
            raise ValueError("epoch physical synthetic counts do not reconcile")
        score = item.get("selection", {}).get("selection_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("epoch selection score is missing or non-finite")
        scores.append(float(score))
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_epoch = best_index + 1
    if report.get("best_epoch") != best_epoch or wrapper.get("best_epoch") != best_epoch:
        raise ValueError("best epoch is not the stable first maximum selection score")
    if (float(report.get("best_score", float("nan"))) != scores[best_index]
            or float(wrapper.get("best_score", float("nan"))) != scores[best_index]):
        raise ValueError("checkpoint/history best score differs from selected epoch")
    if wrapper.get("epoch") != best_epoch:
        raise ValueError("selected checkpoint epoch differs from the full-history best")
    if wrapper.get("history") != history[:best_epoch]:
        raise ValueError("selected checkpoint history is not the best-epoch history prefix")

    data_dir = Path(data_dir).resolve()
    index_path = data_dir / "combined_train.csv"
    audit_path = data_dir / "province_audit.json"
    sampler_path = Path(provenance.get("sampler", "algorithm/out/data/batch11/sampler.json")).resolve()
    expected_files = {"index_sha256": index_path, "audit_sha256": audit_path,
                      "sampler_sha256": sampler_path}
    for key, path in expected_files.items():
        if not path.is_file() or provenance.get(key) != _sha256(path):
            raise ValueError(f"training provenance mismatch for {key}")
    with index_path.open(encoding="utf-8-sig") as stream:
        train_rows = sum(1 for _ in stream) - 1
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise ValueError("audited training index does not contain the expected full physical pool")
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    independent_path = data_dir / "independent_audit.json"
    independent = json.loads(independent_path.read_text(encoding="utf-8-sig"))
    if audit.get("schema") != "province-coverage-data-v1" or independent.get("passed") is not True:
        raise ValueError("province source audit failed or has an unsupported schema")
    for name, identity in audit.get("outputs", {}).items():
        output_path = data_dir / name
        if not output_path.is_file() or _sha256(output_path) != identity.get("sha256"):
            raise ValueError(f"audited training output changed: {name}")
    init_path = Path(provenance.get("init", ""))
    if not init_path.is_file() or provenance.get("init_sha256") != _sha256(init_path):
        raise ValueError("training initialization checkpoint SHA mismatch")
    optional_sources = {
        "char_input_source_sha256": _ROOT / "algorithm" / "lpr_tools" / "train" / "province_char_input.py",
        "augmentation_source_sha256": _ROOT / "algorithm" / "lpr_tools" / "train" / "province_synth.py",
        "font_sha256": Path(r"C:\Windows\Fonts\simhei.ttf"),
    }
    verified_optional_sources = {}
    for key, source_path in optional_sources.items():
        if key in provenance:
            if not source_path.is_file() or provenance[key] != _sha256(source_path):
                raise ValueError(f"training provenance mismatch for {key}")
            verified_optional_sources[key] = True
    if _sha256(tune_path) != audit.get("source_hashes", {}).get("batch11_tune_sha256"):
        raise ValueError("safe tuning source differs from training audit provenance")
    province_counts = Counter()
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            province_counts[row["label"][0]] += 1
    audited_counts = independent.get("combined_train", {}).get("by_province")
    if (sum(province_counts.values()) != EXPECTED_TRAIN_ROWS
            or not isinstance(audited_counts, dict)
            or dict(province_counts) != audited_counts):
        raise ValueError("physical training counts by province differ from the independent audit")
    return {"history_path": str(history_path.resolve()), "history_sha256": _sha256(history_path),
            "epochs_completed": len(history), "minimum_epochs": MIN_EPOCHS[kind],
            "best_epoch": best_epoch, "best_selection_score": scores[best_index],
            "first_maximum_tie_policy": "earliest_epoch", "checkpoint_cursor": 0,
            "audited_train_rows_per_epoch": EXPECTED_TRAIN_ROWS,
            "full_real_physical_coverage_each_epoch": True,
            "physical_counts_by_province_match_audit": True,
            "physical_counts_by_province": dict(sorted(province_counts.items())),
            "synthetic_physical_counts_reconciled": True,
            "provenance": {**provenance, "init_sha256_verified": True,
                           "index_sha256_verified": True, "audit_sha256_verified": True,
                           "sampler_sha256_verified": True, "tune_sha256_verified": True,
                           "optional_sources_verified": verified_optional_sources,
                           "audit_outputs_verified": sorted(audit["outputs"])}}


def _groups(rows):
    result = {}
    for row in rows:
        group = row["subset"].removeprefix("ccpd_")
        result[group] = result.get(group, 0) + 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("box", "crnn", "char"), required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="selected full-epoch best checkpoint/training wrapper")
    parser.add_argument("--tune", type=Path, default=DEFAULT_TUNE)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--char-base", type=Path, default=DEFAULT_CHAR_BASE)
    parser.add_argument("--history", type=Path,
                        help="full history.json; defaults to model checkpoint sibling")
    parser.add_argument("--data-dir", type=Path,
                        default=_ROOT / "algorithm" / "out" / "data" / "batch12")
    args = parser.parse_args(argv)
    result = calibrate(args.kind, args.model, args.tune, model_out=args.model_out,
                       report_path=args.report, char_base_path=args.char_base,
                       history_path=args.history, data_dir=args.data_dir)
    summary = {"kind": args.kind, "model": result["export"]["path"],
               "temperature": result["temperature"],
               "fit_nll": result["fit"]["fit_nll_after"] if result["fit"] else None,
               "report_nll": result["report"]["ctc_nll_after"] if result["report"] else None,
               "province_only_proof": result["char_province_only_proof"] is not None}
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
