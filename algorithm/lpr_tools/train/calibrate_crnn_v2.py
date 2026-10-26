"""Fit one CTC-logit temperature on tune/fit and audit it on tune/report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize import write_manifest_entry
from lpr.recognize.crnn import CrnnNet, _greedy_decode, plate_tensor
from lpr_tools.train.eval_crnn_v2 import CONFIDENCE_EDGES, _confidence_bin, _resolve_crop

BOUNDS = (0.05, 10.0)


def _read_rows(path: Path, wanted_fold: str) -> list[dict[str, str]]:
    required = {"file", "orig_path", "source", "layout", "plate_no", "crop_path",
                "subset", "brightness", "partition", "calibration_fold"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"index must include columns: {sorted(required)}")
        rows = list(reader)
    if any(row["partition"].lower() in {"devhard", "val1000"} for row in rows):
        raise ValueError("calibration index must not include final devhard/val1000 partitions")
    selected = [row for row in rows if row["calibration_fold"] == wanted_fold]
    if not selected:
        raise ValueError(f"index contains no {wanted_fold!r} fold rows")
    for row in selected:
        label = row["plate_no"]
        if len(label) not in (7, 8) or row["layout"] != str(len(label)):
            raise ValueError(f"invalid label/layout for {row['file']}")
        if any(char not in CHAR_TO_INDEX for char in label):
            raise ValueError(f"label has character outside frozen charset: {row['file']}")
    return selected


def _load_image(index_path: Path, row: dict[str, str]) -> np.ndarray:
    image_path = _resolve_crop(index_path, row["crop_path"])
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable crop: {image_path}")
    return image


def _ctc_loss(logits: torch.Tensor, labels: list[str], temperature: float = 1.0) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1] != 1 or logits.shape[2] != NUM_CLASSES + 1:
        raise ValueError("expected CRNN logits shaped T x 1 x (NUM_CLASSES+1)")
    targets = torch.tensor([CHAR_TO_INDEX[c] for c in labels[0]], dtype=torch.long)
    log_probs = (logits / temperature).log_softmax(dim=-1)
    return F.ctc_loss(log_probs, targets, torch.tensor([logits.shape[0]], dtype=torch.long),
                      torch.tensor([len(targets)], dtype=torch.long), blank=NUM_CLASSES,
                      reduction="sum", zero_infinity=False)


def _batched_ctc_loss(logits_list: list[torch.Tensor], labels: list[str],
                      temperature: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Mean per-sample summed CTC NLL, with one batched kernel for equal-length logits."""
    if not logits_list or len(logits_list) != len(labels):
        raise ValueError("one non-empty logits tensor is required per label")
    shape = logits_list[0].shape
    if (len(shape) != 3 or shape[1] != 1 or shape[2] != NUM_CLASSES + 1
            or any(tuple(logits.shape) != tuple(shape) for logits in logits_list)):
        raise ValueError("batched CTC requires same-shaped T x 1 x C logits tensors")
    if any(not label or any(char not in CHAR_TO_INDEX for char in label) for label in labels):
        raise ValueError("CTC labels must be non-empty strings in the frozen charset")
    device, dtype = logits_list[0].device, logits_list[0].dtype
    temp = torch.as_tensor(temperature, dtype=dtype, device=device)
    if temp.ndim != 0 or not torch.isfinite(temp).item() or temp.item() <= 0:
        raise ValueError("temperature must be a finite positive scalar")
    logits = torch.cat(logits_list, dim=1)
    targets = torch.tensor([CHAR_TO_INDEX[c] for label in labels for c in label],
                           dtype=torch.long, device=device)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long,
                                  device=device)
    input_lengths = torch.full((len(labels),), shape[0], dtype=torch.long, device=device)
    return F.ctc_loss((logits / temp).log_softmax(dim=-1), targets, input_lengths,
                      target_lengths, blank=NUM_CLASSES, reduction="sum",
                      zero_infinity=False) / len(labels)


def fit_temperature(logits_list: list[torch.Tensor], labels: list[str],
                    bounds: tuple[float, float] = BOUNDS) -> dict[str, float]:
    """Minimize summed true-label CTC NLL over fit fold with bounded log(T)."""
    lower, upper = map(float, bounds)
    if not (0 < lower < upper and math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("temperature bounds must be finite positive increasing values")
    log_t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=80,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9)
    lo, hi = math.log(lower), math.log(upper)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        bounded = log_t.clamp(lo, hi)
        loss = _batched_ctc_loss(logits_list, labels, bounded.exp())
        # Straight-through derivative outside the interval is unnecessary: clamp grad is zero.
        loss.backward()
        return loss

    optimizer.step(closure)
    optimized_temperature = min(upper, max(lower, math.exp(float(log_t.detach()))))
    temperature = float(torch.tensor(optimized_temperature, dtype=torch.float32).item())
    raw = float(_batched_ctc_loss([x.double() for x in logits_list], labels))
    scaled = float(_batched_ctc_loss([x.double() for x in logits_list], labels, temperature))
    return {"temperature": temperature, "optimized_temperature_float64": optimized_temperature,
            "fit_nll_before": raw, "fit_nll_after": scaled,
            "n": len(labels), "lower_bound": lower, "upper_bound": upper}


def _ece_summary(correct: list[bool], confidence: list[float]) -> dict[str, Any]:
    buckets = [{"lower": lo, "upper": hi, "n": 0, "correct": 0, "confidence_sum": 0.0}
               for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:])]
    brier = 0.0
    for ok, conf in zip(correct, confidence):
        conf = min(1.0, max(0.0, float(conf)))
        bucket = buckets[_confidence_bin(conf)]
        bucket["n"] += 1
        bucket["correct"] += int(ok)
        bucket["confidence_sum"] += conf
        brier += (conf - float(ok)) ** 2
    n = len(correct)
    ece = 0.0
    rendered = []
    for bucket in buckets:
        count = bucket["n"]
        accuracy = bucket["correct"] / count if count else None
        mean_conf = bucket["confidence_sum"] / count if count else None
        if count:
            ece += count / n * abs(accuracy - mean_conf)
        rendered.append({"range": [bucket["lower"], bucket["upper"]], "n": count,
                         "accuracy": accuracy, "mean_confidence": mean_conf})
    return {"n": n, "brier": brier / n, "ece": ece, "bins": rendered}


def _collect_logits(model: CrnnNet, index_path: Path,
                    rows: list[dict[str, str]]) -> list[torch.Tensor]:
    result = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            tensor = plate_tensor(_load_image(index_path, row))
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            logits = model(tensor).detach().cpu()
            if not torch.isfinite(logits).all():
                raise ValueError(f"non-finite logits for {row['file']}")
            result.append(logits)
    return result


def _report_fold(model: CrnnNet, rows: list[dict[str, str]], logits: list[torch.Tensor],
                 temperature: float) -> dict[str, Any]:
    before_nll = after_nll = 0.0
    truth_correct: list[bool] = []
    conf_before: list[float] = []
    conf_after: list[float] = []
    for row, raw_logits in zip(rows, logits):
        truth = row["plate_no"]
        if not torch.equal(raw_logits.argmax(-1), (raw_logits / temperature).argmax(-1)):
            raise AssertionError("positive scalar temperature changed CTC frame argmax")
        before_nll += float(_ctc_loss(raw_logits.double(), [truth]))
        after_nll += float(_ctc_loss(raw_logits.double(), [truth], temperature))
        pred_raw, char_conf_raw = _greedy_decode(raw_logits[:, 0])
        pred_temp, char_conf_temp = _greedy_decode(raw_logits[:, 0] / temperature)
        if pred_raw != pred_temp:
            raise AssertionError("positive scalar temperature changed CTC argmax/string")
        truth_correct.append(pred_raw == truth)
        conf_before.append(min(char_conf_raw) if char_conf_raw else 0.0)
        conf_after.append(min(char_conf_temp) if char_conf_temp else 0.0)
    n = len(rows)
    return {"n": n, "temperature": temperature,
            "ctc_nll_before": before_nll / n, "ctc_nll_after": after_nll / n,
            "argmax_string_invariant": True,
            "confidence_before": _ece_summary(truth_correct, conf_before),
            "confidence_after": _ece_summary(truth_correct, conf_after)}


def _validate_fold_integrity(index_path: Path, fit_rows: list[dict[str, str]],
                             report_rows: list[dict[str, str]]) -> dict[str, Any]:
    fit_origins = {os.path.normcase(os.path.normpath(row["orig_path"])) for row in fit_rows}
    report_origins = {os.path.normcase(os.path.normpath(row["orig_path"])) for row in report_rows}
    if "" in fit_origins or "" in report_origins:
        raise ValueError("fit/report rows must have original image paths")
    if fit_origins & report_origins:
        raise ValueError("fit and report folds share original image paths")
    evidence: dict[str, Any] = {"fit_unique_origins": len(fit_origins),
                                "report_unique_origins": len(report_origins),
                                "origin_path_overlap": 0,
                                "adjacent_audit_verified": False}
    audit_path = index_path.parent / "audit.json"
    if audit_path.is_file():
        if index_path.name.lower() != "tune.csv":
            raise ValueError("when a preparation audit is present, calibration requires tune.csv")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_sha = audit.get("output_sha256", {}).get("tune")
        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if not isinstance(expected_sha, str) or digest.lower() != expected_sha.lower():
            raise ValueError("tune.csv SHA-256 does not match the adjacent preparation audit")
        overlap = audit.get("content_checks", {}).get("fit_report_sha_overlap")
        if overlap != 0:
            raise ValueError("preparation audit does not certify zero fit/report SHA overlap")
        evidence.update(adjacent_audit_verified=True, tune_sha256=digest,
                        fit_report_sha_overlap=overlap,
                        audit_path=str(audit_path.resolve()))
    return evidence


def calibrate(index_path: Path, base_model_path: Path, out_dir: Path, *,
              version: str = "crnn_v2_calibrated") -> dict[str, Any]:
    fit_rows = _read_rows(index_path, "fit")
    report_rows = _read_rows(index_path, "report")
    if any(row["partition"].lower() not in {"tune", "fit", "report", ""}
           for row in fit_rows + report_rows):
        raise ValueError("calibration is restricted to tune fit/report rows")
    fold_integrity = _validate_fold_integrity(index_path, fit_rows, report_rows)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    state = torch.load(base_model_path, map_location="cpu", weights_only=True)
    state = {key: value for key, value in state.items() if key != "ctc_temperature"}
    model = CrnnNet().cpu().eval()
    model.load_state_dict(state, strict=True)
    fit_logits = _collect_logits(model, index_path, fit_rows)
    fitted = fit_temperature(fit_logits, [row["plate_no"] for row in fit_rows])
    report_logits = _collect_logits(model, index_path, report_rows)
    report = _report_fold(model, report_rows, report_logits, fitted["temperature"])

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "crnn_v2_temperature.pt"
    if candidate_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration candidate: {candidate_path}")
    exported = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    temperature_tensor = torch.tensor(fitted["temperature"], dtype=torch.float32)
    fitted["temperature"] = float(temperature_tensor.item())
    exported["ctc_temperature"] = temperature_tensor
    torch.save(exported, candidate_path)
    write_manifest_entry(candidate_path, version=version, train_data=str(index_path.resolve()),
                         val_char_acc=None, input_size=(160, 40))
    result = {"base_model": str(base_model_path.resolve()), "index": str(index_path.resolve()),
              "fit_fold": fitted, "report_fold": report, "fold_integrity": fold_integrity,
              "candidate": str(candidate_path.resolve()),
              "scope_note": "fit/report are development tuning folds, not blind evaluation"}
    (out_dir / "calibration_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="tune index only")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="crnn_v2_calibrated")
    args = parser.parse_args()
    print(json.dumps(calibrate(args.index, args.model, args.out, version=args.version),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
