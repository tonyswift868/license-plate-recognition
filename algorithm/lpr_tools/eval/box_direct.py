"""Directly read padded detection-box crops with the configured CRNN.

This is an offline diagnostic. It does not rectify, deskew, or save crop images.
Every frozen-manifest row remains in the denominator, including decode failures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Keep numerical libraries single-threaded before NumPy/Torch imports.
os.environ.update(dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'))

import cv2
from lpr.boxcrop import crop_box
from lpr.config import ALGORITHM_DIR, load_config
from lpr.locate import locate
from lpr.pipeline import _in_roi, _merge_candidates
from lpr.preprocess import preprocess
from lpr.recognize import load_corner_detector, load_plate_reader
from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path

from .common import effective_config_info, model_identity
from .index_samples import FIXED_HARD_MANIFEST, FIXED_VAL_MANIFEST, fixed_manifest_samples, path_key

INDEX = ALGORITHM_DIR / "data/index/ccpd_index.csv"
CEILING = ALGORITHM_DIR / "out/tmp_arch/ceiling_v2/report.json"
OUTPUT = Path("algorithm/out/receipts/第十一批/box-direct")
PAD_GRID = tuple((pad_x, pad_y) for pad_x in (.05, .10, .15, .25)
                 for pad_y in (.15, .25, .40))
RSS_LIMIT_BYTES = 1_000_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_brightness(index_path, selected_paths):
    values = {}
    with Path(index_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"path", "brightness"}.issubset(reader.fieldnames):
            raise ValueError("CCPD index must contain path and filename brightness columns")
        for raw in reader:
            raw_path = Path(raw["path"])
            if not raw_path.is_absolute():
                raw_path = Path(index_path).resolve().parent / raw_path
            key = path_key(raw_path)
            if key not in selected_paths:
                continue
            if key in values:
                raise ValueError(f"duplicate selected path in CCPD index: {raw_path}")
            token = raw["brightness"].strip()
            values[key] = int(token) if token.isdigit() else None
    if set(values) != selected_paths:
        raise ValueError(f"CCPD index brightness missing for {len(selected_paths-set(values))} fixed rows")
    return values


def _actual_candidates(image, cfg, corner_detector):
    """Mirror the ceiling locator path exactly; deliberately do no rectification."""
    roi = cfg.lane_roi(None)
    pre = preprocess(image, cfg.preprocess, roi)
    classical = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    network = []
    if corner_detector is not None:
        network = [candidate for candidate in corner_detector.detect(
            image, cfg.runtime.max_candidates) if _in_roi(candidate, roi)]
    return _merge_candidates(network, classical, cfg.runtime.max_candidates)[:3]


def _gt_bbox_from_quad(row):
    if row.quad is None or len(row.quad) != 4:
        raise ValueError(f"fixed labeled row has no four-corner GT quad: {row.path}")
    xs = [float(point[0]) for point in row.quad]
    ys = [float(point[1]) for point in row.quad]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    if right <= left or bottom <= top:
        raise ValueError(f"degenerate GT quad bounding box: {row.path}")
    return left, top, right-left, bottom-top


def _read_outcome(text, confidences, color, min_confidence, truth):
    """Apply the production CRNN text validity/status mapping to raw model output."""
    from lpr.pipeline import _decode_text

    confidence = min(confidences) if confidences else None
    decision = _decode_text(text, confidences, color, min_confidence)
    if decision is None:
        status = "NO_READ" if not text else "INVALID_OUTPUT"
        legal = False
        layout_name = None
    else:
        status = decision.status.value
        legal = status in {"OK", "LOW_CONFIDENCE"}
        layout_name = decision.layout_name or None
    matches = sum(a == b for a, b in zip(text, truth, strict=False))
    return {"prediction": text, "layout_name": layout_name,
            "min_confidence": float(confidence) if confidence is not None else None,
            "status": status, "legal": legal, "correct": text == truth,
            "matched_chars": matches, "char_count": len(truth)}


def _read_crnn_native(reader, crop, torch):
    """CRNN's own preprocessing/forward/decode for arbitrary native crop size."""
    # CrnnReader.read() currently guards canonical 252/276 x 80 inputs, while
    # the new crop_box contract intentionally returns native-resolution crops.
    # Reuse the reader's own plate_tensor preprocessing and model path so resize
    # remains CRNN-owned and no evaluator-specific resize is introduced.
    from lpr.charset import NUM_CLASSES
    from lpr.recognize.crnn import CrnnReader, _greedy_decode, plate_tensor

    if not isinstance(reader, CrnnReader):
        raise TypeError("box-direct only adapts the existing CrnnReader; future Reader types need their own native-crop contract")
    tensor = plate_tensor(crop)
    with torch.inference_mode():
        logits = reader.model(tensor[None])[..., :NUM_CLASSES + 1]
        if reader.temperature != 1.0:
            logits = logits / reader.temperature
        if not torch.isfinite(logits).all():
            return "", []
        text, confidences = _greedy_decode(logits[:, 0])
    return text, confidences


def _read_crop(reader, crop, color, min_confidence, truth, torch):
    if crop.size == 0:
        return {"prediction": "", "layout_name": None, "min_confidence": None,
                "status": "BAD_CROP", "legal": False, "correct": False,
                "matched_chars": 0, "char_count": len(truth)}
    text, confidences = _read_crnn_native(reader, crop, torch)
    return _read_outcome(text, confidences, color, min_confidence, truth)


def _select_candidate(candidate_reads):
    """Highest min-character confidence among legal reads; strict > preserves rank ties."""
    selected = None
    for item in candidate_reads:
        if not item["legal"] or item["min_confidence"] is None:
            continue
        if selected is None or item["min_confidence"] > selected["min_confidence"]:
            selected = item
    if selected is None:
        return {"candidate_index": None, "prediction": "", "min_confidence": None,
                "status": "NO_LEGAL_READ", "correct": False,
                "matched_chars": 0, "char_count": (candidate_reads[0]["char_count"]
                                                        if candidate_reads else 0)}
    return {"candidate_index": selected["candidate_index"],
            "prediction": selected["prediction"],
            "min_confidence": selected["min_confidence"],
            "status": selected["status"], "correct": selected["correct"],
            "matched_chars": selected["matched_chars"],
            "char_count": selected["char_count"]}


def _rss_bytes():
    """Best-effort process RSS through platform APIs; no optional dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                                              wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return None


def _locator_signature(config):
    runtime = config.get("runtime", {})
    return {"preprocess": config.get("preprocess"), "locate": config.get("locate"),
            "max_candidates": runtime.get("max_candidates")}


def _ceiling_compatibility(path, manifests, index_sha, cfg, model_ids):
    if path is None or not Path(path).is_file():
        return {"compatible": False, "reason": "ceiling report unavailable"}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"compatible": False, "reason": f"ceiling report unreadable: {type(exc).__name__}"}
    reasons = []
    for name in ("val", "devhard"):
        expected = manifests[name]["manifest_sha256"]
        observed = data.get("manifests", {}).get(name, {}).get("sha256")
        if expected != observed:
            reasons.append(f"{name} manifest SHA mismatch")
    if data.get("index", {}).get("sha256") != index_sha:
        reasons.append("index SHA mismatch")
    archived_config = data.get("config_and_models", {}).get("effective_config")
    if not isinstance(archived_config, dict) or _locator_signature(archived_config) != _locator_signature(cfg.model_dump(mode="json")):
        reasons.append("locator config mismatch")
    archived_models = data.get("config_and_models", {}).get("models", {})
    model_match = {}
    for key, field in (("classifier", "cnn"), ("corner", "corner"), ("plate_reader", "crnn")):
        expected = archived_models.get(key, {}).get("sha256")
        observed = model_ids[field].get("sha256")
        model_match[field] = expected == observed
    # CNN does not participate in either direct box reading or ceiling's isolated CRNN cell.
    if not model_match["corner"]:
        reasons.append("corner model SHA mismatch")
    if not model_match["crnn"]:
        reasons.append("CRNN model SHA mismatch")
    compatible = not reasons
    return {"compatible": compatible, "reason": "; ".join(reasons) if reasons else None,
            "path": str(Path(path).resolve()), "sha256": sha256_file(path),
            "model_sha_matches": model_match, "report": data if compatible else None}


def _group_keys(split, subset, brightness):
    keys = ["all", split, f"by_subset/{subset}", f"by_split_subset/{split}/{subset}"]
    if subset == "ccpd_db" and brightness is not None and brightness <= 19:
        keys += ["ccpd_db_brightness_le_19", f"{split}/ccpd_db_brightness_le_19"]
    if subset in {"ccpd_tilt", "ccpd_rotate"}:
        keys += ["ccpd_tilt_plus_rotate", f"{split}/ccpd_tilt_plus_rotate"]
    return keys


def _new_metrics():
    return Counter()


def _summarize_counts(metrics, ceiling_report, split=None, subset=None):
    rows = {}
    for pad_order, (pad_x, pad_y) in enumerate(PAD_GRID):
        counter = metrics[pad_order]
        item = {key: int(counter.get(key, 0)) for key in (
            "n", "plate_correct", "char_correct", "char_total", "gt_plate_correct",
            "gt_char_correct", "oracle_any_candidate_correct", "no_legal_prediction",
            "decode_errors", "locator_errors")}
        item["plate_accuracy"] = item["plate_correct"] / item["n"] if item["n"] else None
        item["character_accuracy"] = item["char_correct"] / item["char_total"] if item["char_total"] else None
        item["gt_plate_accuracy_diagnostic"] = item["gt_plate_correct"] / item["n"] if item["n"] else None
        item["gt_character_accuracy_diagnostic"] = item["gt_char_correct"] / item["char_total"] if item["char_total"] else None
        item["oracle_any_candidate_accuracy_diagnostic"] = item["oracle_any_candidate_correct"] / item["n"] if item["n"] else None
        item.update({"pad_order": pad_order, "pad_x": pad_x, "pad_y": pad_y})
        rows[f"{pad_order:02d}"] = item
    if ceiling_report is not None:
        if split is not None:
            ceiling_groups = [(split, subset or "all")]
        else:
            # For combined val+hard groups, sum the exact matching ceiling cells.
            ceiling_key = subset or "all"
            ceiling_groups = [(name, ceiling_key) for name in ("val", "devhard")]
        cells = [ceiling_report.get("groups", {}).get(name, {}).get(key, {}).get("actual/crnn")
                 for name, key in ceiling_groups]
        cells = [cell for cell in cells if cell]
        if cells:
            correct = sum(cell["correct"] for cell in cells)
            count = sum(cell["n"] for cell in cells)
            for item in rows.values():
                item["ceiling_actual_crnn_correct"] = correct
                item["ceiling_actual_crnn_n"] = count
                item["ceiling_actual_crnn_accuracy"] = correct/count if count else None
    return rows


def _choose_best_pad(actual_full_correct):
    if len(actual_full_correct) != len(PAD_GRID):
        raise ValueError("best-pad selection requires one count per fixed pad setting")
    # max() preserves the earliest PAD_GRID order for ties; GT metrics are not passed here.
    index = max(range(len(PAD_GRID)), key=lambda i: actual_full_correct[i])
    return {"pad_order": index, "pad_x": PAD_GRID[index][0], "pad_y": PAD_GRID[index][1],
            "correct": int(actual_full_correct[index]),
            "criterion": "maximum exact full-plate correctness over val+devhard actual-selection results; fixed pad order breaks ties"}


def _evaluate_row(sample, split, manifest, brightness, image, decode_error,
                  cfg, corner_detector, reader, torch):
    truth = sample.row.plate_no
    bbox = _gt_bbox_from_quad(sample.row)
    read_error_count = 0
    locator_error = None
    candidates = []
    if image is not None:
        try:
            candidates = _actual_candidates(image, cfg, corner_detector)
        except Exception as exc:  # retain this row and its fixed denominator
            locator_error = f"{type(exc).__name__}:{exc}"
    candidate_info = [{"index": index, "bbox": [int(v) for v in candidate.bbox],
                       "color": candidate.color, "score": float(candidate.score),
                       "source": candidate.source}
                      for index, candidate in enumerate(candidates, 1)]
    pads = []
    for pad_order, (pad_x, pad_y) in enumerate(PAD_GRID):
        if image is None:
            gt_read = {"prediction": "", "min_confidence": None, "status": "DECODE_ERROR",
                       "legal": False, "correct": False, "matched_chars": 0,
                       "char_count": len(truth)}
            candidate_reads = []
        else:
            try:
                gt_crop = crop_box(image, bbox, pad_x, pad_y)
                gt_read = _read_crop(reader, gt_crop, sample.row.plate_color,
                                     cfg.postprocess.min_confidence, truth, torch)
            except Exception as exc:
                read_error_count += 1
                gt_read = {"prediction": "", "min_confidence": None,
                           "status": "READ_ERROR", "error": f"{type(exc).__name__}:{exc}",
                           "legal": False, "correct": False, "matched_chars": 0,
                           "char_count": len(truth)}
            candidate_reads = []
            for index, candidate in enumerate(candidates, 1):
                try:
                    crop = crop_box(image, candidate.bbox, pad_x, pad_y)
                    read = _read_crop(reader, crop, candidate.color,
                                      cfg.postprocess.min_confidence, truth, torch)
                except Exception as exc:
                    read_error_count += 1
                    read = {"prediction": "", "min_confidence": None,
                            "status": "READ_ERROR", "error": f"{type(exc).__name__}:{exc}",
                            "legal": False, "correct": False, "matched_chars": 0,
                            "char_count": len(truth)}
                read["candidate_index"] = index
                read["bbox"] = candidate_info[index-1]["bbox"]
                read["source"] = candidate.source
                read["score"] = float(candidate.score)
                candidate_reads.append(read)
        selected = _select_candidate(candidate_reads)
        pads.append({"pad_order": pad_order, "pad_x": pad_x, "pad_y": pad_y,
                     "ground_truth_bbox": [float(v) for v in bbox],
                     "ground_truth_bbox_source": "quad_axis_aligned_minmax",
                     "ground_truth": gt_read, "candidates": candidate_reads,
                     "actual_selection": selected,
                     "oracle_any_candidate_correct_diagnostic": any(r["correct"] for r in candidate_reads)})
    return {"split": split, "index_split": sample.split, "dataset": sample.dataset,
            "path": sample.row.path, "truth": truth, "plate_color": sample.row.plate_color,
            "subset": sample.subset, "brightness": brightness,
            "ground_truth_bbox": [float(v) for v in bbox],
            "ground_truth_bbox_source": "quad_axis_aligned_minmax",
            "ground_truth_quad": [[float(x), float(y)] for x, y in sample.row.quad],
            "decode_error": decode_error, "locator_error": locator_error,
            "read_error_count": read_error_count,
            "candidates": candidate_info, "pads": pads}


def run(*, index=INDEX, val_manifest=FIXED_VAL_MANIFEST,
        devhard_manifest=FIXED_HARD_MANIFEST, ceiling_report=CEILING,
        config=None, output=OUTPUT, cnn_model=None, corner_model=None,
        crnn_model=None, torch_threads=1, overwrite=False, monitor_every=100):
    if torch_threads not in (1, 2):
        raise ValueError("torch_threads must be 1 or 2")
    if monitor_every < 1:
        raise ValueError("monitor_every must be positive")
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty box-direct output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifests = {"val": Path(val_manifest).resolve(), "devhard": Path(devhard_manifest).resolve()}
    samples_by_split = {}
    identities = {}
    for split, purpose in (("val", "val"), ("devhard", "dev-hard")):
        samples_by_split[split], identities[split] = fixed_manifest_samples(
            index, manifests[split], purpose=purpose)
    samples = [(sample, split) for split in ("val", "devhard")
               for sample in samples_by_split[split]]
    selected_paths = {path_key(sample.row.path) for sample, _split in samples}
    brightness_by_path = _read_brightness(index, selected_paths)

    overrides = {f"runtime__{field}": value for field, value in (
        ("model_path", cnn_model), ("corner_model_path", corner_model),
        ("plate_reader_model_path", crnn_model)) if value is not None}
    overrides["runtime__torch_threads"] = torch_threads
    cfg = load_config(config, **overrides)
    if not cfg.runtime.plate_reader_model_path:
        raise ValueError("config must enable runtime.plate_reader_model_path")
    import torch
    torch.set_num_threads(torch_threads)
    cv2.setNumThreads(1)
    cfg_info = effective_config_info(cfg, config)
    models = {"cnn": model_identity(cfg, "model_path"),
              "corner": model_identity(cfg, "corner_model_path"),
              "crnn": model_identity(cfg, "plate_reader_model_path")}
    index_sha = sha256_file(index)
    ceiling_check = _ceiling_compatibility(ceiling_report, identities, index_sha, cfg, models)
    ceiling_data = ceiling_check.pop("report", None)
    source_sha = sha256_file(Path(__file__))
    algorithm_root = Path(__file__).resolve().parents[2]
    runtime_sources = {name: sha256_file(algorithm_root / "lpr" / filename)
                       for name, filename in (("boxcrop", "boxcrop.py"),
                                              ("crnn", "recognize/crnn.py"),
                                              ("pipeline", "pipeline.py"))}
    metadata = {"status": "starting", "started_utc": datetime.now(UTC).isoformat(),
                "command": list(sys.argv), "config": cfg_info,
                "models": models, "index": {"path": str(Path(index).resolve()), "sha256": index_sha},
                "manifests": identities, "ceiling_compatibility": ceiling_check,
                "tool_source_sha256": source_sha, "runtime_source_sha256": runtime_sources,
                "runtime_threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads(),
                                    "blas_env": {name: os.environ.get(name) for name in (
                                        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
                "pad_grid": [{"pad_order": i, "pad_x": px, "pad_y": py}
                             for i, (px, py) in enumerate(PAD_GRID)],
                "candidate_order": "corner detector outputs first, then classical locate, merged with pipeline._merge_candidates; preserved order is score/rank tie-break order",
                "selection": "highest min-character confidence among syntactically legal reads (OK or LOW_CONFIDENCE); strict-greater update preserves earlier candidate ties",
                "fixed_denominator": len(samples), "records": 0,
                "resource_monitor": {"rss_limit_bytes": RSS_LIMIT_BYTES,
                                     "sample_every_records": monitor_every, "samples": []}}
    meta_path = out / "run_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    corner_detector = (load_corner_detector(cfg.resolve_path(cfg.runtime.corner_model_path))
                       if cfg.runtime.corner_model_path else None)
    reader = load_plate_reader(cfg.resolve_path(cfg.runtime.plate_reader_model_path))
    rss = _rss_bytes()
    if rss is not None:
        metadata["resource_monitor"]["samples"].append({"records": 0, "rss_bytes": rss})
    if rss is not None and rss > RSS_LIMIT_BYTES:
        metadata["status"] = "aborted_memory_limit"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raise MemoryError(f"box-direct process RSS {rss} exceeds 1 GiB limit")

    groups = defaultdict(lambda: [_new_metrics() for _ in PAD_GRID])
    decode_errors, locator_errors, read_errors = [], [], []
    trace_path = out / "trace.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as trace:
        for row_number, (sample, split) in enumerate(samples, 1):
            brightness = brightness_by_path[path_key(sample.row.path)]
            image_pathname = image_path(sample.row, manifests[split])
            image, decode_error = decode_image(image_pathname)
            if decode_error:
                image, retry_error = decode_image(image_pathname)
                decode_error = retry_error
            if decode_error:
                decode_errors.append({"split": split, "path": sample.row.path, "error": decode_error})
            record = _evaluate_row(sample, split, manifests[split], brightness, image,
                                   decode_error, cfg, corner_detector, reader, torch)
            if record["locator_error"]:
                locator_errors.append({"split": split, "path": sample.row.path,
                                       "error": record["locator_error"]})
            if record["read_error_count"]:
                read_errors.append({"split": split, "path": sample.row.path,
                                    "count": record["read_error_count"]})
            for pad_order, pad in enumerate(record["pads"]):
                selected = pad["actual_selection"]
                gt = pad["ground_truth"]
                metric = {"n": 1, "plate_correct": int(selected["correct"]),
                          "char_correct": int(selected["matched_chars"]),
                          "char_total": len(sample.row.plate_no),
                          "gt_plate_correct": int(gt["correct"]),
                          "gt_char_correct": int(gt["matched_chars"]),
                          "oracle_any_candidate_correct": int(pad["oracle_any_candidate_correct_diagnostic"]),
                          "no_legal_prediction": int(selected["candidate_index"] is None),
                          "decode_errors": int(bool(decode_error) and pad_order == 0),
                          "locator_errors": int(bool(record["locator_error"]) and pad_order == 0)}
                for key in _group_keys(split, sample.subset, brightness):
                    groups[key][pad_order].update(metric)
            trace.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            trace.flush()
            metadata["records"] = row_number
            if row_number % monitor_every == 0 or row_number == len(samples):
                rss = _rss_bytes()
                if rss is not None:
                    metadata["resource_monitor"]["samples"].append({"records": row_number,
                                                                      "rss_bytes": rss})
                    if rss > RSS_LIMIT_BYTES:
                        metadata["status"] = "aborted_memory_limit"
                        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                        raise MemoryError(f"box-direct process RSS {rss} exceeds 1 GiB limit")
                meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"processed={row_number}/{len(samples)} rss_bytes={rss}", flush=True)
            del image, record

    combined_metrics = groups["all"]
    best_pad = _choose_best_pad([counter["plate_correct"] for counter in combined_metrics])
    summaries = {}
    for key, value in sorted(groups.items()):
        split = key if key in {"val", "devhard"} else None
        subset = None
        if key.startswith("by_split_subset/"):
            _prefix, split, subset = key.split("/", 2)
        elif key.startswith(("val/", "devhard/")):
            split, subset = key.split("/", 1)
        elif key.startswith("by_subset/"):
            subset = key.split("/", 1)[1]
        summaries[key] = _summarize_counts(value, ceiling_data, split=split, subset=subset)
    metadata["status"] = "complete" if not decode_errors and not locator_errors and not read_errors else "completed_with_errors"
    metadata["finished_utc"] = datetime.now(UTC).isoformat()
    report = {"status": metadata["status"], "method": metadata,
              "record_count": len(samples), "decode_error_count": len(decode_errors),
              "locator_error_count": len(locator_errors),
              "read_error_count": sum(item["count"] for item in read_errors),
              "decode_errors": decode_errors, "locator_errors": locator_errors,
              "read_errors": read_errors,
              "groups": summaries, "best_pad_from_actual_selection": best_pad,
              "trace": str(trace_path.resolve()),
              "notes": {"gt_scope": "GT bbox readings use the float min/max AABB of annotated four corners and manifest GT color; diagnostic only and excluded from pad selection.",
                        "oracle_scope": "Any-candidate-correct is an oracle-only upper bound and is excluded from actual selection and pad selection.",
                        "character_accuracy": "Position-wise matches divided by fixed truth-character count; missing/extra characters do not increase numerator.",
                        "box_crop": "All crops use lpr.boxcrop.crop_box at native resolution. Existing CrnnReader.read rejects noncanonical sizes, so this tool does not call read(native); its adapter reuses the same plate_tensor, temperature, model forward and greedy decoder. Canonical-size parity is tested. This is specific to CrnnReader and does not claim future Reader types support native crops.",
                        "ceiling_comparison": "Only compared when fixed manifests/index, locator config, corner SHA and CRNN SHA match ceiling_v2. Ceiling actual/crnn is first isolated OK among rectified routes; box-direct selects the highest-confidence legal candidate, so differences are not a pure geometry effect."}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(INDEX))
    parser.add_argument("--val", default=str(FIXED_VAL_MANIFEST))
    parser.add_argument("--dev-hard", default=str(FIXED_HARD_MANIFEST))
    parser.add_argument("--ceiling-report", default=str(CEILING))
    parser.add_argument("--config", help="temporary LPR config; defaults to current default.toml")
    parser.add_argument("--cnn-model", help="classifier path override for identity; box route does not invoke CNN")
    parser.add_argument("--corner-model", help="CornerDetector path override")
    parser.add_argument("--crnn-model", help="PlateReader path override")
    parser.add_argument("--threads", choices=(1, 2), type=int, default=1)
    parser.add_argument("--monitor-every", type=int, default=100)
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run(index=args.index, val_manifest=args.val, devhard_manifest=args.dev_hard,
                 ceiling_report=args.ceiling_report, config=args.config, output=args.out,
                 cnn_model=args.cnn_model, corner_model=args.corner_model,
                 crnn_model=args.crnn_model, torch_threads=args.threads,
                 overwrite=args.overwrite, monitor_every=args.monitor_every)
    print(f"report={Path(args.out) / 'report.json'}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
