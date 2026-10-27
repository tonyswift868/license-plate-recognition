"""Prepared one-time G2 runner for the frozen test manifest.

Prepare command (does not run inference):
  python -m lpr_tools.eval.frozen_once --manifest <frozen.csv> --config <model.toml> --out <receipt.json>
Formal execution requires appending --run-frozen-once. The marker is created
exclusively before the first image is evaluated, so an interrupted run cannot be
silently repeated against the frozen set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lpr.config import ALGORITHM_DIR
from lpr.types import RecogStatus
from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path, read_manifest
from .common import config_for_args, effective_config_info, make_recognizer, mode_name

GROUPS = ("clear", "night", "tilt", "dirty")
ONCE_MARKER_DIR = ALGORITHM_DIR / "out" / "receipts" / "frozen-once"


def summarize(rows, results):
    """Summarize fixed G2 groups and automatic-accept precision."""
    if len(rows) != len(results):
        raise ValueError("rows and results must align")
    report = {}
    for group in (*GROUPS, "overall"):
        selected = [(row, result) for row, result in zip(rows, results)
                    if row.category in GROUPS and (group == "overall" or row.category == group)]
        accepted = [(row, result) for row, result in selected if _status(result) == RecogStatus.OK.value]
        correct = sum(_plate(result) == row.plate_no for row, result in selected)
        accepted_correct = sum(_plate(result) == row.plate_no for row, result in accepted)
        report[group] = {
            "n": len(selected), "correct": correct,
            "plate_accuracy": correct / len(selected) if selected else None,
            "auto_accepted": len(accepted), "auto_correct": accepted_correct,
            "auto_precision": accepted_correct / len(accepted) if accepted else None,
            "statuses": dict(Counter(_status(result) for _, result in selected)),
        }
    return report


def _status(result):
    value = result["status"] if isinstance(result, dict) else result.status
    return value.value if hasattr(value, "value") else value


def _plate(result):
    return result["plate_no"] if isinstance(result, dict) else result.plate_no


def _canonical_marker(manifest: Path) -> Path:
    """Stable per-manifest guard, independent of the user-selected receipt path."""
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return ONCE_MARKER_DIR / f"{digest}.json"


def _create_once_marker(marker: Path, details: dict):
    marker.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL is the concurrency and repeat-run guard. An interrupted marker stays consumed.
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(details, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare or run one-time frozen-test G2 evaluation")
    parser.add_argument("--manifest", required=True, help="versioned frozen manifest; configurable for future sets")
    parser.add_argument("--config", required=True, help="model/runtime TOML config; configurable for future models")
    parser.add_argument("--out", required=True, help="new JSON receipt path; existing output is never overwritten")
    parser.add_argument("--run-frozen-once", action="store_true",
                        help="authorize the formal one-time frozen-set run")
    args = parser.parse_args(argv)
    manifest, output = Path(args.manifest), Path(args.out)
    marker = _canonical_marker(manifest)
    try:
        rows = read_manifest(manifest, check_files=True)
        cfg = config_for_args(args)
        config_info = effective_config_info(cfg, args.config)
        if not args.run_frozen_once:
            print(json.dumps({"prepared": True, "inference_run": False,
                              "manifest": str(manifest.resolve()), "rows": len(rows),
                              "config": config_info["config_path"], "output": str(output.resolve()),
                              "once_marker": str(marker.resolve()),
                              "run_command": "python -m lpr_tools.eval.frozen_once --manifest "
                                            f"{manifest} --config {args.config} --out {output} --run-frozen-once"},
                             ensure_ascii=False, indent=2))
            return 0
        if output.exists():
            raise ValueError(f"refusing to overwrite output: {output}")
        _create_once_marker(marker, {"state": "in_progress", "created_utc": datetime.now(timezone.utc).isoformat(),
                                     "manifest": str(manifest.resolve()), "config": config_info["config_path"]})
        recognizer = make_recognizer(args, cfg=cfg)
        results, failures = [], []
        for row in rows:
            path = image_path(row, manifest)
            image, reason = decode_image(path)
            if image is None:
                failures.append({"path": str(path), "reason": reason})
                continue
            result = recognizer.recognize(image)
            # Discard plate_image and candidate arrays immediately; only G2 fields remain resident.
            results.append({"status": result.status.value, "plate_no": result.plate_no})
            del result, image
        if failures:
            raise ValueError(f"frozen manifest has undecodable images: {len(failures)}; one-time marker remains consumed")
        summary = summarize(rows, results)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest": str(manifest.resolve()), "row_count": len(rows),
                  "config": effective_config_info(cfg, args.config), "mode": mode_name(args),
                  "model_version": recognizer.version,
                  "summary": summary,
                  "per_category": {category: summary[category] for category in GROUPS},
                  "once_marker": str(marker.resolve()),
                  "note": "No rerun permitted: the manifest-hash marker is exclusive and independent of --out."}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        marker.write_text(json.dumps({"state": "complete", "manifest_sha256": marker.stem,
                                     "output": str(output.resolve())},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
