"""Evaluate the frozen Batch 16 camera panel with the default 384 detector."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from .detector_size_study import metrics, sha, summary

EXPECTED_INPUT_SHA256 = "62749f6c451abc38c58b61af172dc1fa15e0aaea8d7f4a45edbfddc2543b0641"
SCALES = (0.6, 0.8, 1.0)
INPUT_DEFAULT = Path(__file__).resolve().parents[2] / "out" / "receipts" / "第十六批" / "inputs" / "camera-input.json"


def run(input_path: Path, out_dir: Path) -> dict:
    import cv2
    from loguru import logger
    from lpr.config import DEFAULT_CONFIG_PATH
    from lpr.io_utils import imread
    from lpr.pipeline import PlateRecognizer
    from lpr_tools.eval_det import camera_canvas

    from .common import effective_config_info

    input_path, out_dir = Path(input_path).resolve(), Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {out_dir}")
    input_sha = sha(input_path)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError(f"camera input SHA256 differs from frozen Batch 16 input: {input_sha}")
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 600:
        raise ValueError("frozen camera panel must contain exactly 600 rows")
    required = ("path", "partition", "subset", "truth", "bbox", "quad", "brightness")
    if any(not isinstance(row, dict) or any(key not in row for key in required) for row in rows):
        raise ValueError("camera input row is missing required panel metadata")
    if len({str(row["path"]).replace("\\", "/").casefold() for row in rows}) != len(rows):
        raise ValueError("camera input must contain unique image paths")
    if os.environ.get("LPR_CONFIG"):
        raise RuntimeError("LPR_CONFIG is set; unset it to load the new default with PlateRecognizer.from_config()")

    logger.remove()
    cv2.setNumThreads(1)
    recognizer = PlateRecognizer.from_config()
    detector_size = recognizer.corner_detector.image_size
    if detector_size != 384:
        raise RuntimeError(f"default detector size must be 384; got {detector_size}")
    config_evidence = effective_config_info(recognizer.config)
    default_config = Path(DEFAULT_CONFIG_PATH).resolve()
    config_evidence["default_config_source"] = {
        "path": str(default_config),
        "sha256": sha(default_config),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    by_scale = {str(scale): [] for scale in SCALES}
    with gzip.open(out_dir / "actual.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, 1):
            image = imread(row["path"])
            for scale in SCALES:
                frame, _, _, transform = camera_canvas(image, row["bbox"], row["quad"], scale)
                result = recognizer.recognize(frame)
                actual = {
                    **row,
                    "scale": scale,
                    "status": result.status.value,
                    "plate": result.plate_no,
                    "confidence": float(result.confidence),
                    "route": getattr(result, "route", None),
                    "candidate_index": getattr(result, "candidate_index", None),
                    "rect_index": getattr(result, "rect_index", None),
                    "camera_transform": transform,
                }
                by_scale[str(scale)].append(actual)
                stream.write(json.dumps(actual, ensure_ascii=False, separators=(",", ":")) + "\n")
                del result, frame
            del image
            if index % 50 == 0:
                print(f"camera384 {index}/{len(rows)}", flush=True)

    report = {
        "schema": "batch17-camera384-v1",
        "source": {
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "expected_batch16_sha256": EXPECTED_INPUT_SHA256,
            "rows": len(rows),
            "scales": list(SCALES),
            "canvas_wh": [1280, 720],
            "single_pass_images": True,
            "ground_truth_use": "camera_canvas transform geometry only; recognizer receives generated full camera frame",
        },
        "model_version": recognizer.version,
        "detector_size": detector_size,
        "effective_config": config_evidence,
        "metrics": {
            str(scale): {
                "whole_panel": metrics(by_scale[str(scale)]),
                "groups": summary(by_scale[str(scale)]),
            }
            for scale in SCALES
        },
        "records": str((out_dir / "actual.gz").resolve()),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.input, args.out)
    print(json.dumps({"rows": report["source"]["rows"], "detector_size": report["detector_size"],
                      "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
