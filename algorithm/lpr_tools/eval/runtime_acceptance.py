"""Ordered real-pipeline comparison against immutable JSONL expectations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [r for line in stream if "_meta" not in (r := json.loads(line))]


def compare(actual, expected, tolerance=1e-7):
    differences = {key: {"expected": expected[key], "actual": actual.get(key)}
                   for key in ("status", "plate", "route", "candidate_index", "rect_index")
                   if key in expected and expected[key] != actual.get(key)}
    ac, ec = float(actual["confidence"]), float(expected["confidence"])
    if not (math.isfinite(ac) and math.isfinite(ec)) or abs(ac - ec) > tolerance:
        differences["confidence"] = {"expected": expected["confidence"], "actual": actual["confidence"]}
    return differences


def run(config, manifests, expected_path, output, *, tolerance=1e-7):
    import cv2
    from loguru import logger
    from lpr.io_utils import imread
    from lpr.pipeline import PlateRecognizer
    from lpr_tools.data.manifest import image_path, read_manifest

    from .common import effective_config_info
    logger.remove()
    cv2.setNumThreads(1)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty acceptance directory: {output}")
    expected = _rows(expected_path)
    samples = [(image_path(row, manifest), row) for manifest in manifests for row in read_manifest(manifest)]
    def norm(p):
        return str(p).replace("\\", "/").casefold()
    if [norm(p) for p, _ in samples] != [norm(r["path"]) for r in expected]:
        raise ValueError("manifest order/paths differ from frozen expectations")
    recognizer = PlateRecognizer.from_config() if config is None else PlateRecognizer.from_config(config)
    output.mkdir(parents=True, exist_ok=True)
    mismatches, statuses = [], Counter()
    with gzip.open(output/"actual.jsonl.gz", "wt", encoding="utf-8") as stream:
        for i, ((path, _), frozen) in enumerate(zip(samples, expected, strict=False), 1):
            result = recognizer.recognize(imread(path))
            actual = {"path": str(path), "status": result.status.value, "plate": result.plate_no,
                "confidence": float(result.confidence), "route": getattr(result,"route",None),
                "candidate_index":getattr(result,"candidate_index",None),
                "rect_index":getattr(result,"rect_index",None)}
            diff = compare(actual, frozen, tolerance)
            if diff:
                mismatches.append({"path":str(path),"differences":diff})
            statuses[result.status.value] += 1
            stream.write(json.dumps(actual,ensure_ascii=False,separators=(",", ":"))+"\n")
            del result
            if i % 100 == 0:
                print(f"runtime_acceptance={i}/{len(samples)}", flush=True)
    report = {"rows":len(samples),"difference_count":len(mismatches),"differences":mismatches,
        "statuses":dict(statuses),"model_version":recognizer.version,
        "expected_sha256":hashlib.sha256(Path(expected_path).read_bytes()).hexdigest(),
        "config":effective_config_info(recognizer.config,config),
        "confidence_tolerance":tolerance,"no_threshold_comparison_tolerance":True}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--manifest",action="append",required=True)
    parser.add_argument("--expected",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--confidence-tolerance", type=float, default=1e-7)
    args=parser.parse_args(argv)
    if not math.isfinite(args.confidence_tolerance) or args.confidence_tolerance < 0:
        parser.error("confidence tolerance must be finite and nonnegative")
    result=run(args.config,args.manifest,args.expected,args.out,tolerance=args.confidence_tolerance)
    print(json.dumps({"rows":result["rows"],"difference_count":result["difference_count"]}))
    return int(result["difference_count"]>0)


if __name__ == "__main__":
    raise SystemExit(main())
