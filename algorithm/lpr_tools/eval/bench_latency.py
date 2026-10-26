"""G1: letterbox inputs, warm up five calls, then measure recognition wall time."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lpr.io_utils import list_images
from lpr_tools.data.manifest import image_path, read_manifest
from lpr_tools.data.image_health import decode_image
from .common import (add_backend_args, config_for_args, effective_config_info,
                     latency_stats, letterbox, make_recognizer,
                     mode_name, parse_size, runtime_info)

STAGES = ("preprocess", "locate", "rectify", "segment", "recognize", "postprocess")
WARMUP_CALLS = 5


def benchmark(recognizer, images, *, n=100, size=(1280,720), truths=None):
    if n < 1 or not images or min(size) < 1:
        raise ValueError("benchmark needs n>0, nonempty images and positive size")
    if truths is not None and len(truths) != len(images):
        raise ValueError("truths must align one-to-one with images")
    inputs = [letterbox(image,size) for image in images]
    for i in range(WARMUP_CALLS):
        recognizer.recognize(inputs[i % len(inputs)])
    elapsed, results = [], []
    selected = {"crnn_selected_count": 0, "network_candidate_selected_count": 0,
                "reader_invoked_images": 0, "corner_invoked_images": 0,
                "reader_invoked_calls": 0, "corner_invoked_calls": 0,
                "crnn_selected_correct_count": 0,
                "crnn_selected_with_ground_truth_count": 0}
    for i in range(n):
        start = time.perf_counter()
        sample_index = i % len(inputs)
        result = recognizer.recognize(inputs[sample_index])
        elapsed.append((time.perf_counter()-start)*1000)
        results.append(result)
        provenance = getattr(recognizer, "last_selection", None)
        if provenance is not None:
            selected["crnn_selected_count"] += bool(provenance["crnn_selected"])
            selected["network_candidate_selected_count"] += provenance["source"] == "net"
            if (provenance["crnn_selected"] and truths is not None
                    and truths[sample_index] is not None):
                selected["crnn_selected_with_ground_truth_count"] += 1
                selected["crnn_selected_correct_count"] += result.plate_no == truths[sample_index]
        selected["reader_invoked_images"] += "reader" in result.stage_ms
        selected["corner_invoked_images"] += "corner" in result.stage_ms
        selected["reader_invoked_calls"] += getattr(recognizer, "reader_invoked_calls", 0)
        selected["corner_invoked_calls"] += getattr(recognizer, "corner_invoked_calls", 0)
    stage_names = sorted(set(STAGES).union(*(r.stage_ms for r in results)))
    return {"n":n, "warmup":WARMUP_CALLS, "size":f"{size[0]}x{size[1]}",
            "wall_time":latency_stats(elapsed),
            "stage_ms":{stage:latency_stats([r.stage_ms.get(stage,0.) for r in results]) for stage in stage_names},
            "statuses":dict(Counter(r.status.value for r in results)),
            "network_attribution": {**selected,
                "crnn_selected_correct_count": (selected["crnn_selected_correct_count"]
                                                if truths is not None else None),
                "crnn_selected_accuracy": (
                    selected["crnn_selected_correct_count"] / selected["crnn_selected_with_ground_truth_count"]
                    if selected["crnn_selected_with_ground_truth_count"] else None),
                "crnn_selected_accuracy_status": (
                    "unknown_no_ground_truth" if truths is None else
                    "available" if selected["crnn_selected_with_ground_truth_count"] else "no_crnn_selected_results"),
                "ground_truth_available": truths is not None,
                "ground_truth_note": "CRNN correctness is unknown without manifest ground truth"
                                     if truths is None else None,
                "selection_source": "observed _Attempt at final _build; no-output rows are not selected",
                "invocation_source": "stage_ms key means invoked only; it does not imply selected"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU G1 latency benchmark; fake/uniform runs are smoke tests")
    add_backend_args(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest",help="manifest CSV; paths resolve beside CSV")
    source.add_argument("--images",nargs="+",help="image files or directories")
    parser.add_argument("--n",type=int,default=100)
    parser.add_argument("--size",default="1280x720",help="letterbox target WIDTHxHEIGHT")
    args = parser.parse_args(argv)
    try:
        cfg = config_for_args(args)
        size = parse_size(args.size)
        if args.n < 1:
            raise ValueError("--n must be positive")
        if args.manifest:
            paths = [image_path(row,args.manifest) for row in read_manifest(args.manifest,check_files=False)]
        elif args.images:
            paths = [path for root in args.images for path in list_images(root,recursive=True)]
        elif args.fake or args.uniform:
            paths = []
        else:
            raise ValueError("provide --manifest or --images")
        requested = len(paths)
        images, truths, decode_errors = [], [], []
        truth_by_path = {}
        if args.manifest:
            truth_by_path = {str(image_path(row, args.manifest)): row.plate_no
                             for row in read_manifest(args.manifest, check_files=False)}
        for path in paths:
            image, reason = decode_image(path)
            if image is None:
                decode_errors.append({"path": str(path), "reason": reason})
            else:
                images.append(image)
                truths.append(truth_by_path.get(str(path)) if args.manifest else None)
        decoded = len(images)
        if not paths and not (args.manifest or args.images):
            # Smoke modes without an explicit source use one synthetic frame.
            images = [np.zeros((720,1280,3),np.uint8)]
            truths = [None]
        info = runtime_info(cfg)
        if images:
            recognizer = make_recognizer(args, cfg=cfg)
            report = benchmark(recognizer,images,n=args.n,size=size,
                               truths=truths if args.manifest else None)
            model_version = recognizer.version
        else:
            # Empty manifests, empty directories and all-undecodable inputs are
            # valid benchmark outcomes; retain the requested counts in JSON.
            report = {"n": 0, "warmup": 0, "size": f"{size[0]}x{size[1]}",
                      "wall_time": None, "stage_ms": {}, "statuses": {}}
            model_version = None
        report.update(requested=requested, decoded=decoded, skipped=len(decode_errors),
                      decode_errors=decode_errors)
        report.update(info,mode=mode_name(args),model_version=model_version,
                      evaluation_config=effective_config_info(cfg, args.config),
                      acceptance_eligible=bool(report['n']) and not(args.fake or args.uniform),
                      timing_scope="recognize wall time; image reading and letterbox excluded")
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
