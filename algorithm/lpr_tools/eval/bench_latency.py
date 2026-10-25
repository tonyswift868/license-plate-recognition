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
from .common import (add_backend_args, latency_stats, letterbox, make_recognizer,
                     mode_name, parse_size, runtime_info)

STAGES = ("preprocess", "locate", "rectify", "segment", "recognize", "postprocess")
WARMUP_CALLS = 5


def benchmark(recognizer, images, *, n=100, size=(1280,720)):
    if n < 1 or not images or min(size) < 1:
        raise ValueError("benchmark needs n>0, nonempty images and positive size")
    inputs = [letterbox(image,size) for image in images]
    for i in range(WARMUP_CALLS):
        recognizer.recognize(inputs[i % len(inputs)])
    elapsed, results = [], []
    for i in range(n):
        start = time.perf_counter()
        result = recognizer.recognize(inputs[i % len(inputs)])
        elapsed.append((time.perf_counter()-start)*1000)
        results.append(result)
    stage_names = sorted(set(STAGES).union(*(r.stage_ms for r in results)))
    return {"n":n, "warmup":WARMUP_CALLS, "size":f"{size[0]}x{size[1]}",
            "wall_time":latency_stats(elapsed),
            "stage_ms":{stage:latency_stats([r.stage_ms.get(stage,0.) for r in results]) for stage in stage_names},
            "statuses":dict(Counter(r.status.value for r in results))}


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
        images, decode_errors = [], []
        for path in paths:
            image, reason = decode_image(path)
            if image is None:
                decode_errors.append({"path": str(path), "reason": reason})
            else:
                images.append(image)
        decoded = len(images)
        if not paths and not (args.manifest or args.images):
            # Smoke modes without an explicit source use one synthetic frame.
            images = [np.zeros((720,1280,3),np.uint8)]
        info = runtime_info()
        if images:
            recognizer = make_recognizer(args)
            report = benchmark(recognizer,images,n=args.n,size=size)
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
                      acceptance_eligible=bool(report['n']) and not(args.fake or args.uniform),
                      timing_scope="recognize wall time; image reading and letterbox excluded")
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
