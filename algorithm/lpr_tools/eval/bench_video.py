"""G3: decode every frame; sample recognition at media-time intervals and vote.

End-to-end throughput = decoded frames / wall-clock seconds, including decoding,
recognition and voting. No frame is dropped from the decoding loop.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

import cv2

from lpr.config import load_config
from lpr.voting import PlateVoteSession
from .common import add_backend_args, make_recognizer, mode_name, runtime_info


def benchmark_video(recognizer, video, *, sample_fps=5.):
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("--sample-fps must be finite and positive")
    session = PlateVoteSession.from_config(load_config())
    start = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    decoded = sampled = 0
    next_sample = 0.
    outputs = []
    statuses = Counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video has no valid frame rate")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp = decoded/fps
            decoded += 1
            if timestamp + 1e-9 >= next_sample:
                result = recognizer.recognize(image)
                sampled += 1
                statuses[result.status.value] += 1
                voted = session.feed(result,timestamp)
                if voted is not None:
                    outputs.append(voted)
                # Advance the media-time schedule, even if requested rate is
                # above source FPS; never recognize a decoded frame twice.
                next_sample = (math.floor((timestamp+1e-9)*sample_fps)+1)/sample_fps
        voted = session.flush()
        if voted is not None:
            outputs.append(voted)
    finally:
        capture.release()
    seconds = time.perf_counter()-start
    if decoded == 0:
        raise ValueError("video contains no decodable frames")
    return {"decoded_frames":decoded,"sampled_frames":sampled,"source_fps":fps,
            "sample_fps":sample_fps,"wall_seconds":seconds,
            "end_to_end_fps":decoded/seconds,"recognition_statuses":dict(statuses),
            "vote_outputs":len(outputs),"vote_statuses":dict(Counter(r.status.value for r in outputs)),
            "vote_frames_used":[r.frames_used for r in outputs],
            "timing_scope":"video open, all-frame decoding, sampled recognition, voting and release"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="G3 all-frame decode throughput with sampled recognition and voting")
    add_backend_args(parser)
    parser.add_argument("--video",required=True)
    parser.add_argument("--sample-fps",type=float,default=5.)
    args = parser.parse_args(argv)
    try:
        info = runtime_info()
        recognizer = make_recognizer(args)
        report = benchmark_video(recognizer,args.video,sample_fps=args.sample_fps)
        report.update(info,mode=mode_name(args),model_version=recognizer.version,
                      acceptance_eligible=not(args.fake or args.uniform))
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
