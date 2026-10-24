"""G2 evaluation v0, including explicit fake mode and ground-truth oracle.

Whole-plate equality includes rejected guesses; automatic acceptance requires OK.
Character accuracy is positional matches / sum(max(truth length, predicted length)).
Failure causes are status-based proxies, not proven causal localization recall.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from lpr.config import ALGORITHM_DIR
from lpr.errors import DEFAULT_MESSAGES, ERROR_CODES, failure_result
from lpr.io_utils import imread, imwrite
from lpr.postprocess import decode
from lpr.segment import segment
from lpr.types import PlateCandidate, RectifiedPlate, RecognitionResult, RecogStatus as S
from lpr_tools.data.manifest import CATEGORIES, image_path, read_manifest
from .common import add_backend_args, make_recognizer, mode_name


class OracleRecognizer:
    """Use annotated corners; never invoke locate or the stage-3 rectifier."""
    def __init__(self, recognizer):
        self.recognizer = recognizer

    def recognize(self, image, row):
        if row.category == "np":
            raise ValueError("oracle is undefined for no-plate images")
        if row.quad is None:
            raise ValueError("oracle needs quad ground truth on every plated row")
        if not hasattr(self.recognizer, "classifier"):
            return self.recognizer.recognize(image)
        start = time.perf_counter()
        cfg = self.recognizer.config
        layout = len(row.plate_no)  # oracle also receives true plate length
        w, h = cfg.rectify.size_7 if layout == 7 else cfg.rectify.size_8
        source = np.asarray(row.quad, np.float32)
        if abs(cv2.contourArea(source)) < 1:
            raise ValueError("degenerate oracle quad")
        target = np.asarray([(0,0),(w-1,0),(w-1,h-1),(0,h-1)], np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        plate = cv2.warpPerspective(image, transform, (w,h))
        x,y,bw,bh = cv2.boundingRect(source)
        candidate = PlateCandidate(list(row.quad), (x,y,bw,bh), row.plate_color, 1., "oracle")
        rect = RectifiedPlate(plate, layout, row.plate_color, transform, candidate)
        stages = {"oracle_warp": (time.perf_counter()-start)*1000}
        t = time.perf_counter()
        seg = segment(rect, layout, cfg.segment)
        stages["segment"] = (time.perf_counter()-t)*1000
        if not seg.ok:
            return failure_result(S.SEG_FAILED, elapsed_ms=(time.perf_counter()-start)*1000, stage_ms=stages)
        t = time.perf_counter()
        probs = self.recognizer.classifier.predict_proba(seg.chars)
        stages["recognize"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        dec = decode(probs, layout, row.plate_color, cfg.postprocess)
        stages["postprocess"] = (time.perf_counter()-t)*1000
        return RecognitionResult(dec.status, ERROR_CODES[dec.status], DEFAULT_MESSAGES[dec.status],
            dec.plate_no, row.plate_color, layout, dec.confidence, dec.char_confidences, dec.candidates,
            candidate.bbox, candidate.corners, plate, (time.perf_counter()-start)*1000, stages,
            vehicle_type_hint=dec.vehicle_type_hint, model_version=self.recognizer.version)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(samples):
    plated = [(row, result) for row, result in samples if row.category != "np"]
    correct = sum(result.plate_no == row.plate_no for row, result in plated)
    accepted = [(row, result) for row, result in plated if result.status == S.OK]
    char_correct = sum(sum(a == b for a,b in zip(row.plate_no, result.plate_no or "")) for row,result in plated)
    char_total = sum(max(len(row.plate_no),len(result.plate_no or "")) for row,result in plated)
    np_rows = [(row,result) for row,result in samples if row.category == "np"]
    false_positives = sum(bool(result.plate_no) for _,result in np_rows)
    return {"n": len(plated), "correct": correct, "plate_accuracy": _fraction(correct,len(plated)),
            "char_correct": char_correct, "char_total": char_total,
            "character_accuracy": _fraction(char_correct,char_total), "auto_pass": len(accepted),
            "auto_pass_rate": _fraction(len(accepted),len(plated)),
            "auto_pass_correct": sum(row.plate_no == result.plate_no for row,result in accepted),
            "auto_pass_precision": _fraction(sum(row.plate_no == result.plate_no for row,result in accepted),len(accepted)),
            "np_n": len(np_rows), "false_positives": false_positives,
            "false_positive_rate": _fraction(false_positives,len(np_rows))}


def failure_cause(row, result):
    if row.category == "np":
        return "localization" if result.plate_no else None
    if result.status == S.NO_PLATE:
        return "localization"
    if result.status == S.SEG_FAILED:
        return "segmentation"
    if result.plate_no and result.plate_no != row.plate_no:
        return "recognition"
    if result.status != S.OK:
        return "rejection"
    if result.plate_no != row.plate_no:
        return "recognition"
    return None


def evaluate(manifest, recognizer, out, *, oracle=False):
    rows = read_manifest(manifest, check_files=True)
    if not rows:
        raise ValueError("manifest is empty")
    if oracle and any(r.category != "np" and r.quad is None for r in rows):
        raise ValueError("oracle needs quad ground truth on every plated row")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples, failures = [], []
    causes = Counter({key:0 for key in ("localization","segmentation","recognition","rejection")})
    oracle_recognizer = OracleRecognizer(recognizer)
    for i,row in enumerate(rows):
        image = imread(image_path(row, manifest))
        if oracle and row.category != "np":
            result = oracle_recognizer.recognize(image,row)
        else:
            result = recognizer.recognize(image)
        samples.append((row,result))
        cause = failure_cause(row,result)
        if cause:
            causes[cause] += 1
            filename = f"failure_{i:04d}.png"
            preview = image.copy()
            cv2.putText(preview, f"{cause}: {result.status.value}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,255),1)
            imwrite(out/filename,preview)
            if result.plate_image is not None:
                imwrite(out/f"failure_{i:04d}_plate.png",result.plate_image)
            failures.append({"path":row.path, "truth":row.plate_no, "category":row.category,
                             "cause":cause, "image":filename, "result":result.to_dict()})
    report = {"mode":recognizer.version, "oracle":oracle,
              "oracle_scope":"ground truth quad, color and length; np runs normally" if oracle else None,
              "character_metric":"positional matches / max(true length, predicted length)",
              "attribution":"status-based proxy; not measured localization recall",
              "overall_g2":metrics([(r,s) for r,s in samples if r.category in {"clear","night","tilt","dirty"}]),
              "categories":{category:metrics([(r,s) for r,s in samples if r.category == category]) for category in CATEGORIES},
              "failure_causes":dict(causes),"failures":failures}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def _percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def print_report(report):
    print(f"mode={report['mode']} oracle={report['oracle']}")
    print("category n plate_accuracy character_accuracy auto_pass_rate auto_pass_precision")
    for name,m in [("overall_g2",report["overall_g2"]),*report["categories"].items()]:
        print(f"{name} {m['n']} " + " ".join(_percent(m[key]) for key in
              ("plate_accuracy","character_accuracy","auto_pass_rate","auto_pass_precision")))
    m=report["categories"]["np"]
    print(f"np_false_positives={m['false_positives']}/{m['np_n']} rate={_percent(m['false_positive_rate'])}")
    print("failure_causes=" + json.dumps(report["failure_causes"],ensure_ascii=False))
    print("attribution=" + report["attribution"])
    print("character_metric=" + report["character_metric"])
    print(f"failure_examples={len(report['failures'])}")


def append_scoreboard(path, report, *, split, classifier):
    path = Path(path)
    if path.resolve() == (ALGORITHM_DIR/"reports"/"scoreboard.md").resolve():
        raise ValueError("本批禁止修改正式 scoreboard.md；请传临时副本路径")
    if split != "val":
        raise ValueError("积分板只追加验证集结果；演示请用 --split manifest-val")
    try:
        commit = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        commit = "unknown"
    categories = report["categories"]
    values = [_percent(categories[c]["plate_accuracy"]) for c in ("clear","night","tilt","dirty")]
    m=report["overall_g2"]
    line = "| " + " | ".join([datetime.now(timezone(timedelta(hours=9))).date().isoformat(),commit,classifier,*values,
        _percent(m["character_accuracy"]),"N/A",_percent(m["auto_pass_precision"]),"N/A",
        f"{report['mode']}; oracle={report['oracle']}; validation; status-based attribution"]) + " |\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:
        handle.write(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description="G2 evaluator v0; fake output is integration validation only")
    parser.add_argument("--manifest", default="algorithm/testset/manifest.csv")
    parser.add_argument("--split",choices=("test","val","manifest-val"),default="test",
                        help="val needs ccpd_index.csv; manifest-val explicitly evaluates a supplied validation manifest")
    add_backend_args(parser)
    parser.add_argument("--out",default="algorithm/out/eval")
    parser.add_argument("--scoreboard",metavar="TEMP_COPY",help="append validation row to a temporary scoreboard copy")
    parser.add_argument("--oracle",action="store_true",help="use quad, true color and length; skip locate/rectify")
    args = parser.parse_args(argv)
    if args.split == "val":
        parser.error("--split val 抽样需要 ccpd_index.csv；本批请用构造验证清单 + --split manifest-val")
    if args.scoreboard:
        if Path(args.scoreboard).resolve() == (ALGORITHM_DIR/"reports"/"scoreboard.md").resolve():
            parser.error("本批禁止修改正式 scoreboard.md；请传临时副本路径")
        if args.split != "manifest-val":
            parser.error("--scoreboard 只允许验证清单：--split manifest-val")
    try:
        recognizer = make_recognizer(args,samples_dir=Path(args.manifest).resolve().parent)
        report = evaluate(args.manifest,recognizer,args.out,oracle=args.oracle)
        print_report(report)
        print(f"report={Path(args.out)/'report.json'}")
        if args.scoreboard:
            append_scoreboard(args.scoreboard,report,split="val",classifier=mode_name(args))
            print(f"scoreboard={args.scoreboard}")
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
