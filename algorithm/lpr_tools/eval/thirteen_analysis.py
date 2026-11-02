"""Offline reporting for Batch 13 frozen three-route evaluations.

Reads an existing prediction trace and final flat expected JSONL; it never
scores choices during model inference or changes an automatic/fallback rule.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import ntpath
import re
from collections import defaultdict
from pathlib import Path


def _jsonl(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if "_meta" not in item:
                    yield item


def _norm(path):
    return str(Path(str(path).replace("/", "\\")).resolve()).casefold()


def _read_index(path, selected_paths=None):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            # The shared index is absolute; avoid resolving unselected paths.
            key = (ntpath.normcase(ntpath.normpath(row["path"]))
                   if ntpath.isabs(row["path"]) else _norm(row["path"]))
            if selected_paths is not None and key not in selected_paths:
                continue
            q = [float(v) for v in row["quad"].split()] if row.get("quad") else []
            quad = list(zip(q[::2], q[1::2], strict=False)) if len(q) == 8 else None
            bbox = [float(v) for v in row["bbox"].split()] if row.get("bbox") else None
            if key in result:
                raise ValueError(f"duplicate selected index path: {row['path']}")
            result[key] = {"quad": quad, "bbox": bbox,
                "plate": row.get("plate_no", ""), "subset": row.get("subset", "")}
    return result


def _xywh_iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    iw, ih = max(0., min(ax+aw, bx+bw)-max(ax,bx)), max(0., min(ay+ah,by+bh)-max(ay,by))
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.


def _quad_box(quad):
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]


def _write_mosaic(target, canvas):
    """Encode before writing to support Unicode filenames on Windows."""
    import cv2
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise OSError(f"failed to encode error mosaic: {target}")
    Path(target).write_bytes(encoded.tobytes())


def _diagnostic_crop(image, gt_box, predicted_box=None):
    if not gt_box:
        return image
    boxes = [gt_box] + ([predicted_box] if predicted_box else [])
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    px, py = max(4, (right-left)*.1), max(4, (bottom-top)*.25)
    h, w = image.shape[:2]
    x0, y0 = max(0, int(left-px)), max(0, int(top-py))
    x1, y1 = min(w, int(right+px+1)), min(h, int(bottom+py+1))
    return image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image


def _route_reads(row):
    reads = {"segment": [], "reader": [], "box": []}
    for rect in row.get("rects", []):
        for route in ("segment", "reader"):
            attempt = rect.get(route)
            if attempt and attempt.get("valid") and attempt.get("plate"):
                reads[route].append((float(attempt.get("confidence", 0)), attempt["plate"]))
    for attempt in row.get("boxes", {}).values():
        if attempt and attempt.get("valid") and attempt.get("plate"):
            reads["box"].append((float(attempt.get("confidence", 0)), attempt["plate"]))
    for route, attempts in reads.items():
        best = None
        for attempt in attempts:
            if best is None or attempt[0] > best[0]:
                best = attempt
        reads[route] = best[1] if best else None
    return reads


def _route_has_truth(row, truth):
    """Inspect every valid route attempt for attribution, including low confidence."""
    found={"segment":False,"reader":False,"box":False}
    for rect in row.get("rects",[]):
        for route in ("segment","reader"):
            attempt=rect.get(route)
            if attempt and attempt.get("valid") and attempt.get("plate")==truth:
                found[route]=True
    for attempt in row.get("boxes",{}).values():
        if attempt and attempt.get("valid") and attempt.get("plate")==truth:
            found["box"]=True
    return found


def _category(record, expected, index):
    truth = str(record.get("truth") or "")
    predicted = str(expected.get("plate") or "")
    if not int(record.get("candidate_count", 0)):
        return "无候选"
    meta = index.get(_norm(record["path"]), {})
    # The CCPD index's bbox is its authoritative axis-aligned GT box;
    # enclosing the quadrilateral is only a fallback for older index rows.
    gtbox = meta.get("bbox") or (_quad_box(meta["quad"]) if meta.get("quad") else None)
    candidates = record.get("boxes", {})
    selected_ci = expected.get("candidate_index")
    chosen_box = candidates.get(str(selected_ci)) if selected_ci is not None else None
    chosen_box = chosen_box or candidates.get("0")
    selected_iou = (_xywh_iou(chosen_box["bbox"], gtbox)
                    if chosen_box and chosen_box.get("bbox") and gtbox else None)
    if gtbox and (selected_iou is None or selected_iou < .7):
        return "框偏_IoU<0.7"
    if len(truth) == 8 and len(predicted) not in {0, 8}:
        return "8位牌位数错"
    if len(truth) != len(predicted) and predicted:
        return "牌位数错"
    if len(truth) == len(predicted) and truth and truth[1:] == predicted[1:] and truth[0] != predicted[0]:
        return "只错省份字"
    if len(truth) == len(predicted) and truth and truth[0] == predicted[0] and sum(a != b for a,b in zip(truth[1:],predicted[1:], strict=False)) == 1:
        return "只错其他一位"
    routes = _route_reads(record)
    if any(_route_has_truth(record, truth).values()):
        return "有路线读对但仲裁挑错"
    if all(routes.values()):
        return "三路都读错"
    if any(routes.values()):
        return "有路线缺读且其余全错"
    return "无合法路线输出"


def _selected_candidate_iou(record, expected, index):
    meta=index.get(_norm(record["path"]),{})
    gtbox=meta.get("bbox") or (_quad_box(meta["quad"]) if meta.get("quad") else None)
    boxes=record.get("boxes",{})
    ci=expected.get("candidate_index")
    selected=boxes.get(str(ci)) if ci is not None else None
    selected=selected or boxes.get("0")
    if not gtbox or not selected or not selected.get("bbox"):
        return ci, None
    return ci, _xywh_iou(selected["bbox"],gtbox)


def _rates(rows, truth_key, predicted_key):
    n = correct = 0
    for row in rows:
        truth, pred = row.get(truth_key), row.get(predicted_key)
        if truth:
            n += 1
            correct += int(bool(pred) and pred == truth)
    return {"correct": correct, "n": n, "accuracy": correct/n if n else None}


def analyze(trace_path, expected_path, split_path, index_path, output_dir,
            *, make_mosaics=True):
    from lpr_tools.eval.route_split import path_key
    outdir=Path(output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty analysis output directory: {outdir}")
    traces = list(_jsonl(trace_path))
    expected = {r["path"]: r for r in _jsonl(expected_path)}
    assignments = json.loads(Path(split_path).read_text(encoding="utf-8"))["assignments"]
    index = _read_index(index_path, {_norm(row["path"]) for row in traces})
    if len(index) != len(traces):
        raise ValueError("index does not cover every selected trace path")
    if len(traces) != 3100 or len(expected) != len(traces):
        raise ValueError("analysis requires matched full 3100-row trace and expected output")
    if {r["path"] for r in traces} != set(expected):
        raise ValueError("trace and expectation paths differ")
    rows = []
    for rec in traces:
        out = expected[rec["path"]]
        truth = rec.get("truth") or ""
        row = {"path": rec["path"], "subset": rec.get("subset", "unknown"),
               "partition": "val" if rec.get("split") == "val" else "devhard",
               "half": assignments.get(path_key(rec["path"])),
               "truth": truth, "plate": out.get("plate") or "",
               "status": out.get("status"), "route": out.get("route"),
               "province": truth[:1], "province_pred": (out.get("plate") or "")[:1],
               "reads": _route_reads(rec), "record": rec}
        if row["half"] not in {"fit", "hold"}:
            raise ValueError(f"missing fixed half assignment for {rec['path']}")
        row["correct"] = bool(truth and row["plate"] == truth)
        row["auto"] = out.get("status") == "OK" and out.get("route") is not None
        row["category"] = None if row["correct"] else _category(rec, out, index)
        rows.append(row)

    def score(selected):
        n=len(selected)
        correct=sum(r["correct"] for r in selected)
        accepted=[r for r in selected if r["auto"]]
        ac=sum(r["plate"] == r["truth"] for r in accepted)
        return {"correct":correct,"n":n,"accuracy":correct/n if n else None,
                "auto_correct":ac,"auto_accepted":len(accepted),
                "auto_precision":ac/len(accepted) if accepted else None}

    def route_scores(selected):
        result={}
        for route in ("segment","reader","box"):
            whole=sum(bool(r["reads"][route]) and r["reads"][route]==r["truth"] for r in selected)
            province=sum(bool(r["reads"][route]) and r["reads"][route][0]==r["province"] for r in selected)
            result[route]={"whole_correct":whole,"whole_n":len(selected),
                "whole_accuracy":whole/len(selected) if selected else None,
                "province_char_correct":province,"province_char_n":len(selected),
                "province_char_accuracy":province/len(selected) if selected else None,
                "nonwan_to_wan":sum(r["province"]!="皖" and bool(r["reads"][route]) and r["reads"][route][0]=="皖" for r in selected)}
        return result

    subsets = sorted({r["subset"] for r in rows})
    report = {"schema":"thirteen-offline-analysis-v1", "rows":len(rows),
      "definitions":{"accuracy":"final expected plate equals trace truth / all truth-bearing rows",
       "auto_precision":"correct final OK rows / final OK rows; OK rows only",
       "route_reads":"for each model, highest-confidence valid prediction over all trace candidates/rects (stable traversal ties); descriptive route envelope, not the actual single-candidate invocation",
       "box_top1":"candidate index 0, reported separately as detector Top-1 reference",
       "iou":"axis-aligned xywh IoU of candidate bbox against official CCPD index bbox; if absent, use the GT quad's enclosing bbox",
       "error_bucket_priority":["无候选","框偏_IoU<0.7（选中候选；无选中项时Top1）","8位牌位数错","牌位数错","只错省份字","只错其他一位","有路线读对但仲裁挑错","三路都读错","有路线缺读且其余全错","无合法路线输出"]},
      "source_sha256":{}, "overall":{}, "subsets":{}, "halves":{}, "test_composition":{},
      "auto_subgroups":{}, "province":{}, "errors":{}, "iou_summary":{}}
    groups={"val":[r for r in rows if r["partition"]=="val"],
            "green":[r for r in rows if r["subset"]=="ccpd_green"],
            "devhard":[r for r in rows if r["partition"]=="devhard"],
            "dark":[r for r in rows if r["subset"]=="ccpd_db" and int(r["record"].get("brightness",999))<=19],
            "tilt_rotate":[r for r in rows if r["subset"] in {"ccpd_tilt","ccpd_rotate"}]}
    for name, group in groups.items():
        report["overall"][name]=score(group)
    for subset in subsets:
        report["subsets"][subset]=score([r for r in rows if r["subset"]==subset])
    for half in ("fit","hold"):
        hrows=[r for r in rows if r["half"]==half]
        report["halves"][half]={name:score([r for r in hrows if name=="all" or
          (name=="val" and r["partition"]=="val") or (name=="green" and r["subset"]=="ccpd_green") or
          (name=="devhard" and r["partition"]=="devhard") or (name=="dark" and r["subset"]=="ccpd_db" and int(r["record"].get("brightness",999))<=19) or
          (name=="tilt_rotate" and r["subset"] in {"ccpd_tilt","ccpd_rotate"})])
          for name in ("all","val","green","devhard","dark","tilt_rotate")}
        report["halves"][half]["subsets"]={subset:score([r for r in hrows if r["subset"]==subset])
            for subset in subsets}
        report["halves"][half]["auto_subgroups"]={name:score([r for r in hrows if r in selected])
            for name,selected in {"dark":groups["dark"],"green":groups["green"],
             "fn":[r for r in rows if r["subset"]=="ccpd_fn"],
             "challenge":[r for r in rows if r["subset"]=="ccpd_challenge"],
             "non_皖":[r for r in rows if r["truth"] and r["truth"][0]!="皖"]}.items()}
    # Official test composition: 120 base + 30 green; night=dark, tilt=35+15,
    # dirty=weather. Report component accuracies and weighted clear estimate.
    clear_base=report["subsets"].get("ccpd_base",{})
    clear_green=report["subsets"].get("ccpd_green",{})
    report["test_composition"]={"clear_base120_green30":{
        "base_accuracy":clear_base.get("accuracy"),"green_accuracy":clear_green.get("accuracy"),
        "estimated_correct":120*(clear_base.get("accuracy") or 0)+30*(clear_green.get("accuracy") or 0),
        "accuracy":(120*(clear_base.get("accuracy") or 0)+30*(clear_green.get("accuracy") or 0))/150,"n":150},
        "night_dark":report["overall"]["dark"],
        "tilt_35_rotate_15":{"tilt_accuracy":report["subsets"].get("ccpd_tilt",{}).get("accuracy"),
          "rotate_accuracy":report["subsets"].get("ccpd_rotate",{}).get("accuracy"),
          "estimated_correct":35*(report["subsets"].get("ccpd_tilt",{}).get("accuracy") or 0)+15*(report["subsets"].get("ccpd_rotate",{}).get("accuracy") or 0),
          "accuracy":(35*(report["subsets"].get("ccpd_tilt",{}).get("accuracy") or 0)+15*(report["subsets"].get("ccpd_rotate",{}).get("accuracy") or 0))/50,"n":50},
        "dirty_weather":report["subsets"].get("ccpd_weather",{})}
    subgroup={"dark":groups["dark"],"green":groups["green"],
      "fn":[r for r in rows if r["subset"]=="ccpd_fn"],
      "challenge":[r for r in rows if r["subset"]=="ccpd_challenge"],
      "non_皖":[r for r in rows if r["truth"] and r["truth"][0]!="皖"]}
    for name, selected in subgroup.items():
        report["auto_subgroups"][name]=score(selected)
    provinces=sorted({r["province"] for r in rows if r["truth"]})
    for province in provinces:
        selected=[r for r in rows if r["province"]==province]
        report["province"][province]={"whole":score(selected),"province_char_correct":sum(r["province"]==r["province_pred"] for r in selected),"n":len(selected),
          "province_char_accuracy":sum(r["province"]==r["province_pred"] for r in selected)/len(selected) if selected else None,
          "routes":route_scores(selected)}
    nonwan=[r for r in rows if r["truth"] and r["province"]!="皖"]
    nonwan_pc=sum(r["province"]==r["province_pred"] for r in nonwan)
    report["province"]["non_皖_total"]={"n":len(nonwan),"whole":score(nonwan),
      "province_char_correct":nonwan_pc,"province_char_accuracy":nonwan_pc/len(nonwan) if nonwan else None,
      "routes":route_scores(nonwan)}
    report["iou_summary"]={"top1_B_candidate_count":0,"top1_B_iou_ge_0_7":0,
       "selected_iou_ge_0_7":0,"selected_or_top1_iou_missing":0}
    for r in rows:
        report["iou_summary"]["top1_B_candidate_count"] += int("0" in r["record"].get("boxes",{}))
        meta=index.get(_norm(r["path"]),{})
        gtbox=meta.get("bbox") or (_quad_box(meta["quad"]) if meta.get("quad") else None)
        if not gtbox:
            report["iou_summary"]["selected_or_top1_iou_missing"]+=1
            continue
        boxes=r["record"].get("boxes",{})
        top=boxes.get("0")
        if top and top.get("bbox") and _xywh_iou(top["bbox"],gtbox)>=.7:
            report["iou_summary"]["top1_B_iou_ge_0_7"]+=1
        expected_row=expected[r["path"]]
        ci=expected_row.get("candidate_index")
        chosen=boxes.get(str(ci)) if ci is not None else top
        if chosen and chosen.get("bbox") and _xywh_iou(chosen["bbox"],gtbox)>=.7:
            report["iou_summary"]["selected_iou_ge_0_7"]+=1
    bad=[r for r in rows if not r["correct"]]
    buckets=defaultdict(list)
    for r in bad:
        buckets[r["category"]].append(r)
    report["errors"]={"counts":{k:len(v) for k,v in sorted(buckets.items())},
      "by_subset":{s:{k:sum(r["subset"]==s for r in v) for k,v in buckets.items()} for s in subsets},
      "by_province":{p:{k:sum(r["province"]==p for r in v) for k,v in buckets.items()} for p in provinces},
      "examples":{k:[r["path"] for r in v[:8]] for k,v in buckets.items()},
      "exclusive_total":sum(len(v) for v in buckets.values()),"error_rows":len(bad)}
    outdir.mkdir(parents=True,exist_ok=True)
    for label, path in (("trace",trace_path),("expected",expected_path),("split",split_path),("index",index_path)):
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        report["source_sha256"][label]=digest.hexdigest()
    error_path=outdir/"errors.jsonl.gz"
    with gzip.open(error_path,"wt",encoding="utf-8",newline="\n") as stream:
        for r in bad:
            expected_row=expected[r["path"]]
            ci,iou=_selected_candidate_iou(r["record"],expected_row,index)
            stream.write(json.dumps({"path":r["path"],"subset":r["subset"],
                "province":r["province"],"truth":r["truth"],"pred":r["plate"],
                "status":r["status"],"route":r["route"],"assigned_bucket":r["category"],
                "selected_ci":ci,"selected_iou":iou},ensure_ascii=False,separators=(",",":"))+"\n")
    report["errors"]["detail_file"]={"path":error_path.name,"rows":len(bad),
        "sha256":hashlib.sha256(error_path.read_bytes()).hexdigest()}
    if make_mosaics:
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        report["errors"]["mosaics"]={}
        for category, selected in buckets.items():
            thumbs=[]
            thumb_paths=[]
            for r in selected[:8]:
                image=cv2.imread(r["path"])
                if image is None:
                    continue
                meta = index[_norm(r["path"])]
                gt_box = meta.get("bbox") or (_quad_box(meta["quad"]) if meta.get("quad") else None)
                expected_row = expected[r["path"]]
                chosen_ci = expected_row.get("candidate_index")
                box = r["record"].get("boxes", {}).get(str(chosen_ci) if chosen_ci is not None else "0", {})
                predicted_box = box.get("bbox") if category == "框偏_IoU<0.7" else None
                image = _diagnostic_crop(image, gt_box, predicted_box)
                h,w=image.shape[:2]
                scale=min(1.,100/max(h,w))
                image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
                label=len(thumbs)+1
                cv2.putText(image,str(label),(3,min(image.shape[0]-3,15)),cv2.FONT_HERSHEY_SIMPLEX,
                            .45,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(image,str(label),(3,min(image.shape[0]-3,15)),cv2.FONT_HERSHEY_SIMPLEX,
                            .45,(255,255,255),1,cv2.LINE_AA)
                thumbs.append(image)
                thumb_paths.append(r["path"])
            if thumbs:
                cellw=max(im.shape[1] for im in thumbs)
                cellh=max(im.shape[0] for im in thumbs)
                canvas=np.zeros((cellh*((len(thumbs)+1)//2),cellw*2,3),dtype=np.uint8)
                for i,im in enumerate(thumbs):
                    y=(i//2)*cellh
                    x=(i%2)*cellw
                    canvas[y:y+im.shape[0],x:x+im.shape[1]]=im
                slug=re.sub(r'[<>:"/\\|?*\x00-\x1f]',"_",category).rstrip(" .") or "errors"
                target=outdir/f"errors-{slug}.jpg"
                _write_mosaic(target, canvas)
                report["errors"]["mosaics"][category]={"file":target.name,
                    "view": "GT bbox context crop; localization bucket includes selected/Top1 predicted bbox union; diagnostic only, after all rules frozen",
                    "number_to_path":{str(i+1):path for i,path in enumerate(thumb_paths)},
                    "images":len(thumb_paths),"max_side_px":max(canvas.shape[:2])}
    report["source_sha256"]["analysis_code"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (outdir/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",required=True)
    p.add_argument("--expected",required=True)
    p.add_argument("--split-json",required=True)
    p.add_argument("--index",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--no-mosaics",action="store_true")
    a=p.parse_args(argv)
    result=analyze(a.trace,a.expected,a.split_json,a.index,a.out_dir,make_mosaics=not a.no_mosaics)
    print(json.dumps({"rows":result["rows"],"output":str(Path(a.out_dir)/"analysis.json")},ensure_ascii=False))


if __name__=="__main__":
    raise SystemExit(main())
