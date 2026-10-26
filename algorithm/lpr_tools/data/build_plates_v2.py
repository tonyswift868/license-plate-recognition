"""Build the v2 whole-plate corpus with a frozen-excluded hard training pool.

The v1 crops are hard-linked into v2; only the new hard subset is recomputed.
Run from the worker checkout with ``LPR_CONFIG`` set to a config that enables
``runtime.corner_model_path``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from lpr.config import DEFAULT_CONFIG_PATH, load_config
from lpr.io_utils import decode_image, imread, imwrite
from lpr.locate import box_iou, locate
from lpr.plate_rules import match_layout
from lpr.preprocess import preprocess
from lpr.rectify import rectify_candidates
from lpr.recognize import load_corner_detector
from lpr.pipeline import _in_roi, _merge_candidates
from lpr_tools.data.ccpd import read_index
from lpr_tools.data.build_plates import _warp_gt, FIELDS
from lpr_tools.data.manifest import read_manifest, image_path

V1 = Path(r"D:\keshe2\algorithm\data\plates\v1")
V2 = Path(r"D:\keshe2\algorithm\data\plates\v2")
INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
HARD = {"ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather"}
FROZEN = [Path(r"D:\keshe2\algorithm\data\index\testset_manifest_v1.csv"),
          Path(r"D:\keshe2\workers\03\algorithm\tests\data\golden\golden_manifest.csv"),
          Path(r"D:\keshe2\algorithm\data\index\devhard_v1.csv"),
          Path(r"D:\keshe2\algorithm\data\index\undecodable.csv")]
GOLDEN = FROZEN[1].parent
_CFG = None
_CORNER = None


def _source_group(source: str) -> str:
    """Summarize both legacy v1 tags and subset-tagged v2 hard rows safely."""
    fields = source.split(":")
    return ":".join(fields[:2]) if fields[0] == "hard_train" and len(fields) >= 2 else fields[0]


def _key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _sha(path: str | Path) -> tuple[int, str]:
    h = hashlib.sha256(); size = 0
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            size += len(block); h.update(block)
    return size, h.hexdigest()


def _exclusions() -> tuple[set[str], set[tuple[int, str]]]:
    paths: set[str] = set(); hashes: set[tuple[int, str]] = set()
    for manifest in FROZEN:
        if not manifest.is_file():
            raise FileNotFoundError(f"required exclusion list missing: {manifest}")
        if manifest == FROZEN[-1]:
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "path" not in rows[0]:
                raise ValueError(f"invalid undecodable list: {manifest}")
            entries = [(Path(row["path"].strip()), manifest) for row in rows if row.get("path", "").strip()]
        else:
            entries = [(image_path(row, manifest), manifest) for row in read_manifest(manifest, check_files=True)]
        for candidate, base in entries:
            p = (candidate if candidate.is_absolute() else base.resolve().parent / candidate).resolve()
            paths.add(_key(p))
            if not p.is_file():
                raise FileNotFoundError(f"frozen-list image missing: {p}")
            hashes.add(_sha(p))
    if not GOLDEN.is_dir():
        raise FileNotFoundError(f"golden image folder missing: {GOLDEN}")
    for p in GOLDEN.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            p = p.resolve(); paths.add(_key(p)); hashes.add(_sha(p))
    return paths, hashes


def _init(config_path: str) -> None:
    global _CFG, _CORNER
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    _CFG = load_config(config_path)
    if not _CFG.runtime.corner_model_path:
        raise ValueError("v2 hard pool requires runtime.corner_model_path")
    _CORNER = load_corner_detector(_CFG.resolve_path(_CFG.runtime.corner_model_path))
    command_path = Path("algorithm/out/receipts/第八批/data/build.command.json")
    command = json.loads(command_path.read_text(encoding="utf-8-sig")) if command_path.is_file() else {}
    worker_meta = {"pid": os.getpid(), "corner_model": str(_CFG.resolve_path(_CFG.runtime.corner_model_path)),
                   "launch_source_sha256": command.get("code_sha256"),
                   "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    worker_dir = Path("algorithm/out/receipts/第八批/data/workers")
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / f"{os.getpid()}.json").write_text(json.dumps(worker_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _pipeline_crop(image: np.ndarray, row: dict, cfg, detector):
    try:
        roi = cfg.lane_roi(None)
        classic = locate(preprocess(image, cfg.preprocess, roi), cfg.locate, cfg.runtime.max_candidates)
    except Exception as exc:
        return None, {"corner_detect_calls": 0, "error": f"{type(exc).__name__}: {exc}"}
    try:
        net = [c for c in detector.detect(image, cfg.runtime.max_candidates) if _in_roi(c, roi)]
    except Exception as exc:
        return None, {"corner_detect_calls": 1, "error": f"{type(exc).__name__}: {exc}"}
    fused = _merge_candidates(net, classic, cfg.runtime.max_candidates)
    truth_box = tuple(float(v) for v in row["bbox"].split())
    qualified = [(c, box_iou(c.bbox, truth_box)) for c in fused if box_iou(c.bbox, truth_box) >= .5]
    if not qualified:
        return None, {"corner_detect_calls": 1, "net_candidates": len(net), "fused_candidates": len(fused)}
    # Preserve pipeline order: the first matching fused candidate and its first rectify hypothesis.
    candidate, iou = qualified[0]
    try:
        rect = rectify_candidates(image, candidate, cfg.rectify)[0]
        target = (276, 80) if int(row["layout"]) == 8 else (252, 80)
        crop = rect.image
        if crop.shape[1::-1] != target:
            crop = cv2.resize(crop, target, interpolation=cv2.INTER_AREA if crop.shape[1] > target[0] else cv2.INTER_CUBIC)
    except Exception as exc:
        return None, {"corner_detect_calls": 1, "net_candidates": len(net), "fused_candidates": len(fused),
                      "selected_candidate_source": candidate.source,
                      "error": f"rectify/resize {type(exc).__name__}: {exc}"}
    return (crop, candidate, iou), {"corner_detect_calls": 1, "net_candidates": len(net), "fused_candidates": len(fused)}


def _hard_one(job: tuple[dict, str, str, tuple[int, str]]) -> dict:
    row, root, config_path, expected = job
    src = Path(row["path"])
    try:
        blob = src.read_bytes()
        sha = (len(blob), hashlib.sha256(blob).hexdigest())
        if sha != expected:
            raise ValueError("source changed after preflight")
        image = decode_image(blob)
        if image is None:
            raise ValueError("image decode failed")
        cfg = _CFG if _CFG is not None else load_config(config_path)
        detector = _CORNER if _CORNER is not None else load_corner_detector(cfg.resolve_path(cfg.runtime.corner_model_path))
        relative = hashlib.sha1(str(src).casefold().encode("utf-8")).hexdigest()
        subset = row["subset"].removeprefix("ccpd_")
        gt_rel, pipe_rel = f"train/hard_{relative}_gt.jpg", f"train/hard_{relative}_pipe.jpg"
        quad_values = [float(x) for x in row["quad"].split()]
        quad = tuple(zip(quad_values[::2], quad_values[1::2]))
        out = Path(root)
        gt = _warp_gt(image, quad, int(row["layout"]))
        imwrite(out / gt_rel, gt, [cv2.IMWRITE_JPEG_QUALITY, 95])
        source_tag = f"hard_train:{subset}"
        gt_label = [gt_rel, row["plate_no"], row["layout"], row["plate_color"], source_tag + ":ground_truth", str(src)]
        try:
            result, evidence = _pipeline_crop(image, row, cfg, detector)
        except Exception as exc:
            result, evidence = None, {"corner_detect_calls": 0, "error": f"{type(exc).__name__}: {exc}"}
        if result is None:
            return {"status": "pipeline_missing", "path": str(src), "sha256": sha[1], "subset": subset,
                    "reason": evidence.get("error", "fused candidates failed bbox IoU >= 0.5"), **evidence,
                    "labels": [gt_label]}
        crop, candidate, iou = result
        try:
            imwrite(out / pipe_rel, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as exc:
            pipe_path = out / pipe_rel
            if pipe_path.exists():
                pipe_path.unlink()
            return {"status": "pipeline_missing", "path": str(src), "sha256": sha[1], "subset": subset,
                    "reason": f"candidate write failed: {type(exc).__name__}: {exc}", **evidence,
                    "labels": [gt_label]}
        return {"status": "ok", "path": str(src), "sha256": sha[1], "subset": subset,
                **evidence, "iou": iou,
                "candidate_source": candidate.source,
                "labels": [gt_label,
                           [pipe_rel, row["plate_no"], row["layout"], row["plate_color"], source_tag + ":pipeline:" + candidate.source, str(src)]]}
    except Exception as exc:
        return {"status": "error", "path": str(src), "error": f"{type(exc).__name__}: {exc}", "corner_detect_calls": 0}


def build(*, index: Path = INDEX, v1: Path = V1, target: Path = V2,
          receipt: Path = Path("algorithm/out/receipts/第八批/data"), workers: int = 4,
          limit: int = 0, config: Path | None = None) -> dict:
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing v2 output: {target}")
    if not (v1 / "labels.csv").is_file() or not (v1 / "source_hashes.json").is_file():
        raise FileNotFoundError("plates/v1 corpus or source ledger is incomplete")
    cfg_path = Path(config or os.environ.get("LPR_CONFIG") or DEFAULT_CONFIG_PATH).resolve()
    cfg = load_config(cfg_path)
    if not cfg.runtime.corner_model_path:
        raise ValueError(f"corner network is disabled in {cfg_path}")
    model_path = cfg.resolve_path(cfg.runtime.corner_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"corner network model missing: {model_path}")
    blocked_paths, blocked_hashes = _exclusions()
    ledger = json.loads((v1 / "source_hashes.json").read_text(encoding="utf-8"))
    blocked_ledger = {p for p, item in ledger.items()
                      if _key(p) in blocked_paths or (Path(p).stat().st_size, item["sha256"]) in blocked_hashes}
    base_rows = []
    v1_label_count = 0
    with (v1 / "labels.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("v1 labels.csv schema differs from the six-field contract")
        for n, r in enumerate(reader, 1):
            v1_label_count = n
            rel = Path(r["file"])
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in {"train", "val"}:
                raise ValueError(f"unsafe v1 label path: {r['file']}")
            crop = (v1 / Path(*rel.parts)).resolve()
            if not crop.is_relative_to(v1.resolve()) or not crop.is_file():
                raise FileNotFoundError(f"invalid/missing v1 crop: {crop}")
            if r["layout"] not in {"7", "8"} or len(r["plate_no"]) != int(r["layout"]):
                raise ValueError(f"invalid v1 plate/layout label: {r['file']}")
            if r["orig_path"] not in ledger or ledger[r["orig_path"]]["split"] != rel.parts[0]:
                raise ValueError(f"v1 original path/split disagrees with source ledger: {r['orig_path']}")
            if r["orig_path"] in blocked_ledger:
                continue
            base_rows.append([r[k] for k in FIELDS])
            if n % 50000 == 0:
                print(json.dumps({"phase":"prepare_v1_links","rows_scanned":n,"rows_retained":len(base_rows)}),flush=True)
    known = {item["sha256"] for p, item in ledger.items() if p not in blocked_ledger}
    index_rows = read_index(index)
    selected, skipped = [], Counter()
    frozen_paths_count = len(blocked_paths); frozen_bytes_count = len(blocked_hashes)
    hard_seen = 0
    for row in index_rows:
        if row["subset"] not in HARD or row["has_plate"] != "1" or row["skip_reason"]:
            continue
        if row["split"] != "test":
            skipped["unexpected_split"] += 1; continue
        p = Path(row["path"]).resolve()
        if _key(p) in blocked_paths:
            skipped["frozen_path"] += 1; continue
        try:
            size, digest = _sha(p)
        except OSError:
            skipped["unreadable"] += 1; continue
        if (size, digest) in blocked_hashes:
            skipped["frozen_bytes"] += 1; continue
        if digest in known:
            skipped["duplicate_v1_or_pool"] += 1; continue
        row["path"] = str(p); row["__sha256"] = (size, digest)
        known.add(digest); selected.append(row)
        hard_seen += 1
        if hard_seen % 5000 == 0:
            print(json.dumps({"phase":"hard_source_hash_audit","checked":hard_seen}),flush=True)
    selected.sort(key=lambda r: (r["subset"], r["path"]))
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no eligible hard-training sources after frozen and duplicate exclusions")
    target.mkdir(parents=True, exist_ok=True); receipt.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(base_rows, 1):
        source = (v1 / r[0]).resolve(); dest = target / r[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, dest)
        if n % 50000 == 0:
            print(json.dumps({"phase":"link_v1","linked":n,"total":len(base_rows)}),flush=True)
    for split in ("train", "val"):
        (target / split).mkdir(exist_ok=True)
    (target / "labels.csv").write_text("", encoding="utf-8")
    outcomes, completed = [], []
    state = {"status": "building", "base_v1_labels": len(base_rows), "hard_sources": len(selected),
             "workers": workers, "config": str(cfg_path), "model": str(model_path), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (target / "build_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(cfg_path),)) as pool:
        jobs = ((r, str(target), str(cfg_path), tuple(r["__sha256"])) for r in selected)
        for n, result in enumerate(pool.map(_hard_one, jobs, chunksize=8), 1):
            outcomes.append(result)
            if result["status"] in {"ok", "pipeline_missing"}:
                completed.extend(result["labels"])
            if n % 500 == 0 or n == len(selected):
                state.update(processed=n, accepted=sum(x["status"] in {"ok", "pipeline_missing"} for x in outcomes), paired=sum(x["status"]=="ok" for x in outcomes), pipeline_missing=sum(x["status"]=="pipeline_missing" for x in outcomes), outcomes=dict(Counter(x["status"] for x in outcomes)),
                             elapsed_seconds=round(time.perf_counter()-started, 1))
                (target / "build_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                (receipt / "progress.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(state, ensure_ascii=True), flush=True)
    all_rows = base_rows + completed
    with (target / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(FIELDS); writer.writerows(all_rows)
    (target / "source_hashes.json").write_text(json.dumps({**{p:v for p,v in ledger.items() if p not in blocked_ledger},
       **{x["path"]:{"sha256":x["sha256"],"split":"train","subset":x["subset"]} for x in outcomes if x["status"] in {"ok","pipeline_missing"}}}, ensure_ascii=False, indent=2), encoding="utf-8")
    by_split=Counter(Path(r[0]).parts[0] for r in all_rows); by_source=Counter(_source_group(r[4]) for r in all_rows)
    stats={"v1_labels_read":v1_label_count,"base_v1_rows":len(base_rows),"v1_frozen_overlaps_excluded":v1_label_count-len(base_rows),"hard_sources_eligible":len(selected),"hard_sources_accepted":sum(x["status"] in {"ok","pipeline_missing"} for x in outcomes),
           "hard_sources_paired":sum(x["status"]=="ok" for x in outcomes),"pipeline_missing":sum(x["status"]=="pipeline_missing" for x in outcomes),
           "by_split":dict(by_split),"by_source":dict(by_source),"hard_subsets":dict(Counter(x["subset"] for x in outcomes if x["status"] in {"ok","pipeline_missing"})),
           "outcomes":dict(Counter(x["status"] for x in outcomes)),"excluded":dict(skipped),"excluded_path_count":frozen_paths_count,
           "corner_detect_calls":sum(x.get("corner_detect_calls",0) for x in outcomes),
           "selected_candidate_source_counts":dict(Counter(x["candidate_source"] for x in outcomes if x.get("candidate_source"))),
           "selected_network_candidate_count":sum(x.get("candidate_source")=="net" for x in outcomes),
           "excluded_byte_signatures":frozen_bytes_count,"workers":workers,"corner_model":str(model_path),"corner_model_sha256":_sha(model_path)[1],
           "config":str(cfg_path),"config_sha256":_sha(cfg_path)[1],"seconds":round(time.perf_counter()-started,1)}
    (target/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    (target/"provenance.json").write_text(json.dumps({"base":"plates/v1 hard-linked","hard_subsets":sorted(HARD),
        "validation":"v1 official val only; no hard subset added","exclusions":[str(p) for p in FROZEN],
        "candidate_pipeline":"current lpr.pipeline order: preprocess, classic locate, configured CornerDetector.detect, _merge_candidates, IoU>=0.5, rectify_candidates",
        "size_policy":"resize candidate to truth-label 7/8 layout when predicted layout differs","workers":workers},ensure_ascii=False,indent=2),encoding="utf-8")
    (receipt/"errors.json").write_text(json.dumps([x for x in outcomes if x["status"]=="error"],ensure_ascii=False,indent=2),encoding="utf-8")
    with (receipt/"hard_source_audit.jsonl").open("w",encoding="utf-8") as f:
        for item in outcomes:
            f.write(json.dumps({k:v for k,v in item.items() if k!="labels"},ensure_ascii=False)+"\n")
    final={**state,"status":"complete","processed":len(outcomes),"accepted":sum(x["status"] in {"ok","pipeline_missing"} for x in outcomes),"paired":sum(x["status"]=="ok" for x in outcomes),"pipeline_missing":sum(x["status"]=="pipeline_missing" for x in outcomes),"elapsed_seconds":round(time.perf_counter()-started,1)}
    (target/"build_state.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    (receipt/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    return stats


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,default=INDEX);parser.add_argument("--v1",type=Path,default=V1)
    parser.add_argument("--out",type=Path,default=V2);parser.add_argument("--receipt-dir",type=Path,default=Path("algorithm/out/receipts/第八批/data"))
    parser.add_argument("--workers",type=int,default=4);parser.add_argument("--limit",type=int,default=0);parser.add_argument("--config",type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(build(index=args.index,v1=args.v1,target=args.out,receipt=args.receipt_dir,workers=args.workers,limit=args.limit,config=args.config),ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
