"""Build resumable whole-plate CRNN data from the official CCPD index.

Run from the worker checkout with ``.venv\\Scripts\\python.exe -m lpr_tools.data.build_plates``.
The output is intentionally outside the checkout at algorithm/data/plates/v1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from lpr.config import load_config
from lpr.io_utils import decode_image, imwrite
from lpr.locate import box_iou, locate
from lpr.plate_rules import match_layout
from lpr.preprocess import preprocess
from lpr.rectify import rectify_candidates
from lpr_tools.data.ccpd import read_index
from lpr_tools.data.manifest import read_manifest, image_path

FIELDS = ("file", "plate_no", "layout", "color", "source", "orig_path")
WORKER_CONFIG = None


def _worker_init(config_path: str | None) -> None:
    global WORKER_CONFIG
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    WORKER_CONFIG = load_config(config_path)


def _warp_gt(image: np.ndarray, quad: tuple[tuple[float, float], ...], layout: int) -> np.ndarray:
    width = 252 if layout == 7 else 276
    src = np.float32(quad)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, 79], [0, 79]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, 80))


def _process_one(task: tuple[dict, str, str]) -> dict:
    raw, out_root, config_path = task
    result = {"path": raw["path"], "split": raw["split"], "status": "error", "error": ""}
    try:
        original = Path(raw["path"])
        data = original.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if raw.get("__sha256") and digest != raw["__sha256"]:
            raise ValueError("source_changed_after_hash_audit")
        result["sha256"] = digest
        result["size"] = len(data)
        image = decode_image(data)
        if image is None:
            raise ValueError("decode_failed")
        plate = raw["plate_no"]
        layout = int(raw["layout"])
        color = raw["plate_color"]
        quad_nums = [float(x) for x in raw["quad"].split()]
        quad = tuple(zip(quad_nums[::2], quad_nums[1::2]))
        bbox = tuple(float(x) for x in raw["bbox"].split())
        key = hashlib.sha1((raw["split"] + "\0" + str(original).casefold()).encode("utf-8")).hexdigest()
        split_root = Path(out_root) / raw["split"]
        rows = []
        crop_provenance = [{"file": f"{raw['split']}/{key}_gt.jpg", "orig_path": str(original),
                            "split": raw["split"], "source": "ground_truth", "bbox_iou": 1.0,
                            "candidate_index": "", "hypothesis_index": "", "size_policy": "GT exact"}]
        gt = _warp_gt(image, quad, layout)
        gt_rel = f"{raw['split']}/{key}_gt.jpg"
        imwrite(Path(out_root) / gt_rel, gt, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append((gt_rel, plate, layout, color, "ground_truth", str(original)))

        cfg = WORKER_CONFIG if WORKER_CONFIG is not None else load_config(config_path)
        pipeline_error = ""
        try:
            pre = preprocess(image, cfg.preprocess)
            candidates = locate(pre, cfg.locate, max_candidates=3)
            for ci, candidate in enumerate(candidates):
                candidate_iou = box_iou(candidate.bbox, bbox)
                if candidate_iou < .5:
                    continue
                for hi, rect in enumerate(rectify_candidates(image, candidate, cfg.rectify)):
                    target = (276, 80) if layout == 8 else (252, 80)
                    crop = rect.image
                    if crop.shape[1::-1] != target:
                        crop = cv2.resize(crop, target, interpolation=cv2.INTER_AREA if crop.shape[1] > target[0] else cv2.INTER_CUBIC)
                    rel = f"{raw['split']}/{key}_cand{ci}_hyp{hi}.jpg"
                    imwrite(Path(out_root) / rel, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append((rel, plate, layout, color, f"pipeline_candidate:{ci}:{hi}", str(original)))
                    crop_provenance.append({"file": rel, "orig_path": str(original), "split": raw["split"],
                                            "source": "pipeline_candidate", "bbox_iou": candidate_iou,
                                            "candidate_index": ci, "hypothesis_index": hi,
                                            "predicted_layout": rect.layout, "label_layout": layout,
                                            "size_policy": "resized to GT label layout" if rect.layout != layout else "rectify native size"})
        except Exception as exc:
            # Keep the valid GT crop; the failed pipeline branch contributes no candidates.
            pipeline_error = f"{type(exc).__name__}: {exc}"
        provenance_dir = Path(out_root) / "_crop_provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        (provenance_dir / f"{key}.json").write_text(json.dumps(crop_provenance, ensure_ascii=False), encoding="utf-8")
        result.update(status="ok", labels=rows, pipeline_error=pipeline_error)
    except Exception as exc:  # counted and recorded so one bad image cannot stop a long build
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _frozen_paths_and_hashes(manifests: list[Path], golden_dir: Path) -> tuple[set[str], set[tuple[int, str]]]:
    paths: set[str] = set()
    hashes: set[tuple[int, str]] = set()
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"required frozen manifest does not exist: {manifest}")
        for row in read_manifest(manifest, check_files=True):
            p = image_path(row, manifest).resolve()
            paths.add(os.path.normcase(os.path.normpath(str(p))))
            blob = p.read_bytes()
            hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
    if golden_dir.is_dir():
        for p in golden_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                p = p.resolve()
                paths.add(os.path.normcase(os.path.normpath(str(p))))
                blob = p.read_bytes()
                hashes.add((len(blob), hashlib.sha256(blob).hexdigest()))
    return paths, hashes


def build(index: Path, target: Path, receipt_dir: Path, *, workers: int = 4,
          config: str | None = None, limit_train: int = 0, limit_val: int = 0,
          raw_root: Path = Path(r"D:\keshe2\algorithm\data\raw"),
          frozen_manifests: list[Path] | None = None, golden_dir: Path | None = None) -> dict:
    has_existing_content = target.exists() and any(p.is_file() for p in target.rglob("*"))
    if has_existing_content and not (target / "build_state.json").is_file():
        raise FileExistsError(f"refusing to overwrite existing non-builder output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (target / split).mkdir(parents=True, exist_ok=True)
    if frozen_manifests is None:
        frozen_manifests = [Path(r"D:\keshe2\algorithm\data\index\testset_manifest_v1.csv"),
                            Path(r"D:\keshe2\workers\03\algorithm\tests\data\golden\golden_manifest.csv")]
    if golden_dir is None:
        golden_dir = Path(r"D:\keshe2\workers\03\algorithm\tests\data\golden")
    raw_root = raw_root.resolve()
    frozen_paths, frozen_hashes = _frozen_paths_and_hashes(frozen_manifests, golden_dir)
    rows = read_index(index)
    invalid_label_rows = sum(1 for r in rows if r.get("split") in ("train", "val")
                             and r.get("has_plate") == "1" and not r.get("skip_reason")
                             and (match_layout(r.get("plate_no", "")) is None
                                  or str(len(r.get("plate_no", ""))) != r.get("layout", "")))
    chosen = [r for r in rows if r.get("split") in ("train", "val") and r.get("has_plate") == "1"
              and not r.get("skip_reason") and len(r.get("plate_no", "")) in (7, 8)
              and match_layout(r.get("plate_no", "")) is not None
              and str(len(r.get("plate_no", ""))) == r.get("layout", "")]
    chosen = [r for r in chosen if os.path.normcase(os.path.normpath(str(Path(r["path"]).resolve()))) not in frozen_paths]
    chosen.sort(key=lambda r: (r["split"], r["path"]))
    if limit_train or limit_val:
        tr = [r for r in chosen if r["split"] == "train"]
        va = [r for r in chosen if r["split"] == "val"]
        chosen = (tr[:limit_train] if limit_train else tr) + (va[:limit_val] if limit_val else va)

    # Target roughly 10k validation crops (each source contributes at least its GT crop).
    if not limit_val:
        val_rows = [r for r in chosen if r["split"] == "val"]
        pools: dict[tuple[str, str], list[dict]] = {}
        for row in val_rows:
            pools.setdefault((row["subset"], row["layout"]), []).append(row)
        rng = random.Random(42)
        for pool in pools.values():
            pool.sort(key=lambda row: row["path"])
            rng.shuffle(pool)
        selected_val = []
        strata = sorted(pools)
        rng.shuffle(strata)
        offset = 0
        while len(selected_val) < min(5000, len(val_rows)):
            added = False
            for key in strata:
                if offset < len(pools[key]):
                    selected_val.append(pools[key][offset])
                    added = True
                    if len(selected_val) == 5000:
                        break
            if not added:
                break
            offset += 1
        chosen = [r for r in chosen if r["split"] == "train"] + selected_val
    labels_path = target / "labels.csv"
    stats = Counter()
    error_rows = []
    done_paths = set()
    if labels_path.exists():
        with labels_path.open(encoding="utf-8", newline="") as f:
            done_paths = {row["orig_path"] for row in csv.DictReader(f)}
    label_mode = "a" if labels_path.exists() else "w"
    completed_source_hash: dict[str, dict[str, str]] = {}
    if (target / "source_hashes.json").is_file():
        completed_source_hash = json.loads((target / "source_hashes.json").read_text(encoding="utf-8"))
    owners = {v["sha256"]: v["split"] for v in completed_source_hash.values() if isinstance(v, dict) and "sha256" in v}
    # Reconcile a crash between label/hash checkpoint writes using original bytes.
    if labels_path.exists():
        with labels_path.open(encoding="utf-8", newline="") as f:
            label_rows = list(csv.DictReader(f))
        if tuple(label_rows[0].keys()) != FIELDS if label_rows else False:
            raise RuntimeError("labels.csv schema mismatch; refusing resume")
        by_origin = {str(Path(r["path"]).resolve()): r for r in chosen}
        for origin in done_paths - set(completed_source_hash):
            row = by_origin.get(origin)
            if row is None:
                raise RuntimeError(f"labels.csv contains unknown source path: {origin}")
            blob = Path(origin).read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if (len(blob), digest) in frozen_hashes or (digest in owners and owners[digest] != row["split"]):
                raise RuntimeError(f"cannot safely recover source hash checkpoint: {origin}")
            completed_source_hash[origin] = {"sha256": digest, "split": row["split"]}
            owners[digest] = row["split"]
        for origin in set(completed_source_hash) - done_paths:
            item = completed_source_hash[origin]
            split = item.get("split", "") if isinstance(item, dict) else ""
            key = hashlib.sha1((split + "\0" + origin.casefold()).encode("utf-8")).hexdigest()
            if split:
                for crop in (target / split).glob(key + "_*.jpg"):
                    crop.unlink(missing_ok=True)
            completed_source_hash.pop(origin, None)
        if completed_source_hash:
            temp = target / "source_hashes.json.tmp"
            temp.write_text(json.dumps(completed_source_hash, ensure_ascii=False), encoding="utf-8")
            temp.replace(target / "source_hashes.json")
    selected = [r for r in chosen if str(Path(r["path"]).resolve()) not in done_paths]
    # Pre-hash and deduplicate the complete pending source set before workers may write crops.
    selected_safe = []
    for n, r in enumerate(selected, 1):
        source = Path(r["path"]).resolve()
        try:
            if not source.is_file() or not source.is_relative_to(raw_root):
                stats["unsafe_or_missing_path"] += 1
                continue
            blob = source.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if (len(blob), digest) in frozen_hashes:
                stats["frozen_hash_excluded"] += 1
                continue
            prior_split = owners.get(digest)
            if prior_split is not None:
                if prior_split != r["split"]:
                    stats["cross_split_hash_excluded"] += 1
                else:
                    stats["same_split_duplicate_hash_excluded"] += 1
                continue
            owners[digest] = r["split"]
            r["__sha256"] = digest
            selected_safe.append(r)
        except OSError as exc:
            stats["hash_errors"] += 1
            error_rows.append({"path": str(source), "error": f"{type(exc).__name__}: {exc}"})
        if n % 5000 == 0:
            print(json.dumps({"phase": "source_hash_audit", "checked": n, "pending": len(selected), "accepted": len(selected_safe), "stats": dict(stats)}, ensure_ascii=False), flush=True)
    selected = selected_safe
    started = time.time()
    initial_state = {"status": "building", "selected_sources": len(selected), "workers": workers,
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    temp_state = target / "build_state.json.tmp"
    temp_state.write_text(json.dumps(initial_state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_state.replace(target / "build_state.json")
    with labels_path.open(label_mode, encoding="utf-8", newline="") as labels_file:
        writer = csv.writer(labels_file)
        if label_mode == "w":
            writer.writerow(FIELDS)
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(config,)) as pool:
            tasks = ((r, str(target), config) for r in selected)
            for n, result in enumerate(pool.map(_process_one, tasks, chunksize=8), 1):
                src = result["path"]
                digest = result.get("sha256", "")
                if result["status"] == "ok" and digest:
                    for label in result["labels"]:
                        writer.writerow(label)
                        stats[f"{result['split']}_crops"] += 1
                        stats[f"{result['split']}_source:{label[4]}"] += 1
                    completed_source_hash[str(Path(src).resolve())] = {"sha256": digest, "split": result["split"]}
                    stats[f"{result['split']}_sources"] += 1
                else:
                    stats[f"{result['split']}_errors"] += 1
                    error_rows.append({"path": src, "error": result.get("error", "")})
                if n % 500 == 0 or n == len(selected):
                    labels_file.flush()
                    tmp_hash = target / "source_hashes.json.tmp"
                    tmp_hash.write_text(json.dumps(completed_source_hash, ensure_ascii=False), encoding="utf-8")
                    tmp_hash.replace(target / "source_hashes.json")
                    state = {"processed_this_run": n, "selected_remaining_at_start": len(selected), "stats": dict(stats), "elapsed_seconds": round(time.time()-started, 1), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
                    tmp_state = target / "build_state.json.tmp"
                    tmp_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp_state.replace(target / "build_state.json")
                    print(json.dumps(state, ensure_ascii=False), flush=True)
    (target / "source_hashes.json").write_text(json.dumps(completed_source_hash, ensure_ascii=False), encoding="utf-8")
    (receipt_dir / "errors.json").write_text(json.dumps(error_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter()
    province = Counter()
    greens = Counter()
    sources = Counter()
    splits = Counter()
    with labels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            splits[row["file"].split("/", 1)[0]] += 1
            sources[row["source"]] += 1
            greens[(row["file"].split("/", 1)[0], row["color"] == "green")] += 1
            if row["plate_no"]:
                province[(row["file"].split("/", 1)[0], row["plate_no"][0])] += 1
    summary = {"index": str(index.resolve()), "index_rows": len(rows), "invalid_index_label_rows": invalid_label_rows, "eligible_sources": len(chosen), "frozen_path_count": len(frozen_paths), "frozen_hash_count": len(frozen_hashes), "workers": workers, "stats": dict(stats), "by_split": dict(splits), "by_source": dict(sources), "green_by_split": {f"{s}/{g}": n for (s,g),n in sorted(greens.items())}, "province_by_split": {f"{s}/{p}": n for (s,p),n in sorted(province.items())}, "rare_provinces_under_1000": {f"{s}/{p}": n for (s,p),n in sorted(province.items()) if n < 1000}, "elapsed_seconds": round(time.time()-started, 1)}
    (target / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (receipt_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    all_provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    province_stats = {f"{split}/{p}": province.get((split, p), 0) for split in ("train", "val") for p in all_provinces}
    summary["province_by_split_including_zero"] = province_stats
    (target / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (receipt_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "provenance.json").write_text(json.dumps({"builder": "lpr_tools.data.build_plates", "python": sys.version, "cv2": cv2.__version__, "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "official_split_preserved": True, "hash_policy": "SHA-256 of original bytes; exact length and digest checked against frozen union; train/val source hashes are disjoint; same-split duplicate originals may remain", "rectification": "GT quad direct perspective; pipeline preprocess->locate->rectify_candidates with bbox IoU>=0.5; pipeline crops resized to GT label layout dimensions where predicted layout differs", "label_fields": FIELDS, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    (receipt_dir / "audit.json").write_text(json.dumps({"frozen_manifests": [str(p) for p in frozen_manifests], "frozen_paths": len(frozen_paths), "frozen_hashes": len(frozen_hashes), "cross_split_checked": True, "counts": dict(stats)}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Persist completion even when a resume has zero pending sources.
    final_state = {"status": "complete", "processed_this_run": len(selected),
                   "selected_remaining_at_start": len(selected), "stats": dict(stats),
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    final_state_path = target / "build_state.json.tmp"
    final_state_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")
    final_state_path.replace(target / "build_state.json")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\keshe2\algorithm\data\plates\v1"))
    parser.add_argument("--receipt-dir", type=Path, default=Path("algorithm/out/receipts/第七批/plates"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--limit-train", type=int, default=0, help="small smoke run only")
    parser.add_argument("--limit-val", type=int, default=0, help="small smoke run only")
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\keshe2\algorithm\data\raw"))
    parser.add_argument("--frozen-manifest", type=Path, action="append", default=None)
    parser.add_argument("--golden-dir", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be between 1 and 6")
    summary = build(args.index, args.out, args.receipt_dir, workers=args.workers, config=args.config,
                    limit_train=args.limit_train, limit_val=args.limit_val, raw_root=args.raw_root,
                    frozen_manifests=args.frozen_manifest, golden_dir=args.golden_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
