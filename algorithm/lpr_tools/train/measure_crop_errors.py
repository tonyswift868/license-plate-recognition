"""Measure production fused-candidate crop geometry on the batch8 safe train pool.

The sampler output stores correlated four-corner vectors in the GT canonical
unit-square frame. It is deliberately conditioned on one ordered production
match per source image (first valid rectification with bbox IoU >= .5).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpr.config import load_config
from lpr.locate import locate
from lpr.pipeline import _in_roi, _merge_candidates, _rectifications
from lpr.preprocess import preprocess
from lpr.recognize import load_corner_detector
from lpr_tools.data.image_health import decode_image

SUBSETS = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")
DEFAULT_CONFIG = ROOT / "config" / "default.toml"
DEFAULT_MODEL = ROOT / "models" / "corner_net_v1.pt"
IOU_MIN = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pathkey(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path))).replace("\\", "/")


def _box(quad):
    a = np.asarray(quad, np.float32)
    x, y = a.min(axis=0); right, bottom = a.max(axis=0)
    return float(x), float(y), float(right-x), float(bottom-y)


def _iou(a, b):
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = x*y
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.


def select_sources(train_csv: Path, index_csv: Path, per_subset: int, seed: int):
    """Intersect only GT rows marked train in train.csv with source index."""
    selected = {}
    with train_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source = row.get("source", "")
            if row.get("partition") != "train" or not (source == "ground_truth" or source.endswith(":ground_truth")):
                continue
            selected[pathkey(row["orig_path"])] = row
    if not selected:
        raise ValueError("safe training CSV has no train ground_truth rows")
    indexed = {}
    with index_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = pathkey(row["path"])
            if key in selected:
                indexed[key] = row
    missing = set(selected) - set(indexed)
    if missing:
        raise ValueError(f"source index missing {len(missing)} selected train originals")
    groups = {s: [] for s in SUBSETS}
    seen = set()
    for key, row in selected.items():
        meta = indexed[key]
        subset = row.get("subset") or meta.get("subset", "")
        if subset.startswith("ccpd_"):
            subset = subset[5:]
        if subset not in groups or key in seen:
            continue
        seen.add(key)
        groups[subset].append((key, row, meta))
    rng = random.Random(seed)
    out = []
    for subset in SUBSETS:
        group = sorted(groups[subset], key=lambda x: x[0])
        rng.shuffle(group)
        if len(group) < per_subset:
            raise ValueError(f"subset {subset} has only {len(group)} safe train originals")
        out.extend((subset, *item) for item in group[:per_subset])
    return out


def _quad(value):
    vals = [float(x) for x in value.split()]
    if len(vals) != 8:
        raise ValueError("source quad must contain four xy points")
    return np.asarray(vals, np.float32).reshape(4, 2)


def _stats(values):
    if not values:
        return {f"p{p}": None for p in (50, 75, 90, 95, 99)}
    a = np.asarray(values, np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in (50, 75, 90, 95, 99)}


def measure(train_csv: Path, index_csv: Path, out_dir: Path, *, per_subset=300,
            seed=42, config_path=DEFAULT_CONFIG, model_path=DEFAULT_MODEL, threads=2):
    if out_dir.exists() and any((out_dir/name).exists() for name in ('train_per_image.jsonl','train_sampler.json','summary.json')):
        raise FileExistsError(f'refusing to overwrite geometry evidence: {out_dir}')
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = select_sources(train_csv, index_csv, per_subset, seed)
    cfg = load_config(config_path, runtime__torch_threads=threads)
    detector = load_corner_detector(model_path)
    # Only the detector is loaded; no recognizer/classifier/reader runs.
    class DetectorOnly:
        corner_detector = detector
    rows = []
    aggregate = {s: {"selected": 0, "decoded": 0, "candidate_count_zero": 0,
                     "bbox_hit": 0, "bbox_miss": 0, "vectors": [], "pixel_l2": [],
                     "proposal_l2": [], "raw_norm": [], "mean_pixel_l2": []} for s in SUBSETS}
    for subset, key, source_row, meta in sources:
        group = aggregate[subset]; group["selected"] += 1
        path = Path(meta["path"])
        image, error = decode_image(path)
        rec = {"subset": subset, "partition": "train", "orig_path": str(path),
               "source_sha256": None, "decode_error": error, "candidate_count": 0,
               "match_iou": None, "matched": False}
        if image is None or error:
            rows.append(rec); continue
        rec["source_sha256"] = sha256(path)
        group["decoded"] += 1
        gt = _quad(meta["quad"])
        pre = preprocess(image, cfg.preprocess)
        candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
        net = [c for c in detector.detect(image, cfg.runtime.max_candidates)
               if _in_roi(c, cfg.lane_roi(None))]
        candidates = _merge_candidates(net, candidates, cfg.runtime.max_candidates)
        rec["candidate_count"] = len(candidates)
        if not candidates:
            group["candidate_count_zero"] += 1
            rows.append(rec); continue
        gt_box = _box(gt)
        found = False
        layout = len(source_row["plate_no"])
        tw, th = cfg.rectify.size_7 if layout == 7 else cfg.rectify.size_8
        canonical_gt = np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)], np.float32)
        H = cv2.getPerspectiveTransform(gt, canonical_gt)
        for candidate_rank, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
            iou = _iou(tuple(cand.bbox), gt_box)
            if iou < IOU_MIN:
                continue
            rects = _rectifications(image, cand, cfg.rectify)
            # Respect production ordering: first valid rectification for first
            # bbox-matched candidate. Invert its actual warp matrix; cand.corners
            # is only a proposal and may differ after refinement.
            for hyp_rank, rect in enumerate(rects):
                try:
                    inv = np.linalg.inv(np.asarray(rect.homography, np.float64))
                    out_h, out_w = rect.image.shape[:2]
                    out_quad = np.asarray([(0, 0), (out_w-1, 0), (out_w-1, out_h-1), (0, out_h-1)], np.float32)
                    actual = cv2.perspectiveTransform(out_quad.reshape(1, 4, 2), inv).reshape(4, 2)
                    if not np.isfinite(actual).all() or abs(cv2.contourArea(actual)) < 1:
                        continue
                    found = True
                except (np.linalg.LinAlgError, cv2.error):
                    continue
                delta_px = actual - gt
                norm_gt = delta_px / np.maximum(np.ptp(gt, axis=0), 1.)
                actual_canonical = cv2.perspectiveTransform(actual.reshape(1, 4, 2), H).reshape(4, 2)
                canonical_delta = actual_canonical - canonical_gt
                proposal = np.asarray(cand.corners, np.float32)
                proposal_canonical = cv2.perspectiveTransform(proposal.reshape(1, 4, 2), H).reshape(4, 2)
                per_corner_px = np.linalg.norm(delta_px, axis=1)
                per_corner_unit = np.linalg.norm(canonical_delta, axis=1)
                rec.update({"matched": True, "match_iou": float(iou), "candidate_rank": candidate_rank,
                            "rectification_rank": hyp_rank, "gt_quad_px": gt.tolist(),
                            "actual_quad_px": actual.tolist(), "proposal_quad_px": proposal.tolist(),
                            "delta_xy_over_gt_wh": norm_gt.tolist(),
                            "actual_canonical": actual_canonical.tolist(),
                            "canonical_delta": canonical_delta.tolist(),
                            "canonical_l2_by_corner": per_corner_unit.tolist(),
                            "corner_l2_px": per_corner_px.tolist(),
                            "mean_corner_l2_px": float(per_corner_px.mean()),
                            "proposal_canonical": proposal_canonical.tolist()})
                break
            if found:
                group["bbox_hit"] += 1
                group["vectors"].append(rec["canonical_delta"])
                group["pixel_l2"].append(rec["corner_l2_px"])
                group["proposal_l2"].append(np.linalg.norm(proposal_canonical-canonical_gt, axis=1).tolist())
                group["raw_norm"].append(rec["delta_xy_over_gt_wh"])
                group["mean_pixel_l2"].append(rec["mean_corner_l2_px"])
                break
        if not found:
            group["bbox_miss"] += 1
        rows.append(rec)
    jsonl = out_dir / "train_per_image.jsonl"
    with jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    sampler = {s: np.asarray(aggregate[s]["vectors"], np.float32).reshape(-1, 4, 2).tolist() for s in SUBSETS}
    audit_path = ROOT / "out/data/batch8/audit.json"
    sampler_payload = {"schema": "crop-error-v1", "frame": "GT canonical unit square",
        "corner_order": ["LT", "RT", "RB", "LB"], "selection": "first candidate in production order with a valid rectification and bbox IoU >= 0.5; one vector per source image",
        "fit_partition": "train only", "indices_sha256": {"safe_train_csv": sha256(train_csv),
            "ccpd_index_csv": sha256(index_csv),
            "batch8_data_audit_json": sha256(audit_path) if audit_path.exists() else None},
        "subsets": sampler}
    (out_dir / "train_sampler.json").write_text(json.dumps(sampler_payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    source_files = [ROOT / "lpr/pipeline.py", ROOT / "lpr/locate.py", ROOT / "lpr/preprocess.py",
                    ROOT / "lpr/rectify.py", ROOT / "lpr/corner_net.py",
                    ROOT / "lpr_tools/eval/ceiling.py", Path(__file__).resolve()]
    summary = {"schema": "crop-error-v1", "fit_partition": "train", "sampled_per_subset": per_subset,
        "seed": seed, "iou_threshold": IOU_MIN, "selection": "first production candidate with IoU>=.5 and first valid rectification; one per source image",
        "files": {str(p): sha256(p) for p in (train_csv, index_csv, config_path, model_path, *source_files)},
        "sampler": str(out_dir / "train_sampler.json"), "jsonl": str(jsonl), "subsets": {}}
    summary["files"][str(jsonl)] = sha256(jsonl)
    summary["files"][str(out_dir / "train_sampler.json")] = sha256(out_dir / "train_sampler.json")
    for s, g in aggregate.items():
        n = g["selected"]
        px = np.asarray(g["pixel_l2"], np.float64).reshape(-1).tolist()
        unit = np.linalg.norm(np.asarray(g["vectors"], np.float64).reshape(-1, 4, 2), axis=2).reshape(-1).tolist() if g["vectors"] else []
        summary["subsets"][s] = {"selected": n, "decoded": g["decoded"], "matched": g["bbox_hit"],
            "unmatched": n-g["bbox_hit"], "candidate_count_zero": g["candidate_count_zero"],
            "bbox_iou_hit_rate": g["bbox_hit"]/n if n else None,
            "corner_l2_px_by_corner_quantiles": {i: _stats([v[i] for v in g["pixel_l2"]]) for i in range(4)},
            "raw_delta_xy_over_gt_wh_quantiles": {
                "dx": _stats(np.asarray(g["raw_norm"], np.float64)[..., 0].reshape(-1).tolist()) if g["raw_norm"] else _stats([]),
                "dy": _stats(np.asarray(g["raw_norm"], np.float64)[..., 1].reshape(-1).tolist()) if g["raw_norm"] else _stats([])},
            "mean_corner_l2_px_quantiles": _stats(g["mean_pixel_l2"]),
            "canonical_delta_xy_quantiles": {
                "dx": _stats(np.asarray(g["vectors"], np.float64)[..., 0].reshape(-1).tolist()) if g["vectors"] else _stats([]),
                "dy": _stats(np.asarray(g["vectors"], np.float64)[..., 1].reshape(-1).tolist()) if g["vectors"] else _stats([])},
            "canonical_corner_l2_unit_quantiles": _stats(unit)}
    summary["source_images"] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, default=ROOT / "out/data/batch8/train.csv")
    p.add_argument("--index", type=Path, default=INDEX)
    p.add_argument("--out", type=Path, default=ROOT / "out/receipts/batch9/geometry")
    p.add_argument("--per-subset", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--corner-model", type=Path, default=DEFAULT_MODEL)
    a = p.parse_args()
    result = measure(a.train_csv, a.index, a.out, per_subset=a.per_subset, seed=a.seed,
                     threads=a.threads, config_path=a.config, model_path=a.corner_model)
    print(json.dumps({"source_images": result["source_images"], "sampler": result["sampler"],
                      "subsets": {s: {"selected": x["selected"], "matched": x["matched"], "unmatched": x["unmatched"]}
                                  for s, x in result["subsets"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
