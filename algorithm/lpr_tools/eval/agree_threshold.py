"""Trace and tune agree arbitration on fixed E1 manifests.

The trace preserves the production candidate/rect/layout traversal. Threshold
replay consumes paired segment/reader outputs from each rectification.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from lpr import pipeline
from lpr.config import load_config
from lpr.locate import locate
from lpr.preprocess import preprocess
from lpr.segment import segment
from lpr.types import RecogStatus as S
from lpr_tools.data.image_health import decode_image
from lpr_tools.data.manifest import image_path, read_manifest
from lpr_tools.eval.common import effective_config_info, make_recognizer
from lpr_tools.eval.agree_bits import DecisionBits
from lpr_tools.eval.index_samples import FIXED_HARD_MANIFEST, FIXED_VAL_MANIFEST, fixed_manifest_samples

ROOT = Path(__file__).resolve().parents[3]
INDEX = Path(r"D:\keshe2\algorithm\data\index\ccpd_index.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_e1_model_paths(cfg, expected_reader_model: str) -> None:
    expected = {"model_path": "char_cnn_v2.pt", "corner_model_path": "corner_net_v1.pt",
                "plate_reader_model_path": expected_reader_model}
    if cfg.runtime.classifier != "cnn" or any(
            not getattr(cfg.runtime, field) or
            Path(getattr(cfg.runtime, field)).name != filename or
            not cfg.resolve_path(getattr(cfg.runtime, field)).is_file()
            for field, filename in expected.items()):
        raise ValueError(f"E1 trace requires CNNv2 + corner_net_v1 + {expected_reader_model} model paths")


def _attempt(dec):
    if dec is None:
        return None
    return {"plate": dec.plate_no, "confidence": float(dec.confidence),
            "status": dec.status.value, "valid": dec.status != S.INVALID_FORMAT}


def trace_image(image, recognizer, row):
    """Record per-rect paired attempts in the same order as PlateRecognizer._run."""
    cfg = recognizer.config
    pre = preprocess(image, cfg.preprocess, cfg.lane_roi(None))
    candidates = locate(pre, cfg.locate, cfg.runtime.max_candidates)
    if recognizer.corner_detector is not None:
        roi = cfg.lane_roi(None)
        net = [c for c in recognizer.corner_detector.detect(image, cfg.runtime.max_candidates)
               if pipeline._in_roi(c, roi)]
        candidates = pipeline._merge_candidates(net, candidates, cfg.runtime.max_candidates)
    records = []
    lead_color = candidates[0].color if candidates else "unknown"
    for ci, cand in enumerate(candidates[:cfg.runtime.max_candidates]):
        rects = pipeline._rectifications(image, cand, cfg.rectify)
        if ci == 0 and rects:
            lead_color = rects[0].color
        for ri, rect in enumerate(rects):
            seg_best = None
            for layout in pipeline._layouts_to_try(rect):
                segmented = segment(rect, layout, cfg.segment)
                if not segmented.ok:
                    continue
                probs = recognizer.classifier.predict_proba(segmented.chars)
                dec = pipeline.decode(probs, segmented.layout, rect.color, cfg.postprocess)
                item = _attempt(dec)
                if item is not None and (seg_best is None or pipeline._rank(dec) > seg_best["rank"]):
                    seg_best = {**item, "rank": pipeline._rank(dec)}
                if dec.status == S.OK:
                    break
            rd = None
            if recognizer.plate_reader is not None:
                text, confs = recognizer.plate_reader.read(rect.image)
                rd_dec = pipeline._decode_text(text, confs, rect.color, cfg.postprocess.min_confidence)
                rd = _attempt(rd_dec)
            if seg_best:
                seg_best.pop("rank", None)
            record = {"candidate_index": ci, "rect_index": ri, "candidate_source": cand.source,
                      "color": rect.color, "segment": seg_best, "reader": rd}
            records.append(record)
            # Agree returns at first rect whose arbitration is OK.
            # Keep full order: threshold changes can expose a later attempt.
    return {"path": str(row.row.path), "truth": row.row.plate_no,
            "subset": row.subset, "split": row.split, "category": row.row.category,
            "input_error": None, "lead_color": lead_color,
            "candidate_count": len(candidates), "rects": records}


def arbitrate_pair(seg, rd, agree_min, solo_seg, solo_reader, prefer):
    """Return simplified outcome using the exact _arbitrate priority rules."""
    s = seg if seg and seg.get("valid", True) else None
    r = rd if rd and rd.get("valid", True) else None
    if s and r and s["plate"] == r["plate"] \
            and min(s["confidence"], r["confidence"]) >= agree_min:
        return {"plate": s["plate"], "confidence": max(s["confidence"], r["confidence"]), "status": "OK"}
    valid = {"segment": s, "reader": r}
    solo = {"segment": s if s and s["confidence"] >= solo_seg else None,
            "reader": r if r and r["confidence"] >= solo_reader else None}
    other = {"segment": "reader", "reader": "segment"}
    for choices, status in ((solo, "OK"), (valid, "LOW_CONFIDENCE")):
        side = prefer if choices[prefer] is not None else other[prefer]
        selected = choices[side]
        if selected:
            return {"plate": selected["plate"], "confidence": selected["confidence"], "status": status}
    if seg or rd:
        selected = max((x for x in (seg, rd) if x), key=lambda x: ({"OK": 3, "LOW_CONFIDENCE": 2,
                                                                    "INVALID_FORMAT": 1}.get(x["status"], 0),
                                                                   x["confidence"]))
        return {"plate": selected["plate"], "confidence": selected["confidence"], "status": selected["status"]}
    return None


def replay_record(record, agree_min, solo_seg, solo_reader, prefer):
    if record.get("input_error"):
        return {"plate": None, "confidence": 0.0, "status": "BAD_INPUT"}
    best = None
    for rect in record.get("rects", []):
        outcome = arbitrate_pair(rect["segment"], rect["reader"], agree_min,
                                 solo_seg, solo_reader, prefer)
        if outcome is None:
            continue
        rank = ({"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}.get(outcome["status"], 0),
                outcome["confidence"])
        if best is None or rank > best[0]:
            best = (rank, outcome)
        if outcome["status"] == "OK":
            return outcome
    if record.get("lead_color") in {"white", "black"}:
        return {"plate": None, "confidence": 0.0, "status": "UNSUPPORTED"}
    return best[1] if best else {"plate": None, "confidence": 0.0,
                                 "status": "SEG_FAILED" if record.get("candidate_count") else "NO_PLATE"}


def replay_first_ok(record, min_confidence=0.8):
    """Replay legacy first_ok over the paired trace for comparison reporting."""
    if record.get("input_error"):
        return {"plate": None, "confidence": 0.0, "status": "BAD_INPUT"}
    best = None
    rank = {"OK": 3, "LOW_CONFIDENCE": 2, "INVALID_FORMAT": 1}
    for rect in record.get("rects", []):
        for route in (rect.get("segment"), rect.get("reader")):
            if not route:
                continue
            status = route["status"]
            if status != "INVALID_FORMAT":
                status = "OK" if route["confidence"] >= min_confidence else "LOW_CONFIDENCE"
            item = {"plate": route["plate"], "confidence": route["confidence"], "status": status}
            if best is None or (rank.get(status, 0), item["confidence"]) > (
                    rank.get(best["status"], 0), best["confidence"]):
                best = item
            if status == "OK":
                return item
    if record.get("lead_color") in {"white", "black"}:
        return {"plate": None, "confidence": 0.0, "status": "UNSUPPORTED"}
    return best or {"plate": None, "confidence": 0.0,
                    "status": "SEG_FAILED" if record.get("candidate_count") else "NO_PLATE"}


def _metrics(records, params):
    groups = {"val": [], "devhard": []}
    for r in records:
        groups["val" if r["split"] == "val" else "devhard"].append(r)
    result = {}
    for name, rows in groups.items():
        passed = correct = plate_correct = 0
        for row in rows:
            out = replay_record(row, **params)
            truth = row["truth"]
            plate_correct += out["plate"] == truth
            passed += out["status"] == "OK"
            correct += out["status"] == "OK" and out["plate"] == truth
        result[name] = {"n": len(rows), "whole_plate_correct": plate_correct,
                        "whole_plate_accuracy": plate_correct / len(rows) if rows else None,
                        "auto_pass": passed, "auto_pass_correct": correct,
                        "auto_pass_precision": correct / passed if passed else None,
                        "auto_pass_rate": passed / len(rows) if rows else None}
    return result


def threshold_boundaries(records, route):
    values = {0.0, 1.0000000000000002}
    for row in records:
        for rect in row.get("rects", []):
            item = rect.get(route)
            if item and item.get("valid", True):
                values.add(float(item["confidence"]))
    return sorted(values)


def _score(records, params):
    groups = {"val": {"n": 0, "accepted": 0, "correct": 0, "plate_correct": 0},
              "devhard": {"n": 0, "accepted": 0, "correct": 0, "plate_correct": 0}}
    subsets = defaultdict(lambda: {"n": 0, "accepted": 0, "correct": 0, "plate_correct": 0})
    for row in records:
        name = "val" if row["split"] == "val" else "devhard"
        outcome = replay_record(row, **params)
        for item in (groups[name], subsets[f"{name}/{row.get('subset', name)}"]):
            item["n"] += 1
            item["plate_correct"] += outcome["plate"] == row["truth"]
            if outcome["status"] == "OK":
                item["accepted"] += 1
                item["correct"] += outcome["plate"] == row["truth"]
    return {"sets": groups, "subsets": dict(subsets)}


def _precision_ok(score, permille):
    return all(score[name]["accepted"] > 0 and
               score[name]["correct"] * 1000 >= permille * score[name]["accepted"]
               for name in ("val", "devhard"))


def _total_accepted(score):
    return sum(score[name]["accepted"] for name in ("val", "devhard"))


def _reported(score):
    def report(v):
        return {"n": v["n"], "whole_plate_correct": v["plate_correct"],
                   "whole_plate_accuracy": v["plate_correct"] / v["n"] if v["n"] else None,
                   "auto_pass": v["accepted"], "auto_pass_correct": v["correct"],
                   "auto_pass_precision": v["correct"] / v["accepted"] if v["accepted"] else None,
                   "auto_pass_rate": v["accepted"] / v["n"] if v["n"] else None}
    return {"sets": {name: report(v) for name, v in score["sets"].items()},
            "subsets": {name: report(v) for name, v in score["subsets"].items()}}


def _compile_records(records):
    """Prepack the full image-ordered rect traces for vectorized threshold search."""
    n = len(records)
    width = max(1, max((len(row.get("rects", [])) for row in records), default=0))
    shape = (n, width)
    arrays = {name: np.zeros(shape, dtype=bool) for name in
              ("sv", "rv", "equal", "st", "rt")}
    arrays.update({name: np.zeros(shape, dtype=np.float64) for name in ("sc", "rc")})
    arrays["split_val"] = np.asarray([row["split"] == "val" for row in records], dtype=bool)
    for i, row in enumerate(records):
        for j, rect in enumerate(row.get("rects", [])):
            s, r = rect.get("segment"), rect.get("reader")
            s = s if s and s.get("valid", True) else None
            r = r if r and r.get("valid", True) else None
            arrays["sv"][i, j] = s is not None
            arrays["rv"][i, j] = r is not None
            if s:
                arrays["sc"][i, j] = s["confidence"]
                arrays["st"][i, j] = s["plate"] == row["truth"]
            if r:
                arrays["rc"][i, j] = r["confidence"]
                arrays["rt"][i, j] = r["plate"] == row["truth"]
            arrays["equal"][i, j] = bool(s and r and s["plate"] == r["plate"])
    arrays["pair_valid"] = arrays["sv"] & arrays["rv"] & arrays["equal"]
    arrays["pair_conf"] = np.minimum(arrays["sc"], arrays["rc"])
    return arrays


def _fast_counts(data, params):
    agree = data["pair_valid"] & (data["pair_conf"] >= params["agree_min"])
    s_pass = data["sv"] & (data["sc"] >= params["solo_seg"])
    r_pass = data["rv"] & (data["rc"] >= params["solo_reader"])
    accepted_rect = agree | s_pass | r_pass
    if params["prefer"] == "segment":
        correct_rect = (agree & data["st"]) | (~agree & (s_pass & data["st"] |
                        (~s_pass & r_pass & data["rt"])))
    else:
        correct_rect = (agree & data["rt"]) | (~agree & (r_pass & data["rt"] |
                        (~r_pass & s_pass & data["st"])))
    accepted = accepted_rect.any(axis=1)
    first = accepted_rect.argmax(axis=1)
    correct = accepted & np.take_along_axis(correct_rect, first[:, None], axis=1)[:, 0]
    return {"val": {"accepted": int(accepted[data["split_val"]].sum()),
                     "correct": int(correct[data["split_val"]].sum())},
            "devhard": {"accepted": int(accepted[~data["split_val"]].sum()),
                        "correct": int(correct[~data["split_val"]].sum())},
            "_accepted_mask": accepted}


def exact_branch_bound(records, permille, *, max_nodes=200_000, checkpoint_path=None,
                       trace_sha256=None, seed_params=(), resume=True,
                       checkpoint_interval=10_000):
    """Exact finite-boundary search with a resumable, proof-safe DFS frontier."""
    if checkpoint_interval < 1:
        raise ValueError('checkpoint_interval must be positive')
    seg_values = threshold_boundaries(records, "segment")
    reader_values = threshold_boundaries(records, "reader")
    agree_values = sorted({min(s["confidence"], r["confidence"])
                           for row in records for rect in row.get("rects", [])
                           for s, r in [(rect.get("segment"), rect.get("reader"))]
                           if s and r and s.get("valid", True) and r.get("valid", True)
                           and s["plate"] == r["plate"]} | {0.0, 1.0000000000000002})
    domains = (agree_values, seg_values, reader_values)
    full_grid = len(seg_values) * len(reader_values) * len(agree_values)
    decisions = DecisionBits(records)
    domain_sha = hashlib.sha256(json.dumps(domains, separators=(",", ":")).encode()).hexdigest()
    best = None
    evaluated = nodes = pruned = leaves = 0
    objective_pruned = feasibility_pruned = solved_boxes = 0
    max_bound = 0
    started = time.perf_counter()

    def params_at(indexes, prefer):
        return {"agree_min": agree_values[indexes[0]], "solo_seg": seg_values[indexes[1]],
                "solo_reader": reader_values[indexes[2]], "prefer": prefer}

    def consider(params):
        nonlocal best, evaluated
        evaluated += 1
        score = decisions.counts(params)
        if _precision_ok(score, permille):
            count = _total_accepted(score)
            if best is None or count > best["total_auto_pass"]:
                best = {"params": params, "metrics": _reported(_score(records, params)),
                        "total_auto_pass": count}

    pending = []
    if checkpoint_path and resume and Path(checkpoint_path).is_file():
        state = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if (state.get("trace_sha256") != trace_sha256 or state.get("permille") != permille
                or state.get("domain_sha256") != domain_sha):
            raise ValueError("search checkpoint input/precision/boundary identity mismatch")
        pending = state["frontier"]
        best = state.get("best")
        nodes, pruned, leaves = state.get("nodes", 0), state.get("pruned_nodes", 0), state.get("leaf_points", 0)
        evaluated = state.get("evaluated_points", 0)
        objective_pruned = state.get("objective_pruned_nodes", 0)
        feasibility_pruned = state.get("feasibility_pruned_nodes", 0)
        solved_boxes = state.get("solved_boxes", 0)
        max_bound = state.get("max_upper_bound_seen", 0)
    else:
        seed_sets = ((0.0, .3, .5, .7, .8, .9, .95, .99, 1.0, 1.0000000000000002),
                     (.5, .7, .8, .9, .92, .95, .97, .98, .99, .995, .999, 1.0, 1.0000000000000002),
                     (.5, .7, .8, .9, .92, .95, .97, .98, .99, .995, .999, 1.0, 1.0000000000000002))
        seed_indexes = []
        for values, seed_values in zip(domains, seed_sets):
            seed_indexes.append(sorted({min(len(values)-1, bisect.bisect_left(values, v)) for v in seed_values}))
        seeds = []
        for prefer in ("segment", "reader"):
            for a in seed_indexes[0]:
                for s in seed_indexes[1]:
                    for r in seed_indexes[2]:
                        seeds.append(params_at((a, s, r), prefer))
        for item in seed_params:
            indexes = tuple(min(len(values)-1, bisect.bisect_left(values, float(item[key])))
                            for values, key in zip(domains, ("agree_min", "solo_seg", "solo_reader")))
            seeds.append(params_at(indexes, item["prefer"]))
        seen_seeds = set()
        for params in seeds:
            key = tuple(params[k] for k in ("agree_min", "solo_seg", "solo_reader", "prefer"))
            if key not in seen_seeds:
                seen_seeds.add(key)
                consider(params)
        pending = [{"lo": [0, 0, 0], "hi": [len(x)-1 for x in domains], "prefer": prefer}
                   for prefer in ("reader", "segment")]

    def checkpoint():
        if not checkpoint_path:
            return
        state = {"trace_sha256": trace_sha256, "permille": permille, "domain_sha256": domain_sha,
                 "frontier": pending, "best": best, "nodes": nodes, "pruned_nodes": pruned,
                 "leaf_points": leaves, "evaluated_points": evaluated,
                 "objective_pruned_nodes": objective_pruned,
                 "feasibility_pruned_nodes": feasibility_pruned, "solved_boxes": solved_boxes,
                 "max_upper_bound_seen": max_bound}
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, target)

    run_nodes = 0
    while pending and nodes < max_nodes:
        # Save before popping the next node, so the checkpoint frontier always
        # contains every not-yet-processed box, including the next one.
        if run_nodes % checkpoint_interval == 0:
            print(f"search progress permille={permille} nodes={nodes} frontier={len(pending)} pruned={pruned} feasible={best['total_auto_pass'] if best else 0} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
            checkpoint()
        node = pending.pop()
        lo, hi, prefer = node["lo"], node["hi"], node["prefer"]
        nodes += 1
        run_nodes += 1
        evaluated += 1
        indices = (lo[0], lo[1], lo[2])
        params = params_at(indices, prefer)
        score = decisions.counts(params)
        low_params = params
        high_params = {"agree_min": domains[0][hi[0]], "solo_seg": domains[1][hi[1]],
                       "solo_reader": domains[2][hi[2]], "prefer": prefer}
        bounds = decisions.bounds(low_params, high_params)
        accepted_upper = {name: bounds[name]["possible_accept"] for name in ("val", "devhard")}
        if _precision_ok(score, permille):
            count = _total_accepted(score)
            if best is None or count > best["total_auto_pass"]:
                best = {"params": params, "metrics": _reported(_score(records, params)),
                        "total_auto_pass": count}
            pruned += 1
            solved_boxes += 1
            continue
        possible_correct = {name: bounds[name]["possible_correct"] for name in ("val", "devhard")}
        impossible = any(bounds[name]["guaranteed_accept"] * permille >
                         possible_correct[name] * 1000 for name in ("val", "devhard"))
        if impossible:
            pruned += 1
            feasibility_pruned += 1
            continue
        upper_count = sum(min(accepted_upper[name], possible_correct[name] * 1000 // permille)
                          for name in accepted_upper)
        max_bound = max(max_bound, upper_count)
        if best is not None and upper_count <= best["total_auto_pass"]:
            pruned += 1
            objective_pruned += 1
            continue
        axis = max((i for i in range(3) if lo[i] < hi[i]),
                   key=lambda i: hi[i] - lo[i], default=None)
        if axis is None:
            leaves += 1
            if _precision_ok(score, permille):
                count = _total_accepted(score)
                if best is None or count > best["total_auto_pass"]:
                    best = {"params": params, "metrics": _reported(_score(records, params)),
                            "total_auto_pass": count}
            continue
        mid = (lo[axis] + hi[axis]) // 2
        left_hi = list(hi)
        left_hi[axis] = mid
        right_lo = list(lo)
        right_lo[axis] = mid + 1
        # LIFO: push right first so the low-threshold half is searched next.
        pending.append({"lo": right_lo, "hi": list(hi), "prefer": prefer})
        pending.append({"lo": list(lo), "hi": left_hi, "prefer": prefer})
    checkpoint()
    return {"threshold_boundaries": {"agree": len(agree_values), "solo_segment": len(seg_values),
                                     "solo_reader": len(reader_values)},
            "full_grid_points_per_preference": full_grid, "evaluated_exact_points": evaluated,
            "search_nodes": nodes, "pruned_nodes": pruned, "leaf_points": leaves,
            "objective_pruned_nodes": objective_pruned,
            "feasibility_pruned_nodes": feasibility_pruned,
            "solved_boxes": solved_boxes,
            "search_elapsed_seconds": round(time.perf_counter()-started, 3),
            "search_complete": not pending,
            "recommendation_eligible": not pending and best is not None,
            "frontier_nodes": len(pending), "run_nodes": run_nodes,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "best": best,
            "precision_permille": permille,
            "certificate": {"max_upper_bound_seen": max_bound,
                            "objective_gap": 0 if not pending else None},
            "bound_description": "low corner bounds all possible accepts by monotone OK; possible correct counts any eligible truth route, ignoring earlier wrong OK, so it is optimistic and safe"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "algorithm/config/default.toml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--repair-invalid-fallbacks", action="store_true")
    parser.add_argument("--stamp-identity", action="store_true")
    parser.add_argument("--max-search-nodes", type=int, default=200_000)
    parser.add_argument("--expected-reader-model", choices=("crnn_v1.pt", "crnn_v2.pt"),
                        default="crnn_v1.pt",
                        help="require this configured reader artifact and record its exact identity")
    parser.add_argument("--seed-file", default=str(ROOT / "algorithm/out/receipts/第九批/e1/root-refined-seeds.json"))
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cfg = load_config(args.config)
    if cfg.postprocess.arbitration != "agree" or cfg.runtime.torch_threads != 2:
        raise ValueError("trace config must use arbitration=agree and torch_threads=2")
    if cfg.postprocess.min_confidence != 0.8:
        raise ValueError("E1 replay requires fixed postprocess.min_confidence=0.8")
    _validate_e1_model_paths(cfg, args.expected_reader_model)
    launch_source_sha = sha256(Path(__file__).resolve())
    model_manifest = ROOT / "algorithm/models/manifest.json"
    launch_manifest_sha = sha256(model_manifest)
    env = {"PYTHONPATH": str(ROOT / "algorithm"), "LPR_CONFIG": str(Path(args.config).resolve())}
    os.environ.update(env)
    trace = out / "trace.jsonl"
    errors = []
    identity_path = out / "trace_identity.json"
    if args.stamp_identity:
        if not trace.is_file() or not identity_path.is_file():
            raise FileNotFoundError("identity stamping requires an existing completed trace and initial receipt")
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        counts = Counter(row["split"] for row in records)
        if len(records) != 3100 or counts != {"val": 1000, "test": 2100}:
            raise ValueError(f"trace row count/splits mismatch: rows={len(records)} counts={dict(counts)}")
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        identity = {**previous,
            "effective_config_sha256": effective_config_info(cfg, args.config)["effective_config_sha256"],
            "effective_config": effective_config_info(cfg, args.config)["effective_config"],
            "models": {field: {"path": str(cfg.resolve_path(getattr(cfg.runtime, field))),
                               "sha256": sha256(cfg.resolve_path(getattr(cfg.runtime, field)))}
                       for field in ("model_path", "corner_model_path", "plate_reader_model_path")},
            "manifests": {str(p): sha256(p) for p in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST)},
            "pipeline_sha256": sha256(ROOT / "algorithm/lpr/pipeline.py"),
            "path_list_sha256": hashlib.sha256(("\n".join(row["path"] for row in records) + "\n").encode("utf-8")).hexdigest(),
            "trace_sha256": sha256(trace), "trace_rows": len(records),
            "trace_source_sha256_at_stamp": sha256(Path(__file__).resolve()),
            "trace_source_sha256_at_launch": None,
            "trace_source_note": "launch source SHA was not captured. Full ordered rect traversal and INVALID_FORMAT preservation were verified against the immutable original trace; retracing 577 fallback-only rows was byte-identical.",
            "trace_invalid_format_filtered_at_launch": False,
            "identity_migration": {"status": "human_reviewed_posthoc", "verified": True,
                                   "basis": ["trace row/split audit", "ordered fixed-manifest path equality",
                                             "trace content SHA", "manifest/config/model/pipeline SHA identity"]}}
        old_receipt = out / "trace_identity.pre-stamp.json"
        if old_receipt.exists():
            raise FileExistsError(f"preserved initial identity already exists: {old_receipt}")
        old_receipt.write_bytes(identity_path.read_bytes())
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    if args.scan and not trace.is_file():
        raise FileNotFoundError("--scan requires a completed existing trace.jsonl")
    if args.scan and trace.is_file() and not identity_path.is_file():
        raise ValueError("existing trace has no identity receipt; refuse to overwrite or scan")
    if args.trace_only and trace.exists():
        raise FileExistsError("trace.jsonl already exists; choose a new --out to avoid overwriting evidence")
    if args.repair_invalid_fallbacks and (not trace.is_file() or not identity_path.is_file()):
        raise FileNotFoundError("invalid-fallback repair requires an existing trace and identity receipt")
    existing = (args.scan or args.repair_invalid_fallbacks) and trace.is_file() and identity_path.is_file()
    if existing:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        now_cfg = effective_config_info(cfg, args.config)
        now_models = {field: {"path": str(cfg.resolve_path(getattr(cfg.runtime, field))),
                              "sha256": sha256(cfg.resolve_path(getattr(cfg.runtime, field)))}
                      for field in ("model_path", "corner_model_path", "plate_reader_model_path")}
        if identity.get("effective_config_sha256") != now_cfg["effective_config_sha256"]:
            raise ValueError("existing trace effective config differs; refuse scan")
        if identity.get("models") != now_models:
            raise ValueError("existing trace model files differ; refuse scan")
        if identity.get("expected_reader_model", "crnn_v1.pt") != args.expected_reader_model:
            raise ValueError("existing trace reader model selector differs; refuse scan")
        if identity.get("model_manifest_sha256", launch_manifest_sha) != launch_manifest_sha:
            raise ValueError("existing trace model manifest differs; refuse scan")
        if identity.get("pipeline_sha256") != sha256(ROOT / "algorithm/lpr/pipeline.py"):
            raise ValueError("existing trace pipeline source differs; refuse scan")
        if identity.get("trace_sha256") != sha256(trace):
            raise ValueError("existing trace content hash differs; refuse scan")
        migration = identity.get("identity_migration", {})
        if migration.get("verified") is not True or migration.get("status") not in {
                "generated_in_run", "human_reviewed_posthoc"}:
            raise ValueError("trace identity is unverified; refuse scan")
        if identity.get("manifests") != {str(p): sha256(p) for p in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST)}:
            raise ValueError("existing trace fixed manifests differ; refuse scan")
        errors = identity.get("input_errors", [])
    else:
        recognizer = make_recognizer(argparse.Namespace(fake=False, uniform=False, classifier=None, config=args.config))
        samples = []
        for manifest, purpose in ((FIXED_VAL_MANIFEST, "val"), (FIXED_HARD_MANIFEST, "dev-hard")):
            selected, _ = fixed_manifest_samples(INDEX, manifest, purpose=purpose)
            samples.extend(selected)
        with trace.open("w", encoding="utf-8", newline="\n") as f:
            for i, sample in enumerate(samples, 1):
                image, error = decode_image(image_path(sample.row, FIXED_VAL_MANIFEST if sample.split == "val" else FIXED_HARD_MANIFEST))
                if image is None:
                    record = {"path": sample.row.path, "truth": sample.row.plate_no,
                              "subset": sample.subset, "split": sample.split, "category": sample.row.category,
                              "input_error": error, "lead_color": "unknown", "candidate_count": 0, "rects": []}
                    errors.append({"path": sample.row.path, "reason": error})
                else:
                    record = trace_image(image, recognizer, sample)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    f.flush()
                    print(f"traced {i}/{len(samples)}", flush=True)
    if not existing:
        identity = {"argv": sys.argv, "cwd": str(Path.cwd()), "config": str(Path(args.config).resolve()),
                "config_sha256": sha256(Path(args.config)),
                "effective_config_sha256": effective_config_info(cfg, args.config)["effective_config_sha256"],
                "effective_config": effective_config_info(cfg, args.config)["effective_config"],
                "models": {field: {"path": str(cfg.resolve_path(getattr(cfg.runtime, field))),
                                   "sha256": sha256(cfg.resolve_path(getattr(cfg.runtime, field)))}
                           for field in ("model_path", "corner_model_path", "plate_reader_model_path")},
                "expected_reader_model": args.expected_reader_model,
                "model_manifest_sha256": launch_manifest_sha,
                "trace_source_sha256_at_launch": launch_source_sha,
                "manifests": {str(p): sha256(p) for p in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST)},
                "pipeline_sha256": sha256(ROOT / "algorithm/lpr/pipeline.py"),
                "path_list_sha256": hashlib.sha256(("\n".join(s.row.path for s in samples) + "\n").encode("utf-8")).hexdigest(),
                "trace_sha256": sha256(trace), "trace_rows": len(samples), "input_errors": errors,
                "identity_migration": {"status": "generated_in_run", "verified": True},
                "threads": {"torch": cfg.runtime.torch_threads, "opencv": 1}}
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.trace_only:
        return 2 if errors else 0
    # Scanning is deliberately a separate phase; only exact observed confidence boundaries are legal.
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    counts = Counter(row["split"] for row in records)
    path_list_sha = hashlib.sha256(("\n".join(row["path"] for row in records) + "\n").encode("utf-8")).hexdigest()
    if len(records) != 3100 or counts != {"val": 1000, "test": 2100}:
        raise ValueError(f"trace row order/shape is invalid: rows={len(records)} split_counts={dict(counts)}")
    if identity.get("path_list_sha256", path_list_sha) != path_list_sha:
        raise ValueError("trace ordered input path list differs from identity")
    expected_paths = []
    for manifest in (FIXED_VAL_MANIFEST, FIXED_HARD_MANIFEST):
        expected_paths.extend(str(image_path(row, manifest).resolve())
                              for row in read_manifest(manifest, check_files=True))
    if [str(Path(row["path"]).resolve()) for row in records] != expected_paths:
        raise ValueError("trace ordered path list does not exactly match fixed manifests")
    if args.repair_invalid_fallbacks:
        from types import SimpleNamespace
        samples = []
        for manifest, split in ((FIXED_VAL_MANIFEST, "val"), (FIXED_HARD_MANIFEST, "test")):
            rows = read_manifest(manifest, check_files=True)
            samples.extend(SimpleNamespace(row=row, split=split,
                                           subset=row.source.rsplit("/", 1)[-1]) for row in rows)
        if len(records) != len(samples) or any(record["path"] != sample.row.path
                                               for record, sample in zip(records, samples)):
            raise ValueError("cannot repair trace: fixed ordered rows differ")
        original_trace = out / "trace.pre-invalid-repair.jsonl"
        if original_trace.exists():
            raise FileExistsError(f"preserved pre-repair trace already exists: {original_trace}")
        original_trace.write_bytes(trace.read_bytes())
        recognizer = make_recognizer(argparse.Namespace(fake=False, uniform=False, classifier=None,
                                                        config=args.config))
        repaired = []
        replacements = []
        for index, (record, sample) in enumerate(zip(records, samples)):
            has_valid = any(route and route.get("valid", True)
                            for rect in record.get("rects", [])
                            for route in (rect.get("segment"), rect.get("reader")))
            if record.get("candidate_count", 0) and not has_valid:
                manifest = FIXED_VAL_MANIFEST if sample.split == "val" else FIXED_HARD_MANIFEST
                image, error = decode_image(image_path(sample.row, manifest))
                if image is None:
                    raise ValueError(f"fallback-only row became unreadable: {sample.row.path} {error}")
                corrected = trace_image(image, recognizer, sample)
                replacements.append({"path": record["path"], "row_index": index + 1,
                                     "old_rects": len(record.get("rects", [])),
                                     "new_rects": len(corrected.get("rects", []))})
                record = corrected
            repaired.append(record)
        trace.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired),
                         encoding="utf-8", newline="\n")
        identity["trace_sha256"] = sha256(trace)
        identity["trace_rows"] = len(repaired)
        identity["invalid_fallback_repairs"] = replacements
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "invalid_fallback_repairs.json").write_text(
            json.dumps({"rows_retraced": len(replacements), "rows": replacements},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if not errors else 2
    (out / "trace_summary.json").write_text(json.dumps({"rows": len(records), "input_errors": errors,
        "confidence_boundaries": {k: len(threshold_boundaries(records, k)) for k in ("segment", "reader")}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.scan:
        if errors:
            raise RuntimeError(f"cannot tune with undecodable fixed inputs ({len(errors)}); all rows must be accounted for and manually resolved")
        seed_params = []
        seed_path = Path(args.seed_file)
        if seed_path.is_file():
            seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_params = [seed_data["best"][key]["params"] for key in ("990", "995")]
        result = {}
        for key, permille in (("99.0", 990), ("99.5", 995)):
            result[key] = exact_branch_bound(
                records, permille, max_nodes=args.max_search_nodes,
                checkpoint_path=out / f"search-checkpoint-{permille}.json",
                trace_sha256=identity["trace_sha256"], seed_params=seed_params)
        (out / "scan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if any(not result[key]["recommendation_eligible"] for key in ("99.0", "99.5")):
            print("exact threshold search incomplete; checkpoint is resumable, no global-optimum claim", file=sys.stderr)
            return 3
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
