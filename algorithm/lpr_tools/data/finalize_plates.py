"""Safely cap validation crops while retaining every GT source crop.

Dry-run by default. Use ``--apply`` only after the dataset build has completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from lpr.charset import PROVINCES
from lpr_tools.data.build_plates import FIELDS


def _safe_val_file(root: Path, rel: str, *, must_exist: bool = True) -> Path:
    p = Path(rel)
    if p.is_absolute() or p.drive or ".." in p.parts or not p.parts or p.parts[0].lower() != "val":
        raise ValueError(f"unsafe validation label path: {rel}")
    lexical = root
    for part in p.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError(f"symlink validation path is forbidden: {rel}")
    full = (root / p).resolve(strict=must_exist)
    val_root = (root / "val").resolve(strict=True)
    if not val_root.is_relative_to(root) or not full.is_relative_to(val_root):
        raise ValueError(f"validation path escapes target: {rel}")
    if must_exist and not full.is_file():
        raise FileNotFoundError(full)
    return full


def _candidate_order(rows: list[dict[str, str]], quota: int, seed: int) -> list[dict[str, str]]:
    by_layout: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_layout[row["layout"]][row["orig_path"]].append(row)
    rng = random.Random(seed)
    queues = {}
    for layout, origins in by_layout.items():
        items = []
        for origin, candidates in sorted(origins.items()):
            candidates.sort(key=lambda item: item["file"])
            rng.shuffle(candidates)
            items.append((origin, deque(candidates)))
        rng.shuffle(items)
        queues[layout] = deque(items)
    layouts = sorted(queues, key=lambda value: (value != "7", value))
    selected = []
    while len(selected) < quota and any(queues[k] for k in layouts):
        for layout in layouts:
            q = queues[layout]
            if not q:
                continue
            origin, candidates = q.popleft()
            selected.append(candidates.popleft())
            if candidates:
                q.append((origin, candidates))
            if len(selected) == quota:
                break
    return selected


def _stats(rows: list[dict[str, str]]) -> dict:
    province, green, source, split, layout = Counter(), Counter(), Counter(), Counter(), Counter()
    origins = defaultdict(set)
    for row in rows:
        name = row["file"].split("/", 1)[0]
        split[name] += 1
        source[row["source"]] += 1
        green[(name, row["color"] == "green")] += 1
        layout[(name, row["layout"])] += 1
        if row["plate_no"]:
            province[(name, row["plate_no"][0])] += 1
            origins[(name, row["plate_no"][0])].add(row["orig_path"])
    return {"by_split_crops": dict(split), "by_source": dict(source),
            "green_by_split": {f"{s}/{g}": n for (s, g), n in sorted(green.items())},
            "layout_by_split": {f"{s}/{l}": n for (s, l), n in sorted(layout.items())},
            "province_by_split_crops": {f"{s}/{p}": n for (s, p), n in sorted(province.items())},
            "province_by_split_sources": {f"{s}/{p}": len(v) for (s, p), v in sorted(origins.items())},
            "province_by_split_including_zero": {
                f"{s}/{p}": province.get((s, p), 0) for s in ("train", "val") for p in PROVINCES},
            "rare_provinces_under_1000": {
                f"{s}/{p}": province.get((s, p), 0) for s in ("train", "val") for p in PROVINCES
                if province.get((s, p), 0) < 1000}}


def finalize(target: Path, receipt_dir: Path, *, apply: bool = False,
             cap: int = 10_000, seed: int = 42) -> dict:
    if cap <= 0:
        raise ValueError("cap must be positive")
    root = target.resolve(strict=True)
    if target.is_symlink() or not (root / "labels.csv").is_file():
        raise ValueError("target must be a real built dataset directory with labels.csv")
    labels = root / "labels.csv"
    backup = root / "labels.generated.csv"
    baseline = backup if backup.exists() else labels
    with labels.open(encoding="utf-8", newline="") as handle:
        current_reader = csv.DictReader(handle)
        if tuple(current_reader.fieldnames or ()) != FIELDS:
            raise ValueError("current labels.csv schema mismatch")
        current_rows = list(current_reader)
    with baseline.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("labels.csv schema mismatch")
        rows = list(reader)
    val = [r for r in rows if r["file"].split("/", 1)[0] == "val"]
    gt = [r for r in val if r["source"] == "ground_truth"]
    candidates = [r for r in val if r["source"].startswith("pipeline_candidate:")]
    quota = max(0, cap - len(gt))
    selected_candidates = _candidate_order(candidates, quota, seed)
    kept_names = {r["file"] for r in gt + selected_candidates}
    removed = [r for r in candidates if r["file"] not in kept_names]
    # Validate every retained and removed path before a single file or ledger is changed.
    validated = {r["file"]: _safe_val_file(root, r["file"]) for r in gt + selected_candidates}
    for row in removed:
        validated[row["file"]] = _safe_val_file(root, row["file"], must_exist=False)
    kept_rows = [r for r in rows if r["file"].split("/", 1)[0] != "val" or r["file"] in kept_names]
    result = {"mode": "apply" if apply else "dry_run", "cap": cap, "seed": seed,
              "val_gt_kept": len(gt), "val_candidates_before": len(candidates),
              "val_candidates_kept": len(selected_candidates), "val_crops_after": len(gt)+len(selected_candidates),
              "removed_candidate_files": len(removed), "removed_paths": [r["file"] for r in removed],
              "layout_candidates_kept": dict(Counter(r["layout"] for r in selected_candidates)),
              "stats_after": _stats(kept_rows)}
    if not apply:
        return result

    # A stale snapshot or active/incomplete build must never authorize deletion.
    stats_path = root / "stats.json"
    state_path = root / "build_state.json"
    ledger_path = root / "source_hashes.json"
    if not stats_path.is_file() or not state_path.is_file() or not ledger_path.is_file():
        raise RuntimeError("refusing apply before final build stats, state, and source hash ledger exist")
    stats_now = json.loads(stats_path.read_text(encoding="utf-8"))
    state_now = json.loads(state_path.read_text(encoding="utf-8"))
    ledger_now = json.loads(ledger_path.read_text(encoding="utf-8"))
    actual_by_split = dict(Counter(r["file"].split("/", 1)[0] for r in current_rows))
    if (stats_now.get("by_split") != actual_by_split
            or state_now.get("processed_this_run", -1) < state_now.get("selected_remaining_at_start", 0)
            or set(ledger_now) != {r["orig_path"] for r in current_rows}):
        raise RuntimeError("refusing apply: labels, stats, build state, or source hash ledger is incomplete/stale")

    if not backup.exists():
        fd, backup_tmp_name = tempfile.mkstemp(prefix="labels.generated.", suffix=".tmp", dir=root)
        os.close(fd)
        backup_tmp = Path(backup_tmp_name)
        shutil.copy2(labels, backup_tmp)
        backup_tmp.replace(backup)
    output = root / "labels.csv.tmp"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept_rows)
        handle.flush()
        os.fsync(handle.fileno())
    for row in removed:
        validated[row["file"]].unlink(missing_ok=True)
    if current_rows != kept_rows:
        output.replace(labels)
    else:
        output.unlink(missing_ok=True)

    # Keep per-origin geometry audits aligned with retained label rows.
    audit_root = root / "_crop_provenance"
    val_origins = {r["orig_path"] for r in val}
    val_audit_keys = {hashlib.sha1(("val\0" + origin.casefold()).encode("utf-8")).hexdigest()
                      for origin in val_origins}
    if audit_root.is_dir():
        for key in val_audit_keys:
            audit = audit_root / f"{key}.json"
            if not audit.is_file():
                continue
            items = json.loads(audit.read_text(encoding="utf-8"))
            changed = False
            for item in items:
                keep = item.get("file") in kept_names
                if item.get("retained") != keep:
                    item["retained"] = keep
                    changed = True
            if changed:
                audit.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    summary = _stats(kept_rows)
    summary["by_split"] = summary["by_split_crops"]
    summary.update({"validation_cap": cap, "validation_seed": seed,
                    "validation_gt_kept": len(gt), "validation_candidates_kept": len(selected_candidates),
                    "validation_candidates_removed": len(removed), "generated_stats_file": "stats.generated.json"})
    previous_stats = stats_path
    if previous_stats.exists() and not (root / "stats.generated.json").exists():
        shutil.copy2(previous_stats, root / "stats.generated.json")
    stats_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if not stats_path.exists() or stats_path.read_text(encoding="utf-8") != stats_text:
        stats_path.write_text(stats_text, encoding="utf-8")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "val_finalize.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = root / "provenance.json"
    if provenance.exists():
        details = json.loads(provenance.read_text(encoding="utf-8"))
        details["validation_finalize"] = {"policy": "retain every GT; seed-42 round-robin pipeline candidates by orig_path and 7/8 layout", "cap": cap, "seed": seed, "retained_crops": len(gt)+len(selected_candidates), "removed_candidates": len(removed)}
        provenance_text = json.dumps(details, ensure_ascii=False, indent=2)
        if provenance.read_text(encoding="utf-8") != provenance_text:
            provenance.write_text(provenance_text, encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path(r"D:\keshe2\algorithm\data\plates\v1"))
    parser.add_argument("--receipt-dir", type=Path, default=Path("algorithm/out/receipts/第七批/plates"))
    parser.add_argument("--cap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="write backup, prune files, and replace labels")
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.target, args.receipt_dir, apply=args.apply, cap=args.cap, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
