"""Generate a small, CCPD-shaped synthetic dataset for pipeline development.

Images and filename tokens follow the conventions consumed by
``lpr_tools.data.ccpd``. The pixels and labels are synthetic; this dataset is
only useful for exercising the data flow, not measuring recognition quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from lpr.charset import LETTERS, PROVINCES
from lpr.config import ALGORITHM_DIR
from lpr.io_utils import imwrite
from PIL import Image, ImageDraw, ImageFont

from lpr_tools.data.ccpd import ADS, ALPHABETS, PROVINCES_CCPD
from lpr_tools.synth_scenes import CENTERS7, CENTERS8, draw_plate

IMAGE_SIZE = (720, 1160)  # width, height: portrait CCPD frame
SUBSETS_2019 = ("ccpd_base", "ccpd_db", "ccpd_tilt", "ccpd_blur", "ccpd_np")
SPLITS_2020 = ("train", "val", "test")
HARD_SUBSETS = ("ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
                "ccpd_fn", "ccpd_challenge", "ccpd_weather")
WORKER_ROOT = Path(__file__).resolve().parents[2]
SHARED_INDEX_ROOT = ALGORITHM_DIR / "data/index"


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required exclusion list missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"manifest has no path column: {path}")
        return [row for row in reader if row.get("path")]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_facts(path: str | Path) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def _stable_image_sha(path: str | Path, *, attempts: int = 2) -> tuple[str, tuple[int, int]]:
    """Hash a stable file snapshot; reject concurrent mutation rather than mislabel it."""
    path = Path(path)
    for _ in range(attempts):
        before = _file_facts(path)
        digest = _sha256_file(path)
        after = _file_facts(path)
        if before == after:
            return digest, after
    raise OSError(f"image changed while its bytes were being hashed: {path}")


def _load_hash_cache(cache_path: Path) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    digests: dict[str, str] = {}
    facts: dict[str, tuple[int, int]] = {}
    if not cache_path.is_file():
        return digests, facts
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                saved = json.loads(line)
                key, path, sha = saved["key"], saved["path"], saved["sha256"]
                if _normalized_path(path) != key or len(sha) != 64 or any(
                        char not in "0123456789abcdef" for char in sha.lower()):
                    continue
                fact = (int(saved["size"]), int(saved["mtime_ns"]))
                if fact == _file_facts(path):
                    digests[key], facts[key] = sha.lower(), fact
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # Includes a truncated final line from an interrupted append.
                continue
    return digests, facts


def _path_hashes(rows: list[dict[str, str]]) -> tuple[set[str], set[str], list[str]]:
    paths = {_normalized_path(row["path"]) for row in rows}
    digests: set[str] = set()
    unavailable: list[str] = []
    for row in rows:
        path = row["path"]
        try:
            digests.add(_sha256_file(path))
        except OSError:
            unavailable.append(path)
    return paths, digests, unavailable


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_hard_pool(*, output_dir: str | Path | None = None,
                      index_path: str | Path | None = None,
                      frozen_path: str | Path | None = None,
                      golden_path: str | Path | None = None,
                      devhard_path: str | Path | None = None,
                      undecodable_path: str | Path | None = None,
                      seventh_data_dir: str | Path | None = None,
                      v1_training_csv: str | Path | None = None,
                      seed: int = 42, holdout_per_subset: int = 300) -> dict:
    """Build byte-safe, balanced hard-pool train/holdout CSVs and an audit.

    No pixels are inspected. Image bytes are hashed solely to enforce that no
    listed frozen, golden, fixed-dev, or undecodable image (or byte-identical
    copy) enters training or the tuning holdout.
    """
    if holdout_per_subset < 1 or seed < 0:
        raise ValueError("holdout_per_subset and seed must be positive/nonnegative")
    index_path = Path(index_path or SHARED_INDEX_ROOT / "ccpd_index.csv")
    frozen_path = Path(frozen_path or SHARED_INDEX_ROOT / "testset_manifest_v1.csv")
    golden_path = Path(golden_path or WORKER_ROOT / "algorithm/tests/data/golden/golden_manifest.csv")
    devhard_path = Path(devhard_path or SHARED_INDEX_ROOT / "devhard_v1.csv")
    undecodable_path = Path(undecodable_path or SHARED_INDEX_ROOT / "undecodable.csv")
    output_dir = Path(output_dir or WORKER_ROOT / "algorithm/out/eighth/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / ".image-sha-cache.jsonl"
    digest_cache, digest_facts = _load_hash_cache(cache_path)

    def image_hash(path: str | Path) -> str:
        key = _normalized_path(str(path))
        fact = _file_facts(path)
        if key not in digest_cache or digest_facts.get(key) != fact:
            digest_cache[key], fact = _stable_image_sha(path)
            digest_facts[key] = fact
        return digest_cache[key]

    def cache_hashes(paths: list[str], label: str) -> list[str]:
        """Hash bounded batches in parallel and report durable scan progress."""
        hashes: list[str] = []
        batch_size = 256
        with ThreadPoolExecutor(max_workers=4) as executor:
            for offset in range(0, len(paths), batch_size):
                batch = paths[offset:offset + batch_size]
                batch_hashes = list(executor.map(image_hash, batch))
                hashes.extend(batch_hashes)
                with cache_path.open("a", encoding="utf-8") as handle:
                    for path, digest in zip(batch, batch_hashes, strict=False):
                        key = _normalized_path(path)
                        fact = digest_facts[key]
                        if _file_facts(path) != fact:
                            raise OSError(f"image changed before hash-cache commit: {path}")
                        handle.write(json.dumps({"key": key, "path": path,
                                                 "size": fact[0], "mtime_ns": fact[1],
                                                 "sha256": digest}) + "\n")
                completed = offset + len(batch)
                if completed % 5000 < batch_size or completed == len(paths):
                    print(f"HASH_PROGRESS label={label} completed={completed}/{len(paths)} "
                          f"unique_cached_paths={len(digest_cache)}", flush=True)
        return hashes

    manifests = {"frozen": _read_manifest(frozen_path), "golden": _read_manifest(golden_path),
                 "devhard": _read_manifest(devhard_path), "undecodable": _read_manifest(undecodable_path)}
    excluded_paths: set[str] = set()
    excluded_hashes: set[str] = set()
    source_audit: dict[str, dict] = {}
    for name, rows in manifests.items():
        paths, hashes, unavailable = _path_hashes(rows)
        if unavailable:
            raise ValueError(f"cannot hash {len(unavailable)} {name} exclusion images; first={unavailable[0]}")
        excluded_paths |= paths
        excluded_hashes |= hashes
        source_audit[name] = {"rows": len(rows), "unique_paths": len(paths),
                              "unique_sha256": len(hashes), "manifest_sha256": _sha256_file(
                                  {"frozen": frozen_path, "golden": golden_path,
                                   "devhard": devhard_path, "undecodable": undecodable_path}[name])}

    def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    seventh_data = Path(seventh_data_dir or WORKER_ROOT / "algorithm/out/seventh/data")
    source_names = {"safe_train": "train_sampled_30000.csv", "safe_val": "val_1000.csv",
                    "np_train": "np_train.csv", "np_val": "np_val.csv"}
    source_rows: dict[str, tuple[list[str], list[dict[str, str]]]] = {
        name: read_csv_rows(seventh_data / filename) for name, filename in source_names.items()}
    source_np_audit_path = seventh_data / "np-split-audit.json"
    source_np_audit = json.loads(source_np_audit_path.read_text(encoding="utf-8"))
    source_np_train_sha = hashlib.sha256((seventh_data / "np_train.csv").read_bytes()).hexdigest()
    source_np_val_sha = hashlib.sha256((seventh_data / "np_val.csv").read_bytes()).hexdigest()
    inherited_np_decode_verified = (
        source_np_audit.get("all_selected_decoded_valid") is True
        and source_np_audit.get("csv", {}).get("np_train.csv_sha256") == source_np_train_sha
        and source_np_audit.get("csv", {}).get("np_val.csv_sha256") == source_np_val_sha)
    if not inherited_np_decode_verified:
        raise ValueError("cannot inherit NP decode proof: source batch-7 audit or CSV hashes do not match")

    def rows_hashes(name: str) -> set[str]:
        paths = [row["path"] for row in source_rows[name][1]]
        return set(cache_hashes(paths, name))

    # Keep every training candidate disjoint from the fixed base validation and
    # NP validation sets. Also keep hard cases disjoint from existing train/NP.
    safe_train_hashes = rows_hashes("safe_train")
    safe_val_hashes = rows_hashes("safe_val")
    np_train_hashes_all = rows_hashes("np_train")
    np_val_hashes = rows_hashes("np_val")
    fixed_holdout_hashes = safe_val_hashes | np_val_hashes
    existing_train_hashes = safe_train_hashes | np_train_hashes_all
    _hard_blocked_hashes = excluded_hashes | fixed_holdout_hashes | existing_train_hashes
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"path", "dataset", "subset", "split", "has_plate", "plate_no", "layout",
                    "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
                    "blurriness", "area", "skip_reason"}
        if not required.issubset(fields):
            raise ValueError("ccpd_index missing required corner training columns")
        pool_by_subset: dict[str, list[dict[str, str]]] = {subset: [] for subset in HARD_SUBSETS}
        # The index is streamed and only the seven assigned subsets are kept in
        # memory. Training positives come from the already sampled safe train CSV.
        for row in reader:
            if (row["subset"] in pool_by_subset and row["has_plate"] == "1"
                    and not row["skip_reason"] and len(row["quad"].split()) == 8):
                pool_by_subset[row["subset"]].append(row)
    # Hash only assigned hard-pool candidates; safe train and fixed holdout
    # hashes were already computed from the bounded batch-7 CSV inputs above.
    candidates = [row for subset in HARD_SUBSETS for row in pool_by_subset[subset]
                  if _normalized_path(row["path"]) not in excluded_paths]
    digest_by_path: dict[str, str] = {}
    candidate_paths = [row["path"] for row in candidates]
    try:
        candidate_hashes = cache_hashes(candidate_paths, "hard_candidates")
    except OSError as exc:
        raise ValueError(f"cannot hash a hard-pool candidate: {exc}") from exc
    for row, digest in zip(candidates, candidate_hashes, strict=False):
        digest_by_path[_normalized_path(row["path"])] = digest

    pool_rows: list[dict[str, str]] = []
    seen_digests = set(excluded_hashes | fixed_holdout_hashes | existing_train_hashes)
    pool_counts: dict[str, dict[str, int]] = {}
    dropped_by_subset: dict[str, Counter] = {}
    for subset in HARD_SUBSETS:
        kept: list[dict[str, str]] = []
        dropped = Counter()
        for row in sorted(pool_by_subset[subset], key=lambda item: _normalized_path(item["path"])):
            key = _normalized_path(row["path"])
            if key in excluded_paths:
                dropped["listed_path"] += 1
                continue
            digest = digest_by_path[key]
            if digest in excluded_hashes:
                dropped["listed_bytes"] += 1
            elif digest in fixed_holdout_hashes:
                dropped["fixed_tuning_holdout_bytes"] += 1
            elif digest in existing_train_hashes or digest in seen_digests:
                dropped["duplicate_bytes"] += 1
            else:
                kept.append(row)
                seen_digests.add(digest)
        rng = random.Random(seed)
        rng.shuffle(kept)
        hold_n = min(holdout_per_subset, max(1, int(len(kept) * .10)))
        holdout = kept[:hold_n]
        train = kept[hold_n:]
        for role, selected in (("hard_train", train), ("hard_tuning_holdout", holdout)):
            for row in selected:
                pool_rows.append({**row, "source_split": row["split"], "source_category": role,
                                  "split": "train" if role == "hard_train" else "val"})
        pool_counts[subset] = {"eligible_before_exclusion": len(pool_by_subset[subset]),
                               "available_after_exclusion_and_byte_dedup": len(kept),
                               "train": len(train), "tuning_holdout": len(holdout)}
        dropped_by_subset[subset] = dropped

    # Recheck the final memberships against all four forbidden byte sets.
    hard_train = [r for r in pool_rows if r["source_category"] == "hard_train"]
    hard_holdout = [r for r in pool_rows if r["source_category"] == "hard_tuning_holdout"]
    members = pool_rows
    member_hashes = {digest_by_path[_normalized_path(r["path"])] for r in members}
    if member_hashes & excluded_hashes:
        raise AssertionError("forbidden byte hash remains in a generated training or holdout pool")
    fields_out = [*fields, "source_split", "source_category"]
    _write_csv(output_dir / "hard_pool_train.csv", hard_train, fields_out)
    _write_csv(output_dir / "hard_tuning_holdout.csv", hard_holdout, fields_out)

    def safe_filter(rows: list[dict[str, str]], blocked_hashes: set[str],
                    *, deduplicate: bool = True) -> tuple[list[dict[str, str]], dict[str, int]]:
        selected: list[dict[str, str]] = []
        counts = Counter()
        seen: set[str] = set()
        hashable_rows = [row for row in rows if _normalized_path(row["path"]) not in excluded_paths]
        row_hashes = cache_hashes([row["path"] for row in hashable_rows], "safe_partition")
        hashes_by_key = {_normalized_path(row["path"]): digest
                         for row, digest in zip(hashable_rows, row_hashes, strict=False)}
        for row in rows:
            key = _normalized_path(row["path"])
            if key in excluded_paths:
                counts["listed_path"] += 1
                continue
            digest = hashes_by_key[key]
            if digest in blocked_hashes:
                counts["blocked_content"] += 1
                continue
            if deduplicate and digest in seen:
                counts["duplicate_content"] += 1
                continue
            seen.add(digest)
            selected.append(row)
        return selected, dict(counts)

    # Apply byte-level exclusions to the previous reproducible train/val/NP
    # partitions, preserving the user's authorized NP 80/20 membership.
    train_source_rows = source_rows["safe_train"][1]
    train_safe, train_removed = safe_filter(train_source_rows, excluded_hashes | fixed_holdout_hashes)
    train_hashes = {image_hash(row["path"]) for row in train_safe}
    # Exclude exact copies of the training data from validation and NP; keep
    # their existing split assignment and never rebalance across this boundary.
    val_safe, val_removed = safe_filter(source_rows["safe_val"][1], excluded_hashes | train_hashes)
    np_train_safe, np_train_removed = safe_filter(source_rows["np_train"][1], excluded_hashes | train_hashes | safe_val_hashes)
    np_train_hashes = {image_hash(row["path"]) for row in np_train_safe}
    np_val_safe, np_val_removed = safe_filter(source_rows["np_val"][1], excluded_hashes | train_hashes | np_train_hashes | safe_val_hashes)
    for name, rows in (("safe_train", train_safe), ("safe_val", val_safe),
                       ("np_train_safe", np_train_safe), ("np_val_safe", np_val_safe)):
        fields = source_rows[name.removesuffix("_safe") if name.startswith("np_") else name][0]
        if name == "np_train_safe":
            fields = source_rows["np_train"][0]
        if name == "np_val_safe":
            fields = source_rows["np_val"][0]
        _write_csv(output_dir / f"{name}.csv", rows, fields)
    train_pathsets = {name: {_normalized_path(r["path"]) for r in rows}
                      for name, rows in (("safe_train", train_safe), ("np_train", np_train_safe),
                                         ("hard_train", hard_train))}
    eval_pathsets = {name: {_normalized_path(r["path"]) for r in rows}
                     for name, rows in (("safe_val", val_safe), ("np_val", np_val_safe),
                                        ("hard_holdout", hard_holdout))}
    train_hashsets = {name: {image_hash(r["path"]) for r in rows}
                      for name, rows in (("safe_train", train_safe), ("np_train", np_train_safe),
                                         ("hard_train", hard_train))}
    eval_hashsets = {name: {image_hash(r["path"]) for r in rows}
                     for name, rows in (("safe_val", val_safe), ("np_val", np_val_safe),
                                        ("hard_holdout", hard_holdout))}
    train_paths_all = set().union(*train_pathsets.values())
    eval_paths_all = set().union(*eval_pathsets.values())
    train_hashes_all = set().union(*train_hashsets.values())
    eval_hashes_all = set().union(*eval_hashsets.values())
    path_intersections = {f"{left}__{right}": len(train_pathsets[left] & eval_pathsets[right])
                          for left in train_pathsets for right in eval_pathsets}
    byte_intersections = {f"{left}__{right}": len(train_hashsets[left] & eval_hashsets[right])
                          for left in train_hashsets for right in eval_hashsets}
    if (train_paths_all & eval_paths_all or train_hashes_all & eval_hashes_all
            or (train_hashes_all | eval_hashes_all) & excluded_hashes):
        raise ValueError("final train/eval partitions overlap by path, content, or exclusion manifest")
    safe_hashes = {}
    for name in ("safe_train", "safe_val", "np_train_safe", "np_val_safe"):
        safe_hashes[f"{name}.csv"] = _sha256_file(output_dir / f"{name}.csv")
    safe_count_audit = {"safe_train": {"source_count": len(train_source_rows), "selected": len(train_safe),
                                       "removed": train_removed},
                        "safe_val": {"source_count": len(source_rows['safe_val'][1]), "selected": len(val_safe),
                                     "removed": val_removed},
                        "np_train": {"source_count": len(source_rows['np_train'][1]), "selected": len(np_train_safe),
                                     "removed": np_train_removed},
                        "np_val": {"source_count": len(source_rows['np_val'][1]), "selected": len(np_val_safe),
                                   "removed": np_val_removed}, "csv_sha256": safe_hashes}
    np_audit = {"sampling": {"source_partition": "authorized CCPD-NP train/val from batch 7",
                              "n_train": len(np_train_safe), "n_val": len(np_val_safe)},
                "all_selected_decoded_valid": inherited_np_decode_verified,
                "decode_verification": "inherited: selected rows are subsets of the exact batch-7 NP CSVs whose audit confirms all selected images decoded; this batch filtered memberships and did not decode them again",
                "inherited_decode_audit": {"path": str(source_np_audit_path.resolve()),
                                           "sha256": _sha256_file(source_np_audit_path),
                                           "train_source_csv_sha256": source_np_train_sha,
                                           "val_source_csv_sha256": source_np_val_sha,
                                           "audit_train_csv_sha256": source_np_audit["csv"]["np_train.csv_sha256"],
                                           "audit_val_csv_sha256": source_np_audit["csv"]["np_val.csv_sha256"]},
                "csv": {"np_train.csv_sha256": safe_hashes["np_train_safe.csv"],
                        "np_val.csv_sha256": safe_hashes["np_val_safe.csv"]},
                "manifest_sha256": {
                    name: item["manifest_sha256"] for name, item in source_audit.items()},
                "disjointness": {"np_train_val_intersection": 0,
                                 "np_train_excluded_content_intersection": 0,
                                 "np_val_excluded_content_intersection": 0}}
    safe_count_audit["np_safe_partition"] = np_audit
    # The v1 checkpoint used the selected-train file and the explicit NP pool.
    # Record their intersection with the new four-list exclusion hashes so the
    # inherited initialization's data lineage is visible to reviewers.
    selected_v1_path = Path(v1_training_csv or
                            WORKER_ROOT / "algorithm/out/seventh/runs/main/selected_train.csv")
    v1_rows = read_csv_rows(selected_v1_path)[1] if selected_v1_path.is_file() else []
    v1_hashes: set[str] = set()
    v1_excluded = Counter()
    for row in v1_rows:
        key = _normalized_path(row["path"])
        digest = image_hash(row["path"])
        v1_hashes.add(digest)
        if key in excluded_paths:
            v1_excluded["listed_path"] += 1
        if digest in excluded_hashes:
            v1_excluded["blocked_content"] += 1
    safe_count_audit["corner_net_v1_selected_training_set"] = {
        "source_path": str(selected_v1_path.resolve()), "count": len(v1_rows),
        "unique_content_sha256": len(v1_hashes), "excluded_matches_by_path_or_bytes": dict(v1_excluded),
        "selected_csv_sha256": _sha256_file(selected_v1_path) if selected_v1_path.is_file() else None}
    manifest_sha256 = {name: item["manifest_sha256"] for name, item in source_audit.items()}
    audit = {"seed": seed, "holdout_per_subset_max": holdout_per_subset,
             "index_path": str(index_path.resolve()), "index_sha256": _sha256_file(index_path),
             "exclusion_manifests": source_audit, "excluded_unique_sha256": len(excluded_hashes),
             "manifest_sha256": manifest_sha256,
             "official_train": {"source": "safe_train.csv (seed-42 batch-7 30000 sample, byte-filtered)",
                                "safe_count": len(train_safe)},
             "hard_pool_by_subset": pool_counts,
             "hard_pool_excluded_by_subset": {name: dict(counts) for name, counts in dropped_by_subset.items()},
             "hard_train_count": len(hard_train), "hard_tuning_holdout_count": len(hard_holdout),
             "disjointness": {"holdout_train_path_intersection": len({_normalized_path(r['path']) for r in hard_train}
                                                                        & {_normalized_path(r['path']) for r in hard_holdout}),
                              "selected_forbidden_sha_intersection": len(member_hashes & excluded_hashes),
                              "training_vs_tuning_path_intersection": len(train_paths_all & eval_paths_all),
                              "training_vs_tuning_content_intersection": len(train_hashes_all & eval_hashes_all),
                              "training_vs_exclusion_content_intersection": len(train_hashes_all & excluded_hashes),
                              "tuning_vs_exclusion_content_intersection": len(eval_hashes_all & excluded_hashes),
                              "all_train_eval_path_intersections": path_intersections,
                              "all_train_eval_content_intersections": byte_intersections},
             "train_csv_sha256": _sha256_file(output_dir / "hard_pool_train.csv"),
             "holdout_csv_sha256": _sha256_file(output_dir / "hard_tuning_holdout.csv"),
             "official_train_csv_sha256": safe_hashes["safe_train.csv"]}
    audit["safe_partitions"] = safe_count_audit
    audit["csv_sha256"] = {"train": audit["train_csv_sha256"], "holdout": audit["holdout_csv_sha256"],
                           **safe_hashes}
    audit["content_hash_exclusion_verified"] = (
        audit["disjointness"]["selected_forbidden_sha_intersection"] == 0
        and audit["disjointness"]["training_vs_tuning_content_intersection"] == 0
        and audit["disjointness"]["training_vs_exclusion_content_intersection"] == 0
        and audit["disjointness"]["tuning_vs_exclusion_content_intersection"] == 0)
    audit_path = output_dir / "hard_pool_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")


def _plate_text(rng: random.Random, layout: int) -> str:
    province = rng.choice(PROVINCES)
    letter = rng.choice(LETTERS)
    if layout == 7:
        tail = "".join(rng.choice(LETTERS + list("0123456789")) for _ in range(5))
        return province + letter + tail
    # CCPD2020 small new-energy layout: PROV, LETTER, D/F, ALNUM, DIGIT x 4.
    return province + letter + rng.choice("DF") + rng.choice(LETTERS + list("0123456789")) + "".join(
        rng.choice("0123456789") for _ in range(4)
    )


def _ccpd_indices(text: str) -> str:
    tables = [PROVINCES_CCPD, ALPHABETS] + [ADS] * (len(text) - 2)
    return "_".join(str(table.index(ch)) for table, ch in zip(tables, text, strict=False))


@lru_cache(maxsize=1)
def _province_font():
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"approved Chinese font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), 100)


def _draw_plate(text: str, green: bool) -> np.ndarray:
    """Render one compact plate; Pillow supplies the Chinese province glyph."""
    plate = draw_plate(text, "green" if green else "blue")
    foreground = (12, 12, 12) if green else (245, 245, 245)
    centers = CENTERS8 if len(text) == 8 else CENTERS7
    rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = _province_font()
    center = round(float(centers[0]))
    # Fit the province into its physical slot; font point size is not glyph width.
    mask = Image.new("L", (140, 140), 0)
    painter = ImageDraw.Draw(mask)
    bbox = painter.textbbox((0, 0), text[0], font=font)
    painter.text((-bbox[0], -bbox[1]), text[0], fill=255, font=font)
    mask = mask.crop(mask.getbbox()).resize((43, 88), Image.Resampling.LANCZOS)
    background = tuple(reversed(plate[70, 65].tolist()))
    draw.rectangle((center - 24, 20, center + 24, 120), fill=background)
    canvas.paste(tuple(reversed(foreground)), (center - 21, 26), mask)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _make_sample(text: str | None, subset: str, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (92, 98, 104), dtype=np.uint8)
    # A simple vehicle-like shape gives the crop context and preserves the portrait frame.
    cv2.rectangle(image, (105, 230), (615, 920), (61, 72, 82), -1)
    cv2.rectangle(image, (180, 300), (540, 470), (40, 47, 54), -1)
    cv2.circle(image, (165, 870), 58, (20, 20, 20), -1)
    cv2.circle(image, (555, 870), 58, (20, 20, 20), -1)
    if text is None:
        quad = [(0, 0)] * 4
        tilt = 0.0
    else:
        plate = _draw_plate(text, len(text) == 8)
        ph, pw = plate.shape[:2]
        center = np.array([360.0, 650.0])
        tilt = float(rng.uniform(-24, 24)) if subset == "ccpd_tilt" else float(rng.uniform(-5, 5))
        angle = np.deg2rad(tilt)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        half = np.array([[ -pw/2, -ph/2], [pw/2, -ph/2], [pw/2, ph/2], [-pw/2, ph/2]])
        perspective = (0.12, 0.20) if subset == "ccpd_tilt" else (0.0, 0.05)
        # A tapered top edge adds pronounced perspective to the tilt subset.
        half[0, 0] += pw * rng.uniform(*perspective)
        half[1, 0] -= pw * rng.uniform(*perspective)
        half[2, 0] -= pw * rng.uniform(*perspective)
        half[3, 0] += pw * rng.uniform(*perspective)
        half[0, 1] -= ph * rng.uniform(*perspective)
        half[1, 1] += ph * rng.uniform(*perspective)
        half[2, 1] += ph * rng.uniform(*perspective)
        half[3, 1] -= ph * rng.uniform(*perspective)
        points = np.rint(half @ rot.T + center).astype(np.float32)
        quad = [(int(x), int(y)) for x, y in points]
        src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
        matrix = cv2.getPerspectiveTransform(src, points)
        warped = cv2.warpPerspective(plate, matrix, IMAGE_SIZE)
        mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, IMAGE_SIZE)
        image[mask > 0] = warped[mask > 0]
        if subset == "ccpd_db":
            image = np.clip(image.astype(np.float32) * 0.38, 0, 255).astype(np.uint8)
        elif subset == "ccpd_blur":
            image = cv2.GaussianBlur(image, (13, 13), 3.0)
    noise = rng.normal(0, 2.5, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), quad, tilt


def _filename(text: str | None, quad: list[tuple[int, int]], image: np.ndarray,
              index: int, rng: np.random.Generator, tilt_deg: float) -> tuple[str, dict]:
    if text is None:
        return f"noplate_{index:05d}.jpg", {"area_ratio": 0.0, "tilt_degrees": 0.0,
                                              "brightness_method": "not_applicable", "blurriness_method": "not_applicable"}
    # Filename vertex order is RB, LB, LT, RT; parse_ccpd_name reorders to LT,RT,RB,LB.
    lt, rt, rb, lb = quad
    xs, ys = zip(*quad, strict=False)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    area_ratio = cv2.contourArea(np.asarray(quad, dtype=np.float32)) / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
    area = max(1, round(area_ratio * 1000))  # custom synthetic area token: projected polygon fraction × 1000
    labels = _ccpd_indices(text)
    crop = image[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = int(round(float(np.mean(gray))))
    blurriness = min(255, int(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 100)))
    # Angles are measured from the integer quad; the 90-degree horizontal
    # offset keeps all tokens unsigned, as required by the existing decoder.
    tilt_h = round(90 + np.degrees(np.arctan2(rt[1]-lt[1], rt[0]-lt[0])))
    tilt_v = round(np.degrees(np.arctan2(lb[1]-lt[1], lb[0]-lt[0])))
    return (f"{area:03d}-{tilt_h}_{tilt_v}-"
            f"{x1}&{y1}_{x2}&{y2}-{rb[0]}&{rb[1]}_{lb[0]}&{lb[1]}_"
            f"{lt[0]}&{lt[1]}_{rt[0]}&{rt[1]}-{labels}-{brightness}-{blurriness}.jpg",
            {"area_ratio": float(area_ratio), "tilt_degrees": abs(float(tilt_deg)),
             "area_token": area, "bbox": [x1, y1, x2-x1, y2-y1],
             "tilt_h": tilt_h, "tilt_v": tilt_v, "brightness": brightness, "blurriness": blurriness,
             "area_token_formula": "round(projected quadrilateral area / image area * 1000)",
             "tilt_token_formula": "horizontal=90+top-edge angle, vertical=left-edge angle; degrees from integer quad; official orientation convention pending real-data verification",
             "brightness_method": "mean grayscale of axis-aligned plate bounding crop",
             "blurriness_method": "min(255, round(variance of Laplacian on bounding crop / 100))"})


def _clear_previous_output(out: Path) -> None:
    """Allow repeat runs only when the directory is wholly owned by this tool."""
    if not out.exists() or not any(out.iterdir()):
        return
    manifest = out / "truth.json"
    if not manifest.is_file():
        raise ValueError(f"refusing to overwrite nonempty output without truth.json: {out}")
    try:
        old = json.loads(manifest.read_text(encoding="utf-8"))
        owned = {Path(row["path"]).resolve() for row in old}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot validate existing generated output: {manifest}") from exc
    root = out.resolve()
    if any(path == root or root not in path.parents for path in owned):
        raise ValueError("truth.json contains paths outside its output directory")
    known_split_names = {"train.txt", "val.txt", "test.txt", *(f"{name}.txt" for name in SUBSETS_2019)}
    owned_files = set(owned)
    owned_files.update(p.resolve() for p in (out / "ccpd2019" / "splits").glob("*.txt")
                       if p.name in known_split_names)
    owned_files.add(manifest.resolve())
    if any(path == root or not path.is_relative_to(root) for path in owned_files):
        raise ValueError("managed output resolves outside its output directory")
    actual_files = {p.resolve() for p in out.rglob("*") if p.is_file()}
    if not actual_files.issubset(owned_files):
        extras = sorted(str(p) for p in actual_files - owned_files)
        raise ValueError(f"refusing to overwrite untracked files under output: {extras[:3]}")
    for path in owned_files & actual_files:
        path.unlink()


def generate(out: str | Path, per_subset: int = 50, seed: int = 42) -> list[dict]:
    """Write 5 CCPD2019-like subsets and 3 CCPD2020 splits.

    Returns independent truth rows with absolute image path, encoded filename,
    plate number, quad (LT,RT,RB,LB), dataset/subset/split, and plate flag.
    """
    if not isinstance(per_subset, int) or per_subset < 0 or not isinstance(seed, int) or seed < 0:
        raise ValueError("per_subset and seed must be nonnegative integers")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _clear_previous_output(out)
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)
    truth: list[dict] = []
    split_rows: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    subset_rows: dict[str, list[str]] = {name: [] for name in SUBSETS_2019}
    for subset in SUBSETS_2019:
        for i in range(per_subset):
            has_plate = subset != "ccpd_np"
            text = _plate_text(py_rng, 7) if has_plate else None
            image, quad, tilt = _make_sample(text, subset, rng)
            filename, measured = _filename(text, quad, image, i, rng, tilt)
            relative = Path(subset) / filename
            target = out / "ccpd2019" / relative
            imwrite(target, image)
            rel_posix = relative.as_posix()
            if subset == "ccpd_base":
                split = ("train", "val", "test")[i % 3]
                split_rows[split].append(rel_posix)
            else:
                split = "test"
                subset_rows[subset].append(rel_posix)
            truth.append({"path": str(target.resolve()), "file": filename, "plate_no": text or "",
                          "plate_text": text or "", "quad": [tuple(p) for p in quad], "dataset": "ccpd2019",
                          "subset": subset, "split": split, "has_plate": has_plate, "layout": len(text) if text else 0,
                          "metrics": measured})
    split_dir = out / "ccpd2019" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in split_rows.items():
        (split_dir / f"{name}.txt").write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    for subset, paths in subset_rows.items():
        (split_dir / f"{subset}.txt").write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")

    for split in SPLITS_2020:
        for i in range(per_subset):
            text = _plate_text(py_rng, 8)
            image, quad, tilt = _make_sample(text, "ccpd_base", rng)
            filename, measured = _filename(text, quad, image, i, rng, tilt)
            target = out / "ccpd2020" / "ccpd_green" / split / filename
            imwrite(target, image)
            truth.append({"path": str(target.resolve()), "file": filename, "plate_no": text,
                          "plate_text": text, "quad": [tuple(p) for p in quad], "dataset": "ccpd2020",
                          "subset": "ccpd_green", "split": split, "has_plate": True, "layout": 8,
                          "metrics": measured})

    truth_path = out / "truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="algorithm/data/raw_mock")
    parser.add_argument("--prepare-hard-pool", action="store_true",
                        help="prepare byte-safe eighth-batch train/holdout partitions")
    parser.add_argument("--hard-output", default="algorithm/out/eighth/data")
    parser.add_argument("--holdout-per-subset", type=int, default=300)
    parser.add_argument("--per-subset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.prepare_hard_pool:
            audit = prepare_hard_pool(output_dir=args.hard_output, seed=args.seed,
                                      holdout_per_subset=args.holdout_per_subset)
            print(json.dumps({"hard_pool_by_subset": audit["hard_pool_by_subset"],
                              "safe_partitions": audit["safe_partitions"],
                              "audit": str(Path(args.hard_output) / "hard_pool_audit.json")},
                             ensure_ascii=False, indent=2))
            return 0
        rows = generate(args.out, args.per_subset, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(rows)} CCPD-shaped samples under {Path(args.out)}; seed={args.seed}; per_subset={args.per_subset}")
    print(f"Truth metadata: {Path(args.out) / 'truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
