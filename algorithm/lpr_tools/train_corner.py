"""从零训练 CornerNet；数据仅取安全 train/val CSV，CPU线程受限。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lpr.corner_net import (  # noqa: E402
    ARCHITECTURE, INPUT_SIZE, OUTPUT_STRIDE, CornerModel, letterbox_image,
    CornerNet, RoiConfidenceModel, RoiCornerRefiner, pack_corner_roi_checkpoint,
)
from lpr.io_utils import imread  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "out" / "seventh" / "data"
OUT_DIR = HERE / "out" / "seventh"
DATA_ROOT = Path("D:/keshe2/algorithm/data")
INDEX_CSV = DATA_ROOT / "index" / "ccpd_index.csv"
FROZEN_MANIFEST = DATA_ROOT / "index" / "testset_manifest_v1.csv"
UNDECODABLE_MANIFEST = DATA_ROOT / "index" / "undecodable.csv"
GOLDEN_MANIFEST = HERE / "tests" / "data" / "golden" / "golden_manifest.csv"
TRAIN_CSV = DATA_DIR / "train_sampled_30000.csv"
VAL_CSV = DATA_DIR / "val_1000.csv"
SEED = 42


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/")


def _manifest_paths(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"required exclusion manifest missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"manifest has no path column: {path}")
        return {_path_key(row["path"]) for row in reader if row.get("path")}


def _path_set_sha256(paths: set[str]) -> str:
    payload = "\n".join(sorted(paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_csv_provenance(train_csv: Path, val_csv: Path,
                          sampling_audit: dict[str, Any]) -> dict[str, Any]:
    train_csv, val_csv = Path(train_csv), Path(val_csv)
    train_canonical = _path_key(str(train_csv)) == _path_key(str(TRAIN_CSV))
    val_canonical = _path_key(str(val_csv)) == _path_key(str(VAL_CSV))
    train_hash = hashlib.sha256(train_csv.read_bytes()).hexdigest()
    val_hash = hashlib.sha256(val_csv.read_bytes()).hexdigest()
    train_expected = sampling_audit["train_sample"]["csv_sha256"] if train_canonical else None
    val_expected = sampling_audit["validation"]["csv_sha256"] if val_canonical else None
    if train_canonical and train_hash != train_expected:
        raise ValueError("canonical sampled train CSV hash differs from sampling audit")
    if val_canonical and val_hash != val_expected:
        raise ValueError("canonical validation CSV hash differs from sampling audit")
    return {"train_csv_path": str(train_csv.resolve()), "train_sha256": train_hash,
            "train_csv_is_canonical": train_canonical, "train_sampling_audit_hash_match":
                train_hash == train_expected if train_expected is not None else None,
            "val_csv_path": str(val_csv.resolve()), "val_sha256": val_hash,
            "val_csv_is_canonical": val_canonical, "val_sampling_audit_hash_match":
                val_hash == val_expected if val_expected is not None else None}


def _verify_index_split_membership(train_rows: list[dict[str, str]],
                                   val_rows: list[dict[str, str]],
                                   np_train: list[dict[str, str]],
                                   np_val: list[dict[str, str]],
                                   *, index_path: Path | None = None,
                                   authorized_np_train: set[str] | None = None,
                                   authorized_np_val: set[str] | None = None) -> dict[str, Any]:
    """Verify declared split/labels against source index; NP rows must use the audited override."""
    index_path = index_path or DATA_ROOT / "index" / "ccpd_index.csv"
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    by_path: dict[str, dict[str, str]] = {}
    for row in index_rows:
        key = _path_key(row["path"])
        old = by_path.get(key)
        if old is not None and (old["split"], old["has_plate"], old["subset"]) != (
                row["split"], row["has_plate"], row["subset"]):
            raise ValueError(f"conflicting source index rows for {row['path']}")
        by_path[key] = row

    def check(rows: list[dict[str, str]], declared_split: str, source_split: str,
              has_plate: str, require_np_override: bool = False) -> set[str]:
        keys: set[str] = set()
        for row in rows:
            key = _path_key(row["path"])
            source = by_path.get(key)
            if source is None:
                raise ValueError(f"input path is absent from shared ccpd_index.csv: {row['path']}")
            if row["split"] != declared_split or source["split"] != source_split:
                raise ValueError(f"declared/source split mismatch for {row['path']}: "
                                 f"declared={row['split']} source={source['split']}")
            if row["has_plate"] != has_plate or source["has_plate"] != has_plate:
                raise ValueError(f"declared/source label mismatch for {row['path']}")
            if row["subset"] != source["subset"]:
                raise ValueError(f"declared/source subset mismatch for {row['path']}")
            if require_np_override and (source["subset"] != "ccpd_np" or row.get("source_split") != "test"):
                raise ValueError(f"invalid no-plate exception provenance for {row['path']}")
            keys.add(key)
        return keys

    train_keys = check(train_rows, "train", "train", "1")
    val_keys = check(val_rows, "val", "val", "1")
    train_np_keys = check(np_train, "train", "test", "0", require_np_override=True)
    val_np_keys = check(np_val, "val", "test", "0", require_np_override=True)
    if authorized_np_train is None and np_train:
        authorized_np_train = _manifest_paths(DATA_DIR / "np_train.csv")
    if authorized_np_val is None and np_val:
        authorized_np_val = _manifest_paths(DATA_DIR / "np_val.csv")
    if (authorized_np_train is not None and not train_np_keys.issubset(authorized_np_train)):
        raise ValueError("NP train input includes paths outside the audited safe NP train partition")
    if authorized_np_val is not None and not val_np_keys.issubset(authorized_np_val):
        raise ValueError("NP validation input includes paths outside the audited safe NP validation partition")
    return {"index_path": str(index_path.resolve()),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "train_positive_membership_sha256": _path_set_sha256(train_keys),
            "val_positive_membership_sha256": _path_set_sha256(val_keys),
            "np_train_membership_sha256": _path_set_sha256(train_np_keys),
            "np_val_membership_sha256": _path_set_sha256(val_np_keys),
            "source_membership_checked": True,
            "np_safe_partition_checked": True}


def _csv_bytes(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _official_val_path_set(index_rows: list[dict[str, str]], excluded: set[str],
                           n: int = 1000, seed: int = SEED) -> tuple[set[str], dict[str, Any]]:
    """Mirror tracked index_samples.sample_validation without reparsing the index."""
    if n < 1:
        raise ValueError("validation sample size must be positive")
    category_by_subset = {
        "ccpd_base": "clear", "ccpd_green": "clear", "ccpd_fn": "clear",
        "ccpd_db": "night", "ccpd_tilt": "tilt", "ccpd_rotate": "tilt",
        "ccpd_blur": "dirty", "ccpd_weather": "dirty", "ccpd_challenge": "dirty",
        "ccpd_np": "np",
    }
    pools: dict[str, list[dict[str, str]]] = {}
    for row in index_rows:
        key = _path_key(row["path"])
        if row["split"] != "val" or key in excluded:
            continue
        no_plate = row["subset"] == "ccpd_np" and row["has_plate"] == "0"
        if no_plate:
            if row["skip_reason"] not in {"", "no_plate"}:
                continue
        elif row["has_plate"] != "1" or row["skip_reason"]:
            continue
        if row["subset"] not in category_by_subset:
            raise ValueError(f"unsupported index subset: {row['subset']}")
        pools.setdefault(f"{row['dataset']}/{row['subset']}", []).append(row)
    if not pools:
        raise ValueError("ccpd_index contains no supported val samples")
    rng = random.Random(seed)
    order = sorted(pools)
    for key in order:
        pools[key].sort(key=lambda row: _path_key(row["path"]))
        rng.shuffle(pools[key])
    rng.shuffle(order)
    selected: list[dict[str, str]] = []
    offset = 0
    while len(selected) < min(n, sum(map(len, pools.values()))):
        for key in order:
            if offset < len(pools[key]):
                selected.append(pools[key][offset])
                if len(selected) == n:
                    break
        offset += 1
    return ({_path_key(row["path"]) for row in selected},
            {"split": "val", "seed": seed, "requested_n": n, "selected_n": len(selected),
             "available_n": sum(map(len, pools.values())),
             "allocation": "balanced round-robin by dataset/subset; no replacement",
             "available_by_subset": {key: len(pools[key]) for key in sorted(pools)},
             "selected_by_subset": dict(sorted(Counter(
                 f"{row['dataset']}/{row['subset']}" for row in selected).items())),
             "paths": [row["path"] for row in selected]})


def _select_training_positives(index_rows: list[dict[str, str]], excluded: set[str],
                               *, train_n: int, seed: int = SEED) -> list[dict[str, str]]:
    eligible = [row for row in index_rows if row["split"] == "train" and row["has_plate"] == "1"
                and not row["skip_reason"] and row["subset"] in {"ccpd_base", "ccpd_green"}
                and _path_key(row["path"]) not in excluded]
    green = [row for row in eligible if row["subset"] == "ccpd_green"]
    base = [row for row in eligible if row["subset"] == "ccpd_base"]
    if train_n < 1 or train_n < len(green) or len(base) < train_n - len(green):
        raise ValueError("insufficient train pool for requested positive sample size")
    rng = random.Random(seed)
    selected = rng.sample(base, train_n - len(green)) + green
    rng.shuffle(selected)
    return selected


def _split_np_rows(index_rows: list[dict[str, str]], excluded: set[str],
                   *, train_fraction: float = .8, seed: int = SEED
                   ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("NP train fraction must be between zero and one")
    pool = [row for row in index_rows if row["subset"] == "ccpd_np" and row["split"] == "test"
            and row["has_plate"] == "0" and row["skip_reason"] == "no_plate"
            and _path_key(row["path"]) not in excluded]
    pool.sort(key=lambda row: _path_key(row["path"]))
    random.Random(seed).shuffle(pool)
    cut = int(train_fraction * len(pool))
    return pool[:cut], pool[cut:]


def prepare_corner_data(*, output_dir: Path = DATA_DIR, train_n: int = 30_000,
                        val_n: int = 1000, verify_np_images: bool = False) -> dict[str, Any]:
    """Prepare reproducible training CSVs; only NP is decoded when explicitly requested."""
    inputs = (INDEX_CSV, FROZEN_MANIFEST, GOLDEN_MANIFEST, UNDECODABLE_MANIFEST)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen, golden = _manifest_paths(FROZEN_MANIFEST), _manifest_paths(GOLDEN_MANIFEST)
    known_bad = _manifest_paths(UNDECODABLE_MANIFEST)
    blocked = frozen | golden | known_bad
    with INDEX_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"path", "dataset", "subset", "split", "has_plate", "plate_no", "layout",
                    "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
                    "blurriness", "area", "skip_reason"}
        if not required.issubset(fields):
            raise ValueError("ccpd_index is missing required metadata columns")
        index_rows = list(reader)
    by_key = {_path_key(row["path"]): row for row in index_rows}
    if len(by_key) != len(index_rows):
        raise ValueError("duplicate normalized path in ccpd_index")

    eligible = [row for row in index_rows if row["split"] == "train" and row["has_plate"] == "1"
                and not row["skip_reason"] and row["subset"] in {"ccpd_base", "ccpd_green"}
                and _path_key(row["path"]) not in blocked]
    train_rows = _select_training_positives(index_rows, blocked, train_n=train_n)
    train_keys = {_path_key(row["path"]) for row in train_rows}

    val_keys, val_meta = _official_val_path_set(index_rows, blocked, n=val_n, seed=SEED)
    if len(val_keys) != val_n or not val_keys.issubset(by_key):
        raise ValueError("official validation sampler did not return the requested indexed paths")
    val_rows = [row for row in index_rows if _path_key(row["path"]) in val_keys]
    if len(val_rows) != val_n:
        raise ValueError("validation index-order projection returned an unexpected row count")

    np_train_source, np_val_source = _split_np_rows(index_rows, blocked)
    np_cut = len(np_train_source)
    np_eligible_n = len(np_train_source) + len(np_val_source)
    np_fields = ["path", "dataset", "subset", "split", "source_split", "has_plate", "plate_no",
                 "layout", "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
                 "blurriness", "area", "skip_reason"]
    def np_rows(source: list[dict[str, str]], split: str) -> list[dict[str, str]]:
        result = []
        for original in source:
            row = {key: original.get(key, "") for key in np_fields}
            row["source_split"], row["split"] = original["split"], split
            result.append(row)
        return result
    np_train, np_val = np_rows(np_train_source, "train"), np_rows(np_val_source, "val")
    np_train_keys = {_path_key(row["path"]) for row in np_train}
    np_val_keys = {_path_key(row["path"]) for row in np_val}
    sets = (train_keys, val_keys, np_train_keys, np_val_keys)
    if (train_keys & val_keys or train_keys & np_train_keys or train_keys & np_val_keys
            or val_keys & np_train_keys or val_keys & np_val_keys or np_train_keys & np_val_keys
            or any(paths & blocked for paths in sets)):
        raise ValueError("train/validation paths intersect each other or exclusion manifests")
    if verify_np_images:
        failures = []
        for row in np_train + np_val:
            try:
                imread(row["path"])
            except (OSError, ValueError, cv2.error) as exc:
                failures.append(f"{row['path']}: {exc}")
        if failures:
            raise ValueError("NP image decode validation failed: " + "; ".join(failures[:10]))

    outputs = {
        f"train_sampled_{train_n}.csv": _csv_bytes(train_rows, fields),
        f"val_{val_n}.csv": _csv_bytes(val_rows, fields),
        "np_train.csv": _csv_bytes(np_train, np_fields),
        "np_val.csv": _csv_bytes(np_val, np_fields),
    }
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()}
    disjointness = {"train_val_intersection": len(train_keys & val_keys),
                    "train_frozen_intersection": len(train_keys & frozen),
                    "val_frozen_intersection": len(val_keys & frozen),
                    "train_golden_intersection": len(train_keys & golden),
                    "val_golden_intersection": len(val_keys & golden),
                    "train_known_undecodable_intersection": len(train_keys & known_bad),
                    "val_known_undecodable_intersection": len(val_keys & known_bad),
                    "np_train_val_intersection": len(np_train_keys & np_val_keys),
                    "np_train_frozen_intersection": len(np_train_keys & frozen),
                    "np_val_frozen_intersection": len(np_val_keys & frozen),
                    "np_train_golden_intersection": len(np_train_keys & golden),
                    "np_val_golden_intersection": len(np_val_keys & golden),
                    "np_train_known_undecodable_intersection": len(np_train_keys & known_bad),
                    "np_val_known_undecodable_intersection": len(np_val_keys & known_bad)}
    if any(disjointness.values()):
        raise ValueError(f"split audit failed: {disjointness}")
    np_audit = {"index_path": str(INDEX_CSV.resolve()), "index_sha256": hashlib.sha256(INDEX_CSV.read_bytes()).hexdigest(),
                "frozen_test_manifest": str(FROZEN_MANIFEST.resolve()), "frozen_test_sha256": hashlib.sha256(FROZEN_MANIFEST.read_bytes()).hexdigest(),
                "golden_manifest": str(GOLDEN_MANIFEST.resolve()), "golden_sha256": hashlib.sha256(GOLDEN_MANIFEST.read_bytes()).hexdigest(),
                "undecodable_manifest": str(UNDECODABLE_MANIFEST.resolve()), "undecodable_sha256": hashlib.sha256(UNDECODABLE_MANIFEST.read_bytes()).hexdigest(),
                "sampling": {"seed": SEED, "train_fraction": .8, "train_cut": np_cut,
                             "n_eligible_after_all_exclusions": np_eligible_n, "n_train": len(np_train), "n_val": len(np_val),
                             "source_index_split": "test", "authorized_exception": "Dedicated ccpd_np 80/20 partition authorized for train/val only."},
                "disjointness": disjointness,
                "csv": {"np_train.csv_sha256": hashes["np_train.csv"], "np_val.csv_sha256": hashes["np_val.csv"]},
                "all_selected_decoded_valid": bool(verify_np_images)}
    sampling_audit = {"index_sha256": np_audit["index_sha256"],
                      "train_sample": {"n": len(train_rows), "subset_counts": dict(Counter(r["subset"] for r in train_rows)),
                                       "seed": SEED, "path_intersection_frozen": len(train_keys & frozen),
                                       "csv_sha256": hashes[f"train_sampled_{train_n}.csv"]},
                      "validation": {"n": len(val_rows), "base": sum(r["subset"] == "ccpd_base" for r in val_rows),
                                     "green": sum(r["subset"] == "ccpd_green" for r in val_rows),
                                     "path_intersection_frozen": len(val_keys & frozen),
                                     "csv_sha256": hashes[f"val_{val_n}.csv"], "sampler_metadata": val_meta},
                      "np_train_sha256": hashes["np_train.csv"], "np_val_sha256": hashes["np_val.csv"],
                      "np_images_decoded": len(np_train) + len(np_val) if verify_np_images else 0}
    outputs["np-split-audit.json"] = json.dumps(np_audit, ensure_ascii=False, indent=2).encode("utf-8")
    outputs["sampling-audit.json"] = json.dumps(sampling_audit, ensure_ascii=False, indent=2).encode("utf-8")
    output_dir = Path(output_dir)
    for name, content in outputs.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different prepared data: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = output_dir / name
        if not path.exists():
            path.write_bytes(content)
    return {"output_dir": str(output_dir.resolve()),
            "counts": {"train": len(train_rows), "val": len(val_rows), "np_train": len(np_train), "np_val": len(np_val)},
            "sha256": hashes, "np_images_verified": bool(verify_np_images)}


def read_rows(path: Path, expected_split: str, expected_has_plate: str = "1") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "split", "has_plate", "quad", "skip_reason", "subset"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing/empty dataset fields: {path}")
    for row in rows:
        allowed_skip_reason = "no_plate" if expected_has_plate == "0" else ""
        if row["split"] != expected_split or row["skip_reason"].strip() != allowed_skip_reason:
            raise ValueError(f"unsafe split or malformed row in {path}: {row['path']}")
        if row["has_plate"] != expected_has_plate:
            raise ValueError(f"unexpected plate label in {path}: {row['path']}")
        if expected_has_plate == "0" and (row.get("subset") != "ccpd_np" or row.get("source_split") != "test"):
            raise ValueError(f"unexpected negative sample provenance in {path}: {row['path']}")
        if expected_has_plate == "1" and len(row["quad"].split()) != 8:
            raise ValueError(f"invalid quad in {path}: {row['path']}")
    if len({_path_key(row["path"]) for row in rows}) != len(rows):
        raise ValueError(f"duplicate sample path in {path}")
    return rows


def audit_sources(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]],
                  np_train: list[dict[str, str]] | None = None,
                  np_val: list[dict[str, str]] | None = None,
                  train_csv: Path = TRAIN_CSV, val_csv: Path = VAL_CSV) -> dict[str, Any]:
    train, val = {_path_key(row["path"]) for row in train_rows}, {_path_key(row["path"]) for row in val_rows}
    np_train, np_val = np_train or [], np_val or []
    train_np = {_path_key(row["path"]) for row in np_train}
    val_np = {_path_key(row["path"]) for row in np_val}
    if train & val or train & train_np or train & val_np or val & train_np or val & val_np or train_np & val_np:
        raise ValueError("positive/negative train/validation path leakage")
    audit_path = DATA_DIR / "sampling-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    input_provenance = _input_csv_provenance(train_csv, val_csv, audit)
    if audit["train_sample"]["path_intersection_frozen"] or audit["validation"]["path_intersection_frozen"]:
        raise ValueError("safe sampling audit reports frozen overlap")
    if (np_train or np_val) and not (np_train and np_val):
        raise ValueError("NP negatives require both explicit train and validation lists")
    frozen_path = DATA_ROOT / "index" / "testset_manifest_v1.csv"
    golden_path = HERE / "tests" / "data" / "golden" / "golden_manifest.csv"
    undecodable_path = DATA_ROOT / "index" / "undecodable.csv"
    excluded = _manifest_paths(frozen_path) | _manifest_paths(golden_path) | _manifest_paths(undecodable_path)
    if (train | val | train_np | val_np) & excluded:
        raise ValueError("train/validation rows intersect frozen, golden, or undecodable manifests")
    source_membership = _verify_index_split_membership(train_rows, val_rows, np_train, np_val)
    return {"train_rows": len(train_rows), "val_rows": len(val_rows),
            "np_train_rows": len(np_train), "np_val_rows": len(np_val),
            "train_sha256": input_provenance["train_sha256"],
            "val_sha256": input_provenance["val_sha256"],
            "input_csv_provenance": input_provenance,
            "source_membership": source_membership,
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "frozen_manifest_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "golden_manifest_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
            "undecodable_manifest_sha256": hashlib.sha256(undecodable_path.read_bytes()).hexdigest(),
            "np_pixels_read_during_audit": 0, "hard_dev_images_read": 0,
            "frozen_or_golden_images_read": 0,
            "np_policy": "explicit safe 80/20 NP lists" if np_train else "no NP pixels selected"}


def select_train_subset(rows: list[dict[str, str]], limit: int, seed: int = SEED) -> list[dict[str, str]]:
    """Deterministic proportional subset, preserving every train subset's share."""
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["subset"], []).append(row)
    rng = np.random.default_rng(seed)
    allocation = {key: int(limit * len(group) / len(rows)) for key, group in groups.items()}
    remainder = limit - sum(allocation.values())
    for key in sorted(groups, key=lambda name: (-(limit * len(groups[name]) / len(rows) - allocation[name]), name))[:remainder]:
        allocation[key] += 1
    selected: list[dict[str, str]] = []
    for key, group in groups.items():
        indices = rng.choice(len(group), size=allocation[key], replace=False)
        selected.extend(group[int(index)] for index in indices)
    rng.shuffle(selected)
    return selected


def select_training_positives(rows: list[dict[str, str]], limit: int, seed: int = SEED) -> list[dict[str, str]]:
    """Keep every eligible green example when sampling a positive subset."""
    if limit <= 0 or len(rows) <= limit:
        return rows
    green = [row for row in rows if row["subset"] == "ccpd_green"]
    other = [row for row in rows if row["subset"] != "ccpd_green"]
    if len(green) >= limit:
        return select_train_subset(green, limit, seed)
    selected = green + select_train_subset(other, limit - len(green), seed)
    np.random.default_rng(seed).shuffle(selected)
    return selected


def write_selected_samples(rows: list[dict[str, str]], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _run_step_limit_reached(step: int, run_start_step: int, limit: int) -> bool:
    return limit > 0 and step - run_start_step >= limit


def center_cell(points_norm: np.ndarray, grid_w: int, grid_h: int) -> tuple[int, int]:
    center = np.asarray(points_norm, dtype=np.float32).reshape(4, 2).mean(axis=0)
    ix = int(np.clip(np.floor(center[0] * grid_w), 0, grid_w - 1))
    iy = int(np.clip(np.floor(center[1] * grid_h), 0, grid_h - 1))
    return ix, iy


def encode_relative_target(points_norm: np.ndarray, ix: int, iy: int,
                           grid_w: int, grid_h: int) -> np.ndarray:
    """Model's bounded residual target: .5 + absolute quad - selected cell center."""
    points = np.asarray(points_norm, dtype=np.float32).reshape(4, 2)
    cell = np.tile(np.array([(ix + .5) / grid_w, (iy + .5) / grid_h], np.float32), (4, 1))
    return (points - cell + .5).reshape(8)


def relative_prediction(corner_map: torch.Tensor, ix: int, iy: int,
                        grid_w: int, grid_h: int) -> torch.Tensor:
    pred = corner_map[:, :, iy, ix]
    cell = torch.tensor([(ix + .5) / grid_w, (iy + .5) / grid_h],
                        dtype=pred.dtype, device=pred.device).repeat(4)
    return pred - cell + .5


def gather_quad_loss(corner_map: torch.Tensor, batch_indices: list[int] | torch.Tensor,
                     cell_xs: list[int] | torch.Tensor, cell_ys: list[int] | torch.Tensor,
                     target_abs: np.ndarray | torch.Tensor, grid_w: int, grid_h: int,
                     *, beta: float = .02) -> torch.Tensor | None:
    """Vectorized per-cell relative-coordinate loss; return None for no positives."""
    if len(batch_indices) == 0:
        return None
    device, dtype = corner_map.device, corner_map.dtype
    batch_idx = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
    xs = torch.as_tensor(cell_xs, dtype=torch.long, device=device)
    ys = torch.as_tensor(cell_ys, dtype=torch.long, device=device)
    absolute = torch.as_tensor(target_abs, dtype=dtype, device=device).reshape(-1, 8)
    if not (len(batch_idx) == len(xs) == len(ys) == len(absolute)):
        raise ValueError("quad gather indices and targets must have equal lengths")
    predictions = corner_map.permute(0, 2, 3, 1)[batch_idx, ys, xs]
    centers_xy = torch.stack(((xs.to(dtype) + .5) / grid_w,
                              (ys.to(dtype) + .5) / grid_h), dim=1)
    centers = centers_xy.repeat(1, 4)
    return F.smooth_l1_loss(predictions - centers + .5, absolute - centers + .5, beta=beta)


def _parse_quad(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(item) for item in row["quad"].split()], np.float32).reshape(4, 2)


def _safe_negative_crop(image: np.ndarray, quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """取同一 train 图像中与牌框隔开的随机背景块，作为合成无牌样本。"""
    height, width = image.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    pad = max(2, int(.08 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0-pad), max(0, y0-pad), min(width, x1+pad), min(height, y1+pad)
    candidates = [(0, 0, x0, height), (x1, 0, width, height),
                  (0, 0, width, y0), (0, y1, width, height)]
    candidates = [(a, b, c, d) for a, b, c, d in candidates if c-a >= 24 and d-b >= 24]
    if candidates:
        x_a, y_a, x_b, y_b = candidates[int(rng.integers(len(candidates)))]
        crop_w = int(rng.integers(max(24, min(48, x_b-x_a)), max(25, x_b-x_a+1)))
        crop_h = int(rng.integers(max(24, min(48, y_b-y_a)), max(25, y_b-y_a+1)))
        left = int(rng.integers(x_a, x_b-crop_w+1))
        top = int(rng.integers(y_a, y_b-crop_h+1))
        return image[top:top+crop_h, left:left+crop_w].copy()
    # If the annotated plate leaves no usable border, synthesize a plate-free
    # texture patch from colors sampled from this train image.
    colors = image.reshape(-1, 3)
    palette = colors[rng.integers(0, len(colors), size=8)]
    noise = rng.integers(0, 8, size=(64, 96, 1), dtype=np.uint8)
    texture = palette[rng.integers(0, len(palette), size=(64, 96))].astype(np.int16) + noise.astype(np.int16)
    return np.clip(texture, 0, 255).astype(np.uint8)


def _geometric_augment(image: np.ndarray, points: np.ndarray,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    width, height = INPUT_SIZE
    center = (width / 2., height / 2.)
    strong = rng.random() < .2
    scale = float(rng.uniform(.72, 1.18) if strong else rng.uniform(.82, 1.15))
    angle = float(rng.uniform(-34., 34.) if strong else rng.uniform(-18., 18.))
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    translation = .13 if strong else .08
    affine[:, 2] += (rng.uniform(-translation, translation) * width,
                     rng.uniform(-translation, translation) * height)
    matrix = np.vstack([affine, [0., 0., 1.]])
    if rng.random() < .35:
        src = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
        perspective = .045 if strong else .025
        dst = src + rng.uniform(-perspective, perspective, size=(4, 2)).astype(np.float32) * np.array([width, height], np.float32)
        dst = dst.astype(np.float32, copy=False)
        perspective = cv2.getPerspectiveTransform(src, dst)
        matrix = perspective @ matrix
    out = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped = cv2.perspectiveTransform(points.reshape(1, 4, 2).astype(np.float32), matrix)[0]
    return out, warped


def _appearance_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = image.astype(np.float32)
    if rng.random() < .15:
        img = img * rng.uniform(.38, .68) + rng.uniform(-18., 8.)
    else:
        img = img * rng.uniform(.72, 1.28) + rng.uniform(-24., 24.)
    img = np.clip(img, 0, 255).astype(np.uint8)
    choice = rng.random()
    if choice < .35:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), sigmaX=float(rng.uniform(.3, 1.5)))
    elif choice < .48:
        img = cv2.blur(img, (3, 1))
    return img


def _sample_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))


class CornerDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], *, training: bool, seed: int = SEED,
                 epoch: int = 0, negatives: bool = False) -> None:
        self.rows, self.training, self.seed, self.epoch = rows, training, seed, epoch
        self.negatives = negatives and training
        self.positive_indices = [i for i, row in enumerate(rows) if row["has_plate"] == "1"]

    def __len__(self) -> int:
        return len(self.rows) + (len(self.positive_indices) if self.negatives else 0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        is_synthetic_negative = self.negatives and index >= len(self.rows)
        row_index = self.positive_indices[index - len(self.rows)] if is_synthetic_negative else index
        row = self.rows[row_index]
        rng = _sample_rng(self.seed, self.epoch, index)
        image = imread(row["path"])
        quad = _parse_quad(row) if row["has_plate"] == "1" else None
        has_plate = quad is not None and not is_synthetic_negative
        if is_synthetic_negative:
            assert quad is not None
            image = _safe_negative_crop(image, quad, rng)
            canvas, _ = letterbox_image(image)
            points_norm = np.zeros((4, 2), dtype=np.float32)
        elif has_plate:
            assert quad is not None
            canvas, transform = letterbox_image(image)
            quad[:, 0] = quad[:, 0] * transform.scale_x + transform.pad_left
            quad[:, 1] = quad[:, 1] * transform.scale_y + transform.pad_top
            points_norm = quad / np.array([INPUT_SIZE[0], INPUT_SIZE[1]], np.float32)
        else:
            canvas, _ = letterbox_image(image)
            points_norm = np.zeros((4, 2), dtype=np.float32)
        if self.training and has_plate:
            original_canvas = canvas
            augmented_canvas, augmented = _geometric_augment(canvas, quad.copy(), rng)
            if (np.all(augmented >= 0) and np.all(augmented[:, 0] < INPUT_SIZE[0]) and
                    np.all(augmented[:, 1] < INPUT_SIZE[1]) and cv2.isContourConvex(augmented)):
                canvas = augmented_canvas
                points_norm = augmented / np.array([INPUT_SIZE[0], INPUT_SIZE[1]], np.float32)
            else:
                canvas = original_canvas
            canvas = _appearance_augment(canvas, rng)
        elif self.training:
            canvas = _appearance_augment(canvas, rng)
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.)
        target = torch.from_numpy(points_norm.reshape(8).astype(np.float32))
        meta = {"row_index": row_index, "subset": row["subset"],
                "synthetic_background": bool(is_synthetic_negative)}
        return tensor, target, torch.tensor(float(has_plate), dtype=torch.float32), meta


def _focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    neg_weight = (1 - target).pow(4)
    pos_loss = -(prob.log()) * (1 - prob).pow(2) * positive
    neg_loss = -((1 - prob).log()) * prob.pow(2) * neg_weight * negative
    num_positive = positive.sum().clamp_min(1.)
    return (pos_loss.sum() + neg_loss.sum()) / num_positive


def make_heatmap(points: np.ndarray, grid_w: int, grid_h: int,
                 radius: int = 1, sigma: float = .7) -> tuple[np.ndarray, int, int]:
    ix, iy = center_cell(points, grid_w, grid_h)
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    for y in range(max(0, iy-radius), min(grid_h, iy+radius+1)):
        for x in range(max(0, ix-radius), min(grid_w, ix+radius+1)):
            distance2 = (x-ix)**2 + (y-iy)**2
            heatmap[y, x] = np.exp(-distance2 / (2 * sigma * sigma))
    heatmap[iy, ix] = 1.
    return heatmap, ix, iy


def quad_cell_targets(points_norm: np.ndarray, ix: int, iy: int,
                      grid_w: int, grid_h: int, *, has_plate: bool = True,
                      radius: int = 1) -> list[tuple[int, int, np.ndarray]]:
    """Encode one absolute quad relative to each supervised nearby cell.

    For negatives no coordinate target exists. The returned coordinates retain
    LT,RT,RB,LB order and decode to the same canvas-absolute quad at every cell.
    """
    if not has_plate:
        return []
    targets = []
    for cy in range(max(0, iy-radius), min(grid_h, iy+radius+1)):
        for cx in range(max(0, ix-radius), min(grid_w, ix+radius+1)):
            relative = encode_relative_target(points_norm, cx, cy, grid_w, grid_h)
            if np.all((relative >= 0.) & (relative <= 1.)):
                targets.append((cx, cy, relative))
    return targets


def _checkpoint(path: Path, model: CornerModel, optimizer: torch.optim.Optimizer,
                epoch: int, step: int, config: dict[str, Any], history: list[dict[str, float]]) -> None:
    payload = {"architecture": ARCHITECTURE, "input_size": list(INPUT_SIZE),
               "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
               "epoch": epoch, "step": step, "config": config, "history": history,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(), "saved_at": time.time()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, model: CornerModel,
                     optimizer: torch.optim.Optimizer) -> tuple[int, int, list[dict[str, float]]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("architecture") != ARCHITECTURE or obj.get("input_size") != list(INPUT_SIZE):
        raise ValueError("resume checkpoint architecture/input size mismatch")
    model.load_state_dict(obj["state_dict"])
    optimizer.load_state_dict(obj["optimizer"])
    random.setstate(obj["python_rng"])
    np.random.set_state(obj["numpy_rng"])
    torch.set_rng_state(obj["torch_rng"])
    return int(obj["epoch"]), int(obj["step"]), list(obj.get("history", []))


def validate(model: CornerModel, rows: list[dict[str, str]], batch_size: int = 16) -> dict[str, Any]:
    dataset = CornerDataset(rows, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total, hit50, hit3, top1_detected, error_px, hm_loss = 0, 0, 0, 0, 0., 0.
    positive_scores: list[float] = []
    grid_h, grid_w = INPUT_SIZE[1] // OUTPUT_STRIDE, INPUT_SIZE[0] // OUTPUT_STRIDE
    with torch.inference_mode():
        for images, targets, _, _meta in loader:
            logits, corners = model(images)
            heat_target = torch.zeros_like(logits)
            scores = logits[:, 0].flatten(1)
            order = scores.argsort(dim=1, descending=True)
            for i in range(len(images)):
                gt = targets[i].numpy().reshape(4, 2)
                gt_x, gt_y = center_cell(gt, grid_w, grid_h)
                heat_target[i, 0, gt_y, gt_x] = 1.
                x, y = int(order[i, 0] % grid_w), int(order[i, 0] // grid_w)
                pred = corners[i, :, y, x].numpy().reshape(4, 2)
                peak_score = float(torch.sigmoid(logits[i, 0, y, x]))
                peak_match = _valid_quad(pred) and _quad_iou(pred, gt) >= .5
                positive_scores.append(peak_score if peak_match else 0.)
                top1_detected += int(peak_match)
                corner_errors = np.linalg.norm((pred-gt) * np.array(INPUT_SIZE, np.float32), axis=1)
                if corner_errors.mean() <= 24:
                    hit50 += 1
                top_ok = False
                for loc in order[i, :3]:
                    xx, yy = int(loc % grid_w), int(loc // grid_w)
                    candidate = corners[i, :, yy, xx].numpy().reshape(4, 2)
                    if _valid_quad(candidate) and _quad_iou(candidate, gt) >= .5:
                        top_ok = True
                        break
                hit3 += int(top_ok)
                error_px += float(corner_errors.mean())
                total += 1
            hm_loss += float(_focal_heatmap_loss(logits, heat_target)) * len(images)
    model.train()
    return {"n": total, "top1_corner_le24px": hit50/total,
            "top1_peak_quad_iou50": top1_detected/total, "top3_quad_iou50": hit3/total,
            "mean_corner_error_px": error_px/total, "val_heatmap_loss": hm_loss/total,
            "positive_matched_peak_scores": positive_scores}


def validate_negatives(model: CornerModel, rows: list[dict[str, str]], batch_size: int = 16) -> dict[str, Any]:
    if not rows:
        return {"np_val_n": 0, "np_false_positive_rate_at_0_25": None,
                "np_max_score_mean": None, "negative_max_scores": []}
    loader = DataLoader(CornerDataset(rows, training=False), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for images, _, _, _meta in loader:
            heat, corners = model(images)
            probabilities = torch.sigmoid(heat[:, 0]).cpu().numpy()
            corner_maps = corners.cpu().numpy()
            for probability, corner_map in zip(probabilities, corner_maps):
                local_max = cv2.dilate(probability, np.ones((3, 3), np.uint8))
                ys, xs = np.where(probability >= local_max - 1e-7)
                order = sorted(zip(ys.tolist(), xs.tolist()), key=lambda cell: -float(probability[cell]))
                valid_scores = [float(probability[y, x]) for y, x in order
                                if _valid_quad(corner_map[:, y, x].reshape(4, 2))]
                scores.append(max(valid_scores, default=0.))
    model.train()
    return {"np_val_n": len(scores),
            "np_false_positive_rate_at_0_25": sum(score >= .25 for score in scores) / len(scores),
            "np_max_score_mean": float(np.mean(scores)), "negative_max_scores": scores}


def calibrate_threshold(positive_scores: list[float], negative_scores: list[float]) -> dict[str, float | None]:
    if not positive_scores or not negative_scores:
        return {"calibrated_score_threshold": None, "val_positive_recall_at_threshold": None,
                "val_negative_false_positive_rate_at_threshold": None}
    best_recall, best_threshold = -1., .25
    for threshold in sorted(set([0., 1., *positive_scores, *negative_scores])):
        fp = sum(value >= threshold for value in negative_scores)
        if fp / len(negative_scores) > .01:
            continue
        recall = sum(value >= threshold for value in positive_scores) / len(positive_scores)
        if recall > best_recall or (recall == best_recall and threshold > best_threshold):
            best_recall, best_threshold = recall, threshold
    return {"calibrated_score_threshold": best_threshold,
            "val_positive_recall_at_threshold": sum(value >= best_threshold for value in positive_scores) / len(positive_scores),
            "val_negative_false_positive_rate_at_threshold": sum(value >= best_threshold for value in negative_scores) / len(negative_scores),
            "val_negative_false_positive_rate_cap": .01}


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a * np.array([INPUT_SIZE[0], INPUT_SIZE[1]])).astype(np.float32)
    b = (b * np.array([INPUT_SIZE[0], INPUT_SIZE[1]])).astype(np.float32)
    area_a, area_b = abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b)))
    try:
        intersection, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 0 else 0.


def _valid_quad(points_norm: np.ndarray) -> bool:
    points = np.asarray(points_norm, np.float32).reshape(4, 2) * np.array(INPUT_SIZE, np.float32)
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] >= INPUT_SIZE[0]) or np.any(points[:, 1] >= INPUT_SIZE[1]):
        return False
    if not cv2.isContourConvex(points):
        return False
    area = abs(float(cv2.contourArea(points)))
    if not .00002 * INPUT_SIZE[0] * INPUT_SIZE[1] <= area <= .5 * INPUT_SIZE[0] * INPUT_SIZE[1]:
        return False
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return bool(np.all(edge_lengths >= 1.))


def train(args: argparse.Namespace) -> None:
    threads = min(8, max(1, args.threads))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_rows, val_rows = read_rows(Path(args.train_csv), "train"), read_rows(Path(args.val_csv), "val")
    np_train = read_rows(Path(args.np_train_csv), "train", "0") if args.np_train_csv else []
    np_val = read_rows(Path(args.np_val_csv), "val", "0") if args.np_val_csv else []
    if bool(args.np_train_csv) != bool(args.np_val_csv):
        raise ValueError("specify both --np-train-csv and --np-val-csv")
    audit = audit_sources(train_rows, val_rows, np_train, np_val,
                          Path(args.train_csv), Path(args.val_csv))
    np_audit_path = DATA_DIR / "np-split-audit.json"
    if np_train:
        np_audit = json.loads(np_audit_path.read_text(encoding="utf-8"))
        disjointness = np_audit["disjointness"]
        if (not np_audit["all_selected_decoded_valid"] or
                any(disjointness[key] != 0 for key in disjointness) or
                len(np_train) != np_audit["sampling"]["n_train"] or
                len(np_val) != np_audit["sampling"]["n_val"]):
            raise ValueError("NP split audit failed; refusing to read NP pixels")
        audit["np_split_audit_sha256"] = hashlib.sha256(np_audit_path.read_bytes()).hexdigest()
        audit["np_train_csv_sha256"] = hashlib.sha256(Path(args.np_train_csv).read_bytes()).hexdigest()
        audit["np_val_csv_sha256"] = hashlib.sha256(Path(args.np_val_csv).read_bytes()).hexdigest()
        if (audit["np_train_csv_sha256"] != np_audit["csv"]["np_train.csv_sha256"] or
                audit["np_val_csv_sha256"] != np_audit["csv"]["np_val.csv_sha256"]):
            raise ValueError("NP CSV hash differs from the audited split")
        audit["np_frozen_golden_undecodable_exclusions"] = disjointness
        audit["available_np_train_rows"] = len(np_train)
        audit["available_np_validation_rows"] = len(np_val)
    train_rows = select_training_positives(train_rows, args.limit_train)
    train_rows += np_train
    val_rows += np_val
    val_rows = select_train_subset(val_rows, args.limit_val)
    run_dir = OUT_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_train_csv = run_dir / "selected_train.csv"
    selected_train_sha = write_selected_samples(train_rows, selected_train_csv)
    model = CornerModel()
    with torch.no_grad():
        model.heatmap_head.bias.fill_(-4.6)
    source_model_path = HERE / "lpr" / "corner_net.py"
    source_hashes = {"train_corner_py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "corner_net_py": hashlib.sha256(source_model_path.read_bytes()).hexdigest()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, step, history = 0, 0, []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_DIR / "checkpoints" if args.run_name == "main" else OUT_DIR / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "corner_last.pt"
    resume_source_sha = None
    if args.resume and not last_path.exists():
        raise FileNotFoundError(f"resume requested but no checkpoint exists: {last_path}")
    if args.resume:
        resume_source_sha = hashlib.sha256(last_path.read_bytes()).hexdigest()
        start_epoch, step, history = _load_checkpoint(last_path, model, optimizer)
        if args.run_name == "main" and not (OUT_DIR / "runs" / "main" / "training-audit.json").exists():
            # A diagnostic checkpoint may seed formal training, but its subset metrics
            # must not compete with full-validation checkpoints for best-model selection.
            history = []
    init_path = run_dir / "initial-random-state.pt"
    initial_state_sha = None
    if not args.resume:
        initial_state = {"state_dict": model.state_dict(), "input_size": list(INPUT_SIZE),
                         "architecture": ARCHITECTURE}
        torch.save(initial_state, init_path)
        initial_state_sha = hashlib.sha256(init_path.read_bytes()).hexdigest()
    cfg = {"seed": SEED, "threads": threads, "batch_size": args.batch_size, "lr": args.lr,
           "epochs": args.epochs, "train_csv": str(Path(args.train_csv)), "val_csv": str(Path(args.val_csv)),
           "limit_val": args.limit_val, "heatmap_initial_bias": -4.6,
           "negative_ratio": 1., "coord_weight": args.coord_weight,
           "heatmap_gaussian_radius": 1, "quad_neighbor_supervision_radius": 1,
           "augmentation": "affine+perspective+brightness+blur", "device": "cpu",
           "selected_train_csv": str(selected_train_csv), "selected_train_sha256": selected_train_sha,
           "selected_train_subset_counts": {key: sum(row["subset"] == key for row in train_rows)
                                            for key in sorted({row["subset"] for row in train_rows})},
           "selected_positive_rows": sum(row["has_plate"] == "1" for row in train_rows),
           "selected_np_negative_rows": sum(row["has_plate"] == "0" for row in train_rows),
           "synthetic_background_negative_rows_per_epoch": sum(row["has_plate"] == "1" for row in train_rows),
           "resume_requested": bool(args.resume), "resume_checkpoint_sha256": resume_source_sha,
           "initial_random_state_sha256": initial_state_sha, "source_sha256": source_hashes,
           "command": [sys.executable, *sys.argv]}
    (run_dir / "training-audit.json").write_text(json.dumps({**audit, "config": cfg}, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = run_dir / "training-log.jsonl"
    started = time.time()
    run_start_step = step
    stop_after_steps = False
    for epoch in range(start_epoch, args.epochs):
        dataset = CornerDataset(train_rows, training=True, epoch=epoch, negatives=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(SEED + epoch), drop_last=False)
        model.train()
        running = 0.
        processed_batches = 0
        processed_samples = 0
        consumed_counts = {"positive_examples": 0, "np_negative_examples": 0,
                           "synthetic_background_examples": 0}
        segment_hm = segment_coord = 0.
        for images, targets, labels, meta in loader:
            optimizer.zero_grad(set_to_none=True)
            heat, corner_map = model(images)
            b, _, gh, gw = heat.shape
            hm_targets = torch.zeros_like(heat)
            coord_batch_indices: list[int] = []
            coord_cell_xs: list[int] = []
            coord_cell_ys: list[int] = []
            coord_target_abs: list[np.ndarray] = []
            for i in range(b):
                if labels[i] < .5:
                    continue
                points = targets[i].numpy().reshape(4, 2)
                hm, ix, iy = make_heatmap(points, gw, gh)
                hm_targets[i, 0] = torch.from_numpy(hm)
                for cell_x, cell_y, _target_rel in quad_cell_targets(points, ix, iy, gw, gh,
                                                                    has_plate=True, radius=1):
                    coord_batch_indices.append(i)
                    coord_cell_xs.append(cell_x)
                    coord_cell_ys.append(cell_y)
                    coord_target_abs.append(points.reshape(8))
            loss_hm = _focal_heatmap_loss(heat, hm_targets)
            gathered_loss = gather_quad_loss(corner_map, coord_batch_indices, coord_cell_xs,
                                             coord_cell_ys, np.asarray(coord_target_abs, dtype=np.float32).reshape(-1, 8),
                                             gw, gh)
            loss_coord = gathered_loss if gathered_loss is not None else heat.sum() * 0.
            loss = loss_hm + args.coord_weight * loss_coord
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            running += float(loss.detach())
            step += 1
            processed_batches += 1
            processed_samples += len(images)
            consumed_counts["positive_examples"] += int((labels >= .5).sum())
            consumed_counts["np_negative_examples"] += sum(
                subset == "ccpd_np" and not is_synth
                for subset, is_synth in zip(meta["subset"], meta["synthetic_background"])
            )
            consumed_counts["synthetic_background_examples"] += int(meta["synthetic_background"].sum())
            segment_hm += float(loss_hm.detach())
            segment_coord += float(loss_coord.detach())
            if step % 25 == 0:
                step_record = {"type": "train_step", "epoch_index": epoch, "step": step,
                               "loss_hm_25step_mean": segment_hm / 25,
                               "loss_coord_25step_mean": segment_coord / 25,
                               "loss_total_25step_mean": (segment_hm + args.coord_weight * segment_coord) / 25,
                               "elapsed_sec": time.time() - started}
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                print(json.dumps(step_record), flush=True)
                segment_hm = segment_coord = 0.
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                _checkpoint(last_path, model, optimizer, epoch, step, cfg, history)
            if step % 500 == 0:
                inference = {"state_dict": model.state_dict(), "input_size": list(INPUT_SIZE),
                             "architecture": ARCHITECTURE}
                snapshot = ckpt_dir / f"corner_step_{step}.pt"
                tmp_snapshot = snapshot.with_suffix(snapshot.suffix + ".tmp")
                torch.save(inference, tmp_snapshot)
                tmp_snapshot.replace(snapshot)
                print(json.dumps({"type": "inference_snapshot", "step": step,
                                  "path": str(snapshot)}), flush=True)
            if _run_step_limit_reached(step, run_start_step, args.max_steps_this_run):
                stop_after_steps = True
                break
        positive_val = [row for row in val_rows if row["has_plate"] == "1"]
        negative_val = [row for row in val_rows if row["has_plate"] == "0"]
        val_metrics = validate(model, positive_val, args.batch_size)
        negative_metrics = validate_negatives(model, negative_val, args.batch_size)
        val_metrics.update({key: value for key, value in negative_metrics.items() if key != "negative_max_scores"})
        val_metrics.update(calibrate_threshold(val_metrics.pop("positive_matched_peak_scores"),
                                               negative_metrics["negative_max_scores"]))
        partial_epoch = stop_after_steps
        rec = {"epoch": epoch + (0 if partial_epoch else 1), "epoch_index": epoch,
               "epoch_completed": not partial_epoch, "step": step,
               "steps_this_epoch_run": processed_batches, "samples_seen_this_epoch_run": processed_samples,
               "epoch_fraction": processed_batches / max(1, len(loader)),
               "consumed_examples": consumed_counts,
               "train_loss": running/max(1, processed_batches),
               **val_metrics, "elapsed_sec": time.time()-started}
        history.append(rec)
        audit["last_run_progress"] = {key: rec[key] for key in
                                       ("epoch_index", "epoch_completed", "step", "steps_this_epoch_run",
                                        "samples_seen_this_epoch_run", "epoch_fraction", "consumed_examples")}
        (run_dir / "training-audit.json").write_text(
            json.dumps({**audit, "config": cfg}, ensure_ascii=False, indent=2), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _checkpoint(last_path, model, optimizer, epoch if partial_epoch else epoch+1, step, cfg, history)
        best_metric = max((r["top3_quad_iou50"] for r in history[:-1]), default=-1.)
        if val_metrics["top3_quad_iou50"] >= best_metric:
            inference = {"state_dict": model.state_dict(), "input_size": list(INPUT_SIZE), "architecture": ARCHITECTURE}
            tmp_best = ckpt_dir / "corner_best.pt.tmp"
            torch.save(inference, tmp_best)
            tmp_best.replace(ckpt_dir / "corner_best.pt")
        print(json.dumps(rec), flush=True)
        if stop_after_steps or (args.max_epochs_this_run and (epoch - start_epoch + 1) >= args.max_epochs_this_run):
            break


ROI_WH = (256, 96)
ROI_PAD_PER_SIDE = .25
COARSE_THRESHOLD = .05
COARSE_TOP_K = 2
ROI_POSITIVE_RETRY_LIMIT = 20


class SourceQuadOutOfBounds(ValueError):
    """The indexed quadrilateral itself extends beyond its decoded source image."""


def parse_quad(row: dict[str, str]) -> np.ndarray:
    q = np.asarray([float(x) for x in row["quad"].split()], np.float32).reshape(4, 2)
    if not np.isfinite(q).all():
        raise ValueError(f"invalid quad for {row.get('path')}")
    return q


def parse_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    x, y, w, h = map(float, row["bbox"].split())
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        raise ValueError(f"invalid bbox for {row.get('path')}")
    return x, y, w, h


def _validate_rows(rows: list[dict[str, str]], *, positive: bool,
                   forbidden: set[str], val_keys: set[str]) -> None:
    if not rows:
        raise ValueError("training CSV is empty")
    for row in rows:
        key = _path_key(row["path"])
        if key in forbidden or key in val_keys:
            raise ValueError(f"unsafe or validation-overlapping training path: {row['path']}")
        if row.get("split") != "train":
            raise ValueError(f"ROI training accepts train rows only: {row['path']}")
        if positive:
            if row.get("has_plate") != "1" or row.get("subset") == "ccpd_np":
                raise ValueError(f"expected positive non-NP row: {row['path']}")
            parse_quad(row)
            parse_bbox(row)
        elif not (row.get("subset") == "ccpd_np" and row.get("source_split") == "test"
                  and row.get("has_plate") == "0" and row.get("skip_reason") == "no_plate"):
            raise ValueError(f"NP exception row failed negative contract: {row['path']}")


def _read_safe_pools(train_csv: Path, np_train_csv: Path, val_csv: Path, np_val_csv: Path,
                     limit_train: int, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    positives = read_rows(train_csv, expected_split="train", expected_has_plate="1")
    # Fixed positive pool selection uses the source pilot's seed; stage seeds
    # control crop and augmentation randomness only.
    positives = select_training_positives(positives, limit_train, SEED)
    np_rows = read_rows(np_train_csv, expected_split="train", expected_has_plate="0")
    val_rows = read_rows(val_csv, expected_split="val", expected_has_plate="1")
    np_val_rows = read_rows(np_val_csv, expected_split="val", expected_has_plate="0")
    excluded = (_manifest_paths(FROZEN_MANIFEST)
                | _manifest_paths(GOLDEN_MANIFEST)
                | _manifest_paths(UNDECODABLE_MANIFEST))
    val_keys = {_path_key(row["path"]) for row in val_rows + np_val_rows}
    _validate_rows(positives, positive=True, forbidden=excluded, val_keys=val_keys)
    _validate_rows(np_rows, positive=False, forbidden=excluded, val_keys=val_keys)
    membership_audit = _verify_index_split_membership(positives, val_rows, np_rows, np_val_rows)
    pos_keys = {_path_key(r["path"]) for r in positives}
    np_keys = {_path_key(r["path"]) for r in np_rows}
    if pos_keys & val_keys or np_keys & val_keys:
        raise ValueError("ROI train pool intersects validation rows")
    if pos_keys & np_keys:
        raise ValueError("positive and NP negative pools intersect")
    # Validation is inspected only to prove disjointness; it is not sampled or decoded.
    return positives, np_rows, {"positive_count": len(positives), "np_train_count": len(np_rows),
        "validation_count": len(val_rows), "np_validation_negative_count": len(np_val_rows),
        "positive_val_intersection": 0, "np_val_intersection": 0,
        "excluded_manifest_count": len(excluded), "positive_pool_selection_seed": SEED,
        "stage_random_seed": seed,
        "source_membership_audit": membership_audit}


def _appearance(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.clip(image.astype(np.float32) * rng.uniform(.78, 1.22)
                + rng.uniform(-20., 20.), 0, 255).astype(np.uint8)
    if rng.random() < .28:
        k = int(rng.choice((3, 5)))
        x = cv2.GaussianBlur(x, (k, k), float(rng.uniform(.3, 1.3)))
    return x


def _strong_roi_geometry(image: np.ndarray, target: np.ndarray,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply aug500's train-only 70/20/10 geometry mix to pixels and ordered corners."""
    draw = float(rng.random())
    if draw < .70:
        return image, target, "ordinary"
    mode = "moderate" if draw < .90 else "strong"
    width, height = ROI_WH
    points = np.asarray(target, np.float32).reshape(4, 2) * np.asarray(ROI_WH, np.float32)
    source = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1],
                         [0, height - 1]], np.float32)
    if mode == "moderate":
        angle = float(rng.uniform(-9, 9))
        scale = float(rng.uniform(.94, 1.06))
        translation_x = float(rng.uniform(-.025, .025)) * width
        translation_y = float(rng.uniform(-.025, .025)) * height
        jitter = .025
    else:
        angle = float(rng.uniform(-20, 20))
        scale = float(rng.uniform(.88, 1.12))
        translation_x = float(rng.uniform(-.045, .045)) * width
        translation_y = float(rng.uniform(-.045, .045)) * height
        jitter = .065
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    affine[:, 2] += (translation_x, translation_y)
    matrix = np.vstack((affine, np.asarray([0., 0., 1.], np.float64)))
    destination = source + rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32) * np.asarray(ROI_WH, np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination) @ matrix
    identity = np.eye(3, dtype=np.float64)
    for _attempt in range(4):
        transformed = cv2.perspectiveTransform(points.reshape(1, 4, 2), matrix)[0]
        normalized = transformed / np.asarray(ROI_WH, np.float32)
        if np.all((normalized >= .005) & (normalized <= .995)):
            warped = cv2.warpPerspective(
                image, matrix, ROI_WH, flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            return warped, normalized.reshape(8).astype(np.float32), mode
        matrix = identity + (matrix - identity) * .75
    return image, target, "ordinary_fallback"


def _strong_roi_exposure(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Apply aug500's 35% dark exposure branch or its mild normal branch."""
    work = image.astype(np.float32) / 255.
    if rng.random() < .35:
        gain = float(rng.uniform(.35, .75))
        gamma = float(rng.uniform(.65, 1.25))
        work = np.power(np.clip(work, 0, 1), gamma) * gain
        if rng.random() < .35:
            sigma = float(rng.uniform(.3, 1.0))
            work = cv2.GaussianBlur(work, (3, 3), sigma)
        return np.clip(work * 255, 0, 255).astype(np.uint8), "dark"
    gain = float(rng.uniform(.92, 1.08))
    gamma = float(rng.uniform(.90, 1.10))
    work = np.power(np.clip(work * gain, 0, 1), gamma)
    return np.clip(work * 255, 0, 255).astype(np.uint8), "normal_exposure"


def _load_roi_initial_state(model: RoiConfidenceModel, checkpoint: Path, *,
                            full_model: bool) -> str:
    """Strictly restore either a complete ROI state or its refiner branch."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if full_model:
        model.load_state_dict(state, strict=True)
    else:
        model.refiner.load_state_dict(state, strict=True)
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def _positive_roi(image: np.ndarray, row: dict[str, str], rng: np.random.Generator,
                  geometric: bool) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    quad = parse_quad(row)
    if np.any(quad < 0.) or np.any(quad[:, 0] > w) or np.any(quad[:, 1] > h):
        raise SourceQuadOutOfBounds(
            f"source quad exceeds {w}x{h} image bounds for {row['path']}: "
            f"x=[{quad[:, 0].min():g},{quad[:, 0].max():g}] "
            f"y=[{quad[:, 1].min():g},{quad[:, 1].max():g}]")
    bx, by, bw, bh = parse_bbox(row)
    low, high = quad.min(0), quad.max(0)
    x0b, y0b = min(bx, float(low[0])), min(by, float(low[1]))
    x1b, y1b = max(bx + bw, float(high[0])), max(by + bh, float(high[1]))
    bw, bh = x1b - x0b, y1b - y0b
    pad_l, pad_r = rng.uniform(.10, .38, size=2) * bw
    pad_t, pad_b = rng.uniform(.10, .38, size=2) * bh
    dx, dy = rng.uniform(-.05, .05) * bw, rng.uniform(-.05, .05) * bh
    x0 = max(0, math.floor(x0b - pad_l + dx))
    x1 = min(w, math.ceil(x1b + pad_r + dx))
    y0 = max(0, math.floor(y0b - pad_t + dy))
    y1 = min(h, math.ceil(y1b + pad_b + dy))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("positive ROI crop collapsed")
    canvas, transform = letterbox_image(image[y0:y1, x0:x1], ROI_WH)
    target = quad - np.asarray([x0, y0], np.float32)
    target[:, 0] = target[:, 0] * transform.scale_x + transform.pad_left
    target[:, 1] = target[:, 1] * transform.scale_y + transform.pad_top
    target /= np.asarray(ROI_WH, np.float32)
    if np.any(target < 0) or np.any(target > 1):
        raise ValueError("letterboxed ROI does not contain GT corners")
    canvas = _appearance(canvas, rng)
    if geometric and rng.random() < .6:
        # Small projective warp, applying one matrix to pixels and ordered LT,RT,RB,LB.
        ww, hh = ROI_WH
        pts = target * np.asarray([ww, hh], np.float32)
        src = np.asarray([[0, 0], [ww-1, 0], [ww-1, hh-1], [0, hh-1]], np.float32)
        affine = cv2.getRotationMatrix2D((ww/2, hh/2), float(rng.uniform(-5, 5)),
                                         float(rng.uniform(.9, 1.1)))
        affine[:, 2] += [rng.uniform(-.025, .025)*ww, rng.uniform(-.025, .025)*hh]
        matrix = np.vstack([affine, [0., 0., 1.]])
        if rng.random() < .5:
            dst = src + rng.uniform(-.012, .012, (4, 2)).astype(np.float32) * [ww, hh]
            matrix = cv2.getPerspectiveTransform(src, dst) @ matrix
        transformed = cv2.perspectiveTransform(pts.reshape(1, 4, 2), matrix)[0] / [ww, hh]
        if np.all((transformed >= .005) & (transformed <= .995)):
            canvas = cv2.warpPerspective(canvas, matrix, ROI_WH, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            target = transformed
    return canvas, target.reshape(8).astype(np.float32)


def _sample_positive_roi(rows: list[dict[str, str]], rng: np.random.Generator, *,
                         geometric: bool, invalid_source_paths: set[str],
                         counters: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Retry malformed source labels with a bound and explicit audit counters."""
    for _attempt in range(ROI_POSITIVE_RETRY_LIMIT):
        row = random.choice(rows)
        key = _path_key(row["path"])
        if key in invalid_source_paths:
            counters["source_quad_resample_draws"] += 1
            continue
        image = imread(row["path"])
        if image is None:
            raise OSError(f"failed to decode train positive: {row['path']}")
        try:
            roi, target = _positive_roi(image, row, rng, geometric)
        except SourceQuadOutOfBounds:
            invalid_source_paths.add(key)
            counters["source_quad_resample_draws"] += 1
            continue
        return (torch.from_numpy(roi.transpose(2, 0, 1).copy()).float().div_(255.),
                torch.from_numpy(target))
    counters["source_quad_retry_exhaustions"] += 1
    raise RuntimeError(f"no valid positive ROI after {ROI_POSITIVE_RETRY_LIMIT} bounded draws")


def train_roi_regression(*, train_csv: Path, np_train_csv: Path, val_csv: Path, np_val_csv: Path,
                         output: Path, seed: int = 42, steps: int = 1000,
                         limit_train: int = 12000, batch_size: int = 16,
                         threads: int = 2) -> Path:
    positives, _, audit = _read_safe_pools(
        train_csv, np_train_csv, val_csv, np_val_csv, limit_train, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(threads)
    rng = np.random.default_rng(seed)
    model = RoiCornerRefiner()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    # Match the completed pilot: 200 appearance-only steps, then 800 steps with
    # synchronized mild affine/projective augmentation and a fresh optimizer.
    phase_boundary = min(200, steps)
    invalid_source_paths: set[str] = set()
    source_retry_counters = {"source_quad_resample_draws": 0,
                             "source_quad_retry_exhaustions": 0}
    for step in range(steps):
        if step == phase_boundary and phase_boundary < steps:
            random.seed(seed + 1)
            np.random.seed(seed + 1)
            torch.manual_seed(seed + 1)
            rng = np.random.default_rng(seed + 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        samples = []
        for _ in range(batch_size):
            sample = _sample_positive_roi(
                positives, rng, geometric=(step >= phase_boundary),
                invalid_source_paths=invalid_source_paths, counters=source_retry_counters)
            samples.append(sample)
        x = torch.stack([sample[0] for sample in samples])
        y = torch.stack([sample[1] for sample in samples])
        loss = F.smooth_l1_loss(model(x), y, beta=.015)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    (output.with_suffix(".audit.json")).write_text(json.dumps({**audit, "stage": "roi-regression",
        "steps": steps, "batch_size": batch_size, "seed": seed,
        "input_csv_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (train_csv, np_train_csv, val_csv, np_val_csv)},
        "source_quad_retry_audit": {**source_retry_counters,
            "unique_out_of_bounds_paths": sorted(invalid_source_paths),
            "retry_limit_per_positive_crop": ROI_POSITIVE_RETRY_LIMIT},
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "loss": "SmoothL1(beta=.015), positive quads only"}, indent=2), encoding="utf-8")
    return output


def _roi_bounds(shape: tuple[int, ...], bbox: tuple[float, float, float, float], pad: float):
    image_h, image_w = shape[:2]
    x, y, box_w, box_h = bbox
    return (
        max(0, math.floor(x - pad * box_w)),
        max(0, math.floor(y - pad * box_h)),
        min(image_w, math.ceil(x + box_w + pad * box_w)),
        min(image_h, math.ceil(y + box_h + pad * box_h)),
    )


def _outside_gt_crop(image: np.ndarray, row: dict[str, str], rng: np.random.Generator):
    h, w = image.shape[:2]
    quad = parse_quad(row)
    bx, by, bw, bh = parse_bbox(row)
    low, high = quad.min(0), quad.max(0)
    l, t = min(bx, float(low[0])), min(by, float(low[1]))
    r, b = max(bx+bw, float(high[0])), max(by+bh, float(high[1]))
    margin_x = .12 * (r - l)
    margin_y = .12 * (b - t)
    exclusion = (l - margin_x, t - margin_y, r + margin_x, b + margin_y)
    scale = float(rng.uniform(1.2, 1.7))
    cw = min(w, max(16, int((r-l)*scale*rng.uniform(.92, 1.08))))
    ch = min(h, max(16, int((b-t)*scale*rng.uniform(.92, 1.08))))
    for _ in range(40):
        x = int(rng.integers(0, max(1, w - cw + 1)))
        y = int(rng.integers(0, max(1, h - ch + 1)))
        disjoint_x = x + cw <= exclusion[0] or x >= exclusion[2]
        disjoint_y = y + ch <= exclusion[1] or y >= exclusion[3]
        if disjoint_x or disjoint_y:
            return image[y:y + ch, x:x + cw]
    raise RuntimeError("no GT-exterior background crop found after bounded retries")


def train_roi_confidence(*, train_csv: Path, np_train_csv: Path, val_csv: Path, np_val_csv: Path,
                         coarse_checkpoint: Path, output: Path,
                         refiner_checkpoint: Path | None = None,
                         roi_checkpoint: Path | None = None,
                         strong_augmentation: bool = False,
                         seed: int = 44, steps: int = 400, batch_size: int = 16,
                         limit_train: int = 12000, np_limit: int = 512,
                         threads: int = 2) -> Path:
    positives, np_pool, audit = _read_safe_pools(
        train_csv, np_train_csv, val_csv, np_val_csv, limit_train, seed)
    np_rows = random.Random(seed).sample(np_pool, min(np_limit, len(np_pool)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(threads)
    rng = np.random.default_rng(seed)
    detector = CornerNet.from_file(coarse_checkpoint, version="roi-train-np-proposals")
    detector.score_threshold = COARSE_THRESHOLD
    proposals: dict[str, list[Any]] = {}
    proposal_counts: list[dict[str, Any]] = []
    proposal_samples: list[dict[str, Any]] = []
    for row in np_rows:
        image = imread(row["path"])
        if image is None:
            raise OSError(f"failed to decode safe NP train image: {row['path']}")
        candidates = detector.detect(image, max_candidates=COARSE_TOP_K)
        proposals[_path_key(row["path"])] = candidates
        proposal_counts.append({"path": _path_key(row["path"]),
                                "candidate_count": len(candidates)})
        proposal_samples.append({
            "path": row["path"],
            "candidates": [{"score": float(candidate.score),
                            "bbox": list(candidate.bbox),
                            "corners": np.asarray(candidate.corners).tolist()}
                           for candidate in candidates],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    proposal_cache_path = output.with_suffix(".np-proposals.json")
    proposal_cache = {
        "coarse_checkpoint_sha256": hashlib.sha256(coarse_checkpoint.read_bytes()).hexdigest(),
        "score_threshold": COARSE_THRESHOLD,
        "max_candidates": COARSE_TOP_K,
        "samples": proposal_samples,
    }
    proposal_cache_path.write_text(json.dumps(proposal_cache, indent=2), encoding="utf-8")
    proposal_cache_sha256 = hashlib.sha256(proposal_cache_path.read_bytes()).hexdigest()
    model = RoiConfidenceModel()
    if strong_augmentation:
        if roi_checkpoint is None:
            raise ValueError("strong augmentation requires a complete --roi-checkpoint")
        initial_checkpoint = roi_checkpoint
        refiner_lr, confidence_lr = 1e-5, 1e-4
        initial_full_model = True
    else:
        if refiner_checkpoint is None:
            raise ValueError("standard confidence training requires --refiner-checkpoint")
        initial_checkpoint = refiner_checkpoint
        refiner_lr, confidence_lr = 2e-5, 3e-4
        initial_full_model = False
    initial_checkpoint_sha256 = _load_roi_initial_state(
        model, initial_checkpoint, full_model=initial_full_model)
    optimizer = torch.optim.AdamW(
        [{"params": model.refiner.parameters(), "lr": refiner_lr},
         {"params": model.confidence.parameters(), "lr": confidence_lr}],
        weight_decay=1e-4)
    model.train()
    source_counts = {"np_weak_candidate": 0, "np_fallback": 0,
        "gt_exterior_background": 0, "source_quad_resample_draws": 0,
        "source_quad_retry_exhaustions": 0, "ordinary": 0,
        "moderate": 0, "strong": 0, "ordinary_fallback": 0,
        "dark": 0, "normal_exposure": 0}
    invalid_source_paths: set[str] = set()
    for _step in range(steps):
        crops: list[torch.Tensor] = []
        labels: list[float] = []
        quads: list[torch.Tensor] = []
        for _ in range(8):
            crop_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            positive_tensor, positive_quad = _sample_positive_roi(
                positives, crop_rng, geometric=not strong_augmentation,
                invalid_source_paths=invalid_source_paths, counters=source_counts)
            if strong_augmentation:
                roi_image = positive_tensor.numpy().transpose(1, 2, 0) * 255.
                roi_image = np.clip(roi_image, 0, 255).astype(np.uint8)
                roi_image, target, geometry_mode = _strong_roi_geometry(
                    roi_image, positive_quad.numpy(), crop_rng)
                roi_image, exposure_mode = _strong_roi_exposure(roi_image, crop_rng)
                source_counts[geometry_mode] += 1
                source_counts[exposure_mode] += 1
                positive_tensor = torch.from_numpy(
                    roi_image.transpose(2, 0, 1).copy()).float().div_(255.)
                positive_quad = torch.from_numpy(target.astype(np.float32))
            crops.append(positive_tensor)
            labels.append(1.)
            quads.append(positive_quad)
        for _ in range(4):
            row = random.choice(np_rows)
            image = imread(row["path"])
            if image is None:
                raise OSError(row["path"])
            candidates = proposals[_path_key(row["path"])]
            if candidates:
                candidate = random.choice(candidates)
                x0, y0, x1, y1 = _roi_bounds(image.shape, candidate.bbox, ROI_PAD_PER_SIDE)
                crop = image[y0:y1, x0:x1]
                source_counts["np_weak_candidate"] += 1
            else:
                reference = random.choice(positives)
                quad = parse_quad(reference)
                bx, by, bw, bh = parse_bbox(reference)
                low, high = quad.min(0), quad.max(0)
                plate_w = max(bx + bw, float(high[0])) - min(bx, float(low[0]))
                plate_h = max(by + bh, float(high[1])) - min(by, float(low[1]))
                scale = float(rng.uniform(1.2, 1.7))
                crop_h = min(image.shape[0], max(2, int(plate_h * scale * rng.uniform(.92, 1.08))))
                crop_w = min(image.shape[1], max(2, int(plate_w * scale * rng.uniform(.92, 1.08))))
                x = int(rng.integers(0, max(1, image.shape[1] - crop_w + 1)))
                y = int(rng.integers(0, max(1, image.shape[0] - crop_h + 1)))
                crop = image[y:y + crop_h, x:x + crop_w]
                source_counts["np_fallback"] += 1
            canvas, _ = letterbox_image(crop, ROI_WH)
            canvas = _appearance(canvas, rng)
            crops.append(torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.))
            labels.append(0.)
            quads.append(torch.zeros(8))
        for _ in range(4):
            row = random.choice(positives)
            image = imread(row["path"])
            if image is None:
                raise OSError(row["path"])
            crop = _outside_gt_crop(image, row, rng)
            canvas, _ = letterbox_image(crop, ROI_WH)
            canvas = _appearance(canvas, rng)
            crops.append(torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.))
            labels.append(0.)
            quads.append(torch.zeros(8))
            source_counts["gt_exterior_background"] += 1
        order = torch.randperm(len(crops)).tolist()
        x = torch.stack([crops[index] for index in order])
        y = torch.tensor([labels[index] for index in order])
        q = torch.stack([quads[index] for index in order])
        pred, logits = model(x)
        cls = F.binary_cross_entropy_with_logits(logits, y)
        # Critical mask: negatives train confidence only; their zero placeholder quads are never regressed.
        positive_mask = y > .5
        reg = F.smooth_l1_loss(pred[positive_mask], q[positive_mask], beta=.015)
        loss = cls + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), output)
    proposal_inventory = hashlib.sha256(json.dumps(
        proposal_counts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    confidence_audit = {
        **audit,
        "stage": "roi-confidence",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "np_train_rows_sampled": len(np_rows),
        "strong_augmentation": strong_augmentation,
        "augmentation_schedule": ({
            "geometry_mix": {"ordinary": .70, "moderate": .20, "strong": .10},
            "moderate": {"rotation_degrees": [-9, 9], "scale": [.94, 1.06],
                         "translation_fraction": .025, "perspective_fraction": .025},
            "strong": {"rotation_degrees": [-20, 20], "scale": [.88, 1.12],
                       "translation_fraction": .045, "perspective_fraction": .065},
            "dark_probability": .35, "dark_gain": [.35, .75], "dark_gamma": [.65, 1.25],
            "dark_blur_probability": .35,
        } if strong_augmentation else {"mode": "standard mild ROI geometry"}),
        "initial_roi_checkpoint": str(initial_checkpoint.resolve()),
        "initial_roi_checkpoint_sha256": initial_checkpoint_sha256,
        "optimizer_learning_rates": {"refiner": refiner_lr, "confidence": confidence_lr},
        "proposal_threshold": COARSE_THRESHOLD,
        "proposal_top_k": COARSE_TOP_K,
        "roi_padding_per_side": ROI_PAD_PER_SIDE,
        "negative_source_counts": source_counts,
        "np_proposal_inventory_sha256": proposal_inventory,
        "np_proposal_cache_path": str(proposal_cache_path.resolve()),
        "np_proposal_cache_sha256": proposal_cache_sha256,
        "np_proposal_images_with_candidates": sum(
            row["candidate_count"] > 0 for row in proposal_counts),
        "np_proposal_count": sum(row["candidate_count"] for row in proposal_counts),
        "source_quad_retry_audit": {
            "unique_out_of_bounds_paths": sorted(invalid_source_paths),
            "retry_limit_per_positive_crop": ROI_POSITIVE_RETRY_LIMIT,
            "resample_draws": source_counts["source_quad_resample_draws"],
            "retry_exhaustions": source_counts["source_quad_retry_exhaustions"],
        },
        "input_csv_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (train_csv, np_train_csv, val_csv, np_val_csv)
        },
        "coarse_checkpoint_sha256": hashlib.sha256(coarse_checkpoint.read_bytes()).hexdigest(),
        "refiner_checkpoint_sha256": None if strong_augmentation else initial_checkpoint_sha256,
        "full_roi_initial_checkpoint_sha256": initial_checkpoint_sha256 if strong_augmentation else None,
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "loss": "BCEWithLogits all samples + SmoothL1(beta=.015) positive-mask only",
    }
    (output.with_suffix(".audit.json")).write_text(
        json.dumps(confidence_audit, indent=2), encoding="utf-8")
    return output


def package_roi(coarse: Path, roi: Path, output: Path, confidence_threshold: float = .8) -> Path:
    package = pack_corner_roi_checkpoint(
        coarse,
        roi,
        output,
        proposal_threshold=COARSE_THRESHOLD,
        final_confidence_threshold=confidence_threshold,
        roi_padding_per_side=ROI_PAD_PER_SIDE,
    )
    audit = {
        "stage": "package",
        "coarse_checkpoint": str(coarse.resolve()),
        "coarse_checkpoint_sha256": hashlib.sha256(coarse.read_bytes()).hexdigest(),
        "roi_checkpoint": str(roi.resolve()),
        "roi_checkpoint_sha256": hashlib.sha256(roi.read_bytes()).hexdigest(),
        "package_checkpoint": str(package.resolve()),
        "package_checkpoint_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "proposal_threshold": COARSE_THRESHOLD,
        "final_confidence_threshold": confidence_threshold,
        "roi_padding_per_side": ROI_PAD_PER_SIDE,
    }
    package.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return package


def _average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(states) < 2:
        raise ValueError("coarse averaging requires at least two checkpoints")
    reference_names = set(states[0])
    if not reference_names:
        raise ValueError("checkpoint state_dict is empty")
    if any(set(state) != reference_names for state in states[1:]):
        raise ValueError("coarse checkpoints have different tensor names")
    averaged: dict[str, torch.Tensor] = {}
    for name in states[0]:
        tensors = [state[name] for state in states]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"checkpoint value is not a tensor: {name}")
        reference = tensors[0]
        if any(tensor.shape != reference.shape for tensor in tensors[1:]):
            raise ValueError(f"coarse checkpoints have different tensor shapes: {name}")
        if any(tensor.dtype != reference.dtype for tensor in tensors[1:]):
            raise ValueError(f"coarse checkpoints have different tensor dtypes: {name}")
        if not torch.is_floating_point(reference):
            raise ValueError(f"coarse checkpoint tensor is not floating point: {name}")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError(f"coarse checkpoint tensor contains non-finite values: {name}")
        mean = torch.stack([tensor.detach().cpu() for tensor in tensors], dim=0).mean(dim=0)
        if not torch.isfinite(mean).all():
            raise ValueError(f"averaged coarse tensor contains non-finite values: {name}")
        averaged[name] = mean
    return averaged


def average_coarse_checkpoints(checkpoints: list[Path], output: Path) -> Path:
    if len(checkpoints) < 2:
        raise ValueError("average-coarse requires at least two --average-checkpoints")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    loaded = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must be a metadata wrapper: {checkpoint_path}")
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError(f"checkpoint architecture mismatch: {checkpoint_path}")
        if tuple(checkpoint.get("input_size", ())) != INPUT_SIZE:
            raise ValueError(f"checkpoint input_size mismatch: {checkpoint_path}")
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint has no state_dict: {checkpoint_path}")
        loaded.append(state)
    averaged = _average_state_dicts(loaded)
    CornerModel().load_state_dict(averaged, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": averaged, "architecture": ARCHITECTURE,
                "input_size": list(INPUT_SIZE)}, output)
    audit = {
        "method": "arithmetic mean with equal checkpoint weights",
        "architecture": ARCHITECTURE,
        "input_size": list(INPUT_SIZE),
        "inputs": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in checkpoints},
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tensor_count": len(averaged),
        "external_pretrained_weights": False,
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    return output




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true",
                        help="write deterministic train/val/NP CSVs and audit, then exit")
    parser.add_argument("--prepare-output", type=Path,
                        help="output directory for --prepare-data (default: algorithm/out/seventh/data)")
    parser.add_argument("--verify-np-images", action="store_true",
                        help="decode every selected NP image before writing an audit marked valid")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--val-csv", default=str(VAL_CSV))
    parser.add_argument("--np-train-csv", default="")
    parser.add_argument("--np-val-csv", default="")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--coord-weight", type=float, default=12.)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-run", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--max-steps-this-run", type=int, default=0)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--stage", choices=("roi-regression", "roi-confidence", "package", "average-coarse"),
                        help="run an explicit ROI/package/checkpoint-average stage instead of CornerNet training")
    parser.add_argument("--roi-output", type=Path,
                        help="output checkpoint path for an ROI, package, or average-coarse stage")
    parser.add_argument("--average-checkpoints", nargs="+", type=Path,
                        help="same-architecture coarse checkpoints to average with equal weights")
    parser.add_argument("--coarse-checkpoint", type=Path)
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument("--roi-checkpoint", type=Path)
    parser.add_argument("--roi-seed", type=int)
    parser.add_argument("--roi-steps", type=int)
    parser.add_argument("--roi-limit-train", type=int, default=12000)
    parser.add_argument("--roi-batch-size", type=int, default=16,
                        help="ROI stages default to 16 samples per batch; main detector batch-size is unchanged")
    parser.add_argument("--roi-np-limit", type=int, default=512)
    parser.add_argument("--roi-strong-augmentation", action="store_true",
                        help="resume full ROI confidence state with aug500 geometry/exposure schedule")
    parser.add_argument("--roi-confidence-threshold", type=float, default=.8)
    args = parser.parse_args()
    if args.stage:
        if args.prepare_data or args.verify_np_images:
            parser.error("ROI stages cannot be combined with --prepare-data/--verify-np-images")
        if args.roi_output is None:
            parser.error("selected stage requires --roi-output")
        try:
            if args.stage == "average-coarse":
                if not args.average_checkpoints:
                    parser.error("average-coarse requires --average-checkpoints")
                result = average_coarse_checkpoints(args.average_checkpoints, args.roi_output)
            elif args.stage == "roi-regression":
                result = train_roi_regression(
                    train_csv=Path(args.train_csv),
                    np_train_csv=Path(args.np_train_csv or DATA_DIR / "np_train.csv"),
                    val_csv=Path(args.val_csv),
                    np_val_csv=Path(args.np_val_csv or DATA_DIR / "np_val.csv"),
                    output=args.roi_output,
                    seed=args.roi_seed if args.roi_seed is not None else 42,
                    steps=args.roi_steps if args.roi_steps is not None else 1000,
                    limit_train=args.roi_limit_train, batch_size=args.roi_batch_size, threads=args.threads)
            elif args.stage == "roi-confidence":
                if args.coarse_checkpoint is None:
                    parser.error("roi-confidence requires --coarse-checkpoint")
                if args.roi_strong_augmentation and args.roi_checkpoint is None:
                    parser.error("--roi-strong-augmentation requires full-state --roi-checkpoint")
                if not args.roi_strong_augmentation and args.refiner_checkpoint is None:
                    parser.error("standard roi-confidence requires --refiner-checkpoint")
                confidence_seed = args.roi_seed
                if confidence_seed is None:
                    confidence_seed = 157 if args.roi_strong_augmentation else 44
                confidence_steps = args.roi_steps
                if confidence_steps is None:
                    confidence_steps = 500 if args.roi_strong_augmentation else 400
                result = train_roi_confidence(
                    train_csv=Path(args.train_csv),
                    np_train_csv=Path(args.np_train_csv or DATA_DIR / "np_train.csv"),
                    val_csv=Path(args.val_csv),
                    np_val_csv=Path(args.np_val_csv or DATA_DIR / "np_val.csv"),
                    coarse_checkpoint=args.coarse_checkpoint,
                    refiner_checkpoint=args.refiner_checkpoint,
                    roi_checkpoint=args.roi_checkpoint,
                    strong_augmentation=args.roi_strong_augmentation,
                    output=args.roi_output,
                    seed=confidence_seed,
                    steps=confidence_steps,
                    batch_size=args.roi_batch_size, limit_train=args.roi_limit_train,
                    np_limit=args.roi_np_limit, threads=args.threads)
            else:
                if args.coarse_checkpoint is None or args.roi_checkpoint is None:
                    parser.error("package requires --coarse-checkpoint and --roi-checkpoint")
                result = package_roi(args.coarse_checkpoint, args.roi_checkpoint,
                                     args.roi_output, args.roi_confidence_threshold)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps({"stage": args.stage, "output": str(result)}, ensure_ascii=False))
        return
    if args.verify_np_images and not args.prepare_data:
        parser.error("--verify-np-images requires --prepare-data")
    if args.prepare_data:
        try:
            result = prepare_corner_data(output_dir=args.prepare_output or DATA_DIR,
                                         verify_np_images=args.verify_np_images)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.batch_size < 1 or args.epochs < 1 or args.threads < 1:
        parser.error("epochs, batch-size, and threads must be positive")
    train(args)


if __name__ == "__main__":
    main()
