"""Safe province holdout and online province sampling metadata (no image cache)."""
from __future__ import annotations

import array
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lpr.charset import PROVINCES
from lpr.config import ALGORITHM_DIR

from lpr_tools.train import box_data

ROOT = Path(__file__).resolve().parents[3]
DATA = ALGORITHM_DIR / "data"
INDEX = DATA / "index" / "ccpd_index.csv"
V2 = DATA / "plates" / "v2"
BATCH10 = ROOT / "algorithm" / "out" / "data" / "batch10"
BATCH11 = ROOT / "algorithm" / "out" / "data" / "batch11"
DEFAULT_OUT = ROOT / "algorithm" / "out" / "data" / "batch12"
CACHE_INDEX = ALGORITHM_DIR / "data/boxcrop_cache/box_reader_v1/index.csv"
SEED = 12042
HOLDOUT_FIELDS = ("path", "province", "label", "subset", "brightness", "quad", "sha256", "split")
SYNTH_FIELDS = ("path", "label", "subset", "brightness", "quad", "sha256",
                "target_province", "seed")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class ProvinceRepeatPlan:
    """Additional source repeats; baseline full-pool visits remain untouched."""

    additions: array.array
    unique_by_province: Mapping[str, int]
    added_by_province: Mapping[str, int]
    seed: int
    cap_per_province: int
    cap_per_source: int


def holdout_size(eligible_count: int, *, cap: int = 20, fraction: float = .30,
                 synthetic_only_below: int = 5) -> int:
    """Return the deterministic per-province quota, with tiny groups synthetic-only."""
    if eligible_count < 0 or cap < 0 or not 0 <= fraction <= 1 or synthetic_only_below < 0:
        raise ValueError("invalid holdout policy")
    if eligible_count < synthetic_only_below:
        return 0
    return min(cap, int(eligible_count * fraction))


def province_repeat_additions(rows: Sequence[Mapping[str, str]], *, seed: int,
                              epoch: int, province_cap: int = 5000,
                              per_source_cap: int = 20) -> ProvinceRepeatPlan:
    """Add balanced repeat visits for non-皖 sources, bounded per source/province.

    Every source is already visited once by the base full-pool epoch. This plan
    only returns *additional* visits and never changes the nine batch11 group
    baseline. The total additions for each non-皖 province are at most 5,000;
    any one source gets at most 19 additional visits (20 total including base).
    皖 remains at its natural distribution. A province absent from real data is
    reported as zero and must be supplied by online synthesis.
    """
    if epoch < 0 or province_cap < 0 or per_source_cap < 1:
        raise ValueError("invalid province repeat limits")
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        label = row.get("label", "")
        province = label[0] if label else ""
        if province in PROVINCES and province != "皖":
            positions[province].append(index)
    additions = array.array("I", [0]) * len(rows)
    added: dict[str, int] = {}
    unique = {p: len(positions.get(p, ())) for p in PROVINCES}
    max_extra_each = per_source_cap - 1
    for province in PROVINCES:
        members = positions.get(province, [])
        budget = min(province_cap, len(members) * max_extra_each)
        base, remainder = divmod(budget, len(members)) if members else (0, 0)
        # Deterministic hash ranking rotates the fractional remainder by epoch.
        ranked = sorted(members, key=lambda i: hashlib.sha256(
            f"{seed}:{epoch}:{province}:{i}".encode()).digest())
        for i in members:
            additions[i] = base
        for i in ranked[:remainder]:
            additions[i] += 1
        if any(additions[i] > max_extra_each for i in members):
            raise AssertionError("per-source province repeat cap exceeded")
        added[province] = sum(additions[i] for i in members)
        if added[province] > province_cap:
            raise AssertionError("per-province extra repeat cap exceeded")
    return ProvinceRepeatPlan(additions, unique, added, seed, province_cap, per_source_cap)


def prepare_batch12(*, out_dir: Path = DEFAULT_OUT, index_path: Path = INDEX,
                    tune_path: Path | None = None, seed: int = SEED,
                    holdout_cap: int = 20, holdout_fraction: float = .30,
                    synthetic_per_province: int = 30) -> dict[str, object]:
    """Create province holdout and deterministic synthesis recipes, never images."""
    out_dir, index_path = Path(out_dir), Path(index_path)
    if not 0 <= holdout_fraction <= 1 or holdout_cap < 0 or synthetic_per_province < 1:
        raise ValueError("invalid province data limits")
    tune_path = Path(tune_path) if tune_path else BATCH11 / "tune.csv"
    for path in (index_path, tune_path, BATCH10 / "audit.json", V2 / "source_hashes.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_path, synth_path, audit_path = (out_dir / name for name in
                                            ("province_holdout.csv", "synth_eval.csv", "province_audit.json"))
    train_cache_path, extra_train_path = out_dir / "train_cache.csv", out_dir / "extra_train.csv"
    combined_train_path = out_dir / "combined_train.csv"
    for path in (holdout_path, synth_path, audit_path, train_cache_path, extra_train_path, combined_train_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing province data: {path}")

    old_audit = json.loads((BATCH10 / "audit.json").read_text(encoding="utf-8-sig"))
    if old_audit.get("source_files", {}).get(str(index_path.resolve())) != _sha_file(index_path):
        raise ValueError("CCPD index differs from the pinned batch10 audit")
    ledger, ledger_sha = box_data._hash_ledger(V2 / "source_hashes.json")
    protected_paths, protected_hashes, protected_meta = box_data._protected_manifests()

    old_pool_rows = _read_rows(BATCH10 / "pool.csv")
    old_tune_rows = _read_rows(tune_path)
    old_pool_paths = {box_data._path_key(row["path"]) for row in old_pool_rows}
    old_pool_shas = {row["sha256"].lower() for row in old_pool_rows}
    old_tune_paths = {box_data._path_key(row["path"]) for row in old_tune_rows}
    old_tune_shas = {row["sha256"].lower() for row in old_tune_rows}
    # Keep only compact metadata for eligible non-皖 sources; do not decode images.
    by_province: dict[str, list[dict[str, str]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    seen_path: set[str] = set()
    seen_sha: set[str] = set()
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream):
            if source.get("has_plate") != "1" or source.get("skip_reason", "").strip():
                continue
            label = source.get("plate_no", "")
            province = label[:1]
            if province not in PROVINCES or province == "皖":
                continue
            counters["non_anwei_train_candidates"] += 1
            path = source.get("path", "").strip()
            key = box_data._path_key(path)
            if key in seen_path:
                counters["duplicate_path"] += 1
                continue
            if key in protected_paths:
                counters["excluded_protected_path"] += 1
                continue
            digest = ledger.get(key)
            if digest is None:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"CCPD source missing from ledger and disk: {path}")
                digest = box_data.sha256_file(path)
                counters["directly_hashed"] += 1
            else:
                counters["ledger_hash_reused"] += 1
            if digest in protected_hashes:
                counters["excluded_protected_sha"] += 1
                continue
            if digest in seen_sha:
                counters["duplicate_sha"] += 1
                continue
            seen_path.add(key)
            seen_sha.add(digest)
            subset = source.get("subset", "")
            if subset.startswith("ccpd_"):
                subset = subset.removeprefix("ccpd_")
            by_province[province].append({
                "path": path, "province": province, "label": label,
                "subset": subset, "brightness": source.get("brightness", ""),
                "quad": box_data._quad_string(source.get("quad", "")), "sha256": digest,
                "split": source.get("split", ""),
            })

    held: list[dict[str, str]] = []
    province_counts: dict[str, dict[str, int]] = {}
    holdout_paths: set[str] = set()
    holdout_hashes: set[str] = set()
    holdout_seen_pool_paths: set[str] = set()
    holdout_seen_pool_shas: set[str] = set()
    holdout_seen_tune_shas: set[str] = set()
    for province in PROVINCES:
        values = by_province.get(province, [])
        n = len(values)
        # <5 has synthetic-only evaluation as specified; floor matches a strict
        # 30% sample and avoids claiming a real-image province holdout.
        count = holdout_size(n, cap=holdout_cap, fraction=holdout_fraction)
        ranked = sorted(values, key=lambda row: hashlib.sha256(
            f"{seed}:{province}:{row['sha256']}".encode()).digest())
        # Prefer an origin not used by the previous reader's pool, making the
        # province holdout fresh for the continued checkpoint. If the eligible
        # unseen bucket cannot meet the fixed quota, fill from the complete safe
        # candidate set and report overlap explicitly.
        fresh = [row for row in ranked if row["sha256"] not in old_pool_shas
                 and row["sha256"] not in old_tune_shas]
        selected = fresh[:count]
        if len(selected) < count:
            selected_hashes = {row["sha256"] for row in selected}
            fallback = [row for row in ranked if row["sha256"] not in selected_hashes
                        and row["sha256"] not in old_tune_shas]
            selected.extend(fallback[:count - len(selected)])
        held.extend(selected)
        selected_keys = {box_data._path_key(row["path"]) for row in selected}
        holdout_paths.update(selected_keys)
        holdout_hashes.update(row["sha256"] for row in selected)
        holdout_seen_pool_paths.update(selected_keys & old_pool_paths)
        holdout_seen_pool_shas.update(row["sha256"] for row in selected if row["sha256"] in old_pool_shas)
        holdout_seen_tune_shas.update(row["sha256"] for row in selected if row["sha256"] in old_tune_shas)
        province_counts[province] = {"eligible_non_anwei": n, "holdout": len(selected),
                                     "remaining_real": n - len(selected),
                                     "synthetic_only": int(n < 5)}
    with holdout_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HOLDOUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(held, key=lambda row: (row["province"], row["sha256"])))

    tune_rows = old_tune_rows
    tune_hashes = set()
    safe_tune: list[dict[str, str]] = []
    for row in tune_rows:
        digest = row.get("sha256", "").lower()
        path = row.get("path", "")
        if (not digest or box_data._path_key(path) in protected_paths or digest in protected_hashes
                or box_data._path_key(path) in holdout_paths or digest in holdout_hashes):
            continue
        if digest in tune_hashes:
            continue
        tune_hashes.add(digest)
        safe_tune.append(row)
    if len(safe_tune) < synthetic_per_province:
        raise ValueError("not enough unique safe tune crops for synthesis recipes")
    recipes: list[dict[str, str]] = []
    for province in PROVINCES:
        candidates = [row for row in safe_tune if row.get("label", "")[:1] != province]
        if len(candidates) < synthetic_per_province:
            # Allow the same province as source only when the tune pool itself is
            # too small; replacement still uses an independently selected seed.
            candidates = safe_tune
        ranked = sorted(candidates, key=lambda row: hashlib.sha256(
            f"{seed}:synth:{province}:{row['sha256']}".encode()).digest())
        for ordinal, row in enumerate(ranked[:synthetic_per_province]):
            recipes.append({
                "path": row["path"], "label": row["label"], "subset": row.get("subset", ""),
                "brightness": row.get("brightness", ""), "quad": box_data._quad_string(row["quad"]),
                "sha256": row["sha256"], "target_province": province,
                "seed": str(seed + PROVINCES.index(province) * synthetic_per_province + ordinal),
            })
    with synth_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SYNTH_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(recipes)

    # Preserve the audited packed-cache schema exactly, removing only held-out
    # or otherwise protected same-byte origins. This is metadata, not a new cache.
    with CACHE_INDEX.open("r", encoding="utf-8-sig", newline="") as source, \
            train_cache_path.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        cache_fields = reader.fieldnames
        if not cache_fields or not {"path", "label", "sha256", "subset"}.issubset(cache_fields):
            raise ValueError("shared cache index schema changed")
        writer = csv.DictWriter(target, fieldnames=cache_fields, lineterminator="\n")
        writer.writeheader()
        retained_cache = Counter()
        retained_cache_paths: set[str] = set()
        retained_cache_shas: set[str] = set()
        for row in reader:
            key, digest = box_data._path_key(row["path"]), row["sha256"].lower()
            if (key in holdout_paths or digest in holdout_hashes or key in protected_paths
                    or digest in protected_hashes or key in old_tune_paths or digest in old_tune_shas):
                continue
            writer.writerow(row)
            retained_cache_paths.add(key)
            retained_cache_shas.add(digest)
            province = row["label"][:1]
            if province in PROVINCES:
                retained_cache[province] += 1

    # Sources absent from the audited cache are listed for optional GPU decode;
    # no CPU image cache or decoded image is created here.
    cache_paths = {box_data._path_key(row["path"]) for row in old_pool_rows}
    extras: list[dict[str, str]] = []
    for province in PROVINCES:
        for row in sorted(by_province.get(province, []), key=lambda item: item["sha256"]):
            key, digest = box_data._path_key(row["path"]), row["sha256"]
            if (row["split"] != "train" or key in cache_paths or key in holdout_paths
                    or digest in holdout_hashes or key in protected_paths or digest in protected_hashes
                    or key in old_tune_paths or digest in old_tune_shas):
                continue
            extras.append(row)
    with extra_train_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HOLDOUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(extras)

    with train_cache_path.open("r", encoding="utf-8", newline="") as source, \
            combined_train_path.open("w", encoding="utf-8", newline="") as target:
        cache_reader = csv.DictReader(source)
        assert cache_reader.fieldnames == cache_fields
        writer = csv.DictWriter(target, fieldnames=cache_fields, lineterminator="\n")
        writer.writeheader()
        for row in cache_reader:
            writer.writerow(row)
        for row in extras:
            digest = row["sha256"].lower()
            if digest in retained_cache_shas or box_data._path_key(row["path"]) in retained_cache_paths:
                raise AssertionError("extra source duplicates an audited cache source")
            writer.writerow({
                "path": row["path"], "label": row["label"], "subset": row["subset"],
                "brightness": row["brightness"], "sha256": digest, "quad": row["quad"],
                "cache_pack": "", "pack_offset": "", "pack_length": "",
                "x0": "", "y0": "", "orig_w": "", "orig_h": "",
            })

    audit: dict[str, object] = {
        "schema": "province-coverage-data-v1", "seed": seed,
        "source_hashes": {"ccpd_index_sha256": _sha_file(index_path),
                          "pinned_source_ledger_sha256": ledger_sha,
                          "protected_manifests": protected_meta,
                          "batch11_tune_sha256": _sha_file(tune_path)},
        "protection_policy": "all valid CCPD splits are eligible for province holdout; every five protected manifest excluded by normalized source path and source-byte SHA; duplicate source bytes deduplicated. Additional training sources are restricted to split=train.",
        "holdout_policy": "per province min(20, floor(eligible_non_anwei*0.30)); provinces with fewer than five eligible originals receive zero real holdout and use synthetic evaluation only",
        "province_counts": province_counts,
        "holdout": {"rows": len(held), "unique_paths": len(holdout_paths),
                    "unique_sha256": len(holdout_hashes), "path": str(holdout_path.resolve()),
                    "sha256": _sha_file(holdout_path),
                    "seen_in_batch11_pool_paths": len(holdout_seen_pool_paths),
                    "seen_in_batch11_pool_sha256": len(holdout_seen_pool_shas),
                    "seen_in_batch11_tune_sha256": len(holdout_seen_tune_shas),
                    "train_cache_rows": sum(retained_cache.values()),
                    "train_cache_by_province": dict(retained_cache),
                    "extra_train_rows": len(extras),
                    "extra_train_by_province": dict(Counter(row["province"] for row in extras)),
                    "combined_train_rows": sum(retained_cache.values()) + len(extras),
                    "combined_train_by_province": dict(retained_cache + Counter(row["province"] for row in extras))},
        "synthetic_eval": {"rows": len(recipes), "per_province": synthetic_per_province,
                           "province_counts": dict(Counter(row["target_province"] for row in recipes)),
                           "path": str(synth_path.resolve()), "sha256": _sha_file(synth_path),
                           "image_bytes_written": 0},
        "candidate_filter_counts": dict(counters),
        "repeat_policy": {"baseline": "all existing full-pool sources receive their ordinary batch11 group exposure; this plan adds only non-皖 province repeats",
                          "province_additional_cap": 5000, "per_source_total_cap": 20,
                          "missing_real_province": "zero repeat; synthesis must fill the class"},
        "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": _sha_file(p)}
                    for p in (holdout_path, synth_path, train_cache_path, extra_train_path, combined_train_path)},
        "images_created_or_cached": 0,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


__all__ = ["ProvinceRepeatPlan", "holdout_size", "province_repeat_additions", "prepare_batch12"]
