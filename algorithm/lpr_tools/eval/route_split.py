"""Fixed seed, label-independent half split shared by three-route diagnostics."""
import hashlib
import json
import posixpath
import random
from collections import defaultdict
from pathlib import Path


def path_key(path):
    return posixpath.normpath(str(path).replace("\\", "/")).casefold()


def split_assignments(records, seed=42, image_sha_by_path=None):
    """Sort strata and paths, shuffle with one seeded RNG, use the first half.

    Fixed membership/subset define strata. Optional source-byte SHA keeps exact
    copies together, rebalancing with singleton rows in the seeded order.
    Truth, predictions, confidence and pixel measurements are never used.
    Odd strata put the extra item in hold. Duplicate normalized paths fail.
    """
    strata = defaultdict(list)
    seen = set()
    for row in records:
        key = path_key(row["path"])
        if key in seen:
            raise ValueError(f"duplicate split path: {row['path']}")
        if row["split"] not in {"val", "devhard"}:
            raise ValueError("split requires fixed val/devhard membership")
        seen.add(key)
        strata[(row["split"], row["subset"])].append(key)
    rng = random.Random(seed)
    assignments = {}
    orders = {}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        orders[stratum] = keys
        boundary = len(keys) // 2
        assignments.update((key, "fit" if index < boundary else "hold")
                           for index, key in enumerate(keys))
    if image_sha_by_path is not None:
        hashes = {path_key(path): sha for path, sha in image_sha_by_path.items()}
        if set(hashes) != seen or any(not sha for sha in hashes.values()):
            raise ValueError("content split requires every image SHA exactly once")
        groups = defaultdict(list)
        for key, sha in hashes.items():
            groups[sha].append(key)
        repeated = set()
        for keys in groups.values():
            if len(keys) > 1:
                # Preserve the initial seeded assignment of the canonical path.
                half = assignments[min(keys)]
                repeated.update(keys)
                assignments.update((key, half) for key in keys)
        for stratum in sorted(orders):
            keys = orders[stratum]
            difference = len(keys) // 2 - sum(assignments[key] == "fit" for key in keys)
            if difference == 0:
                continue
            source_half, target_half = ("hold", "fit") if difference > 0 else ("fit", "hold")
            ordered = keys if difference > 0 else reversed(keys)
            available = [key for key in ordered if key not in repeated
                         and assignments[key] == source_half]
            if len(available) < abs(difference):
                raise ValueError("cannot balance exact stratum halves without splitting duplicate bytes")
            for key in available[:abs(difference)]:
                assignments[key] = target_half
        assert_content_separation(assignments, hashes)
    return assignments


def assert_content_separation(assignments, image_sha_by_path):
    """Reject same-byte images appearing in opposite halves."""
    halves = defaultdict(set)
    for path, sha in image_sha_by_path.items():
        if not sha:
            raise ValueError("image SHA is required for content separation")
        halves[sha].add(assignments[path_key(path)])
    if any(len(values) > 1 for values in halves.values()):
        raise ValueError("same image bytes cross fit/hold; half split is not independent")


def write_split(records, image_sha_by_path, trace_sha256, target, seed=42):
    """Persist the content-safe assignment once for both evaluation tools."""
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    records = list(records)
    assignments = split_assignments(records, seed, image_sha_by_path)
    counts = defaultdict(int)
    for record in records:
        counts[f"{record['split']}_{assignments[path_key(record['path'])]}"] += 1
    amendment = Path(__file__).with_name("fallback_split_v1.json")
    canonical = json.dumps(assignments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {"schema": "fallback-refine-split-v1", "seed": seed,
        "strategy": "seeded stratum path split; exact-byte groups; singleton rebalance",
        "trace_sha256": trace_sha256, "assignments": assignments,
        "assignment_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "counts": dict(counts), "split_amendment": str(amendment.resolve()),
        "split_amendment_sha256": hashlib.sha256(amendment.read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
