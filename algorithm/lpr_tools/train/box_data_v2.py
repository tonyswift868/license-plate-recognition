"""Safe metadata preparation and exact group-balanced exposure plans for box-reader v2.

No images are copied or decoded here. The existing verified batch10 cache is
reused by reference, while a held-out green validation slice is added to tune.
"""
from __future__ import annotations

import array
import csv
import hashlib
import io
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lpr.config import ALGORITHM_DIR

from lpr_tools.train import box_data

ROOT = Path(__file__).resolve().parents[3]
DATA = ALGORITHM_DIR / "data"
V2 = DATA / "plates" / "v2"
CACHE_INDEX = ALGORITHM_DIR / "data/boxcrop_cache/box_reader_v1/index.csv"
BATCH10 = ROOT / "algorithm" / "out" / "data" / "batch10"
DEFAULT_OUT = ROOT / "algorithm" / "out" / "data" / "batch11"
CCPD_INDEX = DATA / "index" / "ccpd_index.csv"
GROUPS = ("base", "green", "db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
TUNE_FIELDS = ("path", "label", "subset", "brightness", "quad", "sha256", "calibration_fold")
SEED = 11042
GREEN_TUNE_COUNT = 300


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpochSamplingPlan:
    """Compact source multiplicities; each source remains represented once."""

    multiplicity: array.array
    group_exposures: Mapping[str, int]
    group_unique: Mapping[str, int]
    unique_samples: int
    total_exposures: int
    epoch: int
    seed: int


def build_balanced_multiplicity(subsets: Sequence[str], *, seed: int, epoch: int,
                                groups: Sequence[str] = GROUPS) -> EpochSamplingPlan:
    """Balance exposure counts while covering every unique source at least once.

    Every group is brought to the largest unique group size. Lower-frequency
    groups receive integer copies per source and a deterministic, epoch-specific
    choice of sources for the remainder. The returned unsigned-int array has one
    entry per input source, in original index order; callers can reuse one decoded
    tensor for its multiplicity within a cache block.
    """
    if len(subsets) == 0:
        raise ValueError("cannot balance an empty training pool")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    group_order = tuple(groups)
    if len(set(group_order)) != len(group_order) or not group_order:
        raise ValueError("groups must be a nonempty unique sequence")
    members: dict[str, list[int]] = {name: [] for name in group_order}
    for index, subset in enumerate(subsets):
        if subset not in members:
            raise ValueError(f"unknown training subset {subset!r} at index {index}")
        members[subset].append(index)
    counts = {name: len(indices) for name, indices in members.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"balanced epoch requires every group: {counts}")
    target = max(counts.values())
    multiplicity = array.array("I", [1]) * len(subsets)
    for group, indices in members.items():
        copies, remainder = divmod(target, len(indices))
        for index in indices:
            multiplicity[index] = copies
        if remainder:
            # Hash ordering is stable across Python versions and needs no second
            # large shuffled list. Only the selected remainder gets one extra.
            ranked = sorted(indices, key=lambda i: hashlib.sha256(
                f"{seed}:{epoch}:{group}:{i}".encode("ascii")).digest())
            for index in ranked[:remainder]:
                multiplicity[index] += 1
    exposures = {group: sum(multiplicity[i] for i in indices)
                 for group, indices in members.items()}
    if len(set(exposures.values())) != 1:
        raise AssertionError(f"internal group-balance error: {exposures}")
    return EpochSamplingPlan(multiplicity, exposures, counts, len(subsets),
                             sum(exposures.values()), epoch, seed)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _candidate_green_rows(index_path: Path, *, train_hashes: set[str],
                          tune_hashes: set[str], protected_paths: set[str],
                          protected_hashes: set[str], holdout_paths: set[str],
                          holdout_hashes: set[str], ledger: Mapping[str, str]
                          ) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return safe eight-character green val rows, directly hashing only misses."""
    candidates: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    seen_sha: set[str] = set()
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream):
            if (source.get("split") != "val" or source.get("subset") != "ccpd_green"
                    or source.get("has_plate") != "1" or source.get("skip_reason", "").strip()):
                continue
            counts["green_val_candidates"] += 1
            path = source.get("path", "").strip()
            key = box_data._path_key(path)
            if key in protected_paths:
                counts["excluded_protected_path"] += 1
                continue
            if key in holdout_paths:
                counts["excluded_batch8_holdout_path"] += 1
                continue
            if len(source.get("plate_no", "")) != 8:
                counts["excluded_non_eight_character"] += 1
                continue
            value = ledger.get(key)
            if value is None:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"green validation image missing: {path}")
                value = sha256_file(path)
                counts["directly_hashed_green_validation"] += 1
            if value in protected_hashes:
                counts["excluded_protected_sha"] += 1
                continue
            if value in holdout_hashes:
                counts["excluded_batch8_holdout_sha"] += 1
                continue
            if value in train_hashes:
                counts["excluded_training_sha"] += 1
                continue
            if value in tune_hashes:
                counts["excluded_existing_tune_sha"] += 1
                continue
            if value in seen_sha:
                counts["excluded_duplicate_sha"] += 1
                continue
            seen_sha.add(value)
            candidates.append({
                "path": path,
                "label": source["plate_no"],
                "subset": "green",
                "brightness": source.get("brightness", ""),
                "quad": box_data._quad_string(source["quad"]),
                "sha256": value,
            })
    return candidates, dict(counts)


def prepare_batch11(*, out_dir: Path = DEFAULT_OUT, green_tune_count: int = GREEN_TUNE_COUNT,
                    seed: int = SEED, index_path: Path = CCPD_INDEX,
                    cache_index: Path = CACHE_INDEX) -> dict[str, object]:
    """Create a fresh metadata-only training reference, tune CSV and audit.

    Existing batch10 pool/tune and the packed image cache are pinned by SHA.
    Green tune examples come from the unprotected CCPD green validation split,
    which v1 never trained on. The batch8 seven-hard-group tune remains present.
    """
    out_dir, index_path, cache_index = Path(out_dir), Path(index_path), Path(cache_index)
    if green_tune_count <= 0:
        raise ValueError("green_tune_count must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path, tune_path = BATCH10 / "pool.csv", BATCH10 / "tune.csv"
    old_audit_path, old_sampler_path = BATCH10 / "audit.json", BATCH10 / "sampler.json"
    packed_audit_path = cache_index.parent / "audit.json"
    for path in (pool_path, tune_path, old_audit_path, old_sampler_path,
                 cache_index, packed_audit_path, index_path, V2 / "source_hashes.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (out_dir / "audit.json").exists():
        raise FileExistsError(f"refusing to overwrite existing batch11 audit in {out_dir}")

    old_audit = json.loads(old_audit_path.read_text(encoding="utf-8-sig"))
    if old_audit.get("schema") != "box-reader-pool-v1":
        raise ValueError("unsupported batch10 isolation audit")
    pool_sha, old_tune_sha = sha256_file(pool_path), sha256_file(tune_path)
    if (pool_sha != old_audit["output_hashes"]["pool.csv"]
            or old_tune_sha != old_audit["output_hashes"]["tune.csv"]):
        raise ValueError("batch10 pool/tune bytes differ from their isolation audit")
    if sha256_file(old_sampler_path) != old_audit["output_hashes"]["sampler.json"]:
        raise ValueError("batch10 jitter sampler differs from its isolation audit")
    if any(old_audit["content_checks"][key] != 0 for key in
           ("train_tune_sha_overlap", "train_protected_sha_overlap", "tune_protected_sha_overlap")):
        raise ValueError("batch10 audit records an existing isolation overlap")
    if sha256_file(index_path) != old_audit["source_files"][str(index_path.resolve())]:
        raise ValueError("CCPD source index differs from batch10 provenance")
    if sha256_file(cache_index) != json.loads(packed_audit_path.read_text(encoding="utf-8-sig"))["cache_index_sha256"]:
        raise ValueError("shared packed cache index changed")
    packed_audit = json.loads(packed_audit_path.read_text(encoding="utf-8-sig"))
    if packed_audit.get("pool_sha256") != pool_sha or packed_audit.get("pool_count") != 210616:
        raise ValueError("shared packed cache is not the audited full batch10 pool")
    source_ledger_path = V2 / "source_hashes.json"
    ledger, ledger_sha = box_data._hash_ledger(source_ledger_path)

    train_rows = _read_csv(pool_path)
    old_tune_rows = _read_csv(tune_path)
    train_hashes = {row["sha256"].lower() for row in train_rows}
    tune_hashes = {row["sha256"].lower() for row in old_tune_rows}
    if len(train_rows) != 210616 or len(train_hashes) != 210616:
        raise ValueError("batch10 training pool changed or contains duplicate bytes")
    if len(old_tune_rows) != 2100 or len(tune_hashes) != 2100:
        raise ValueError("batch10 hard-group tune set changed")
    if train_hashes & tune_hashes:
        raise ValueError("batch10 train/tune bytes overlap")
    if any(len(row["label"]) == 8 for row in old_tune_rows):
        raise ValueError("batch10 hard-group tune unexpectedly includes 8-character examples")

    protected_paths, protected_hashes, protected_meta = box_data._protected_manifests()
    holdout_path = ROOT / "algorithm" / "out" / "data" / "batch8" / "holdout_origins.csv"
    holdout_rows = _read_csv(holdout_path)
    holdout_paths = {box_data._path_key(row["orig_path"]) for row in holdout_rows}
    holdout_hashes = {row["sha256"].lower() for row in holdout_rows}
    if len(holdout_rows) != 2100 or train_hashes & holdout_hashes or tune_hashes != holdout_hashes:
        raise ValueError("batch8 holdout no longer matches its safe isolation contract")

    green_candidates, green_counts = _candidate_green_rows(
        index_path, train_hashes=train_hashes, tune_hashes=tune_hashes,
        protected_paths=protected_paths, protected_hashes=protected_hashes,
        holdout_paths=holdout_paths, holdout_hashes=holdout_hashes, ledger=ledger)
    if len(green_candidates) < green_tune_count:
        raise ValueError(f"only {len(green_candidates)} safe unused green validation rows; "
                         f"requested {green_tune_count}")
    chooser = random.Random(seed)
    chooser.shuffle(green_candidates)
    selected_green = green_candidates[:green_tune_count]
    chooser.shuffle(selected_green)
    half = green_tune_count // 2
    for position, row in enumerate(selected_green):
        row["calibration_fold"] = "fit" if position < half else "report"
    green_shas = {row["sha256"] for row in selected_green}
    green_paths = {box_data._path_key(row["path"]) for row in selected_green}
    if green_shas & (train_hashes | tune_hashes | holdout_hashes | protected_hashes):
        raise AssertionError("selected green tune bytes overlap a protected fold")
    if green_paths & (protected_paths | holdout_paths):
        raise AssertionError("selected green tune paths overlap a protected fold")
    all_tune_rows = old_tune_rows + selected_green
    tune_fold_shas = {
        fold: {row["sha256"] for row in all_tune_rows if row["calibration_fold"] == fold}
        for fold in ("fit", "report")
    }
    tune_fold_counts = {fold: len(values) for fold, values in tune_fold_shas.items()}
    if tune_fold_counts != {"fit": 1200, "report": 1200}:
        raise ValueError(f"batch11 tune fold sizes changed: {tune_fold_counts}")
    fold_overlap = len(tune_fold_shas["fit"] & tune_fold_shas["report"])
    if fold_overlap:
        raise ValueError(f"batch11 tune fit/report same-byte overlap: {fold_overlap}")

    tune_out = out_dir / "tune.csv"
    rendered_tune = io.StringIO(newline="")
    writer = csv.DictWriter(rendered_tune, fieldnames=TUNE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(old_tune_rows)
    writer.writerows(sorted(selected_green, key=lambda row: row["path"].casefold()))
    tune_bytes = rendered_tune.getvalue().encode("utf-8")
    if tune_out.exists():
        if tune_out.read_bytes() != tune_bytes:
            raise FileExistsError(f"refusing to replace nonmatching tune metadata: {tune_out}")
    else:
        tune_out.write_bytes(tune_bytes)

    group_counts = Counter(row["subset"] for row in train_rows)
    if tuple(sorted(group_counts)) != tuple(sorted(GROUPS)):
        raise ValueError(f"unexpected training groups: {dict(group_counts)}")
    synthetic_subsets = [group for group, count in group_counts.items() for _ in range(count)]
    example_plan = build_balanced_multiplicity(synthetic_subsets, seed=seed, epoch=0)
    if example_plan.unique_samples != 210616 or example_plan.total_exposures != 899739:
        raise AssertionError("unexpected balanced epoch plan for current audited pool")

    sampler_source = json.loads(old_sampler_path.read_text(encoding="utf-8-sig"))
    sampler_out = out_dir / "sampler.json"
    sampler_text = json.dumps(sampler_source, ensure_ascii=False, separators=(",", ":")) + "\n"
    if sampler_out.exists():
        if sampler_out.read_text(encoding="utf-8") != sampler_text:
            raise FileExistsError(f"refusing to replace nonmatching jitter sampler: {sampler_out}")
    else:
        sampler_out.write_text(sampler_text, encoding="utf-8")
    manifest_audit = []
    for item in protected_meta:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"protected manifest changed during preparation: {path}")
        manifest_audit.append(item)
    audit: dict[str, object] = {
        "schema": "box-reader-v2-metadata-v1",
        "source_hashes": {
            "batch10_pool_sha256": pool_sha,
            "batch10_tune_sha256": old_tune_sha,
            "batch10_sampler_sha256": sha256_file(old_sampler_path),
            "ccpd_index_sha256": sha256_file(index_path),
            "packed_cache_index_sha256": sha256_file(cache_index),
            "packed_cache_tar_sha256": packed_audit["cache_tar_sha256"],
            "packed_cache_audit_sha256": sha256_file(packed_audit_path),
            "pinned_source_ledger_sha256": ledger_sha,
            "batch8_holdout_sha256": sha256_file(holdout_path),
        },
        "protected_manifests": manifest_audit,
        "protection_policy": "all five named manifests plus worker03 golden alias checked by normalized path and direct source-byte SHA; batch8 holdout path and SHA excluded",
        "training": {
            "kind": "reuse existing packed-cache index; no image data or new cache created",
            "cache_index": str(cache_index.resolve()),
            "cache_tar": packed_audit["cache_path"],
            "unique_rows": len(train_rows),
            "unique_sha256": len(train_hashes),
            "by_group": dict(sorted(group_counts.items())),
            "train_tune_sha_overlap": len(train_hashes & (tune_hashes | green_shas)),
            "train_protected_sha_overlap": 0,
            "train_batch8_holdout_sha_overlap": len(train_hashes & holdout_hashes),
        },
        "tune": {
            "path": str(tune_out.resolve()),
            "rows": len(old_tune_rows) + len(selected_green),
            "unique_sha256": len(tune_hashes | green_shas),
            "old_safe_hard_rows": len(old_tune_rows),
            "new_green_rows": len(selected_green),
            "green_label_length_counts": dict(Counter(len(row["label"]) for row in selected_green)),
            "green_by_fold": dict(Counter(row["calibration_fold"] for row in selected_green)),
            "green_by_brightness_le_19": sum(bool(row["brightness"].isdigit()) and int(row["brightness"]) <= 19 for row in selected_green),
            "fit_report_fold_rows": tune_fold_counts,
            "green_pool_candidates": len(green_candidates),
            "green_candidate_filter_counts": green_counts,
            "train_tune_sha_overlap": len(train_hashes & (tune_hashes | green_shas)),
            "tune_protected_sha_overlap": len((tune_hashes | green_shas) & protected_hashes),
            "prior_batch8_hard_tune_sha_overlap": len(tune_hashes & holdout_hashes),
            "green_batch8_holdout_sha_overlap": len(green_shas & holdout_hashes),
            "fold_sha_overlap": fold_overlap,
        },
        "balanced_epoch": {
            "group_unique_counts": dict(sorted(group_counts.items())),
            "per_group_target_exposures": max(group_counts.values()),
            "group_exposures": dict(example_plan.group_exposures),
            "unique_rows_each_epoch": example_plan.unique_samples,
            "exposures_each_epoch": example_plan.total_exposures,
            "exposures_for_60_epochs": example_plan.total_exposures * 60,
            "allocation_method": "Each source multiplicity >=1; divide per-group target by unique count; assign one additional repetition to a SHA-256-ranked deterministic epoch remainder. Each group's total equals the largest unique group.",
        },
        "outputs": {
            "tune.csv": {"bytes": tune_out.stat().st_size, "sha256": sha256_file(tune_out)},
            "sampler.json": {"bytes": sampler_out.stat().st_size, "sha256": sha256_file(sampler_out)},
        },
        "images_created_or_copied": 0,
        "green_selection_seed": seed,
        "notes": [
            "The green holdout is selected from CCPD green split=val rows that are absent from batch10 training and the five protected manifests; green test rows are never considered.",
            "Existing batch10 seven-hard-group tune origins and their calibration folds are preserved.",
            "Each balanced epoch includes all 210616 unique training origins and repeats lower-frequency groups to 99971 exposures per group.",
        ],
    }
    if any(audit["training"][key] != 0 for key in
           ("train_tune_sha_overlap", "train_protected_sha_overlap", "train_batch8_holdout_sha_overlap")):
        raise AssertionError("training metadata intersects a held-out/protected fold")
    if (audit["tune"]["train_tune_sha_overlap"] != 0
            or audit["tune"]["tune_protected_sha_overlap"] != 0
            or audit["tune"]["green_batch8_holdout_sha_overlap"] != 0
            or audit["tune"]["prior_batch8_hard_tune_sha_overlap"] != 2100):
        raise AssertionError("tune metadata intersects an excluded fold")
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--green-tune-count", type=int, default=GREEN_TUNE_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare_batch11(out_dir=args.out, green_tune_count=args.green_tune_count,
                                     seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
