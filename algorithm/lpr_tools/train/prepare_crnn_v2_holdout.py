"""Prepare isolated CRNN v2 training and hard calibration data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from lpr.config import ALGORITHM_DIR

SUBSETS = ("blur", "challenge", "db", "fn", "rotate", "tilt", "weather")
FIELDS = ("file", "orig_path", "source", "layout", "plate_no", "crop_path", "subset", "brightness", "partition", "calibration_fold")
DATA = ALGORITHM_DIR / "data"
V2 = DATA / "plates/v2"
V1 = DATA / "plates/v1"
OUT = Path("algorithm/out/data/batch8")
PROTECTED = (
    ("testset", DATA / "index/testset_manifest_v1.csv"),
    ("golden", Path("algorithm/tests/data/golden/golden_manifest.csv")),
    ("worker03_golden", ALGORITHM_DIR / "tests/data/golden/golden_manifest.csv"),
    ("devhard", DATA / "index/devhard_v1.csv"),
    ("val1000", DATA / "index/val1000_v1.csv"),
    ("undecodable", DATA / "index/undecodable.csv"),
)
V1_EXCLUDED = Path("algorithm/out/receipts/batch7/excluded-train-origins.txt")
V2_LABELS_SHA = "b93c3e7795ff0a6a8cdeb5f1321ac5bf76a0c784cb80ab90c6ae687b1418aaad"
V2_LEDGER_SHA = "2a992d2b90ac2ab60c8eb6bdf4b5b89331cff784655bd4d8350c2bd73d329986"


def normpath(path):
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ccpd_brightness(original_path):
    fields = Path(original_path).name.rsplit(".", 1)[0].split("-")
    try:
        return int(fields[-2]) if len(fields) >= 3 else None
    except ValueError:
        return None


def hard_subset(source):
    parts = source.split(":", 2)
    return parts[1] if len(parts) == 3 and parts[0] == "hard_train" else None


def choose_holdout(candidates, protected_hashes, v1_train_hashes, *, per_subset=300, fit_per_subset=150, seed=42):
    if per_subset < 2 or not 0 < fit_per_subset < per_subset:
        raise ValueError("holdout and fit sizes are invalid")
    rng = random.Random(seed)
    selected, used, eligible = {}, set(), {}
    for subset in SUBSETS:
        by_hash = {}
        for origin, digest in candidates.get(subset, ()):
            digest = digest.lower()
            if len(digest) != 64:
                raise ValueError(f"invalid SHA-256 for {Path(origin).name}")
            if digest in protected_hashes or digest in v1_train_hashes:
                continue
            origin = normpath(origin)
            if digest not in by_hash or origin < by_hash[digest]:
                by_hash[digest] = origin
        eligible[subset] = len(by_hash)
        pool = sorted((digest, origin) for digest, origin in by_hash.items() if digest not in used)
        rng.shuffle(pool)
        chosen = pool[:per_subset]
        if len(chosen) != per_subset:
            raise ValueError(f"{subset}: need {per_subset} unique origins, found {len(chosen)}")
        used.update(digest for digest, _ in chosen)
        rng.shuffle(chosen)
        for i, (digest, origin) in enumerate(chosen):
            selected[origin] = {"subset": subset, "sha256": digest,
                                "calibration_fold": "fit" if i < fit_per_subset else "report"}
    if len(used) != per_subset * len(SUBSETS):
        raise AssertionError("holdout SHA groups are not globally unique")
    return selected, eligible


def load_ledger(path):
    digest = sha256_file(path)
    if digest != V2_LEDGER_SHA:
        raise ValueError(f"v2 source ledger changed: {digest}")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for original, record in raw.items():
        value = record.get("sha256", "").lower()
        if len(value) != 64 or record.get("split") not in ("train", "val"):
            raise ValueError("malformed source_hashes entry")
        result[normpath(original)] = (value, record["split"])
    return result, digest


def protected_sources():
    paths, metas = set(), []
    for label, filename in PROTECTED:
        if not filename.is_file():
            raise FileNotFoundError(f"protected list missing: {filename}")
        count = 0
        with filename.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise ValueError(f"protected list has no path column: {filename}")
            for row in reader:
                value = row.get("path", "").strip()
                if value:
                    count += 1
                    paths.add(normpath(value))
        metas.append({"name": label, "path": str(filename.resolve()), "rows": count,
                      "sha256": sha256_file(filename)})
    hashes = set()
    for source in paths:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"protected source missing: {path.name}")
        hashes.add(sha256_file(path))
    return paths, hashes, metas


def v1_training_hashes(ledger, v1root=V1):
    if not V1_EXCLUDED.is_file():
        raise FileNotFoundError("batch7 train exclusion list is missing")
    excluded = {normpath(p.strip()) for p in V1_EXCLUDED.read_text(encoding="utf-8-sig").splitlines() if p.strip()}
    origins, labels = set(), Path(v1root) / "labels.csv"
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["file"].startswith("train/"):
                origin = normpath(row["orig_path"])
                if origin not in excluded:
                    origins.add(origin)
    missing = origins - ledger.keys()
    if missing:
        raise ValueError(f"{len(missing)} v1 training origins lack v2 ledger hashes")
    hashes = {ledger[p][0] for p in origins}
    return hashes, {"origins": len(origins), "unique_sha256": len(hashes),
                    "labels_sha256": sha256_file(labels),
                    "excluded_origins_sha256": sha256_file(V1_EXCLUDED),
                    "excluded_origin_count": len(excluded)}


def output_row(row, root, subset, partition, fold=""):
    return {"file": row["file"], "orig_path": row["orig_path"], "source": row["source"],
            "layout": row["layout"], "plate_no": row["plate_no"],
            "crop_path": str((root / row["file"]).resolve()), "subset": subset,
            "brightness": ccpd_brightness(row["orig_path"]), "partition": partition,
            "calibration_fold": fold}


def prepare(data=V2, v1=V1, out=OUT, *, seed=42, per_subset=300, fit_per_subset=150):
    data, v1, out = Path(data).resolve(), Path(v1).resolve(), Path(out)
    labels, ledger_file = data / "labels.csv", data / "source_hashes.json"
    labels_sha = sha256_file(labels)
    if labels_sha != V2_LABELS_SHA:
        raise ValueError(f"v2 labels changed: {labels_sha}")
    ledger, ledger_sha = load_ledger(ledger_file)
    protected_paths, protected_hashes, protected_meta = protected_sources()
    v1_hashes, v1_meta = v1_training_hashes(ledger, v1)

    candidates = {subset: {} for subset in SUBSETS}
    train_origins, val_origins = set(), set()
    seen = Counter()
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS[:5]).issubset(reader.fieldnames):
            raise ValueError("v2 labels schema lacks expected fields")
        for row in reader:
            origin = normpath(row["orig_path"])
            record = ledger.get(origin)
            split = "train" if row["file"].startswith("train/") else "val" if row["file"].startswith("val/") else None
            if record is None or split is None or record[1] != split:
                raise ValueError("crop path split/source does not match v2 ledger")
            seen[split] += 1
            (train_origins if split == "train" else val_origins).add(origin)
            subset = hard_subset(row["source"])
            if split == "train" and subset:
                if subset not in candidates:
                    raise ValueError(f"unknown hard subset {subset}")
                candidates[subset].setdefault(origin, record[0])

    eligible_sources = {s: [(p, h) for p, h in candidates[s].items()
                            if p not in protected_paths and h not in protected_hashes and h not in v1_hashes]
                        for s in SUBSETS}
    selected, eligible = choose_holdout(eligible_sources, protected_hashes, v1_hashes,
                                        per_subset=per_subset, fit_per_subset=fit_per_subset, seed=seed)
    selected_by_hash = {item["sha256"]: item for item in selected.values()}
    if len(selected_by_hash) != len(selected) or selected_by_hash.keys() & v1_hashes:
        raise AssertionError("selected holdout content aliases")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "tune.csv", "val.csv", "holdout_origins.csv", "audit.json"):
        if (out / name).exists():
            raise FileExistsError(f"refusing to overwrite {out / name}")
    counts = {name: Counter() for name in ("train", "tune", "val")}
    excluded = Counter()
    tune_fold_rows = {s: Counter() for s in SUBSETS}
    val_folded = Counter()
    written_val_origins = set()
    streams, writers = {}, {}
    try:
        for name in ("train", "tune", "val"):
            streams[name] = (out / f"{name}.csv").open("w", encoding="utf-8", newline="")
            writers[name] = csv.DictWriter(streams[name], fieldnames=FIELDS, lineterminator="\n")
            writers[name].writeheader()
        with labels.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                origin = normpath(row["orig_path"])
                digest = ledger[origin][0]
                protected = origin in protected_paths or digest in protected_hashes
                split = "train" if row["file"].startswith("train/") else "val"
                if split == "train":
                    if protected:
                        excluded["protected_train_rows"] += 1
                    elif digest in selected_by_hash:
                        selection = selected_by_hash[digest]
                        s, fold = selection["subset"], selection["calibration_fold"]
                        writers["tune"].writerow(output_row(row, data, s, "tune", fold))
                        counts["tune"][s] += 1
                        tune_fold_rows[s][fold] += 1
                    else:
                        s = hard_subset(row["source"]) or ""
                        writers["train"].writerow(output_row(row, data, s, "train"))
                        counts["train"][s or "nonhard"] += 1
                elif protected:
                    val_folded["protected_path_rows" if origin in protected_paths else "protected_sha_rows"] += 1
                elif digest in selected_by_hash:
                    val_folded["holdout_sha_rows"] += 1
                else:
                    writers["val"].writerow(output_row(row, data, "", "val"))
                    counts["val"][row["source"]] += 1
                    written_val_origins.add(origin)
    finally:
        for stream in streams.values():
            stream.close()

    fold_origins = {s: Counter() for s in SUBSETS}
    for selection in selected.values():
        fold_origins[selection["subset"]][selection["calibration_fold"]] += 1
    for s in SUBSETS:
        if fold_origins[s] != Counter({"fit": fit_per_subset, "report": per_subset-fit_per_subset}):
            raise AssertionError(f"{s} fit/report SHA counts are wrong")
    partition_hashes = {}
    for name in ("train", "tune", "val"):
        with (out / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            partition_hashes[name] = {
                ledger[normpath(row["orig_path"])][0] for row in csv.DictReader(stream)
            }
    if partition_hashes["train"] & partition_hashes["tune"]:
        raise AssertionError("train/tune SHA overlap")
    if partition_hashes["train"] & partition_hashes["val"]:
        raise AssertionError("train/val SHA overlap")
    if partition_hashes["tune"] & partition_hashes["val"]:
        raise AssertionError("tune/val SHA overlap")
    train_hashes = partition_hashes["train"]
    fit_hashes = {v["sha256"] for v in selected.values() if v["calibration_fold"] == "fit"}
    report_hashes = {v["sha256"] for v in selected.values() if v["calibration_fold"] == "report"}
    if fit_hashes & report_hashes:
        raise AssertionError("fit/report SHA overlap")
    if any(ledger[p][0] in protected_hashes for p in written_val_origins):
        raise AssertionError("protected SHA in val.csv")

    with (out / "holdout_origins.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("orig_path", "subset", "sha256", "brightness", "partition", "calibration_fold"), lineterminator="\n")
        writer.writeheader()
        for origin, selection in sorted(selected.items(), key=lambda p: (p[1]["subset"], p[0])):
            writer.writerow({"orig_path": origin, "subset": selection["subset"], "sha256": selection["sha256"],
                             "brightness": ccpd_brightness(origin), "partition": "tune",
                             "calibration_fold": selection["calibration_fold"]})

    dark = Counter()
    brightness_unparsed = Counter()
    for origin, selection in selected.items():
        value = ccpd_brightness(origin)
        if value is None:
            brightness_unparsed[selection["subset"]] += 1
        elif value <= 19:
            dark[selection["subset"]] += 1
    paths = {name: str((out/f"{name}.csv").resolve()) for name in ("train", "tune", "val")}
    paths["holdout_origins"] = str((out/"holdout_origins.csv").resolve())
    audit = {
        "version": "batch8-holdout-v1", "seed": seed, "source_data": str(data),
        "labels_sha256": labels_sha, "source_ledger_sha256": ledger_sha,
        "ledger_scope": "v2 labels train/val origins compared with source_hashes.json; training pool not fully rehashed",
        "crop_rows_seen": dict(seen), "origins_seen": {"train": len(train_origins), "val": len(val_origins)},
        "v1_initializer_training_sources": v1_meta,
        "protected_manifests": protected_meta,
        "protected_unique_paths": len(protected_paths), "protected_direct_sha256_count": len(protected_paths),
        "protected_unique_sha256": len(protected_hashes),
        "protected_hash_scope": "manifest-listed testset/golden/devhard/val1000/undecodable originals hashed only; no decode/inference",
        "hard_origin_candidates": {s: len(candidates[s]) for s in SUBSETS},
        "eligible_unique_sha_after_protected_and_v1": eligible,
        "selected": {"origins_per_subset": per_subset, "unique_sha256": len(selected_by_hash),
                     "fit_per_subset": fit_per_subset, "report_per_subset": per_subset-fit_per_subset,
                     "fold_origins_by_subset": {s: dict(fold_origins[s]) for s in SUBSETS},
                     "brightness_le_19_by_subset": dict(dark),
                     "brightness_unparsed_by_subset": dict(brightness_unparsed)},
        "excluded_rows": {"train": dict(excluded), "val": dict(val_folded)},
        "output": {"paths": paths, "train_rows": sum(counts["train"].values()),
                   "train_rows_by_subset": dict(counts["train"]),
                   "tune_crop_rows": sum(counts["tune"].values()),
                   "tune_crop_rows_by_subset": dict(counts["tune"]),
                   "tune_crop_rows_by_subset_fold": {s: dict(tune_fold_rows[s]) for s in SUBSETS},
                   "val_rows": sum(counts["val"].values()),
                   "val_rows_by_source": dict(counts["val"]),
                   "val_origins": len(written_val_origins)},
        "content_checks": {"train_selected_sha_overlap": len(train_hashes & selected_by_hash.keys()),
                           "partition_unique_sha256": {n: len(v) for n, v in partition_hashes.items()},
                           "partition_pairwise_sha256_overlap": {
                               "train_tune": len(partition_hashes["train"] & partition_hashes["tune"]),
                               "train_val": len(partition_hashes["train"] & partition_hashes["val"]),
                               "tune_val": len(partition_hashes["tune"] & partition_hashes["val"]),
                           },
                           "fit_report_sha_overlap": len(fit_hashes & report_hashes),
                           "selected_sha_seen_by_v1": len(selected_by_hash.keys() & v1_hashes),
                           "protected_train_rows": excluded["protected_train_rows"],
                           "protected_hash_overlap_train": 0, "protected_hash_overlap_tune": 0,
                           "protected_val_rows_removed": sum(val_folded.values())},
        "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "crop_policy": "CSV references read-only absolute v2 crop paths; no image copy/rewrite",
        "brightness_policy": "penultimate hyphen field in CCPD original basename; <=19 is darkest requested band",
    }
    (out/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=V2)
    parser.add_argument("--v1-data", type=Path, default=V1)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-subset", type=int, default=300)
    parser.add_argument("--fit-per-subset", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.v1_data, args.out, seed=args.seed,
                             per_subset=args.per_subset, fit_per_subset=args.fit_per_subset),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
