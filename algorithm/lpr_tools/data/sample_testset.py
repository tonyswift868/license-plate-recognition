"""Build a deterministic v1 G2 test manifest from a CCPD index."""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from lpr_tools.data.ccpd import read_index
from lpr_tools.data.manifest import ManifestRow, validate_manifest, write_manifest

SEED = 42
# Delivered ccpd_db (10,132 rows): Q10=17, Q25=22, median=28.
# <20 selects its dark tail (1,678 rows), excluding the bright/overexposed mode.
DEFAULT_BRIGHTNESS_THRESHOLD = 20
QUOTAS = {"clear_base": 120, "clear_2020": 30, "night": 50,
          "tilt": 50, "dirty": 50, "np": 20}
DIRTY_SUBSETS = {"ccpd_blur", "ccpd_weather", "ccpd_challenge"}


def _quad(value: str) -> tuple[tuple[float, float], ...] | None:
    if not value:
        return None
    nums = [float(v) for v in value.split()]
    if len(nums) != 8:
        raise ValueError("index quad must contain eight coordinates")
    return tuple(zip(nums[::2], nums[1::2]))


def _index_image_path(raw_path: str, index_path: Path) -> str:
    p = Path(raw_path)
    return str((p if p.is_absolute() else index_path.resolve().parent / p).resolve())


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _default_local_labels(index_path: Path) -> Path:
    root = index_path.resolve().parent.parent
    canonical = root / "raw" / "local" / "labels.csv"
    legacy = root / "local" / "labels.csv"
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


def _manifest_row(record: dict[str, str], category: str, index_path: Path, note: str = "") -> ManifestRow:
    color = record.get("plate_color") or "unknown"
    if color not in {"blue", "yellow", "green", "white", "black", "unknown"}:
        color = "unknown"
    return ManifestRow(_index_image_path(record["path"], index_path), record.get("plate_no", "") or "", color,
                       category, record.get("subset", ""), _quad(record.get("quad", "")), note)


def _choose(candidates: list[dict[str, str]], n: int, rng: random.Random,
            label: str, allow_shortfall: bool, shortfalls: dict[str, int]) -> list[dict[str, str]]:
    ordered = sorted(candidates, key=lambda r: r["path"].casefold())
    if len(ordered) < n:
        missing = n - len(ordered)
        if not allow_shortfall:
            raise ValueError(f"{label}: need {n} test samples, found {len(ordered)} (shortfall {missing})")
        shortfalls[label] = missing
    return rng.sample(ordered, min(n, len(ordered)))


def _local_rows(index_path: Path, local_labels: Path | None,
                index_splits: dict[str, str]) -> list[ManifestRow]:
    labels = Path(local_labels) if local_labels else _default_local_labels(index_path)
    if not labels.is_file():
        return []
    output = []
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not ({"file", "path"} & set(reader.fieldnames)):
            raise ValueError("local labels.csv must contain a file or path column")
        for line_no, item in enumerate(reader, 2):
            raw_path = (item.get("path") or item.get("file") or "").strip()
            if not raw_path:
                raise ValueError(f"local labels row {line_no}: empty file/path")
            p = Path(raw_path)
            path = str((p if p.is_absolute() else labels.resolve().parent / p).resolve())
            indexed_split = index_splits.get(_path_key(path))
            if indexed_split in {"train", "val"}:
                raise ValueError(f"local labels row {line_no}: path belongs to index {indexed_split} split: {path}")
            color = item.get("plate_color", "unknown") or "unknown"
            output.append(ManifestRow(path, item.get("plate_no", "") or "", color,
                                      "local", "local", None, item.get("note", "") or ""))
    return output


def _reviewed_dirty(review_path: Path, candidates: list[dict[str, str]],
                    index_path: Path) -> list[tuple[dict[str, str], str]]:
    pool = {_path_key(_index_image_path(r['path'], index_path)): r for r in candidates}
    selected, seen = [], set()
    with review_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['path', 'note']:
            raise ValueError('dirty review CSV must contain path,note columns')
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError('invalid dirty review row width')
            image = Path(raw['path'])
            key = _path_key(image if image.is_absolute() else review_path.resolve().parent / image)
            if key in seen:
                raise ValueError('duplicate dirty review image')
            seen.add(key)
            if key not in pool:
                raise ValueError('dirty review image must be a supported dirty-subset test image')
            if 'AI 目检，待tonyswift868复核' not in raw['note']:
                raise ValueError('dirty review note must state AI 目检，待tonyswift868复核')
            selected.append((pool[key], raw['note']))
    if len(selected) != QUOTAS['dirty']:
        raise ValueError(f'dirty review needs exactly {QUOTAS["dirty"]} images, got {len(selected)}')
    return selected


def build_testset(index_path: str | Path, out_path: str | Path, *,
                  brightness_threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
                  seed: int = SEED, local_labels: str | Path | None = None,
                  allow_shortfall: bool = False,
                  dirty_review: str | Path | None = None) -> list[ManifestRow]:
    index_path, out_path = Path(index_path), Path(out_path)
    records = read_index(index_path)
    path_splits: dict[str, set[str]] = {}
    for record in records:
        # Canonicalize once per index row; selected manifest rows are checked
        # again when written. Re-resolving all 366k rows for each lookup is costly.
        record['path'] = _index_image_path(record['path'], index_path)
        key = record['path'].casefold()
        path_splits.setdefault(key, set()).add(record.get("split", ""))
    leaked = [path for path, splits in path_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"index path appears in multiple splits: {leaked[0]}")
    if len(path_splits) != len(records):
        raise ValueError("duplicate image path in CCPD index")
    index_splits = {path: next(iter(splits)) for path, splits in path_splits.items()}
    rng, shortfalls = random.Random(seed), {}
    # The delivered base has no test split. Its only authorized holdout is
    # 120 val images; every other CCPD category stays on official test rows.
    test = [r for r in records if r.get("split") == "test"]
    usable = [r for r in test if r.get("has_plate") == "1" and not r.get("skip_reason")]
    selected: list[ManifestRow] = []

    base = [r for r in records if r.get("split") == "val" and r.get("has_plate") == "1"
            and not r.get("skip_reason") and r.get("dataset") == "ccpd2019" and r.get("subset") == "ccpd_base"]
    green = [r for r in usable if r.get("dataset") == "ccpd2020" and r.get("subset") == "ccpd_green"]
    night = [r for r in usable if r.get("dataset") == "ccpd2019" and r.get("subset") == "ccpd_db"
             and r.get("brightness", "").isdigit() and int(r["brightness"]) < brightness_threshold]
    tilt = [r for r in usable if r.get("dataset") == "ccpd2019" and r.get("subset") in {"ccpd_tilt", "ccpd_rotate"}]
    dirty = [r for r in usable if r.get("dataset") == "ccpd2019" and r.get("subset") in DIRTY_SUBSETS]
    np_rows = [r for r in test if r.get("subset") == "ccpd_np" and r.get("skip_reason") == "no_plate"
               and r.get("has_plate") == "0"]

    for label, candidates, count, category in (
        ("clear_base", base, QUOTAS["clear_base"], "clear"),
        ("clear_2020", green, QUOTAS["clear_2020"], "clear"),
        ("night", night, QUOTAS["night"], "night"),
        ("tilt", tilt, QUOTAS["tilt"], "tilt"),
        ("dirty", dirty, QUOTAS["dirty"], "dirty"),
        ("np", np_rows, QUOTAS["np"], "np"),
    ):
        if category == 'dirty' and dirty_review is not None:
            selected.extend(_manifest_row(r, category, index_path, note)
                            for r, note in _reviewed_dirty(Path(dirty_review), candidates, index_path))
            continue
        chosen = _choose(candidates, count, rng, label, allow_shortfall, shortfalls)
        note = ('待人工确认' if category == 'dirty' else
                'base val holdout; seed=42; exclude from training/validation' if label == 'clear_base' else '')
        selected.extend(_manifest_row(r, category, index_path, note)
                        if category != "np" else ManifestRow(_index_image_path(r["path"], index_path), "", "unknown", "np", r.get("subset", ""), None,
                                                               "no_plate") for r in chosen)

    selected.extend(_local_rows(index_path, Path(local_labels) if local_labels else None, index_splits))
    validate_manifest(selected, base_dir=out_path.resolve().parent, check_files=False)
    # Writing occurs only after all quotas, local labels, and manifest checks pass.
    write_manifest(out_path, selected)
    counts = Counter(row.category for row in selected)
    print("category_counts=" + ", ".join(f"{key}:{counts.get(key, 0)}" for key in
          ("clear", "night", "tilt", "dirty", "np", "local")))
    if shortfalls:
        print("shortfall=" + ", ".join(f"{k}:{v}" for k, v in shortfalls.items()))
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build G2 v1: base val holdout plus official test images")
    parser.add_argument("--index", required=True, help="ccpd_index.csv")
    parser.add_argument("--out", required=True, help="output manifest CSV")
    parser.add_argument("--brightness-threshold", type=int, default=DEFAULT_BRIGHTNESS_THRESHOLD,
                        help=f"night iff CCPD brightness < threshold (default {DEFAULT_BRIGHTNESS_THRESHOLD})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--local-labels", help="optional labels.csv; by default infer raw/local/labels.csv from index path")
    parser.add_argument("--allow-shortfall", action="store_true", help="demo only: write available rows and report quota shortfalls")
    parser.add_argument('--dirty-review', help='CSV path,note of 50 visually reviewed dirty test images; required for formal output')
    args = parser.parse_args(argv)
    if not args.allow_shortfall and not args.dirty_review:
        parser.error('formal testset requires --dirty-review; unreviewed random dirty images are demo-only')
    try:
        build_testset(args.index, args.out, brightness_threshold=args.brightness_threshold,
                      seed=args.seed, local_labels=args.local_labels, allow_shortfall=args.allow_shortfall,
                      dirty_review=args.dirty_review)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
