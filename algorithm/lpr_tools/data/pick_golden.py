"""Select a deterministic, test-only golden image set from a CCPD index."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
from pathlib import Path

from lpr_tools.data.ccpd import read_index
from lpr.plate_rules import match_layout

LABEL_FIELDS = ("file", "plate_no", "plate_color", "category")
SEED = 42
# (subset, expected dataset, count, output category, plate color)
SELECTIONS = (
    ("ccpd_base", "ccpd2019", 10, "clear", "blue"),
    ("ccpd_green", "ccpd2020", 2, "clear", "green"),
    ("ccpd_db", "ccpd2019", 2, "night", "blue"),
    ("ccpd_tilt", "ccpd2019", 2, "tilt", "blue"),
    ("ccpd_blur", "ccpd2019", 2, "dirty", "blue"),
    ("ccpd_np", "ccpd2019", 2, "np", "unknown"),
)


def _safe_output(out: Path) -> Path:
    result = out.resolve()
    # Protect the canonical regression fixture directory, including descendants.
    protected = (Path(__file__).resolve().parents[2] / "tests" / "data" / "golden").resolve()
    if result == protected or protected in result.parents:
        raise ValueError(f"refusing to write protected golden test data: {result}")
    return result


def _eligible(row: dict[str, str], subset: str, dataset: str) -> bool:
    if row.get("split") != "test" or row.get("subset") != subset or row.get("dataset") != dataset:
        return False
    if subset == "ccpd_np":
        return row.get("has_plate") == "0" and row.get("skip_reason") == "no_plate"
    return (row.get("has_plate") == "1" and not row.get("skip_reason", "").strip()
            and match_layout(row.get("plate_no", "")) is not None)


def _canonical_path(raw: str, index_path: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else index_path.parent / path).resolve()


def _safe_remove_staging(staging: Path, staging_root: Path) -> None:
    root = staging_root.resolve()
    candidate = staging.resolve()
    if candidate == root or root not in candidate.parents or not candidate.name.startswith("golden-pick-"):
        raise RuntimeError(f"refusing unsafe staging cleanup: {candidate}")
    shutil.rmtree(candidate)


def pick_golden(index: str | Path, out: str | Path, *, seed: int = SEED) -> list[dict[str, str]]:
    """Copy the fixed 20-image sample and write labels.csv; refuse existing targets."""
    index_path = Path(index).resolve()
    target = _safe_output(Path(out))
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    rows = read_index(index_path)
    split_by_path: dict[str, str] = {}
    for row in rows:
        if not row.get("path", "").strip():
            raise ValueError("index contains an empty image path")
        canonical = _canonical_path(row.get("path", ""), index_path)
        key = str(canonical).casefold()
        previous = split_by_path.setdefault(key, row.get("split", ""))
        if previous != row.get("split", ""):
            raise ValueError(f"same image assigned to multiple splits: {canonical}")
    rng = random.Random(seed)
    chosen: list[tuple[dict[str, str], str, str]] = []
    # Check every bucket and every source file before creating any output.
    for subset, dataset, count, category, color in SELECTIONS:
        candidates = [r for r in rows if _eligible(r, subset, dataset)]
        candidates.sort(key=lambda r: (r.get("path", ""), r.get("plate_no", "")))
        if len(candidates) < count:
            raise ValueError(f"{subset}: need {count} eligible test rows, found {len(candidates)}")
        chosen.extend((r, category, color) for r in rng.sample(candidates, count))
    resolved = [_canonical_path(r["path"], index_path) for r, _, _ in chosen]
    canonical_selected: set[str] = set()
    for source in resolved:
        key = str(source).casefold()
        if key in canonical_selected:
            raise ValueError(f"same image selected more than once: {source}")
        canonical_selected.add(key)
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing source image(s): " + ", ".join(missing))

    # Names include source and bucket so same basenames from distinct datasets never collide.
    names: list[str] = []
    for (row, category, _), source in zip(chosen, resolved):
        base = source.name
        prefix = f"{row['dataset']}_{row['subset']}"
        name = f"{prefix}_{base}"
        if name in names:
            raise ValueError(f"duplicate source image selected: {source}")
        names.append(name)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(__file__).resolve().parents[2] / "out"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="golden-pick-", dir=staging_root))
    try:
        labels: list[dict[str, str]] = []
        for (row, category, color), source, name in zip(chosen, resolved, names):
            shutil.copy2(source, staging / name)
            labels.append({"file": name,
                           "plate_no": "" if category == "np" else row["plate_no"],
                           "plate_color": color,
                           "category": category})
        with (staging / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(labels)
        # Atomic directory rename; target must still be absent.
        staging.replace(target)
        print(f"golden_images={len(labels)} seed={seed} out={target}")
        for subset, _, count, category, _ in SELECTIONS:
            print(f"{subset} ({category}): {count}")
        return labels
    except Exception:
        if staging.exists():
            _safe_remove_staging(staging, staging_root)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick deterministic test-split golden samples")
    parser.add_argument("--index", required=True, help="ccpd_index.csv")
    parser.add_argument("--out", required=True, help="new output directory for images and labels.csv")
    args = parser.parse_args(argv)
    try:
        pick_golden(args.index, args.out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
