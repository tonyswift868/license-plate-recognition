"""Select a deterministic, test-only golden image set from a CCPD index."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
from pathlib import Path

from lpr_tools.data.ccpd import read_index
from lpr_tools.data.manifest import ManifestRow, parse_quad, read_manifest, write_manifest
from lpr.plate_rules import match_layout

LABEL_FIELDS = ("file", "plate_no", "plate_color", "category", "source")
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


def _manifest_candidates(manifest_path: Path) -> list[tuple[dict[str, str], str, str]]:
    """Limit picks to images explicitly frozen in the supplied test manifest."""
    candidates: list[tuple[dict[str, str], str, str]] = []
    for row in read_manifest(manifest_path, check_files=True):
        source = row.source.strip()
        for subset, dataset, _, category, color in SELECTIONS:
            if source != subset or row.category != category:
                continue
            if subset == "ccpd_np":
                if not row.plate_no:
                    candidates.append(({"path": str(_canonical_path(row.path, manifest_path)),
                                        "plate_no": "", "source": source, "dataset": dataset,
                                        "subset": subset, "plate_color": row.plate_color,
                                        "quad": "", "selection_origin": "frozen_manifest"}, category, color))
            elif row.plate_no and match_layout(row.plate_no) is not None:
                candidates.append(({"path": str(_canonical_path(row.path, manifest_path)),
                                    "plate_no": row.plate_no, "source": source,
                                    "plate_color": row.plate_color, "dataset": dataset,
                                    "subset": subset, "quad": " ".join(
                                        f"{x:g} {y:g}" for x, y in row.quad) if row.quad else "",
                                    "selection_origin": "frozen_manifest"}, category, color))
    return candidates


def _index_candidates(rows: list[dict[str, str]], subset: str, dataset: str,
                      category: str, color: str) -> list[tuple[dict[str, str], str, str]]:
    result = []
    for row in rows:
        if not _eligible(row, subset, dataset):
            continue
        copied = dict(row)
        copied["source"] = subset
        copied["selection_origin"] = "golden_only"
        result.append((copied, category, color))
    return result


def _safe_remove_staging(staging: Path, staging_root: Path) -> None:
    root = staging_root.resolve()
    candidate = staging.resolve()
    if candidate == root or root not in candidate.parents or not candidate.name.startswith("golden-pick-"):
        raise RuntimeError(f"refusing unsafe staging cleanup: {candidate}")
    shutil.rmtree(candidate)


def pick_golden(index: str | Path | None, out: str | Path, *, seed: int = SEED,
                manifest: str | Path | None = None) -> list[dict[str, str]]:
    """Copy the fixed 20-image sample and write labels.csv; refuse existing targets.

    With ``manifest``, base picks must come from its clear holdout. Other buckets
    prefer it and may fall back to official test rows when ``index`` is supplied.
    Every original path is reserved in the output golden_manifest.csv.
    """
    index_path = Path(index).resolve() if index is not None else None
    target = _safe_output(Path(out))
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    if manifest is not None:
        manifest_path = Path(manifest).resolve()
        source_candidates = _manifest_candidates(manifest_path)
        if index_path is None:
            index_path = manifest_path
            rows = []
        else:
            rows = read_index(index_path)
            split_by_path: dict[str, str] = {}
            for row in rows:
                if not row.get("path", "").strip():
                    raise ValueError("index contains an empty image path")
                canonical = _canonical_path(row["path"], index_path)
                key = str(canonical).casefold()
                previous = split_by_path.setdefault(key, row.get("split", ""))
                if previous != row.get("split", ""):
                    raise ValueError(f"same image assigned to multiple splits: {canonical}")
    else:
        if index_path is None:
            raise ValueError("either --index or --manifest is required")
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
        source_candidates = []
    rng = random.Random(seed)
    chosen: list[tuple[dict[str, str], str, str]] = []
    # Check every bucket and every source file before creating any output.
    for subset, dataset, count, category, color in SELECTIONS:
        manifest_rows = [r for r, cat, _ in source_candidates
                         if r.get("source") == subset and cat == category]
        candidates = list(manifest_rows)
        if subset != "ccpd_base" and len(candidates) < count and index_path is not None and manifest is not None:
            existing = {str(_canonical_path(r["path"], index_path)).casefold() for r in candidates}
            fallback = [r for r, _, _ in _index_candidates(rows, subset, dataset, category, color)
                        if str(_canonical_path(r["path"], index_path)).casefold() not in existing]
            candidates.extend(fallback)
        elif manifest is None:
            candidates = [r for r, _, _ in _index_candidates(rows, subset, dataset, category, color)]
        candidates.sort(key=lambda r: (r.get("path", ""), r.get("plate_no", "")))
        if len(candidates) < count:
            raise ValueError(f"{subset}: need {count} eligible rows, found {len(candidates)}")
        pool = candidates
        # Prefer frozen rows before filling any remaining slots from official test.
        if manifest is not None:
            frozen = [r for r in candidates if r.get("selection_origin") == "frozen_manifest"]
            fallback = [r for r in candidates if r.get("selection_origin") != "frozen_manifest"]
            picked = rng.sample(frozen, min(count, len(frozen)))
            if len(picked) < count:
                picked.extend(rng.sample(fallback, count - len(picked)))
        else:
            picked = rng.sample(pool, count)
        chosen.extend((r, category, color) for r in picked)
    assert index_path is not None
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
    staging_root = Path(__file__).resolve().parents[2] / "out" / "receipts" / "第四批"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="golden-pick-", dir=staging_root))
    try:
        labels: list[dict[str, str]] = []
        for (row, category, color), source, name in zip(chosen, resolved, names):
            shutil.copy2(source, staging / name)
            labels.append({"file": name,
                           "plate_no": "" if category == "np" else row["plate_no"],
                           "plate_color": color,
                           "category": category,
                           "source": row.get("source") or row.get("subset", "")})
        with (staging / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(labels)
        golden_rows = []
        for (row, category, color), source in zip(chosen, resolved):
            golden_rows.append(ManifestRow(
                path=str(source), plate_no="" if category == "np" else row["plate_no"],
                plate_color=color, category=category,
                source=row.get("source") or row.get("subset", ""),
                quad=parse_quad(row.get("quad", "")),
                note=row.get("selection_origin", "golden_only")))
        write_manifest(staging / "golden_manifest.csv", golden_rows, check_files=True)
        # Copy from the approved same-volume scratch area; targets may live on
        # another volume, where an atomic rename is not available on Windows.
        shutil.copytree(staging, target)
        _safe_remove_staging(staging, staging_root)
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
    parser.add_argument("--index", help="ccpd_index.csv for official-test fallback candidates")
    parser.add_argument("--manifest", help="frozen test manifest; prefer its matching sources")
    parser.add_argument("--out", required=True, help="new output directory for images and labels.csv")
    args = parser.parse_args(argv)
    try:
        if not args.index and not args.manifest:
            parser.error("either --index or --manifest is required")
        pick_golden(args.index, args.out, manifest=args.manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
