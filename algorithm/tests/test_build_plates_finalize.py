import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from lpr_tools.data.build_plates import FIELDS
from lpr_tools.data.finalize_plates import finalize


def _row(file: str, plate: str, source: str, origin: str) -> dict[str, str]:
    return {"file": file, "plate_no": plate, "layout": str(len(plate)),
            "color": "green" if len(plate) == 8 else "blue", "source": source, "orig_path": origin}


def _write_dataset(root: Path, rows: list[dict[str, str]]) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = root / row["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"crop:" + row["file"].encode())
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    ledger = {origin: {"sha256": hashlib.sha256(origin.encode()).hexdigest(),
                       "split": "train" if origin.startswith("train") else "val"}
              for origin in {row["orig_path"] for row in rows}}
    (root / "source_hashes.json").write_text(json.dumps(ledger), encoding="utf-8")
    split_counts = Counter(row["file"].split("/", 1)[0] for row in rows)
    (root / "stats.json").write_text(json.dumps({"by_split": split_counts}), encoding="utf-8")
    (root / "build_state.json").write_text(json.dumps({"processed_this_run": len(ledger),
                                                        "selected_remaining_at_start": len(ledger)}), encoding="utf-8")
    (root / "provenance.json").write_text('{"builder":"test"}', encoding="utf-8")
    (root / "_crop_provenance").mkdir(exist_ok=True)
    (root / "_crop_provenance" / "train.json").write_text('[{"file":"train/t.jpg","split":"train"}]', encoding="utf-8")


def test_finalize_caps_candidates_balanced_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    origins = {key: f"source-{key}.jpg" for key in "abcdef"}
    plates = {"a": "京A12345", "b": "沪B23456", "c": "粤C34567",
              "d": "沪BD12345", "e": "粤CE23456", "f": "京DF34567"}
    rows = [_row("train/t.jpg", "京A12345", "ground_truth", "train-source.jpg")]
    for key in origins:
        rows.append(_row(f"val/{key}_gt.jpg", plates[key], "ground_truth", origins[key]))
    candidates = [("a", 0), ("a", 1), ("b", 0), ("c", 0),
                  ("d", 0), ("d", 1), ("e", 0), ("f", 0)]
    for key, n in candidates:
        rows.append(_row(f"val/{key}_cand{n}.jpg", plates[key], f"pipeline_candidate:0:{n}", origins[key]))
    _write_dataset(root, rows)
    ledger = (root / "source_hashes.json").read_bytes()
    train_before = (root / "labels.csv").read_text(encoding="utf-8")

    plan = finalize(root, tmp_path / "receipt", cap=10, seed=42)
    assert plan["mode"] == "dry_run"
    assert plan["val_gt_kept"] == 6 and plan["val_crops_after"] == 10
    assert len(list((root / "val").glob("*.jpg"))) == 14

    first = finalize(root, tmp_path / "receipt", apply=True, cap=10, seed=42)
    with (root / "labels.csv").open(encoding="utf-8", newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    kept = [row for row in final_rows if row["file"].startswith("val/")]
    kept_candidates = [row for row in kept if row["source"].startswith("pipeline_candidate:")]
    assert len(kept) == 10 and len(kept_candidates) == 4
    assert Counter(row["layout"] for row in kept_candidates) == {"7": 2, "8": 2}
    assert len({row["orig_path"] for row in kept_candidates}) == 4
    assert {row["orig_path"] for row in kept if row["source"] == "ground_truth"} == set(origins.values())
    assert all((root / row["file"]).is_file() for row in kept)
    assert (root / "labels.generated.csv").read_text(encoding="utf-8") == train_before
    assert (root / "source_hashes.json").read_bytes() == ledger
    assert (root / "_crop_provenance" / "train.json").read_text(encoding="utf-8") == '[{"file":"train/t.jpg","split":"train"}]'
    assert len(list((root / "val").glob("*.jpg"))) == 10

    labels_after = (root / "labels.csv").read_bytes()
    files_after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jpg"))
    second = finalize(root, tmp_path / "receipt", apply=True, cap=10, seed=42)
    assert second == first
    assert (root / "labels.csv").read_bytes() == labels_after
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jpg")) == files_after
    assert json.loads((root / "stats.json").read_text(encoding="utf-8"))["validation_candidates_kept"] == 4


def test_finalize_rejects_path_escape_before_deleting(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rows = [_row("val/gt.jpg", "京A12345", "ground_truth", "v.jpg"),
            _row("val/../outside.jpg", "京A12345", "pipeline_candidate:0:0", "v.jpg")]
    _write_dataset(root, rows)
    outside = root / "outside.jpg"
    outside.write_bytes(b"keep")
    before = (root / "labels.csv").read_bytes()
    with pytest.raises(ValueError, match="unsafe validation label path"):
        finalize(root, tmp_path / "receipt", apply=True, cap=1)
    assert (root / "labels.csv").read_bytes() == before
    assert outside.read_bytes() == b"keep"
    assert (root / "val" / "gt.jpg").is_file()


def test_apply_refuses_incomplete_build_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rows = [_row("val/gt.jpg", "京A12345", "ground_truth", "v.jpg"),
            _row("val/cand.jpg", "京A12345", "pipeline_candidate:0:0", "v.jpg")]
    _write_dataset(root, rows)
    (root / "stats.json").unlink()
    labels_before = (root / "labels.csv").read_bytes()
    files_before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jpg"))
    with pytest.raises(RuntimeError, match="before final build stats"):
        finalize(root, tmp_path / "receipt", apply=True, cap=1)
    assert (root / "labels.csv").read_bytes() == labels_before
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jpg")) == files_before
