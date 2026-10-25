import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpr.io_utils import imwrite
import lpr_tools.data.build_plates as builder
from lpr.config import load_config
from lpr_tools.data.ccpd import INDEX_FIELDS


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 15, (160, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 55), (230, 115), (255, 70, 20), -1)
    cv2.rectangle(image, (80, 55), (230, 115), (240, 240, 240), 2)
    imwrite(path, image)


def test_build_plates_hash_splits_layouts_and_resume(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []

    def add(name: str, split: str, subset: str, plate: str, color: str, seed: int) -> Path:
        path = raw / name
        _image(path, seed)
        row = dict.fromkeys(INDEX_FIELDS, "")
        row.update(path=str(path.resolve()), dataset="ccpd2019", subset=subset, split=split,
                   has_plate="1", plate_no=plate, layout=str(len(plate)), plate_color=color,
                   quad="80 55 230 55 230 115 80 115", bbox="80 55 151 61", skip_reason="")
        rows.append(row)
        return path

    source = add("train.jpg", "train", "ccpd_base", "京A12345", "blue", 1)
    same_split_copy = raw / "train_copy.jpg"
    shutil.copyfile(source, same_split_copy)
    duplicate = rows[0].copy()
    duplicate["path"] = str(same_split_copy.resolve())
    rows.append(duplicate)
    add("val_green.jpg", "val", "ccpd_green", "沪BD12345", "green", 2)
    val_copy = raw / "val_copy.jpg"
    shutil.copyfile(source, val_copy)
    cross_split = rows[0].copy()
    cross_split.update(path=str(val_copy.resolve()), split="val", subset="ccpd_green")
    rows.append(cross_split)
    frozen = add("frozen.jpg", "train", "ccpd_base", "浙C23456", "blue", 3)
    frozen_copy = raw / "frozen_copy.jpg"
    shutil.copyfile(frozen, frozen_copy)
    frozen_duplicate = rows[-1].copy()
    frozen_duplicate["path"] = str(frozen_copy.resolve())
    rows.append(frozen_duplicate)
    add("val_blue.jpg", "val", "ccpd_base", "粤E45678", "blue", 4)

    index = tmp_path / "index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    frozen_manifest = tmp_path / "frozen.csv"
    with frozen_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path",))
        writer.writeheader()
        writer.writerow({"path": str(frozen.resolve())})
    golden = tmp_path / "golden"
    golden.mkdir()
    out, receipts = tmp_path / "out", tmp_path / "receipts"

    builder.build(index, out, receipts, workers=1, raw_root=raw,
                  frozen_manifests=[frozen_manifest], golden_dir=golden)
    with (out / "labels.csv").open(encoding="utf-8", newline="") as handle:
        first = list(csv.DictReader(handle))
    assert tuple(first[0]) == ("file", "plate_no", "layout", "color", "source", "orig_path")
    assert all((out / row["file"]).is_file() for row in first)
    assert all(cv2.imdecode(np.fromfile(out / row["file"], np.uint8), cv2.IMREAD_COLOR).shape[:2] ==
               (80, 252 if row["layout"] == "7" else 276) for row in first)
    assert not any("frozen" in row["orig_path"] for row in first)
    source_hashes = json.loads((out / "source_hashes.json").read_text(encoding="utf-8"))
    assert len({item["sha256"] for item in source_hashes.values() if item["split"] == "train"}
               & {item["sha256"] for item in source_hashes.values() if item["split"] == "val"}) == 0
    assert any(row["color"] == "green" and row["layout"] == "8" for row in first)
    assert any(row["color"] == "blue" and row["layout"] == "7" for row in first)

    # A second invocation resumes the same source ledger without duplicating labels.
    builder.build(index, out, receipts, workers=1, raw_root=raw,
                  frozen_manifests=[frozen_manifest], golden_dir=golden)
    with (out / "labels.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(first)


def test_missing_frozen_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required frozen manifest"):
        builder._frozen_paths_and_hashes([tmp_path / "missing.csv"], tmp_path / "golden")


def test_pipeline_failure_keeps_only_labeled_gt_crop(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw.jpg"
    _image(raw, 7)
    row = {"path": str(raw.resolve()), "split": "train", "plate_no": "京A12345", "layout": "7",
           "plate_color": "blue", "quad": "80 55 230 55 230 115 80 115", "bbox": "80 55 151 61"}
    previous = builder.WORKER_CONFIG
    builder.WORKER_CONFIG = load_config()
    monkeypatch.setattr(builder, "preprocess", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")))
    try:
        out = tmp_path / "out"
        result = builder._process_one((row, str(out), None))
    finally:
        builder.WORKER_CONFIG = previous
    assert result["status"] == "ok"
    assert result["pipeline_error"].startswith("RuntimeError")
    assert len(result["labels"]) == 1
    assert (out / result["labels"][0][0]).is_file()
