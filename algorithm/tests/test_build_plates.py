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
from lpr_tools.data.manifest import ManifestRow, write_manifest


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
    write_manifest(frozen_manifest, [ManifestRow(str(frozen.resolve()), "浙C23456", "blue", "clear", "fixture")])
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
    state = json.loads((out / "build_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["processed_this_run"] == state["selected_remaining_at_start"] == 0


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


def test_malformed_frozen_manifest_is_rejected_before_any_crop_write(tmp_path):
    import pytest
    from lpr_tools.data.build_plates import _frozen_paths_and_hashes
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("file,plate_no\nimage.jpg,京A12345\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest columns"):
        _frozen_paths_and_hashes([invalid], tmp_path / "no-golden")


def test_pipeline_version_filters_iou_and_matches_true_layout_size(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lpr_tools.data.build_plates as module
    from lpr.config import load_config
    from lpr.io_utils import imread
    original = tmp_path / 'original.jpg'
    _image(original, 72)
    good = SimpleNamespace(bbox=(80, 55, 151, 61))
    bad = SimpleNamespace(bbox=(0, 0, 20, 10))
    calls = []
    monkeypatch.setattr(module, 'WORKER_CONFIG', load_config())
    monkeypatch.setattr(module, 'preprocess', lambda *_: None)
    monkeypatch.setattr(module, 'locate', lambda *args, **kwargs: [good, bad])
    def rects(image, candidate, cfg):
        calls.append(candidate)
        return [SimpleNamespace(image=np.full((80, 252, 3), 100, np.uint8), layout=7)]
    monkeypatch.setattr(module, 'rectify_candidates', rects)
    raw = {'path': str(original), 'split': 'train', 'plate_no': '沪BD12345',
           'layout': '8', 'plate_color': 'green', 'quad': '80 55 230 55 230 115 80 115',
           'bbox': '80 55 151 61'}
    out = tmp_path / 'dataset'
    result = module._process_one((raw, str(out), None))
    assert result['status'] == 'ok' and len(result['labels']) == 2
    assert calls == [good]
    assert result['labels'][1][4] == 'pipeline_candidate:0:0'
    candidate_crop = imread(out / result['labels'][1][0])
    assert candidate_crop.shape == (80, 276, 3)
    assert candidate_crop.min() == candidate_crop.max() == 100
