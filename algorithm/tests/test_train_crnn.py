import csv
from pathlib import Path

from lpr_tools.train.train_crnn import _read_rows


def test_excluded_origin_removes_train_views_only(tmp_path: Path):
    root = tmp_path / "plates"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    origin = tmp_path / "source.jpg"
    rows = [
        ["train/a.jpg", "皖ADX640", "7", "blue", "ground_truth", str(origin)],
        ["train/b.jpg", "皖ADX640", "7", "blue", "pipeline_candidate:1:0", str(origin)],
        ["val/c.jpg", "皖ADX640", "7", "blue", "ground_truth", str(origin)],
    ]
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)
    excluded = {str(origin.resolve()).casefold()}
    assert _read_rows(root, "train", excluded) == []
    val = _read_rows(root, "val", excluded)
    assert len(val) == 1 and val[0]["plate_no"] == "皖ADX640"


def test_invalid_label_fails_closed(tmp_path: Path):
    root = tmp_path / "plates"
    (root / "train").mkdir(parents=True)
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerow(["train/a.jpg", "皖ADX0O0", "7", "blue", "ground_truth", str(tmp_path / "s.jpg")])
    try:
        _read_rows(root, "train")
    except ValueError as exc:
        assert "invalid plate label" in str(exc)
    else:
        raise AssertionError("invalid labels must stop training")
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from lpr_tools.train.train_crnn import train


def _write_plate(root: Path, split: str, name: str, source: str = "ground_truth"):
    path = root / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((80, 252, 3), 127, np.uint8))
    return [f"{split}/{name}", "皖A12345", "7", "blue", source, str(root / "origin.jpg")]


def test_tiny_ctc_training_resume_completes_planned_epochs(tmp_path: Path):
    data = tmp_path / "plates"
    rows = [_write_plate(data, "train", "a.png"), _write_plate(data, "train", "b.png")]
    rows += [_write_plate(data, "val", "v.png")]
    with (data / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)
    out = tmp_path / "models" / "crnn_v1.pt"
    first = train([data], out, epochs=2, batch_size=2, samples_per_epoch=2,
                  num_workers=0, stop_after=1)
    assert len(first) == 1
    continued = train([data], out, epochs=2, batch_size=2, samples_per_epoch=2,
                      num_workers=0, resume=out.with_suffix(".checkpoint.pt"))
    assert len(continued) == 2
    assert torch.load(out, map_location="cpu", weights_only=True)
    assert len(out.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()) == 2


def test_resume_matches_uninterrupted_run(tmp_path: Path):
    data = tmp_path / "plates"
    rows = [_write_plate(data, "train", "a.png"), _write_plate(data, "train", "b.png")]
    rows += [_write_plate(data, "val", "v.png")]
    with (data / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)
    continuous = tmp_path / "continuous" / "model.pt"
    split = tmp_path / "split" / "model.pt"
    train([data], continuous, epochs=2, batch_size=2, samples_per_epoch=2, num_workers=0)
    train([data], split, epochs=2, batch_size=2, samples_per_epoch=2,
          num_workers=0, stop_after=1)
    train([data], split, epochs=2, batch_size=2, samples_per_epoch=2, num_workers=0,
          resume=split.with_suffix(".checkpoint.pt"))
    state_a = torch.load(continuous, map_location="cpu", weights_only=True)
    state_b = torch.load(split, map_location="cpu", weights_only=True)
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    logs_a = [__import__("json").loads(line) for line in continuous.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    logs_b = [__import__("json").loads(line) for line in split.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    assert [row["metrics"] for row in logs_a] == [row["metrics"] for row in logs_b]


def test_explicit_worker_change_resume_matches_zero_worker_run(tmp_path: Path):
    data = tmp_path / "plates"
    rows = [_write_plate(data, "train", "a.png"), _write_plate(data, "train", "b.png")]
    rows += [_write_plate(data, "val", "v.png")]
    with (data / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)
    continuous = tmp_path / "continuous" / "model.pt"
    source = tmp_path / "source" / "model.pt"
    migrated = tmp_path / "migrated" / "model.pt"
    train([data], continuous, epochs=2, batch_size=2, samples_per_epoch=2, num_workers=0)
    train([data], source, epochs=2, batch_size=2, samples_per_epoch=2,
          num_workers=0, stop_after=1)
    source_best_bytes = source.read_bytes()
    train([data], migrated, epochs=2, batch_size=2, samples_per_epoch=2,
          num_workers=1, resume=source.with_suffix(".checkpoint.pt"),
          resume_execution_change=True)
    assert source.read_bytes() == source_best_bytes
    state_a = torch.load(continuous, map_location="cpu", weights_only=True)
    state_b = torch.load(migrated, map_location="cpu", weights_only=True)
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    logs_a = [__import__("json").loads(line) for line in continuous.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    logs_b = [__import__("json").loads(line) for line in migrated.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    assert [row["metrics"] for row in logs_a] == [row["metrics"] for row in logs_b]
    ckpt = torch.load(migrated.with_suffix(".checkpoint.pt"), map_location="cpu", weights_only=True)
    assert ckpt["metadata"]["execution_lineage"][-1]["to_num_workers"] == 1
    assert ckpt["metadata"]["execution_lineage"][-1]["checkpoint_sha256"]
    assert ckpt["metadata"]["execution_lineage"][-1]["source_best_sha256"]
    assert migrated.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()[0]
import csv
import shutil
from pathlib import Path

import pytest
import torch

from lpr_tools.train.train_crnn import train


def _make_tiny_data(root: Path):
    rows = [_write_plate(root, "train", "a.png"), _write_plate(root, "train", "b.png")]
    rows += [_write_plate(root, "val", "v.png")]
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "plate_no", "layout", "color", "source", "orig_path"])
        writer.writerows(rows)


def test_execution_change_resume_rejects_core_changes_and_overwrite(tmp_path: Path):
    data = tmp_path / "plates"
    _make_tiny_data(data)
    exclude_a = tmp_path / "exclude_a.txt"
    exclude_a.write_text("", encoding="utf-8")
    source = tmp_path / "source" / "model.pt"
    train([data], source, epochs=2, batch_size=2, samples_per_epoch=2, num_workers=0,
          stop_after=1, exclude_origins=exclude_a)
    checkpoint = source.with_suffix(".checkpoint.pt")

    with pytest.raises(ValueError, match="execution-only resume"):
        train([data], tmp_path / "bad_lr" / "model.pt", epochs=2, batch_size=2,
              samples_per_epoch=2, lr=.002, num_workers=1, resume=checkpoint,
              exclude_origins=exclude_a, resume_execution_change=True)
    cloned = tmp_path / "cloned_plates"
    shutil.copytree(data, cloned)
    with pytest.raises(ValueError, match="execution-only resume"):
        train([cloned], tmp_path / "bad_data" / "model.pt", epochs=2, batch_size=2,
              samples_per_epoch=2, num_workers=1, resume=checkpoint,
              exclude_origins=exclude_a, resume_execution_change=True)
    exclude_b = tmp_path / "exclude_b.txt"
    exclude_b.write_text("different configuration\n", encoding="utf-8")
    with pytest.raises(ValueError, match="execution-only resume"):
        train([data], tmp_path / "bad_exclude" / "model.pt", epochs=2, batch_size=2,
              samples_per_epoch=2, num_workers=1, resume=checkpoint,
              exclude_origins=exclude_b, resume_execution_change=True)
    with pytest.raises(ValueError, match="distinct output checkpoint"):
        train([data], source, epochs=2, batch_size=2, samples_per_epoch=2,
              num_workers=1, resume=checkpoint, exclude_origins=exclude_a,
              resume_execution_change=True)
    existing = tmp_path / "existing" / "model.pt"
    existing.parent.mkdir()
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="new output directory"):
        train([data], existing, epochs=2, batch_size=2, samples_per_epoch=2,
              num_workers=1, resume=checkpoint, exclude_origins=exclude_a,
              resume_execution_change=True)
    assert source.is_file() and torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"] == 1
    assert existing.read_bytes() == b"keep"
