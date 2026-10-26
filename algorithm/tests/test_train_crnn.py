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


def test_real_green_boost_is_real_8_char_only():
    from lpr_tools.train.train_crnn import _build_sampler, _record_draw_counts, _weights
    rows = [
        {"plate_no": "皖A12345", "source": "ground_truth", "layout": "7"},
        {"plate_no": "皖A123456", "source": "ground_truth", "layout": "8"},
        {"plate_no": "皖A123456", "source": "pipeline_candidate:0:0", "layout": "8"},
        {"plate_no": "皖A123456", "source": "font_synthetic", "layout": "8"},
    ]
    expected = _weights(rows, real_green_boost=8)
    sampler = _build_sampler(expected, samples_per_epoch=100, seed=42)
    assert sampler.weights.tolist() == [1.0, 8.0, 8.0, 1.0]
    assert sampler.num_samples == 100 and sampler.replacement is True
    counts = {}
    _record_draw_counts(counts, ("ground_truth", "pipeline_candidate:0:0", "font_synthetic"),
                        torch.tensor([8, 7, 8]))
    assert counts == {"ground_truth/layout8": 1, "pipeline_candidate/layout7": 1,
                      "font_synthetic/layout8": 1}


def test_from_scratch_weight_continuation_records_local_lineage_and_fresh_run(tmp_path: Path):
    import hashlib
    import torch
    from lpr.recognize.crnn import CrnnNet
    from lpr.recognize import read_manifest
    from lpr_tools.train.train_crnn import train

    data = tmp_path / "plates"
    _make_tiny_data(data)
    initial = tmp_path / "run1" / "best.pt"
    initial.parent.mkdir()
    torch.save(CrnnNet().state_dict(), initial)
    original_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    output = tmp_path / "run2" / "model.pt"
    history = train([data], output, version="crnn_run2", epochs=1, batch_size=2,
                    samples_per_epoch=2, num_workers=0, lr=.0003,
                    init_weights=initial, init_training_run="algorithm/out/models/batch7/crnn_run1_workers1",
                    real_green_boost=8)
    assert len(history) == 1
    metadata = torch.load(output.with_suffix(".checkpoint.pt"), map_location="cpu",
                          weights_only=True)["metadata"]
    assert metadata["initialization"]["kind"] == "local_training_continuation"
    assert metadata["initialization"]["sha256"] == original_sha
    assert metadata["initialization"]["source_training_run"].endswith("crnn_run1_workers1")
    assert metadata["real_green_boost"] == 8.0
    assert metadata["input_size"] == [160, 40]
    assert metadata["pretrained_weights"] is None
    from lpr.recognize.crnn import INPUT_SIZE
    assert read_manifest(output.parent)["model.pt"]["input_size"] == list(INPUT_SIZE)
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == original_sha
    train_log = [__import__("json").loads(line) for line in output.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    assert sum(train_log[0]["actual_draw_counts"].values()) == 2
    assert train_log[0]["actual_draw_counts"] == {"ground_truth/layout7": 2}


def test_local_init_weight_training_resumes_identically(tmp_path: Path):
    import hashlib
    import torch
    from lpr.recognize.crnn import CrnnNet
    from lpr_tools.train.train_crnn import train

    data = tmp_path / "plates"
    _make_tiny_data(data)
    initial = tmp_path / "local_scratch" / "best.pt"
    initial.parent.mkdir()
    torch.manual_seed(17)
    torch.save(CrnnNet().state_dict(), initial)
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    common = dict(epochs=2, batch_size=2, samples_per_epoch=2, num_workers=0,
                  lr=.0003, real_green_boost=8)
    continuous = tmp_path / "continuous_init" / "model.pt"
    split = tmp_path / "split_init" / "model.pt"
    train([data], continuous, init_weights=initial, init_training_run="local_scratch", **common)
    train([data], split, init_weights=initial, init_training_run="local_scratch",
          stop_after=1, **common)
    train([data], split, resume=split.with_suffix(".checkpoint.pt"), **common)
    state_a = torch.load(continuous, map_location="cpu", weights_only=True)
    state_b = torch.load(split, map_location="cpu", weights_only=True)
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    logs_a = [__import__("json").loads(line) for line in continuous.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    logs_b = [__import__("json").loads(line) for line in split.with_suffix(".training.json").read_text(encoding="utf-8").splitlines()]
    assert [row["metrics"] for row in logs_a] == [row["metrics"] for row in logs_b]
    metadata = torch.load(split.with_suffix(".checkpoint.pt"), map_location="cpu",
                          weights_only=True)["metadata"]
    assert metadata["initialization"]["sha256"] == initial_sha
    assert metadata["real_green_boost"] == 8
