from __future__ import annotations

import csv
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lpr.charset import CHAR_TO_INDEX
from lpr.recognize import file_sha256
from lpr.recognize.crnn import CrnnNet
from lpr_tools.train import train_crnn_v2 as trainer


SUBSETS = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
FIELDS = ("plate_no", "partition", "layout", "subset", "source", "crop_path", "brightness")
TEST_TMP = Path(__file__).resolve().parents[1] / "out" / "test_tmp"


@pytest.fixture
def local_tmp():
    path = TEST_TMP / ("crnn_v2_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_crop(path: Path, label: str, seed: int) -> None:
    width = 276 if len(label) == 8 else 252
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 220, size=(80, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _index_row(root: Path, label: str, partition: str, subset: str,
               source: str, seed: int, brightness: str = "") -> dict[str, str]:
    crop = root / f"{partition}_{subset or 'base'}_{source.replace(':', '_')}_{seed}.png"
    _write_crop(crop, label, seed)
    return {"plate_no": label, "partition": partition, "layout": str(len(label)),
            "subset": subset, "source": source, "crop_path": str(crop),
            "brightness": brightness}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_audited_data(root: Path):
    index_dir, fonts = root / "index", root / "fonts"
    index_dir.mkdir()
    train_rows, tune_rows, val_rows = [], [], []
    seed = 1
    for group in (*SUBSETS, "base7", "real8"):
        label = "皖A12345" if group != "real8" else "皖AD12345"
        subset = "" if group in {"base7", "real8"} else group
        for source in ("ground_truth", "pipeline_candidate:0:0"):
            train_rows.append(_index_row(root, label, "train", subset, source, seed))
            seed += 1
    for group in SUBSETS:
        tune_rows.append(_index_row(root, "皖A12345", "tune", group,
                                    "ground_truth", seed, "19" if group == "db" else "80"))
        seed += 1
    val_rows.append(_index_row(root, "皖A12345", "val", "", "ground_truth", seed))
    _write_csv(index_dir / "train.csv", train_rows)
    _write_csv(index_dir / "tune.csv", tune_rows)
    _write_csv(index_dir / "val.csv", val_rows)

    fonts.mkdir()
    font_rows = []
    for i, label in enumerate(("皖A54321", "皖AD54321")):
        name = f"train/font_{i}.png"
        _write_crop(fonts / name, label, 100 + i)
        font_rows.append({"file": name, "plate_no": label, "source": "font_synthetic", "orig_path": ""})
    with (fonts / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "plate_no", "source", "orig_path"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(font_rows)

    checks = {
        "train_selected_sha_overlap": 0, "fit_report_sha_overlap": 0,
        "selected_sha_seen_by_v1": 0, "protected_hash_overlap_train": 0,
        "protected_hash_overlap_tune": 0,
        "partition_pairwise_sha256_overlap": {"train_tune": 0, "train_val": 0, "tune_val": 0},
    }
    audit = {
        "output_sha256": {name: file_sha256(index_dir / f"{name}.csv")
                          for name in ("train", "tune", "val")},
        "content_checks": checks,
    }
    (index_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    init = root / "crnn_v1.pt"
    torch.manual_seed(71)
    torch.save(CrnnNet().state_dict(), init)
    return index_dir, fonts, init


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _tune(values: dict[str, float]):
    return {group + "_GT": {"accuracy": values[group]} for group in SUBSETS}


def test_sampling_weights_preserve_group_shares_and_gt_pipeline_ratio():
    rows = []
    for group in (*SUBSETS, "base7", "real8"):
        label = "皖A12345" if group != "real8" else "皖AD12345"
        rows.extend([
            trainer.Sample("gt", label, group, True),
            trainer.Sample("pipeline", label, group, False),
        ])
    rows.append(trainer.Sample("font", "皖A12345", "font", False))

    weights = trainer.sampling_weights(rows)
    totals = defaultdict(float)
    gt_totals = defaultdict(float)
    for row, weight in zip(rows, weights.tolist()):
        totals[row.group] += weight
        if row.gt:
            gt_totals[row.group] += weight

    assert sum(weights.tolist()) == pytest.approx(1.0)
    for group, expected in trainer.MIXTURE.items():
        assert totals[group] == pytest.approx(expected)
        if group != "font":
            assert gt_totals[group] / totals[group] == pytest.approx(.65)
            assert (totals[group] - gt_totals[group]) / totals[group] == pytest.approx(.35)
    assert trainer.MIXTURE["base7"] == pytest.approx(.12)
    assert trainer.MIXTURE["real8"] == pytest.approx(.10)
    assert trainer.MIXTURE["font"] == pytest.approx(.08)


def test_augmentation_is_seeded_uint8_and_shape_preserving():
    image = np.random.default_rng(9).integers(0, 256, size=(80, 252, 3), dtype=np.uint8)
    first = trainer.augment(image.copy(), np.random.default_rng(1234))
    second = trainer.augment(image.copy(), np.random.default_rng(1234))

    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert np.array_equal(first, second)


def test_plateau_stops_only_after_minimum_epochs_and_patience():
    plateau = [{"tune": _tune({group: .82 for group in SUBSETS})} for _ in range(6)]
    assert trainer.stop_reason(plateau, patience=3, minimum_epochs=6).startswith("db:")

    too_short = plateau[:5]
    assert trainer.stop_reason(too_short, patience=3, minimum_epochs=6) is None

    recovered = [{"tune": _tune({group: .91 for group in SUBSETS})}
                 for _ in range(6)]
    assert trainer.stop_reason(recovered, patience=3, minimum_epochs=6) is None

    last_epoch_improved = plateau[:5] + [
        {"tune": _tune({group: .83 for group in SUBSETS})}
    ]
    assert trainer.stop_reason(last_epoch_improved, patience=3, minimum_epochs=6) is None


def test_selection_is_lexicographic_worst_subset_first():
    high_mean_low_floor = _tune({group: (.89 if group == "db" else .99) for group in SUBSETS})
    lower_mean_high_floor = _tune({group: (.90 if group == "db" else .91) for group in SUBSETS})

    score_a = trainer.selection_score(high_mean_low_floor, {"all": {"accuracy": .99}})
    score_b = trainer.selection_score(lower_mean_high_floor, {"all": {"accuracy": .80}})

    assert score_a[1] > score_b[1]
    assert score_b > score_a


def test_audit_rejects_nonzero_protected_overlap_before_reading_rows(local_tmp):
    index_dir = local_tmp / "index"
    index_dir.mkdir()
    for name in ("train", "tune", "val"):
        _write_csv(index_dir / f"{name}.csv", [])
    checks = {key: 0 for key in (
        "train_selected_sha_overlap", "fit_report_sha_overlap", "selected_sha_seen_by_v1",
        "protected_hash_overlap_train", "protected_hash_overlap_tune")}
    checks["protected_hash_overlap_tune"] = 1
    audit = {
        "output_sha256": {name: file_sha256(index_dir / f"{name}.csv")
                          for name in ("train", "tune", "val")},
        "content_checks": checks,
    }
    (index_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    init = local_tmp / "init.pt"
    torch.save(CrnnNet().state_dict(), init)

    with pytest.raises(ValueError, match="data leakage audit failed"):
        trainer.train(index_dir, local_tmp / "fonts", init, local_tmp / "out",
                      epochs=2, samples_per_epoch=10, batch_size=5, num_workers=0)


def test_audit_rejects_nonzero_partition_pairwise_sha_overlap(local_tmp):
    index_dir, fonts, init = _make_audited_data(local_tmp)
    audit_path = index_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["content_checks"]["partition_pairwise_sha256_overlap"]["train_val"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="data leakage audit failed"):
        trainer.train(index_dir, fonts, init, local_tmp / "out",
                      epochs=2, samples_per_epoch=10, batch_size=5, num_workers=0)


def test_tiny_training_resume_matches_uninterrupted_and_records_hashes(local_tmp):
    index_dir, fonts, init = _make_audited_data(local_tmp)
    whole_out, split_out = local_tmp / "whole", local_tmp / "split"
    kwargs = dict(epochs=2, samples_per_epoch=10, batch_size=5, lr=.0003,
                  seed=42, num_workers=0)

    whole_result = trainer.train(index_dir, fonts, init, whole_out, **kwargs)
    first_part = trainer.train(index_dir, fonts, init, split_out, stop_after=1, **kwargs)
    assert first_part["epochs_completed"] == 1
    resumed = trainer.train(index_dir, fonts, init, split_out, resume=True, **kwargs)
    assert resumed["epochs_completed"] == 2
    assert resumed["best_epoch"] == whole_result["best_epoch"]
    assert resumed["best_score"] == whole_result["best_score"]

    whole_checkpoint = torch.load(whole_out / "checkpoint.pt", map_location="cpu", weights_only=True)
    split_checkpoint = torch.load(split_out / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert whole_checkpoint["epoch"] == split_checkpoint["epoch"] == 2
    _assert_tree_equal(whole_checkpoint["model"], split_checkpoint["model"])
    _assert_tree_equal(whole_checkpoint["optimizer"], split_checkpoint["optimizer"])
    _assert_tree_equal(whole_checkpoint["torch_rng"], split_checkpoint["torch_rng"])
    assert whole_checkpoint["history"][1]["actual_draw_counts"] == split_checkpoint["history"][1]["actual_draw_counts"]

    metadata = json.loads((whole_out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trainer_sha256"] == file_sha256(Path(trainer.__file__))
    assert metadata["helper_sha256"] == file_sha256(Path(trainer.__file__).with_name("train_crnn.py"))
    assert metadata["reader_sha256"] == file_sha256(
        Path(trainer.__file__).resolve().parents[2] / "lpr/recognize/crnn.py")
    assert metadata["initialization"]["sha256"] == file_sha256(init)
    assert metadata["mixture"] == trainer.MIXTURE
