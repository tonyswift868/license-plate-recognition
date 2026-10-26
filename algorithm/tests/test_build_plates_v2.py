from types import SimpleNamespace
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lpr.types import PlateCandidate
from lpr_tools.data import build_plates_v2
from lpr_tools.data.manifest import ManifestRow, write_manifest


def test_source_group_accepts_v1_and_subset_tagged_hard_rows():
    assert build_plates_v2._source_group("ground_truth") == "ground_truth"
    assert build_plates_v2._source_group("pipeline_candidate:2:1") == "pipeline_candidate"
    assert build_plates_v2._source_group("hard_train:blur:ground_truth") == "hard_train:blur"
    assert build_plates_v2._source_group("hard_train:rotate:pipeline:net") == "hard_train:rotate"


def test_exclusion_union_covers_all_four_lists_paths_and_byte_copies(tmp_path, monkeypatch):
    import lpr_tools.data.build_plates_v2 as builder

    frozen_dir = tmp_path / "golden"
    frozen_dir.mkdir()
    source = tmp_path / "frozen.jpg"
    source.write_bytes(b"same-frozen-image-bytes")
    same_bytes = tmp_path / "copy.jpg"
    same_bytes.write_bytes(source.read_bytes())
    paths = [tmp_path / f"list-{i}.csv" for i in range(3)]
    row = ManifestRow(str(source), "皖A12345", "blue", "clear", "fixture", None, "")
    for path in paths:
        write_manifest(path, [row])
    (frozen_dir / "golden.jpg").write_bytes(source.read_bytes())
    undecodable = tmp_path / "undecodable.csv"
    undecodable.write_text("path,dataset,subset,split,reason\n" + str(source) + ",x,y,test,bad\n", encoding="utf-8")
    monkeypatch.setattr(builder, "FROZEN", [*paths, undecodable])
    monkeypatch.setattr(builder, "GOLDEN", frozen_dir)
    blocked_paths, blocked_hashes = builder._exclusions()
    assert builder._key(source) in blocked_paths
    assert builder._key(same_bytes) not in blocked_paths
    assert (len(source.read_bytes()), hashlib.sha256(source.read_bytes()).hexdigest()) in blocked_hashes


def test_missing_frozen_input_fails_closed(tmp_path, monkeypatch):
    import lpr_tools.data.build_plates_v2 as builder
    monkeypatch.setattr(builder, "FROZEN", [tmp_path / "missing.csv"])
    with pytest.raises(FileNotFoundError, match="required exclusion list missing"):
        builder._exclusions()


def _candidate(source: str, x: int) -> PlateCandidate:
    return PlateCandidate([(x, 0), (x + 100, 0), (x + 100, 40), (x, 40)],
                          (x, 0, 100, 40), "blue", .9, source)


def test_pipeline_crop_uses_network_first_fused_iou_candidate(monkeypatch):
    net = _candidate("net", 10)
    classic = _candidate("edge", 12)
    calls = []
    class Detector:
        def detect(self, image, max_candidates):
            calls.append((image, max_candidates))
            return [net]
    monkeypatch.setattr(build_plates_v2, "preprocess", lambda image, cfg, roi: object())
    monkeypatch.setattr(build_plates_v2, "locate", lambda pre, cfg, max_candidates: [classic])
    monkeypatch.setattr(build_plates_v2, "rectify_candidates", lambda image, candidate, cfg: [
        SimpleNamespace(layout=7, image=np.zeros((80, 252, 3), np.uint8))])
    crop, evidence = build_plates_v2._pipeline_crop(
        np.zeros((50, 150, 3), np.uint8), {"bbox": "10 0 100 40", "layout": "7"},
        SimpleNamespace(preprocess=1, locate=2, runtime=SimpleNamespace(max_candidates=3), rectify=4,
                        lane_roi=lambda lane: None), Detector())
    assert len(calls) == 1
    assert crop[1] is net
    assert crop[2] == 1.0
    assert evidence == {"corner_detect_calls": 1, "net_candidates": 1, "fused_candidates": 1}


def test_pipeline_crop_reports_missing_iou_without_fabricating_crop(monkeypatch):
    classic = _candidate("edge", 0)
    class Detector:
        def detect(self, image, max_candidates):
            return []
    monkeypatch.setattr(build_plates_v2, "preprocess", lambda image, cfg, roi: object())
    monkeypatch.setattr(build_plates_v2, "locate", lambda pre, cfg, max_candidates: [classic])
    monkeypatch.setattr(build_plates_v2, "rectify_candidates", lambda *args: (_ for _ in ()).throw(AssertionError()))
    crop, evidence = build_plates_v2._pipeline_crop(
        np.zeros((50, 150, 3), np.uint8), {"bbox": "200 0 100 40", "layout": "7"},
        SimpleNamespace(preprocess=1, locate=2, runtime=SimpleNamespace(max_candidates=3), rectify=4,
                        lane_roi=lambda lane: None), Detector())
    assert crop is None
    assert evidence == {"corner_detect_calls": 1, "net_candidates": 0, "fused_candidates": 1}


def test_pipeline_crop_counts_network_when_rectification_fails(monkeypatch):
    net = _candidate("net", 10)
    class Detector:
        def detect(self, image, max_candidates):
            return [net]
    monkeypatch.setattr(build_plates_v2, "preprocess", lambda image, cfg, roi: object())
    monkeypatch.setattr(build_plates_v2, "locate", lambda pre, cfg, max_candidates: [])
    monkeypatch.setattr(build_plates_v2, "rectify_candidates", lambda *args: (_ for _ in ()).throw(ValueError("bad quad")))
    crop, evidence = build_plates_v2._pipeline_crop(
        np.zeros((50, 150, 3), np.uint8), {"bbox": "10 0 100 40", "layout": "7"},
        SimpleNamespace(preprocess=1, locate=2, runtime=SimpleNamespace(max_candidates=3), rectify=4,
                        lane_roi=lambda lane: None), Detector())
    assert crop is None
    assert evidence["corner_detect_calls"] == 1
    assert evidence["selected_candidate_source"] == "net"
    assert "bad quad" in evidence["error"]


def test_hard_worker_keeps_gt_when_pipeline_crop_write_fails(tmp_path, monkeypatch):
    import lpr_tools.data.build_plates_v2 as builder
    source = tmp_path / "hard.jpg"
    source.write_bytes(b"source")
    size, digest = builder._sha(source)
    cfg = SimpleNamespace(runtime=SimpleNamespace(corner_model_path="corner.pt", max_candidates=3),
                          resolve_path=lambda _: tmp_path / "corner.pt")
    monkeypatch.setattr(builder, "_CFG", cfg)
    monkeypatch.setattr(builder, "_CORNER", object())
    monkeypatch.setattr(builder, "decode_image", lambda blob: np.zeros((20, 30, 3), np.uint8))
    monkeypatch.setattr(builder, "_warp_gt", lambda *args: np.zeros((80, 252, 3), np.uint8))
    monkeypatch.setattr(builder, "_pipeline_crop", lambda *args: (
        (np.zeros((80, 252, 3), np.uint8), _candidate("net", 1), .8),
        {"corner_detect_calls": 1, "net_candidates": 1, "fused_candidates": 1}))
    writes = []
    def write(path, image, params):
        writes.append(Path(path))
        if str(path).endswith("_pipe.jpg"):
            raise OSError("fixture candidate write failure")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"gt crop")
    monkeypatch.setattr(builder, "imwrite", write)
    row = {"path": str(source), "split": "test", "subset": "ccpd_blur", "plate_no": "皖A12345",
           "layout": "7", "plate_color": "blue", "quad": "1 1 20 1 20 10 1 10", "bbox": "1 1 20 10"}
    out = tmp_path / "v2"
    result = builder._hard_one((row, str(out), "unused.toml", (size, digest)))
    assert result["status"] == "pipeline_missing"
    assert len(result["labels"]) == 1 and result["labels"][0][4] == "hard_train:blur:ground_truth"
    assert writes[0].is_file() and not writes[1].exists()


def test_tiny_v2_build_filters_frozen_copies_and_keeps_val_from_v1(tmp_path, monkeypatch):
    import lpr_tools.data.build_plates_v2 as builder

    v1, target = tmp_path / "v1", tmp_path / "v2"
    (v1 / "train").mkdir(parents=True); (v1 / "val").mkdir()
    frozen_original = tmp_path / "frozen-original.jpg"; frozen_original.write_bytes(b"devhard-image")
    val_original = tmp_path / "val-original.jpg"; val_original.write_bytes(b"ordinary val image")
    rows = [
        ["train/frozen-gt.jpg", "皖A12345", "7", "blue", "ground_truth", str(frozen_original)],
        ["train/frozen-pipe.jpg", "皖A12345", "7", "blue", "pipeline_candidate:0:0", str(frozen_original)],
        ["val/ordinary-gt.jpg", "皖A54321", "7", "blue", "ground_truth", str(val_original)],
    ]
    with (v1 / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(builder.FIELDS); writer.writerows(rows)
    for row in rows:
        (v1 / row[0]).write_bytes(b"crop")
    source_ledger = {str(frozen_original): {"sha256": builder._sha(frozen_original)[1], "split": "train"},
                     str(val_original): {"sha256": builder._sha(val_original)[1], "split": "val"}}
    (v1 / "source_hashes.json").write_text(json.dumps(source_ledger), encoding="utf-8")

    duplicate_copy = tmp_path / "hard-duplicate.jpg"; duplicate_copy.write_bytes(frozen_original.read_bytes())
    excluded_path = tmp_path / "frozen-path.jpg"; excluded_path.write_bytes(b"path only exclusion")
    allowed = tmp_path / "hard-paired.jpg"; allowed.write_bytes(b"unique hard source")
    allowed_missing = tmp_path / "hard-gt-only.jpg"; allowed_missing.write_bytes(b"unique source with no pipeline warp")
    def sample(path):
        return {"path": str(path), "split": "test", "subset": "ccpd_blur", "has_plate": "1", "skip_reason": "",
                "plate_no": "皖A12345", "layout": "7", "plate_color": "blue",
                "quad": "1 1 20 1 20 10 1 10", "bbox": "1 1 20 10"}
    index_rows = [sample(p) for p in (allowed, allowed_missing, duplicate_copy, excluded_path)]
    monkeypatch.setattr(builder, "read_index", lambda _: index_rows)
    monkeypatch.setattr(builder, "_exclusions", lambda: ({builder._key(excluded_path)}, {builder._sha(frozen_original)}))
    cfg_path = tmp_path / "corner.toml"; cfg_path.write_text("corner_model_path", encoding="utf-8")
    model = tmp_path / "corner_net_v1.pt"; model.write_bytes(b"corner model")
    cfg = SimpleNamespace(runtime=SimpleNamespace(corner_model_path="models/corner_net_v1.pt", max_candidates=3),
                          resolve_path=lambda _: model)
    monkeypatch.setattr(builder, "load_config", lambda _: cfg)
    monkeypatch.setattr(builder.os, "link", lambda src, dst: shutil.copyfile(src, dst))

    class InlinePool:
        def __init__(self, max_workers, initializer, initargs):
            assert max_workers == 2
            assert initializer is builder._init
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def map(self, func, jobs, chunksize):
            assert func is builder._hard_one and chunksize == 8
            for row, out, config, expected in jobs:
                digest = expected[1]; key = hashlib.sha1(row["path"].casefold().encode()).hexdigest()
                subset = row["subset"].removeprefix("ccpd_")
                gt = f"train/hard_{key}_gt.jpg"; pipe = f"train/hard_{key}_pipe.jpg"
                (Path(out) / gt).write_bytes(b"gt crop")
                gt_label = [gt, row["plate_no"], row["layout"], row["plate_color"], f"hard_train:{subset}:ground_truth", row["path"]]
                if Path(row["path"]) == allowed_missing:
                    yield {"status":"pipeline_missing","path":row["path"],"sha256":digest,"subset":subset,
                           "corner_detect_calls":1,"reason":"no IoU candidate","labels":[gt_label]}
                else:
                    (Path(out) / pipe).write_bytes(b"pipeline crop")
                    pipe_label = [pipe, row["plate_no"], row["layout"], row["plate_color"], f"hard_train:{subset}:pipeline:net", row["path"]]
                    yield {"status":"ok","path":row["path"],"sha256":digest,"subset":subset,
                           "corner_detect_calls":1,"candidate_source":"net","labels":[gt_label,pipe_label]}
    monkeypatch.setattr(builder, "ProcessPoolExecutor", InlinePool)

    stats = builder.build(index=tmp_path / "index.csv", v1=v1, target=target,
                          receipt=tmp_path / "receipt", workers=2, config=cfg_path)
    with (target / "labels.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); output_rows = list(reader)
        assert tuple(reader.fieldnames) == builder.FIELDS
    assert stats["v1_frozen_overlaps_excluded"] == 2
    assert stats["hard_sources_accepted"] == 2 and stats["hard_sources_paired"] == 1
    assert stats["pipeline_missing"] == 1
    assert stats["excluded"] == {"frozen_bytes": 1, "frozen_path": 1}
    assert len(output_rows) == 4
    assert all(row["file"].startswith("val/") and not row["source"].startswith("hard_train:")
               for row in output_rows if row["file"].startswith("val/"))
    assert not any(row["orig_path"] in {str(frozen_original), str(duplicate_copy), str(excluded_path)}
                   for row in output_rows)
    assert all((target / row["file"]).is_file() for row in output_rows)
    audit = (tmp_path / "receipt" / "hard_source_audit.jsonl").read_text(encoding="utf-8")
    assert '"status": "pipeline_missing"' in audit and "no IoU candidate" in audit
