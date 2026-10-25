"""Validation sampling and localization metric contracts, using independent fixtures."""
import csv
import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpr.io_utils import imwrite
from lpr.testing import FakeRecognizer
from lpr_tools.data.ccpd import INDEX_FIELDS
from lpr_tools.data.manifest import read_manifest, write_manifest
from lpr_tools.eval.index_samples import path_key, read_samples, sample_dev_hard, sample_validation
from lpr_tools.eval.eval_accuracy import evaluate, main
import lpr_tools.eval.eval_accuracy as ev
import lpr_tools.eval.locate_recall as lr
import lpr_tools.eval.index_samples as index_samples_module


@pytest.fixture(autouse=True)
def isolate_eval_defaults(tmp_path, monkeypatch):
    """Fence evaluator discovery and reject real model/index access in tests."""
    algo = tmp_path / "algorithm"
    golden = algo / "tests" / "data" / "golden" / "golden_manifest.csv"
    golden.parent.mkdir(parents=True)
    golden.write_text("path,plate_no,plate_color,category,source,quad,note\n", encoding="utf-8")
    monkeypatch.setattr(ev, "ALGORITHM_DIR", algo)
    make_recognizer = ev.make_recognizer

    def guarded_make_recognizer(args, *args_, **kwargs):
        assert args.fake or args.uniform, "unit tests must never load the configured real model"
        return make_recognizer(args, *args_, **kwargs)

    monkeypatch.setattr(ev, "make_recognizer", guarded_make_recognizer)
    read_samples_impl = index_samples_module.read_samples

    def guarded_read_samples(index_path):
        index_path = Path(index_path).resolve()
        assert index_path.is_relative_to(tmp_path.resolve()), \
            f"unit test attempted to read a non-temporary index: {index_path}"
        return read_samples_impl(index_path)

    monkeypatch.setattr(index_samples_module, "read_samples", guarded_read_samples)
    monkeypatch.setattr(ev, "read_samples", guarded_read_samples)


def index_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [("ccpd2019", "ccpd_base", "val", 5),
             ("ccpd2019", "ccpd_db", "val", 3),
             ("ccpd2020", "ccpd_green", "val", 2),
             ("ccpd2019", "ccpd_np", "val", 2),
             ("ccpd2019", "ccpd_base", "train", 3),
             ("ccpd2019", "ccpd_base", "test", 3)]
    for dataset, subset, split, count in specs:
        for i in range(count):
            relative = f"images/{dataset}/{subset}/{split}/{i}.png"
            image = np.full((240,400,3), 90, np.uint8)
            np_sample = subset == "ccpd_np"
            if not np_sample:
                cv2.rectangle(image, (120,150), (260,185), (170,70,10), -1)
                cv2.putText(image, "A12345", (125,177), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255),2)
            imwrite(root/relative, image)
            row = dict.fromkeys(INDEX_FIELDS, "")
            row.update(path=relative, dataset=dataset, subset=subset, split=split,
                       has_plate="0" if np_sample else "1",
                       plate_no="" if np_sample else "陕AD12345" if dataset == "ccpd2020" else "陕A12345",
                       layout="" if np_sample else "8" if dataset == "ccpd2020" else "7",
                       plate_color="green" if dataset == "ccpd2020" else "blue",
                       bbox="120 150 140 35" if not np_sample else "",
                       quad="120 150 260 150 260 185 120 185" if not np_sample else "",
                       brightness="20", skip_reason="no_plate" if np_sample else "")
            rows.append(row)
    index = root/"ccpd_index.csv"
    save_index(index, rows)
    return index, rows


def save_index(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader(); writer.writerows(rows)


def test_balanced_val_sampling_stable_and_without_leakage(tmp_path):
    index, raw = index_fixture(tmp_path)
    samples, metadata = sample_validation(index, n=8)
    assert len(samples) == 8 and all(s.split == "val" for s in samples)
    assert Counter(s.stratum for s in samples) == {
        "ccpd2019/ccpd_base":2, "ccpd2019/ccpd_db":2,
        "ccpd2020/ccpd_green":2, "ccpd2019/ccpd_np":2}
    assert samples == sample_validation(index, n=8)[0]
    save_index(index, list(reversed(raw)))
    assert samples == sample_validation(index, n=8)[0]
    assert len({s.row.path for s in samples}) == 8
    assert all(Path(s.row.path).is_absolute() for s in samples)
    assert metadata["selected_n"] == 8 and metadata["available_n"] == 12
    assert len(sample_validation(index, n=999)[0]) == 12
    assert sample_validation(index, n=8, seed=41)[0] != samples
    with pytest.raises(ValueError, match="positive"):sample_validation(index, n=0)
    save_index(index, [r for r in raw if r["split"] != "val"])
    with pytest.raises(ValueError, match="no supported val"):sample_validation(index)


def test_val_exclusion_and_dev_hard_are_stratified(tmp_path):
    index, rows = index_fixture(tmp_path)
    hard = {"ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate",
            "ccpd_fn", "ccpd_challenge", "ccpd_weather"}
    base = next(r for r in rows if r["subset"] == "ccpd_db" and r["split"] == "val")
    added = []
    for subset in sorted(hard):
        for i in range(3):
            row = dict(base, subset=subset, split="test", path=f"images/hard/{subset}/{i}.png")
            imwrite(tmp_path/row["path"], np.zeros((240,400,3),np.uint8))
            added.append(row)
    save_index(index, rows + added)
    excluded = tmp_path/"frozen.csv"
    from lpr_tools.data.manifest import ManifestRow, write_manifest
    aliased = tmp_path/"unused"/".."/added[0]["path"]
    write_manifest(excluded,[ManifestRow(str(aliased),"陕A12345","blue","night","test")])
    selected, info = sample_dev_hard(index,n=2,seed=42,exclude_manifest=excluded)
    assert len(selected)==14
    assert set(info["selected_by_subset"].values())=={2}
    assert info["split"] == "dev-hard/index-splits"
    assert all(s.subset in hard for s in selected)
    assert all(s.split == "test" for s in selected if s.subset != "ccpd_db")
    assert all(info["index_splits_by_subset"][subset].get("test", 0) >= 2
               for subset in hard - {"ccpd_db"})
    assert path_key(tmp_path/added[0]["path"]) not in {path_key(s.row.path) for s in selected}
    base_val = next(r for r in rows if r["subset"] == "ccpd_base" and r["split"] == "val")
    frozen = tmp_path/"clear-test.csv"
    write_manifest(frozen,[ManifestRow(base_val["path"],"陕A12345","blue","clear","test")])
    clear, _ = sample_validation(index,n=99,seed=42,exclude_manifest=frozen,subsets={"ccpd_base"})
    assert clear and {s.subset for s in clear} == {"ccpd_base"}
    assert path_key(tmp_path/base_val["path"]) not in {path_key(s.row.path) for s in clear}


def test_exclusion_manifest_union_deduplicates_canonical_aliases(tmp_path):
    index, rows = index_fixture(tmp_path/"input")
    from lpr_tools.data.manifest import ManifestRow, write_manifest
    base = tmp_path/"input"
    first, second = tmp_path/"frozen-a.csv", tmp_path/"frozen-b.csv"
    base_rows = [r for r in rows if r["subset"] == "ccpd_base" and r["split"] == "val"]
    canonical = base/base_rows[0]["path"]
    alias = base/"unused"/".."/base_rows[0]["path"]
    second_path = base/base_rows[1]["path"]
    write_manifest(first,[ManifestRow(str(canonical),"陕A12345","blue","clear","test")])
    write_manifest(second,[ManifestRow(str(alias),"陕A12345","blue","clear","test"),
                           ManifestRow(str(second_path),"陕A12345","blue","clear","test")])
    selected, metadata = sample_validation(index,n=99,exclude_manifest=[first,second],
                                           subsets={"ccpd_base"})
    assert len(selected)==3
    assert metadata["excluded_n"]==2
    assert metadata["exclusion_sources"]==[str(first.resolve()),str(second.resolve())]
    remaining={path_key(sample.row.path) for sample in selected}
    assert path_key(canonical) not in remaining
    assert path_key(second_path) not in remaining


def test_duplicate_path_split_alias_and_skipped_labels(tmp_path):
    index, rows = index_fixture(tmp_path)
    bad = dict(rows[0], path="images/skipped.png", skip_reason="placeholder")
    save_index(index, rows+[bad])
    assert len(read_samples(index)) == len(rows)
    alias = dict(rows[0], path="images/../"+rows[0]["path"], split="train")
    save_index(index, rows+[alias])
    with pytest.raises(ValueError, match="conflicting index path"):sample_validation(index)
    save_index(index, [dict(rows[0], bbox="1 2 -3 4")])
    with pytest.raises(ValueError, match="bbox"):read_samples(index)
    save_index(index, [dict(rows[0], plate_color="ultraviolet")])
    with pytest.raises(ValueError, match="plate color"):read_samples(index)


def test_top3_iou_boundary_and_original_coordinates(cfg, monkeypatch):
    monkeypatch.setattr(lr, "preprocess", lambda image, config: "preprocessed")
    truth = (300,150,100,25)
    seen = []
    boxes = [(0,0,10,10), (10,10,10,10), truth, truth]
    def locate(pre, config, max_candidates):
        seen.append((pre, max_candidates))
        return [SimpleNamespace(bbox=box) for box in boxes]
    monkeypatch.setattr(lr, "locate", locate)
    assert lr.locate_match(None, truth, cfg) == {"hit": True, "best_iou":1., "candidates":3}
    boxes[:] = [(0,0,10,10)]*3 + [truth]
    assert lr.locate_match(None, truth, cfg)["hit"] is False
    boxes[:] = [(300,150,50,25)]
    assert lr.locate_match(None, truth, cfg)["best_iou"] == .5
    assert lr.locate_match(None, truth, cfg)["hit"] is True
    boxes[:] = [(300,150,49,25)]
    assert lr.locate_match(None, truth, cfg)["hit"] is False
    assert all(call == ("preprocessed",3) for call in seen)
    assert lr.summarize([], excluded_np=2)["overall"]["recall"] is None


def test_evaluate_location_uses_index_excludes_np_and_not_fake_bbox(tmp_path, monkeypatch):
    index, _ = index_fixture(tmp_path)
    selected, sampling = sample_validation(index, n=4)
    manifest = tmp_path/"validation.csv"
    write_manifest(manifest, [s.row for s in selected], check_files=True)
    seen = []
    def match(image, bbox, config):
        seen.append(bbox)
        return {"hit":True, "best_iou":1., "candidates":3}
    monkeypatch.setattr(ev, "locate_match", match)
    report = evaluate(manifest, FakeRecognizer(), tmp_path/"out", index=index,
                      locate_recall=True, sampling=sampling)
    assert len(seen) == 3 and all(b == (120.,150.,140.,35.) for b in seen)
    assert report["locate_recall"]["overall"] == {"hits":3,"total":3,"recall":1.}
    assert report["locate_recall"]["excluded_np"] == 1
    assert len(report["subsets"]) == 4
    assert len(report["locate_recall"]["subsets"]) == 3
    assert json.loads((tmp_path/"out/report.json").read_text(encoding="utf-8")) == report


def test_cli_val_uniform_sampling_and_scoreboard(tmp_path, monkeypatch, capsys):
    index, _ = index_fixture(tmp_path/"input")
    monkeypatch.setattr(ev,"locate_match",lambda *args:{"hit":True,"best_iou":1.,"candidates":3})
    score = tmp_path/"scoreboard-copy.md"
    score.write_text("temporary copy\n",encoding="utf-8")
    exclude = tmp_path/"exclude.csv"
    write_manifest(exclude, [])
    golden_exclude = tmp_path/"algo"/"tests"/"data"/"golden"/"golden_manifest.csv"
    write_manifest(golden_exclude, [])
    monkeypatch.setattr(ev,"ALGORITHM_DIR",tmp_path/"algo")
    out = tmp_path/"run"
    assert main(["--split","val","--index",str(index),"--n","8","--uniform",
                 "--exclude-manifest",str(exclude),
                 "--locate-recall","--out",str(out),"--scoreboard",str(score)]) == 0
    report = json.loads((out/"report.json").read_text(encoding="utf-8"))
    assert report["mode"].endswith("uniform") and report["sampling"]["selected_n"] == 8
    assert report["sampling"]["exclusion_sources"] == [str(exclude.resolve()),str(golden_exclude.resolve())]
    assert report["locate_recall"]["overall"]["total"] == 6
    assert all("/val/" in row.path for row in read_manifest(out/"validation_manifest.csv"))
    score_line = score.read_text(encoding="utf-8").splitlines()[-1]
    fields = [field.strip() for field in score_line.split("|")[1:-1]]
    assert len(fields) == 12 and fields[2] == "uniform" and fields[8] == "100.00%"
    assert fields[1] != "unknown" and fields[0].count("-") == 2
    assert "Top-3 locate recall" in capsys.readouterr().out


def test_locate_only_does_not_create_recognizer(tmp_path, monkeypatch, capsys):
    index, _ = index_fixture(tmp_path/"input")
    exclude = tmp_path/"exclude.csv"
    write_manifest(exclude, [])
    monkeypatch.setattr(ev,"make_recognizer",lambda *args,**kwargs: (_ for _ in ()).throw(
        AssertionError("locate-only must not load a recognizer")))
    monkeypatch.setattr(ev,"read_samples",lambda *args,**kwargs: (_ for _ in ()).throw(
        AssertionError("CLI-sampled truth must be reused without rereading the full index")))
    monkeypatch.setattr(ev,"locate_match",lambda *args:{"hit":True,"best_iou":1.,"candidates":3})
    out = tmp_path/"locate"
    assert main(["--split","val","--index",str(index),"--n","4","--exclude-manifest",str(exclude),
                 "--locate-only","--out",str(out)]) == 0
    report=json.loads((out/"report.json").read_text(encoding="utf-8"))
    assert report["mode"]=="locate-only" and report["locate_recall"]["overall"]["total"]==3
    assert "mode=locate-only" in capsys.readouterr().out


def test_val_cli_fails_closed_when_exclusion_is_missing(tmp_path, capsys):
    index, _ = index_fixture(tmp_path/"input")
    with pytest.raises(SystemExit) as exc:
        main(["--split","val","--index",str(index),"--locate-only","--exclude-manifest",
              str(tmp_path/"missing.csv"),"--out",str(tmp_path/"out")])
    assert exc.value.code == 2
    assert not (tmp_path/"out").exists()
    assert "validation exclusion manifest not found" in capsys.readouterr().err


def test_missing_gt_and_cli_input_errors(tmp_path, capsys):
    index, _ = index_fixture(tmp_path)
    selected, _ = sample_validation(index,n=4)
    orphan = tmp_path/"orphan.png"; imwrite(orphan,np.zeros((240,400,3),np.uint8))
    manifest = tmp_path/"orphan.csv"
    plated = next(s.row for s in selected if s.row.category != "np")
    write_manifest(manifest,[replace(plated,path=str(orphan))])
    with pytest.raises(ValueError,match="ground truth"):
        evaluate(manifest,FakeRecognizer(),tmp_path/"out",index=index,locate_recall=True)
    assert not (tmp_path/"out").exists()
    exclude = tmp_path/"exclude.csv"
    write_manifest(exclude, [])
    for argv in (["--split","val","--index",str(tmp_path/"missing.csv"),
                  "--exclude-manifest",str(exclude)],
                 ["--split","val","--index",str(index),"--n","0","--exclude-manifest",str(exclude)],
                 ["--split","val","--index",str(index),"--exclude-manifest",str(exclude),
                  "--scoreboard",str(tmp_path/"algorithm"/"reports"/"scoreboard.md")]):
        with pytest.raises(SystemExit) as exc:main(argv)
        assert exc.value.code == 2
