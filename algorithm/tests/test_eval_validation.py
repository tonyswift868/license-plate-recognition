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
from lpr_tools.eval.index_samples import read_samples, sample_validation
from lpr_tools.eval.eval_accuracy import evaluate, main
import lpr_tools.eval.eval_accuracy as ev
import lpr_tools.eval.locate_recall as lr


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
    out = tmp_path/"run"
    assert main(["--split","val","--index",str(index),"--n","8","--uniform",
                 "--locate-recall","--out",str(out),"--scoreboard",str(score)]) == 0
    report = json.loads((out/"report.json").read_text(encoding="utf-8"))
    assert report["mode"].endswith("uniform") and report["sampling"]["selected_n"] == 8
    assert report["locate_recall"]["overall"]["total"] == 6
    assert all("/val/" in row.path for row in read_manifest(out/"validation_manifest.csv"))
    score_line = score.read_text(encoding="utf-8").splitlines()[-1]
    fields = [field.strip() for field in score_line.split("|")[1:-1]]
    assert len(fields) == 12 and fields[2] == "uniform" and fields[8] == "100.00%"
    assert fields[1] != "unknown" and fields[0].count("-") == 2
    assert "Top-3 locate recall" in capsys.readouterr().out


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
    for argv in (["--split","val","--index",str(tmp_path/"missing.csv")],
                 ["--split","val","--index",str(index),"--n","0"],
                 ["--split","val","--index",str(index),"--scoreboard","algorithm/reports/scoreboard.md"]):
        with pytest.raises(SystemExit) as exc:main(argv)
        assert exc.value.code == 2
