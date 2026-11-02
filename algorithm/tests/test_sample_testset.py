import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from lpr.io_utils import imwrite
from lpr_tools.data.ccpd import INDEX_FIELDS
from lpr_tools.data.manifest import read_manifest
from lpr_tools.data.sample_testset import build_testset

PLATE = "陕A12345"


def make_index(path, *, short=False):
    rows = []
    def add(dataset, subset, count, **kw):
        for i in range(count):
            row = dict.fromkeys(INDEX_FIELDS, "")
            split = kw.get("split", "test")
            image = path.parent / dataset / subset / f"{split}-{i}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            imwrite(image, np.full((8, 16, 3), 127, dtype=np.uint8))
            row.update(path=str(image), dataset=dataset, subset=subset,
                       split="test", has_plate="1", plate_no=PLATE, plate_color="blue",
                       quad="1 2 3 2 3 4 1 4")
            row.update(kw)
            rows.append(row)
    add("ccpd2019", "ccpd_base", 119 if short else 121, split="val")
    add("ccpd2020", "ccpd_green", 30)
    add("ccpd2019", "ccpd_db", 50, brightness="10")
    add("ccpd2019", "ccpd_tilt", 25)
    add("ccpd2019", "ccpd_rotate", 25)
    add("ccpd2019", "ccpd_blur", 25)
    add("ccpd2019", "ccpd_weather", 25)
    add("ccpd2019", "ccpd_np", 20, has_plate="0", plate_no="", skip_reason="no_plate")
    add("ccpd2019", "ccpd_base", 20, split="train")
    add("ccpd2019", "ccpd_db", 1, brightness="20", split="val")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_quota_seed_leakage_and_skip_rules(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    out1, out2 = tmp_path / "one.csv", tmp_path / "two.csv"
    build_testset(index, out1)
    build_testset(index, out2)
    rows = read_manifest(out1)
    assert rows == read_manifest(out2)
    assert {cat: sum(r.category == cat for r in rows) for cat in ("clear", "night", "tilt", "dirty", "np")} == {
        "clear": 150, "night": 50, "tilt": 50, "dirty": 50, "np": 20}
    assert all(not Path(r.path).name.startswith('train-') for r in rows)
    assert sum(Path(r.path).name.startswith('val-') for r in rows) == 120
    assert all(r.category == 'clear' and r.source == 'ccpd_base'
               for r in rows if Path(r.path).name.startswith('val-'))
    assert all(r.note == "待人工确认" for r in rows if r.category == "dirty")
    assert all(not r.plate_no for r in rows if r.category == "np")


def test_undecodable_candidate_is_skipped_and_counted(tmp_path, capsys):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    records = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    broken = next(Path(row["path"]) for row in records if row["subset"] == "ccpd_base" and row["split"] == "val")
    broken.write_bytes(b"not an image")
    result = build_testset(index, tmp_path / "manifest.csv")
    assert len([row for row in result if row.category == "clear" and row.source == "ccpd_base"]) == 120
    assert str(broken.resolve()) not in {row.path for row in result}
    assert "undecodable_skipped=1" in capsys.readouterr().out


def test_default_shortfall_does_not_write_and_demo_reports_shortfall(tmp_path, capsys):
    index = tmp_path / "ccpd_index.csv"
    make_index(index, short=True)
    out = tmp_path / "manifest.csv"
    with pytest.raises(ValueError, match="clear_base.*shortfall 1"):
        build_testset(index, out)
    assert not out.exists()
    build_testset(index, out, allow_shortfall=True)
    assert "shortfall=clear_base:1" in capsys.readouterr().out


def test_local_labels_all_paths_and_optional_path_column(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    labels_dir = tmp_path / "raw" / "local"
    labels_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        imwrite(labels_dir / name, np.full((8, 16, 3), 127, dtype=np.uint8))
    labels = labels_dir / "labels.csv"
    labels.write_text("path,plate_no,plate_color\na.jpg,陕A12345,blue\nb.jpg,陕A12345,blue\n", encoding="utf-8")
    # Default convention derives raw/local from raw/index/ccpd_index.csv.
    derived_index_dir = tmp_path / "raw" / "index"
    derived_index_dir.mkdir()
    derived_index = derived_index_dir / "ccpd_index.csv"
    make_index(derived_index)
    rows = build_testset(derived_index, tmp_path / "manifest.csv")
    local = [r for r in rows if r.category == "local"]
    assert len(local) == 2
    assert {r.path for r in local} == {str((labels_dir / "a.jpg").resolve()), str((labels_dir / "b.jpg").resolve())}


def test_canonical_algorithm_data_index_derives_raw_local(tmp_path):
    data = tmp_path / "algorithm" / "data"
    index_dir = data / "index"
    index_dir.mkdir(parents=True)
    make_index(index_dir / "ccpd_index.csv")
    local_dir = data / "raw" / "local"
    local_dir.mkdir(parents=True)
    imwrite(local_dir / "local.jpg", np.full((8, 16, 3), 127, dtype=np.uint8))
    (local_dir / "labels.csv").write_text(
        "file,plate_no,plate_color\nlocal.jpg,陕A12345,blue\n", encoding="utf-8")
    rows = build_testset(index_dir / "ccpd_index.csv", tmp_path / "manifest.csv")
    local = [r for r in rows if r.category == "local"]
    assert len(local) == 1 and local[0].path == str((local_dir / "local.jpg").resolve())


def test_file_column_override_and_threshold(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    imwrite(tmp_path / "local.jpg", np.full((8, 16, 3), 127, dtype=np.uint8))
    labels = tmp_path / "my_labels.csv"
    labels.write_text("file,plate_no,plate_color\nlocal.jpg,陕A12345,blue\n", encoding="utf-8")
    rows = build_testset(index, tmp_path / "manifest.csv", local_labels=labels, brightness_threshold=10,
                         allow_shortfall=True)
    assert not any(r.category == "night" for r in rows)
    assert next(r for r in rows if r.category == "local").path == str((tmp_path / "local.jpg").resolve())


def test_duplicate_local_path_rejected_before_write(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    labels = tmp_path / "labels.csv"
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    green_path = next(row["path"] for row in rows if row["subset"] == "ccpd_green")
    labels.write_text(f"file,plate_no,plate_color\n{green_path},陕A12345,blue\n", encoding="utf-8")
    out = tmp_path / "manifest.csv"
    with pytest.raises(ValueError, match="duplicate"):
        build_testset(index, out, local_labels=labels)
    assert not out.exists()


def test_local_labels_cannot_reference_index_train_or_val(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    train_path = next(r["path"] for r in rows if r["split"] == "train")
    labels = tmp_path / "labels.csv"
    labels.write_text(f"file,plate_no,plate_color\n{train_path},陕A12345,blue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to index train split"):
        build_testset(index, tmp_path / "manifest.csv", local_labels=labels)


def test_relative_np_path_resolves_beside_index(tmp_path):
    index_dir = tmp_path / "raw" / "index"
    index_dir.mkdir(parents=True)
    index = index_dir / "ccpd_index.csv"
    make_index(index)
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    relative_np_dir = tmp_path / "raw" / "ccpd2019" / "ccpd_np"
    relative_np_dir.mkdir(parents=True)
    for row in rows:
        if row["subset"] == "ccpd_np":
            shutil.copyfile(row["path"], relative_np_dir / Path(row["path"]).name)
            row["path"] = "../ccpd2019/ccpd_np/" + Path(row["path"]).name
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = build_testset(index, tmp_path / "manifest.csv")
    no_plates = [r for r in result if r.category == "np"]
    assert len(no_plates) == 20
    assert all(r.path.startswith(str((index_dir / ".." / "ccpd2019").resolve())) for r in no_plates)


def test_same_image_assigned_to_test_and_train_is_rejected(tmp_path):
    index = tmp_path / "ccpd_index.csv"
    make_index(index)
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    train = next(r for r in rows if r["split"] == "train")
    train["path"] = next(r["path"] for r in rows if r["split"] == "val" and r["subset"] == "ccpd_base")
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="multiple splits"):
        build_testset(index, tmp_path / "manifest.csv")


def review_csv(index, path):
    with index.open(encoding='utf-8', newline='') as handle:
        dirty = [r for r in csv.DictReader(handle) if r['subset'] in {'ccpd_blur', 'ccpd_weather'}]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'note'])
        writer.writeheader()
        writer.writerows({'path': r['path'], 'note': 'AI 目检，待tonyswift868复核；测试用遮挡标注'} for r in dirty)


def test_reviewed_dirty_exact_selection_and_note(tmp_path):
    index = tmp_path/'index.csv'
    make_index(index)
    review = tmp_path/'review.csv'
    review_csv(index, review)
    result = build_testset(index, tmp_path/'manifest.csv', dirty_review=review)
    dirty = [r for r in result if r.category == 'dirty']
    assert len(dirty) == 50
    assert all('AI 目检，待tonyswift868复核' in r.note for r in dirty)
    assert {Path(r.path).parent.name for r in dirty} == {'ccpd_blur', 'ccpd_weather'}


@pytest.mark.parametrize('problem', ['duplicate', 'outside', 'note', 'shortfall'])
def test_bad_review_does_not_write(tmp_path, problem):
    index = tmp_path/'index.csv'
    make_index(index)
    review = tmp_path/'review.csv'
    review_csv(index, review)
    with review.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if problem == 'duplicate':
        rows[1] = rows[0].copy()
    elif problem == 'outside':
        rows[0]['path'] = '/data/not-reviewed.jpg'
    elif problem == 'note':
        rows[0]['note'] = 'random choice'
    else:
        rows.pop()
    with review.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path','note'])
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path/'manifest.csv'
    with pytest.raises(ValueError):
        build_testset(index, out, dirty_review=review)
    assert not out.exists()


def test_formal_cli_requires_review(tmp_path):
    from lpr_tools.data.sample_testset import main
    with pytest.raises(SystemExit) as exc:
        main(['--index',str(tmp_path/'absent.csv'),'--out',str(tmp_path/'manifest.csv')])
    assert exc.value.code == 2
    assert not (tmp_path/'manifest.csv').exists()
