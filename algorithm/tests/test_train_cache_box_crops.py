from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import uuid

import numpy as np
import pytest

from lpr_tools.train import cache_box_crops as cache
from lpr_tools.train.box_data import sha256_file

TEST_TMP = Path(__file__).resolve().parents[1] / "out" / "test_tmp"


@pytest.fixture
def local_tmp():
    root = TEST_TMP.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("cache_box_crops_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"refusing to remove test path outside {root}: {resolved}")
        shutil.rmtree(resolved)


def test_prepare_cache_writes_original_resolution_index_and_reuses_same_pool(local_tmp, monkeypatch):
    tmp_path = local_tmp
    pool = tmp_path / "pool.csv"
    with pool.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "subset", "brightness",
                                                    "sha256", "quad", "source"), lineterminator="\n")
        writer.writeheader()
        writer.writerow({"path": "synthetic-source.jpg", "label": "皖A12345", "subset": "db",
                         "brightness": "10", "sha256": "a" * 64,
                         "quad": "2 2 8 2 8 6 2 6", "source": "test"})
    monkeypatch.setattr(cache, "_read_image", lambda _path: np.full((10, 12, 3), 127, dtype=np.uint8))
    target = tmp_path / "cache"
    report = cache.prepare_cache(pool, target, quality=70, limit=1_000_000)
    assert report["samples"] == 1
    assert report["quality"] == 70
    assert report["pool_sha256"] == sha256_file(pool)
    assert report["cache_index_sha256"] == sha256_file(target / "index.csv")
    with (target / "index.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["cache_file"].endswith("a" * 64 + ".jpg")
    assert Path(row["cache_file"]).is_file()
    assert json.loads((target / "audit.json").read_text(encoding="utf-8"))["gt_bbox_pad"] == [0.5, 0.5]
    # Existing audited files are reused; no second image decode occurs.
    assert cache.prepare_cache(pool, target, quality=70) == report

