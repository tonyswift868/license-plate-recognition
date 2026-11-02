"""团队六字段适配、并发初始化与提交快照导出的回归测试。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from lpr import pipeline
from lpr.errors import ModelLoadError, failure_result
from lpr.types import RecogStatus
from lpr_tools import team_export
from pydantic import BaseModel

GOLDEN_NAME = "ccpd2019_ccpd_base_0110201149425-92_87-285&529_479&589-485&598_279&587_277&524_483&535-0_0_5_24_17_33_33-105-19.jpg"
GOLDEN_PLATE = "皖AF0T99"


class BackendRecognitionResult(BaseModel):
    """与团队 backend/app/schemas/recognition.py 的六字段一致。"""
    plate_no: str
    confidence: float = 0.0
    image_path: str | None = None
    stage_ms: dict[str, int] = {}
    success: bool = True
    message: str | None = None


@pytest.fixture(autouse=True)
def isolated_singleton(monkeypatch):
    monkeypatch.setattr(pipeline, "_backend_recognizer", None)


@pytest.fixture
def image_bytes():
    return cv2.imencode(".png", np.zeros((32, 64, 3), np.uint8))[1].tobytes()


@pytest.mark.parametrize("status", list(RecogStatus))
def test_six_fields_status_types_and_rounding(monkeypatch, image_bytes, status):
    result = failure_result(status)
    result.plate_no = "皖A12345"
    result.confidence = 0.9
    result.stage_ms = {"locate": 1.49, "reader": 2.5}
    lanes = []

    def recognize(image, *, lane_id):
        lanes.append(lane_id)
        assert image.shape == (32, 64, 3)
        return result

    monkeypatch.setattr(pipeline.PlateRecognizer, "from_config", lambda: SimpleNamespace(recognize=recognize))
    actual = pipeline.recognize(image_bytes, lane_id=7)
    assert set(actual) == {"plate_no", "confidence", "image_path", "stage_ms", "success", "message"}
    assert type(actual["plate_no"]) is str
    assert type(actual["confidence"]) is float
    assert actual["image_path"] is None
    assert type(actual["success"]) is bool
    assert all(type(k) is str and type(v) is int for k, v in actual["stage_ms"].items())
    assert actual["stage_ms"]["locate"] == 1 and actual["stage_ms"]["reader"] == 3
    assert actual["success"] is (status == RecogStatus.OK)
    assert actual["message"] == (None if status == RecogStatus.OK else result.message)
    assert actual["plate_no"] == result.plate_no
    assert BackendRecognitionResult(**actual).model_dump() == actual
    assert lanes == [7]


@pytest.mark.parametrize("data", [None, b"", b"not an image"])
def test_invalid_image_does_not_load_or_raise(monkeypatch, data):
    def unexpected_load():
        raise AssertionError("无效图片不应加载模型")

    monkeypatch.setattr(pipeline.PlateRecognizer, "from_config", unexpected_load)
    result = pipeline.recognize(data)
    assert result["success"] is False and result["plate_no"] == ""
    assert "无法解码" in result["message"]
    BackendRecognitionResult(**result)


@pytest.mark.parametrize("error", [ImportError("missing torch"), ModelLoadError("bad manifest"), ValueError("bad config")])
def test_load_failure_wraps_cause_and_next_call_retries(monkeypatch, image_bytes, error):
    calls = []

    def load():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return SimpleNamespace(recognize=lambda *a, **kw: failure_result(RecogStatus.NO_PLATE))

    monkeypatch.setattr(pipeline.PlateRecognizer, "from_config", load)
    with pytest.raises(RuntimeError, match="pip install -r requirements-algo.txt") as exc:
        pipeline.recognize(image_bytes)
    assert exc.value.__cause__ is error and str(error) in str(exc.value)
    assert pipeline._backend_recognizer is None
    assert pipeline.recognize(image_bytes)["success"] is False
    pipeline.recognize(image_bytes)
    assert len(calls) == 2


def test_concurrent_first_call_loads_once(monkeypatch, image_bytes):
    barrier = threading.Barrier(8)
    calls = []

    def load():
        calls.append(1)
        time.sleep(0.03)
        return SimpleNamespace(recognize=lambda *a, **kw: failure_result(RecogStatus.NO_PLATE))

    def invoke(_):
        barrier.wait(timeout=10)
        return pipeline.recognize(image_bytes)

    monkeypatch.setattr(pipeline.PlateRecognizer, "from_config", load)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(8)))
    assert len(calls) == 1 and all(r["success"] is False for r in results)


def test_default_golden_matches_backend_contract():
    image = Path(__file__).parent / "data/golden" / GOLDEN_NAME
    result = pipeline.recognize(image.read_bytes())
    assert result["success"] is True and result["plate_no"] == GOLDEN_PLATE
    assert BackendRecognitionResult(**result).message is None


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@pytest.fixture
def export_repos(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "team"
    for repo in (source, dest):
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.name", "Export Test")
        _git(repo, "config", "user.email", "export@example.invalid")
        _git(repo, "config", "core.autocrlf", "false")
    manifest = {}
    for name in team_export.MODEL_NAMES[:-1]:
        manifest[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest()}
    for name in team_export.FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(json.dumps(manifest).encode() if target.name == "manifest.json" else target.name.encode())
    for directory in team_export.DIRECTORIES:
        target = source / directory / "committed.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("committed", encoding="utf-8")
    for name in ("algorithm/README.md", "algorithm/__init__.py", "algorithm/lpr/recognize.py", "algorithm/train/stub.py"):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("team", encoding="utf-8")
    for repo in (source, dest):
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "fixture")
    return source, dest


def test_export_uses_commit_and_exact_allowlist(export_repos):
    source, dest = export_repos
    (source / "algorithm/lpr/committed.py").write_text("dirty", encoding="utf-8")
    (source / "algorithm/lpr/untracked.py").write_text("untracked", encoding="utf-8")
    report = team_export.export(source, "HEAD", dest)
    assert (dest / "algorithm/lpr/committed.py").read_text() == "committed"
    assert not (dest / "algorithm/lpr/untracked.py").exists()
    assert report["deleted"] == ["algorithm/lpr/recognize.py", "algorithm/train/stub.py"]
    assert all(not (dest / name).exists() for name in report["deleted"])
    assert (dest / "algorithm/README.md").read_text() == "team"
    assert (dest / "algorithm/__init__.py").read_text() == "team"
    assert {p.name for p in (dest / "algorithm/models").iterdir()} == set(team_export.MODEL_NAMES)
    assert all(report["models"][n]["manifest_matches"] for n in team_export.MODEL_NAMES[:-1])


def test_export_rejects_bad_hash_before_writing(export_repos):
    source, dest = export_repos
    (source / "algorithm/models/plate_det_v1.pt").write_bytes(b"corrupt")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "bad hash")
    with pytest.raises(ValueError, match="SHA256"):
        team_export.export(source, "HEAD", dest)
    assert not (dest / "algorithm/models").exists()
    assert (dest / "algorithm/lpr/recognize.py").exists()


def test_export_rejects_dirty_destination(export_repos):
    source, dest = export_repos
    (dest / "algorithm/lpr/recognize.py").write_text("local edit", encoding="utf-8")
    with pytest.raises(ValueError, match="未提交修改"):
        team_export.export(source, "HEAD", dest)
    assert (dest / "algorithm/lpr/recognize.py").read_text() == "local edit"


def test_export_paths_cannot_escape(tmp_path):
    for name in ("../elsewhere", "algorithm/lpr/../../outside", "/absolute", "algorithm/README.md", "algorithm/lpr/..\\escape"):
        assert not team_export._allowed(name)
    with pytest.raises(ValueError, match="越界"):
        team_export._target(tmp_path, "../escape")
