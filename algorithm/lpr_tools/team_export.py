"""Export a committed M2 snapshot to the team's independent repository.

Only the fixed allowlist is written; stale tracked files are removed with git rm.
Run from any directory: python team_export.py --repo SOURCE --dest TEAM [--ref HEAD].
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

DIRECTORIES = tuple(f"algorithm/{name}/" for name in ("lpr", "lpr_tools", "tests", "config"))
MODEL_NAMES = ("plate_det_v1.pt", "char_cnn_v2.pt", "crnn_v4.pt", "box_reader_v3.pt", "manifest.json")
FILES = ("algorithm/reports/scoreboard.md", "algorithm/pyproject.toml",
         *(f"algorithm/models/{name}" for name in MODEL_NAMES))


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _allowed(name: str) -> bool:
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and (name in FILES or name.startswith(DIRECTORIES)))


def _target(dest: Path, name: str) -> Path:
    path = dest / name
    # Resolve existing parents too: a symlink/junction may not escape the destination.
    if not path.resolve().is_relative_to(dest):
        raise ValueError(f"导出路径越界：{name}")
    return path


def export(repo: Path, ref: str, dest: Path) -> dict:
    repo, dest = repo.resolve(), dest.resolve()
    if repo == dest or dest.is_relative_to(repo) or repo.is_relative_to(dest):
        raise ValueError("源仓库和团队仓库必须是互不包含的独立目录")
    if Path(_git(dest, "rev-parse", "--show-toplevel").decode().strip()).resolve() != dest:
        raise ValueError("--dest 必须是团队 git 仓库根目录")
    commit = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    archive = _git(repo, "archive", "--format=tar", commit, "--", *DIRECTORIES, *FILES)
    payload: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            if member.isdir():
                continue
            if not member.isfile() or not _allowed(member.name):
                raise ValueError(f"不支持的归档条目：{member.name}")
            payload[member.name] = stream.extractfile(member).read()
    if not set(FILES) <= payload.keys() or any(not any(n.startswith(d) for n in payload) for d in DIRECTORIES):
        raise ValueError("提交缺少导出清单中的文件或目录")
    manifest = json.loads(payload["algorithm/models/manifest.json"])
    models = {}
    for name in MODEL_NAMES:
        digest = hashlib.sha256(payload[f"algorithm/models/{name}"]).hexdigest()
        matches = digest == manifest.get(name, {}).get("sha256") if name != "manifest.json" else None
        models[name] = {"sha256": digest, "manifest_matches": matches}
        if matches is False:
            raise ValueError(f"模型 SHA256 与 manifest 不一致：{name}")
    paths = (*DIRECTORIES, *FILES, "algorithm/train/")
    if _git(dest, "status", "--porcelain", "--untracked-files=all", "--", *paths).strip():
        raise ValueError("团队导出路径存在未提交修改，请先提交或人工处理")
    tracked = _git(dest, "ls-files", "-z", "--", *DIRECTORIES, "algorithm/train/").decode().split("\0")
    deleted = sorted(name for name in tracked if name and name not in payload)
    for name in (*payload, *deleted):
        _target(dest, name)
    if deleted:
        _git(dest, "rm", "--", *deleted)
    for name, data in sorted(payload.items()):
        target = _target(dest, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    report = {"commit": commit, "exported": sorted(payload), "deleted": deleted, "models": models,
              "manifest_note": "manifest.json 自身无自校验记录；其 SHA256 单独列出"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    export(args.repo, args.ref, args.dest)


if __name__ == "__main__":
    main()
