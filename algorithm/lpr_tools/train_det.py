"""Prepare an audited full detector pool, then train with bounded RAM/VRAM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path


WORKER = Path(__file__).resolve().parents[2]
PROJECT = WORKER.parent.parent
OUT = WORKER / "algorithm" / "out" / "eleventh"
DATA = OUT / "data"
INDEX = PROJECT / "algorithm" / "data" / "index"
RAW_CACHE = WORKER / "algorithm" / "out" / "eighth" / "data" / ".image-sha-cache.jsonl"
CAMERA_AUGMENT_SCALES = (.4, .6, .8, 1.0, 1.3, 1.6)
CAMERA_SCALE_FALLBACK_POLICY = "when a sampled scale cannot keep the GT bbox fully inside the 1280x720 canvas, retry lower configured scales in descending order; raise if none fit"

EXCLUSIONS = {
    "frozen": INDEX / "testset_manifest_v1.csv",
    "golden": WORKER / "algorithm" / "tests" / "data" / "golden" / "golden_manifest.csv",
    "devhard": INDEX / "devhard_v1.csv",
    "val1000": INDEX / "val1000_v1.csv",
    "undecodable": INDEX / "undecodable.csv",
}
INDEX_CSV = INDEX / "ccpd_index.csv"
HARD_TRAIN = WORKER / "algorithm" / "out" / "eighth" / "data" / "hard_pool_train.csv"
HARD_HOLDOUT = WORKER / "algorithm" / "out" / "eighth" / "data" / "hard_tuning_holdout.csv"
NP_TRAIN = WORKER / "algorithm" / "out" / "eighth" / "data" / "np_train_safe.csv"
NP_VAL = WORKER / "algorithm" / "out" / "eighth" / "data" / "np_val_safe.csv"

FIELDS = ["path", "dataset", "subset", "split", "source_split", "has_plate", "plate_no",
          "layout", "plate_color", "quad", "bbox", "tilt_h", "tilt_v", "brightness",
          "blurriness", "area", "skip_reason", "source_category", "sha256", "width", "height"]


def path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/").lower()


def file_sha(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    return file_sha(str(path))


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-16000")
    conn.execute("CREATE TABLE IF NOT EXISTS sha_cache (path_key TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS forbidden_path (path_key TEXT PRIMARY KEY, sources TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS forbidden_sha (sha256 TEXT PRIMARY KEY, sources TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS selected_sha (sha256 TEXT PRIMARY KEY, path_key TEXT NOT NULL, source TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS candidate (seq INTEGER PRIMARY KEY, source TEXT NOT NULL, row_json TEXT NOT NULL, sha256 TEXT NOT NULL, path_key TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS holdout (seq INTEGER PRIMARY KEY, source TEXT NOT NULL, row_json TEXT NOT NULL, sha256 TEXT NOT NULL, path_key TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS needed_path (path_key TEXT PRIMARY KEY)")
    conn.commit()
    return conn


def cached_sha(conn: sqlite3.Connection, path: str) -> tuple[str, bool]:
    stat = os.stat(path)
    key = path_key(path)
    row = conn.execute("SELECT size, mtime_ns, sha256 FROM sha_cache WHERE path_key=?", (key,)).fetchone()
    if row and row[0] == stat.st_size and row[1] == stat.st_mtime_ns:
        return row[2], True
    sha = file_sha(path)
    conn.execute("INSERT OR REPLACE INTO sha_cache VALUES (?, ?, ?, ?, ?)",
                 (key, os.path.abspath(path), stat.st_size, stat.st_mtime_ns, sha))
    return sha, False


def load_hash_cache(conn: sqlite3.Connection) -> dict[str, int]:
    imported = 0
    relevant = 0
    if not RAW_CACHE.is_file():
        raise FileNotFoundError(f"reusable SHA cache missing: {RAW_CACHE}")
    with RAW_CACHE.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            imported += 1
            if conn.execute("SELECT 1 FROM needed_path WHERE path_key=?", (item["key"],)).fetchone():
                conn.execute("INSERT OR IGNORE INTO sha_cache VALUES (?, ?, ?, ?, ?)",
                             (item["key"], item["path"], int(item["size"]), int(item["mtime_ns"]), item["sha256"]))
                relevant += 1
            if imported % 10000 == 0:
                conn.commit()
    conn.commit()
    return {"source": str(RAW_CACHE), "source_sha256": file_digest(RAW_CACHE),
            "records_seen": imported, "relevant_records_imported": relevant}


def load_forbidden(conn: sqlite3.Connection) -> dict[str, dict[str, int | str]]:
    audit = {}
    for name, manifest in EXCLUSIONS.items():
        if not manifest.is_file():
            raise FileNotFoundError(f"required protected manifest missing: {manifest}")
        n = 0
        sha_set: set[str] = set()
        for item in csv_rows(manifest):
            value = item.get("path", "")
            if not value:
                continue
            key = path_key(value)
            conn.execute("INSERT INTO forbidden_path(path_key, sources) VALUES (?, ?) ON CONFLICT(path_key) DO UPDATE SET sources=forbidden_path.sources || ',' || excluded.sources",
                         (key, name))
            conn.execute("INSERT OR IGNORE INTO needed_path VALUES (?)", (key,))
            n += 1
        audit[name] = {"path": str(manifest), "sha256": file_digest(manifest), "rows": n}
        conn.commit()
    return audit


def candidates(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {"base_green_train": 0, "hard_pool_train": 0, "np_train": 0,
              "hard_holdout": 0, "np_val": 0}
    conn.execute("DELETE FROM candidate")
    conn.execute("DELETE FROM selected_sha")
    conn.execute("DELETE FROM holdout")
    conn.execute("DELETE FROM needed_path")
    conn.execute("DELETE FROM forbidden_path")
    conn.execute("DELETE FROM forbidden_sha")
    seq = 0
    # Register protected paths before importing only relevant rows from the old ledger.
    for name, manifest in EXCLUSIONS.items():
        for item in csv_rows(manifest):
            if item.get("path"):
                key = path_key(item["path"])
                conn.execute("INSERT OR IGNORE INTO forbidden_path VALUES (?, ?)", (key, name))
                conn.execute("INSERT OR IGNORE INTO needed_path VALUES (?)", (key,))
    # All shared-index train rows from base and green; do not sample.
    with INDEX_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if item.get("split") != "train" or item.get("subset") not in {"ccpd_base", "ccpd_green"}:
                continue
            counts["base_green_train"] += 1
            seq += 1
            row = dict(item)
            row.update(split="train", source_split="train", source_category="base_green")
            conn.execute("INSERT INTO candidate VALUES (?, ?, ?, ?, ?)",
                         (seq, "base_green", json.dumps(row, ensure_ascii=False), "", path_key(item["path"])))
            conn.execute("INSERT OR IGNORE INTO needed_path VALUES (?)", (path_key(item["path"]),))
    # Confirm holdouts against current bytes; they are excluded by both path and content.
    for source, file, split_count_key in (("hard_holdout", HARD_HOLDOUT, "hard_holdout"),
                                          ("np_val", NP_VAL, "np_val")):
        for item in csv_rows(file):
            counts[split_count_key] += 1
            row = dict(item)
            row["source_category"] = source
            conn.execute("INSERT INTO holdout VALUES (?, ?, ?, ?, ?)",
                         (counts["hard_holdout"] + counts["np_val"], source,
                          json.dumps(row, ensure_ascii=False), "", path_key(item["path"])))
            conn.execute("INSERT OR IGNORE INTO needed_path VALUES (?)", (path_key(item["path"]),))
    # Full hard pool and NP 2,400 membership from audited batch-8 data.
    for source, file in (("hard_pool", HARD_TRAIN), ("np_train", NP_TRAIN)):
        for item in csv_rows(file):
            counts["hard_pool_train" if source == "hard_pool" else "np_train"] += 1
            seq += 1
            row = dict(item)
            row["source_category"] = "hard_pool" if source == "hard_pool" else "np"
            row.setdefault("source_split", row.get("split", "train"))
            conn.execute("INSERT INTO candidate VALUES (?, ?, ?, ?, ?)",
                         (seq, source, json.dumps(row, ensure_ascii=False), "", path_key(item["path"])))
            conn.execute("INSERT OR IGNORE INTO needed_path VALUES (?)", (path_key(item["path"]),))
    conn.commit()
    return counts


def hash_protected_inputs(conn: sqlite3.Connection, protected: dict) -> dict[str, dict]:
    """Hash five fixed manifests and both independent holdout partitions after cache import."""
    sha_members: dict[str, set[str]] = {name: set() for name in EXCLUSIONS}
    for name, manifest in EXCLUSIONS.items():
        for item in csv_rows(manifest):
            if item.get("path"):
                sha, _ = cached_sha(conn, item["path"])
                sha_members[name].add(sha)
                conn.execute("INSERT OR IGNORE INTO forbidden_sha VALUES (?, ?)", (sha, name))
        protected[name]["unique_sha256"] = len(sha_members[name])
    holdout_counts = {}
    # This membership has only 2,701 rows. Finish the SELECT before updating its
    # table: updating a table under an active SQLite scan can revisit rows.
    members = conn.execute("SELECT seq, source, row_json, path_key FROM holdout ORDER BY seq").fetchall()
    for batch_start in range(0, len(members), 100):
        for seq, source, encoded, key in members[batch_start:batch_start+100]:
            row = json.loads(encoded)
            sha, _ = cached_sha(conn, row["path"])
            row["sha256"] = sha
            if not conn.execute("SELECT 1 FROM forbidden_path WHERE path_key=?", (key,)).fetchone():
                conn.execute("INSERT OR IGNORE INTO forbidden_path VALUES (?, ?)", (key, source))
            conn.execute("INSERT OR IGNORE INTO forbidden_sha VALUES (?, ?)", (sha, source))
            conn.execute("UPDATE holdout SET row_json=?, sha256=? WHERE seq=?", (json.dumps(row, ensure_ascii=False), sha, seq))
            holdout_counts[source] = holdout_counts.get(source, 0) + 1
    conn.commit()
    return {"fixed_manifest_unique_sha256": {name: len(values) for name, values in sha_members.items()},
            "holdout_memberships": holdout_counts}


def jpeg_size(path: str) -> tuple[int, int]:
    """Read only JPEG marker headers; return original (width, height), without decoding pixels."""
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with open(path, "rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG image: {path}")
        while True:
            prefix = stream.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue
            marker_byte = stream.read(1)
            while marker_byte == b"\xff":
                marker_byte = stream.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in {0xD8, 0xD9, 0x01, *range(0xD0, 0xD8)}:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError(f"invalid JPEG segment length: {path}")
            if marker in sof:
                data = stream.read(5)
                if len(data) != 5:
                    break
                height, width = int.from_bytes(data[1:3], "big"), int.from_bytes(data[3:5], "big")
                if width <= 0 or height <= 0:
                    raise ValueError(f"invalid JPEG dimensions: {path}")
                return width, height
            stream.seek(length - 2, os.SEEK_CUR)
    raise ValueError(f"JPEG SOF marker not found: {path}")


def reduced_to_original(x: float, y: float, original_width: int, original_height: int,
                        decoded_width: int, decoded_height: int) -> tuple[float, float]:
    """Map decoded coordinates exactly, including odd dimensions after reduced JPEG decode."""
    if min(original_width, original_height, decoded_width, decoded_height) <= 0:
        raise ValueError("image dimensions must be positive")
    return (x * original_width / decoded_width, y * original_height / decoded_height)


def detector_geometry_is_valid(row: dict[str, str], width: int, height: int) -> bool:
    """Detector labels remain usable when skip_reason only rejects OCR text."""
    import math
    if row.get("has_plate") != "1":
        return True
    try:
        quad = [float(value) for value in row.get("quad", "").split()]
        box = [float(value) for value in row.get("bbox", "").split()]
    except ValueError:
        return False
    if len(quad) != 8 or len(box) != 4 or not all(math.isfinite(v) for v in quad + box):
        return False
    xs, ys = quad[0::2], quad[1::2]
    x, y, box_w, box_h = box
    if box_w <= 0 or box_h <= 0 or x < 0 or y < 0:
        return False
    if min(xs) < -1 or min(ys) < -1 or max(xs) > width or max(ys) > height:
        return False
    # CCPD's separately annotated bbox need not enclose every annotated vertex.
    return x+box_w <= width+1 and y+box_h <= height+1


def assemble(conn: sqlite3.Connection) -> tuple[dict, list[str]]:
    # This is a replacement export. A resumed preparation must not treat a
    # previous partial export's selected ledger as newly encountered duplicates.
    conn.execute("DELETE FROM selected_sha")
    conn.commit()
    report = {"cache_hit": 0, "fresh_hash": 0, "excluded_path": 0, "excluded_content": 0,
              "duplicate_content": 0, "selected": 0, "selected_by_source": {}}
    output = DATA / "train_pool.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        cursor = conn.execute("SELECT seq, source, row_json, path_key FROM candidate ORDER BY seq")
        for seq, source, encoded, key in cursor:
            row = json.loads(encoded)
            path = row["path"]
            sha, hit = cached_sha(conn, path)
            report["cache_hit" if hit else "fresh_hash"] += 1
            if conn.execute("SELECT 1 FROM forbidden_path WHERE path_key=?", (key,)).fetchone():
                report["excluded_path"] += 1
                continue
            if conn.execute("SELECT 1 FROM forbidden_sha WHERE sha256=?", (sha,)).fetchone():
                report["excluded_content"] += 1
                continue
            if conn.execute("SELECT 1 FROM selected_sha WHERE sha256=?", (sha,)).fetchone():
                report["duplicate_content"] += 1
                continue
            width, height = jpeg_size(path)
            if not detector_geometry_is_valid(row, width, height):
                report["invalid_detector_geometry"] = report.get("invalid_detector_geometry", 0)+1
                continue
            conn.execute("INSERT INTO selected_sha VALUES (?, ?, ?)", (sha, key, source))
            row.update(sha256=sha, width=width, height=height)
            writer.writerow(row)
            report["selected"] += 1
            report["selected_by_source"][source] = report["selected_by_source"].get(source, 0) + 1
            if seq % 10000 == 0:
                conn.commit()
    conn.commit()
    if conn.execute("SELECT COUNT(*) FROM selected_sha").fetchone()[0] != report["selected"]:
        raise AssertionError("selected ledger and CSV export differ")
    holdout_output = DATA / "holdout.csv"
    with holdout_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for _seq, _source, encoded, sha, _key in conn.execute("SELECT seq, source, row_json, sha256, path_key FROM holdout ORDER BY seq"):
            row = json.loads(encoded)
            width, height = jpeg_size(row["path"])
            row.update(sha256=sha, width=width, height=height)
            writer.writerow(row)
    report["selected_by_subset"] = {}
    with output.open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            label = f"{item.get('source_category', '')}:{item.get('subset', '')}"
            report["selected_by_subset"][label] = report["selected_by_subset"].get(label, 0) + 1
    return report, [str(output), file_digest(output)], [str(holdout_output), file_digest(holdout_output)]


def prepare_pool(*, fresh: bool = False) -> None:
    if not DATA.resolve().is_relative_to((WORKER / "algorithm" / "out" / "eleventh").resolve()):
        raise ValueError("data output must remain in this worker")
    if (DATA / "prepare_audit.json").exists() and not fresh:
        raise FileExistsError("completed data audit exists; explicit --fresh required for rebuilding")
    DATA.mkdir(parents=True, exist_ok=True)
    db_path = DATA / "sha-ledger.sqlite3"
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(db_path) + suffix).unlink()
            except FileNotFoundError:
                pass
    conn = init_db(db_path)
    started = time.time()
    # Populate minimal memberships first, so the 92MB historic cache is filtered in SQLite
    # and only hashes relevant to this batch's candidates/manifests/holdouts are retained.
    counts = candidates(conn)
    protected = {name: {"path": str(path), "sha256": file_digest(path)} for name, path in EXCLUSIONS.items()}
    cache_meta = load_hash_cache(conn)
    protected_hashes = hash_protected_inputs(conn, protected)
    assemble_report, train_file, holdout_file = assemble(conn)
    overlap = conn.execute("SELECT COUNT(*) FROM selected_sha JOIN forbidden_sha USING(sha256)").fetchone()[0]
    if overlap:
        raise AssertionError("training content overlaps protected/holdout membership")
    index_sha = file_digest(INDEX_CSV)
    audit = {
        "schema": 1,
        "created_unix": int(time.time()),
        "index": {"path": str(INDEX_CSV), "sha256": index_sha},
        "input_csvs": {str(p): file_digest(p) for p in (HARD_TRAIN, HARD_HOLDOUT, NP_TRAIN, NP_VAL)},
        "reused_hash_cache": cache_meta,
        "protected_manifests": protected,
        "protected_hashes": protected_hashes,
        "source_counts": counts,
        "assembly": assemble_report,
        "train_pool": {"path": train_file[0], "sha256": train_file[1]},
        "holdout": {"path": holdout_file[0], "sha256": holdout_file[1]},
        "database_bytes": db_path.stat().st_size,
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": ["No image decode, resize, copy, or model/tuning access was performed."],
    }
    audit_path = DATA / "prepare_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    conn.close()
    export_holdout_partitions(DATA)


def export_holdout_partitions(data_dir: Path) -> dict:
    paths = {"hard": data_dir/"holdout_hard.csv", "np": data_dir/"np_val.csv"}
    streams = {key: path.open("w", encoding="utf-8", newline="") for key, path in paths.items()}
    counts = {"hard": 0, "np": 0}
    try:
        writers = {key: csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                   for key, stream in streams.items()}
        for writer in writers.values(): writer.writeheader()
        for row in csv_rows(data_dir/"holdout.csv"):
            key = "hard" if row["has_plate"] == "1" else "np"
            writers[key].writerow(row); counts[key] += 1
    finally:
        for stream in streams.values(): stream.close()
    if counts != {"hard": 2100, "np": 601}:
        raise ValueError(f"unexpected independent holdout partitions: {counts}")
    metadata = {key: {"path": str(path), "n": counts[key], "sha256": file_digest(path)}
                for key, path in paths.items()}
    (data_dir/"holdout_partitions.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


# Model imports stay lazy so data preparation never loads Torch/CUDA.
def process_memory() -> dict[str, int]:
    if os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return {"working_set": peak, "peak_working_set": peak}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    value = Counters(); value.cb = ctypes.sizeof(value)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb):
        raise OSError("cannot measure process working set")
    return {"working_set": int(value.WorkingSetSize), "peak_working_set": int(value.PeakWorkingSetSize)}


def available_memory() -> int:
    """One startup check, plus safety checks only during authorized training."""
    if os.name != "nt":
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    import ctypes
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    *[(name, ctypes.c_ulonglong) for name in
                      ("total", "available", "total_page", "available_page",
                       "total_virtual", "available_virtual", "extended")]]
    status = Status(); status.length = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Status)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("cannot measure available memory")
    return int(status.available)


def limit_cpu_working_set(limit_bytes: int) -> bool:
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    # This only limits the current process, never another app or the machine.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t,
                                                   ctypes.c_size_t, wintypes.DWORD]
    kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    result = kernel32.SetProcessWorkingSetSizeEx(
        kernel32.GetCurrentProcess(), ctypes.c_size_t(1024**2),
        ctypes.c_size_t(limit_bytes), wintypes.DWORD(4))  # QUOTA_LIMITS_HARDWS_MAX_ENABLE
    return bool(result)


def _parse_numbers(value: str, count: int) -> list[float]:
    import math
    values = list(map(float, value.split()))
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError(f"expected {count} finite label values")
    return values


class CompactPool:
    """Paths and contiguous float labels, rather than hundreds of thousands of dicts."""
    def __init__(self, path: Path, *, limit: int = 0):
        from array import array
        import sys
        self.paths, self.subsets, self.labels = [], [], array("f")
        for row in csv_rows(path):
            positive = row["has_plate"] == "1"
            geometry = (_parse_numbers(row["bbox"], 4) + _parse_numbers(row["quad"], 8)
                        if positive else [0.] * 12)
            self.paths.append(row["path"])
            self.subsets.append(sys.intern(row["subset"]))
            self.labels.extend([float(positive), *geometry, float(row["width"]), float(row["height"])])
            if limit and len(self.paths) >= limit:
                break
        if not self.paths:
            raise ValueError("empty audited training pool")

    def __len__(self):
        return len(self.paths)


def read_training_sample(pool: CompactPool, index: int, epoch: int, size: int,
                         *, augment: bool = True, camera_augment: bool = False):
    import cv2
    import numpy as np
    from lpr.corner_net import letterbox_image
    path = pool.paths[index]
    raw = pool.labels[index*15:(index+1)*15]
    positive, ow, oh = bool(raw[0]), raw[13], raw[14]
    flags = cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imread(path, flags)
    if image is None:
        image = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if image is None:
        raise OSError(f"training JPEG cannot decode: {path}")
    ih, iw = image.shape[:2]
    boxes = np.zeros((0, 4), np.float32); corners = np.zeros((0, 4, 2), np.float32)
    source_box = source_points = None
    if positive:
        x, y, w, h = raw[1:5]
        if min(w, h, ow, oh) <= 0:
            raise ValueError(f"invalid source dimensions/box: {path}")
        source_box = np.asarray([x*iw/ow, y*ih/oh, w*iw/ow, h*ih/oh], np.float32)
        source_points = np.asarray(raw[5:13], np.float32).reshape(4, 2)
        source_points[:, 0] *= iw/ow; source_points[:, 1] *= ih/oh
    rng = np.random.default_rng(42 + epoch*1_000_003 + index)
    use_camera_canvas = bool(camera_augment and rng.random() < .8)
    if use_camera_canvas:
        from lpr_tools.eval_det import camera_canvas
        if positive:
            camera_box, camera_points = source_box, source_points
        else:
            # The dummy geometry lets the shared camera transform place a negative
            # image consistently; it is never copied into detector targets.
            camera_box = np.asarray([iw/2-.5, ih/2-.5, 1., 1.], np.float32)
            camera_points = np.asarray([[iw/2-.5, ih/2-.5], [iw/2+.5, ih/2-.5],
                                        [iw/2+.5, ih/2+.5], [iw/2-.5, ih/2+.5]], np.float32)
        camera_scale = float(rng.choice(CAMERA_AUGMENT_SCALES))
        try:
            camera_image, mapped_box, mapped_points, _camera_meta = camera_canvas(
                image, camera_box, camera_points, camera_scale)
        except ValueError as exc:
            if not str(exc).startswith("scaled GT bbox+quad"):
                raise
            # Oversized synthetic plates cannot fit while remaining complete;
            # retry lower sampled scales and fail only if none can preserve GT.
            valid_fallbacks = [value for value in reversed(CAMERA_AUGMENT_SCALES)
                               if value < camera_scale]
            last_error = None
            for fallback in valid_fallbacks:
                try:
                    camera_image, mapped_box, mapped_points, _camera_meta = camera_canvas(
                        image, camera_box, camera_points, fallback)
                    camera_scale = fallback
                    break
                except ValueError as exc:
                    if not str(exc).startswith("scaled GT bbox+quad"):
                        raise
                    last_error = exc
            else:
                if last_error is not None:
                    raise last_error
                raise
        canvas, transform = letterbox_image(camera_image, (size, size))
        if positive:
            x, y, w, h = mapped_box
            edges = np.asarray([[x, y], [x+w, y+h]], np.float32)
            points = np.asarray(mapped_points, np.float32).reshape(4, 2)
            for values in (edges, points):
                values[:, 0] = values[:, 0]*transform.scale_x + transform.pad_left
                values[:, 1] = values[:, 1]*transform.scale_y + transform.pad_top
                values[:] = np.clip(values, 0, size-1)
            boxes, corners = edges.reshape(1, 4), points[None]
    else:
        canvas, transform = letterbox_image(image, (size, size))
        if positive:
            edges = np.asarray([[source_box[0], source_box[1]],
                                [source_box[0]+source_box[2], source_box[1]+source_box[3]]], np.float32)
            points = source_points.copy()
            for values in (edges, points):
                values[:, 0] = values[:, 0]*transform.scale_x + transform.pad_left
                values[:, 1] = values[:, 1]*transform.scale_y + transform.pad_top
                values[:] = np.clip(values, 0, size-1)
            boxes = edges.reshape(1, 4)
            corners = points[None]
    if positive:
        if min(boxes[0, 2:] - boxes[0, :2]) < 1:
            raise ValueError(f"source box collapses after clipping: {path}")
    if augment:
        if rng.random() < .5:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            if positive:
                boxes[:, [0, 2]] = size-1-boxes[:, [2, 0]]
                corners[..., 0] = size-1-corners[..., 0]
                corners = corners[:, [1, 0, 3, 2]]
        if rng.random() < .65:
            affine = cv2.getRotationMatrix2D((size/2, size/2), float(rng.uniform(-12, 12)),
                                             float(rng.uniform(.85, 1.15)))
            affine[:, 2] += rng.uniform(-.04, .04, 2)*size
            matrix = np.vstack((affine, [0., 0., 1.])).astype(np.float32)
            matrix[2, :2] = rng.uniform(-.00015, .00015, 2)
            accept = True
            if positive:
                x0, y0, x1, y1 = boxes[0]
                rectangle = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                moved = cv2.perspectiveTransform(rectangle[None], matrix)[0]
                shifted = cv2.perspectiveTransform(corners, matrix)
                accept = bool(np.isfinite(shifted).all() and (shifted >= 0).all()
                              and (shifted < size).all() and (moved >= 0).all() and (moved < size).all())
                if accept:
                    boxes = np.concatenate((moved.min(0), moved.max(0)))[None]
                    corners = shifted
            if accept:
                canvas = cv2.warpPerspective(canvas, matrix, (size, size), borderValue=(0, 0, 0))
        gain = float(rng.uniform(.45, 1.65))
        bias = float(rng.uniform(-18, 18))
        table = np.clip(np.arange(256)*gain+bias, 0, 255).astype(np.uint8)
        canvas = cv2.LUT(canvas, table)
        if rng.random() < .25:
            gamma = float(rng.uniform(.65, 1.6))
            lut = np.round(255*(np.arange(256)/255.)**gamma).astype(np.uint8)
            canvas = cv2.LUT(canvas, lut)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), boxes, corners, pool.subsets[index]


def make_targets(samples, size: int, device):
    import numpy as np
    import torch
    grid = size//8
    heat = np.zeros((len(samples), 1, grid, grid), np.float32)
    indices, batch_indices, offsets, whs, relative = [], [], [], [], []
    for b, sample in enumerate(samples):
        boxes, corners = sample[1], sample[2]
        for box, points in zip(boxes, corners):
            xy = (box[:2]+box[2:])/16.; wh = (box[2:]-box[:2])/8.
            ix, iy = np.floor(xy).astype(int)
            if not (0 <= ix < grid and 0 <= iy < grid) or min(wh) <= 0:
                raise ValueError("augmented object outside output grid")
            radius = max(1, min(7, int(min(wh)*.25)))
            sigma = max(.7, radius/2.)
            ys, xs = np.mgrid[max(0, iy-radius):min(grid, iy+radius+1),
                              max(0, ix-radius):min(grid, ix+radius+1)]
            patch = np.exp(-((xs-ix)**2+(ys-iy)**2)/(2*sigma*sigma))
            heat[b, 0, max(0, iy-radius):min(grid, iy+radius+1),
                 max(0, ix-radius):min(grid, ix+radius+1)] = patch
            indices.append(iy*grid+ix); batch_indices.append(b)
            offsets.append(xy-[ix, iy]); whs.append(wh)
            center = (box[:2]+box[2:])/2
            relative.append((points-center)/(box[2:]-box[:2]))
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(
        np.asarray(x), dtype=dtype).to(device=device, non_blocking=True)
    return {"heatmap": tensor(heat), "index": tensor(indices, torch.long),
            "batch": tensor(batch_indices, torch.long),
            "offset": tensor(offsets).reshape(-1, 2), "wh": tensor(whs).reshape(-1, 2),
            "relative": tensor(relative).reshape(-1, 8)}


def background_negative(sample):
    """Keep background negatives without replacing any full-pool positive exposure."""
    import cv2
    import numpy as np
    rgb, boxes, corners, _subset = sample
    if len(boxes) != 1:
        return None
    size = rgb.shape[-1]
    lower = np.minimum(boxes[0, :2], corners[0].min(0))
    upper = np.maximum(boxes[0, 2:], corners[0].max(0))
    x0, y0 = np.floor(lower-2).astype(int)
    x1, y1 = np.ceil(upper+2).astype(int)
    strips = [(0, 0, max(0, x0), size), (min(size, x1), 0, size, size),
              (0, 0, size, max(0, y0)), (0, min(size, y1), size, size)]
    for left, top, right, bottom in sorted(strips, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True):
        if min(right-left, bottom-top) < 16 or (right-left)*(bottom-top) < size*size*.05:
            continue
        crop = rgb[:, top:bottom, left:right].transpose(1, 2, 0)
        if float(crop.std()) < 5:
            continue
        image = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return (np.ascontiguousarray(image.transpose(2, 0, 1)), np.zeros((0, 4), np.float32),
                np.zeros((0, 4, 2), np.float32), "background_negative")
    return None


def detector_loss(outputs, target):
    import torch
    import torch.nn.functional as F
    logits = outputs["heatmap"].float(); truth = target["heatmap"]
    probability = logits.sigmoid().clamp(1e-5, 1-1e-5)
    positive = truth.eq(1); negative = truth.lt(1)
    focal = -(probability.log()*(1-probability).pow(2)*positive
              +(1-probability).log()*probability.pow(2)*(1-truth).pow(4)*negative).sum()
    count = target["index"].numel(); focal = focal/max(1, count)
    if not count:
        empty = outputs["box"].sum()*0+outputs["corners"].sum()*0
        return focal+empty, {"heatmap": focal.detach(), "box": empty.detach(), "corners": empty.detach()}
    def gather(x):
        return x.float().flatten(2).transpose(1, 2)[target["batch"], target["index"]]
    raw = gather(outputs["box"])
    offset = raw[:, :2].sigmoid(); wh = raw[:, 2:].clamp(-6, 7).exp()
    offset_loss = F.smooth_l1_loss(offset, target["offset"], beta=.1)
    size_loss = F.smooth_l1_loss(raw[:, 2:], target["wh"].log(), beta=.2)
    pred0, pred1 = offset-wh/2, offset+wh/2
    gt0, gt1 = target["offset"]-target["wh"]/2, target["offset"]+target["wh"]/2
    intersection = (torch.minimum(pred1, gt1)-torch.maximum(pred0, gt0)).clamp_min(0).prod(-1)
    union = wh.prod(-1)+target["wh"].prod(-1)-intersection
    enclosing = (torch.maximum(pred1, gt1)-torch.minimum(pred0, gt0)).prod(-1).clamp_min(1e-6)
    giou = intersection/union.clamp_min(1e-6)-(enclosing-union)/enclosing
    box_loss = offset_loss+.5*size_loss+3*(1-giou).mean()
    corner_loss = F.smooth_l1_loss(gather(outputs["corners"]).tanh(), target["relative"], beta=.05)
    return focal+box_loss+.5*corner_loss, {
        "heatmap": focal.detach(), "box": box_loss.detach(), "corners": corner_loss.detach()}


def _save_atomic(payload, path: Path):
    import torch
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def record_finite_loss(history, loss):
    import torch
    history.logical_and_(torch.isfinite(loss.detach()))


def check_finite_history(history):
    if not bool(history):
        raise FloatingPointError("nonfinite detector loss in training history; last preserved")


def source_hashes() -> dict[str, str]:
    return {str(p): file_digest(p) for p in
            (Path(__file__).resolve(), WORKER/"algorithm/lpr/plate_det.py")}


def validate_init_resume_options(init_from: Path | None, resume: Path | None) -> None:
    if init_from is not None and resume is not None:
        raise ValueError("--init-from and --resume are mutually exclusive")


def minimum_complete_epochs(*, stage_parent: bool) -> int:
    return 10 if stage_parent else 40


def stage_learning_rates(*, stage_parent: bool) -> tuple[float, float]:
    return (2e-5, 2e-4) if stage_parent else (1e-4, 1e-3)


def build_stage_parent(path: Path, sha256: str, *, parent_epochs: int | None,
                       parent_exposures: int | None, parent_train_pool_sha256: str | None,
                       release_manifest_sha256: str | None) -> dict:
    return {"model_path": str(path), "model_sha256": sha256,
            "parent_training": {"verified_epoch_completed": parent_epochs,
                                "exposures": parent_exposures,
                                "train_pool_sha256": parent_train_pool_sha256,
                                "release_manifest_sha256": release_manifest_sha256},
            "stage_start_epoch": 0, "stage_start_training_wall_seconds": 0.,
            "optimizer_state": "new", "rng_state": "new"}


def validate_camera_source(payload, *, current_sha: str, allow_change: bool, reason: str | None = None):
    old_sha = payload.get("camera_source_sha256")
    if old_sha is None or old_sha == current_sha:
        return None
    if not allow_change or not reason or not reason.strip():
        raise ValueError("camera augmentation source changed; explicit allow flag and reason required")
    return {"old": old_sha, "new": current_sha, "reason": reason}


def validate_resume_sources(payload, *, allow_change: bool, reason: str | None = None):
    current = source_hashes()
    previous = payload["source_hashes"]
    if set(previous) != set(current):
        raise ValueError("resume source membership differs")
    changes = {path: {"old": old, "new": current[path]}
               for path, old in previous.items() if current[path] != old}
    if changes and (not allow_change or not reason or not reason.strip()):
        raise ValueError("resume source changed; explicit allow flag and reason required")
    return changes


def _dump_profile(args):
    if not args.profile or not args._profile:
        return
    import torch
    torch.cuda.synchronize()
    numeric = []
    for cpu, events in args._profile:
        numeric.append({"cpu_ms": cpu, "gpu_ms": {
            name: events[i].elapsed_time(events[i+1]) for i, name in enumerate(
                ("input_targets", "forward", "loss", "backward_clip", "optimizer"))}})
    (args.out_dir/f"profile-{args._phase_unix}.json").write_text(json.dumps({
        "image_size": args.image_size, "batch_size": args.batch_size,
        "ram_cap_mb": args.ram_cap_mb, "io_threads": args.io_threads,
        "freeze_backbone_blocks": args.freeze_backbone_blocks, "steps": numeric}, indent=2), encoding="utf-8")


def _checkpoint(model, optimizer, scaler, args, epoch, cursor, exposures, pool_sha):
    import torch
    from lpr.plate_det import ARCHITECTURE
    check_finite_history(args._finite_history)
    training_config = {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items() if not k.startswith("_")}
    training_config.update({"out_dir": str(args.out_dir), "data_dir": str(args.data_dir),
                            "resume": str(args.resume) if args.resume else None,
                            "init_from": str(args.init_from) if args.init_from else None,
                            "stage_learning_rates": list(args._base_lrs),
                            "camera_augment_scale_policy": CAMERA_SCALE_FALLBACK_POLICY
                            if args.camera_augment else None})
    return {"architecture": ARCHITECTURE, "input_size": [args.image_size]*2,
            "score_threshold": .25, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch_completed": epoch, "batch_cursor": cursor, "exposures": exposures,
            "training_wall_seconds": args._wall_before+time.monotonic()-args._wall_start,
            "epoch_elapsed_seconds": (args._epoch_before+time.monotonic()-args._epoch_start if cursor else 0.),
            "epoch_subset_counts": dict(args._epoch_counts) if cursor else {},
            "epoch_background_unrecorded_upper_bound": args._background_unknown if cursor else 0,
            "resume_history": args._resume_history,
            "stage_parent": args._stage_parent,
            "camera_source_sha256": file_digest(Path(__file__).with_name("eval_det.py")),
            "train_pool_sha256": pool_sha, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if args.device == "cuda" else None,
            "source_hashes": source_hashes(),
            "resource_limits": {"working_set_cap": args.ram_cap_mb*1_000_000,
                                "cuda_reserved_guard": args.cuda_reserved_cap_mb*1_000_000,
                                "cuda_fraction": args.cuda_fraction, "workers": 0, "io_threads": args.io_threads},
            "imagenet": "torchvision ResNet18_Weights.IMAGENET1K_V1; stem and layer1..3",
            "training_config": training_config}


def run_training(args):
    """One authorized GPU phase, or a bounded CPU smoke; no automatic launch/wait loop."""
    if not args.smoke and (not args.start_training or args.device != "cuda"):
        raise ValueError("formal training needs explicit --start-training --device cuda authorization")
    validate_init_resume_options(args.init_from, args.resume)
    if args.workers != 0:
        raise ValueError("this memory-bounded trainer uses DataLoader workers0; use --io-threads1..3")
    # Leave launcher/shell headroom inside the configured process working-set cap.
    hard_limit = 1_000_000_000 if args.smoke else args.ram_cap_mb*1_000_000
    if args.low_memory_start and (args.workers != 0 or args.io_threads > 2):
        raise ValueError("low-memory start requires workers0 and at most2 IO threads")
    if not args.smoke:
        minimum_free = max(hard_limit+2*1024**3,
                           (3 if args.low_memory_start else 4)*1024**3)
        free = available_memory()
        print(json.dumps({"startup_free_bytes": free, "minimum_free_bytes": minimum_free,
                          "working_set_cap": hard_limit, "io_threads": args.io_threads}), flush=True)
        if free < minimum_free:
            raise RuntimeError("startup free memory cannot retain2GiB reserve; no retry loop")
    hard_ws_applied = limit_cpu_working_set(hard_limit)
    if os.name == "nt" and not hard_ws_applied:
        raise RuntimeError("cannot apply own hard working-set cap")
    import math
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    import torch
    from lpr.plate_det import PlateDetModel, PlateDetector, ARCHITECTURE
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.manual_seed(42)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the shared .venv-gpu interpreter")
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
        torch.backends.cudnn.benchmark = True
    os.environ["TORCH_HOME"] = str(OUT/"cache"/"torch")
    torch.hub.set_dir(str(OUT/"cache"/"torch"/"hub"))
    audit = json.loads((args.data_dir/"prepare_audit.json").read_text(encoding="utf-8-sig"))
    train_csv = args.data_dir/"train_pool.csv"
    pool_sha = file_digest(train_csv)
    if pool_sha != audit["train_pool"]["sha256"]:
        raise ValueError("training pool differs from preparation audit")
    pool = CompactPool(train_csv)
    if len(pool) != audit["assembly"]["selected"]:
        raise ValueError("training rows differ from the audited full pool")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    last = args.out_dir/"last.pt"
    if last.exists() and args.resume is None:
        raise FileExistsError("last checkpoint exists; pass explicit --resume")
    memory_format = torch.channels_last if args.device == "cuda" else torch.contiguous_format
    stage_parent = None
    model = PlateDetModel(pretrained=args.resume is None and args.init_from is None and not args.random_smoke).to(
        device=args.device, memory_format=memory_format)
    for block in model.features[:args.freeze_backbone_blocks]:
        block.requires_grad_(False)
    args._stage_parent = None
    args._base_lrs = stage_learning_rates(stage_parent=args.init_from is not None)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args._base_lrs[0]},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("features.")], "lr": args._base_lrs[1]}],
        weight_decay=1e-4, fused=args.device == "cuda", foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
    epoch_completed, cursor, exposures = 0, 0, 0
    args._wall_before, epoch_before = 0., 0.
    args._resume_history, restored_counts = [], None
    args._background_unknown = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload["architecture"] != ARCHITECTURE or payload["train_pool_sha256"] != pool_sha:
            raise ValueError("resume architecture/pool mismatch")
        changes = validate_resume_sources(payload, allow_change=args.allow_code_change_resume,
                                          reason=args.resume_change_reason)
        camera_change = validate_camera_source(
            payload, current_sha=file_digest(Path(__file__).with_name("eval_det.py")),
            allow_change=args.allow_code_change_resume, reason=args.resume_change_reason)
        if payload["input_size"] != [args.image_size]*2 and not args.allow_resize_resume:
            raise ValueError("input resize resume requires --allow-resize-resume")
        model.load_state_dict(payload["state_dict"], strict=True)
        # Optimizer loading uses the saved fused flag to place scalar step
        # counters; update that flag before loading a pre-optimization checkpoint.
        for group in payload["optimizer"]["param_groups"]:
            group.update(fused=args.device == "cuda", foreach=False)
        optimizer.load_state_dict(payload["optimizer"]); scaler.load_state_dict(payload["scaler"])
        epoch_completed, cursor, exposures = payload["epoch_completed"], payload["batch_cursor"], payload["exposures"]
        args._stage_parent = payload.get("stage_parent")
        saved_rates = payload.get("training_config", {}).get("stage_learning_rates")
        args._base_lrs = tuple(saved_rates) if saved_rates else stage_learning_rates(
            stage_parent=args._stage_parent is not None)
        if not args.smoke and args.epochs < minimum_complete_epochs(stage_parent=args._stage_parent is not None):
            raise ValueError(f"formal training requires at least{minimum_complete_epochs(stage_parent=args._stage_parent is not None)} complete epochs for this stage")
        if exposures != epoch_completed*len(pool)+cursor:
            raise ValueError("resume exposures do not match complete epochs plus cursor")
        args._resume_history = payload.get("resume_history", []) + [{
            "unix": int(time.time()), "epoch_completed": epoch_completed, "cursor": cursor,
            "source_changes": changes, "reason": args.resume_change_reason or "phase resume",
            "old_input_size": payload["input_size"], "new_input_size": [args.image_size]*2,
            "old_batch_size": payload["training_config"]["batch_size"], "new_batch_size": args.batch_size,
            "new_ram_cap_mb": args.ram_cap_mb, "new_io_threads": args.io_threads}]
        if camera_change:
            args._resume_history[-1]["camera_source_change"] = camera_change
        restored_counts = payload.get("epoch_subset_counts") if cursor else None
        if cursor:
            args._background_unknown = payload.get("epoch_background_unrecorded_upper_bound", 0)
            if restored_counts is None:
                args._background_unknown = 2*math.ceil(cursor/payload["training_config"]["batch_size"])
        args._wall_before = payload.get("training_wall_seconds", 0.)
        epoch_before = payload.get("epoch_elapsed_seconds", 0.) if cursor else 0.
        torch.set_rng_state(payload["torch_rng"].cpu())
        if args.device == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu())
    elif args.init_from:
        parent_path = args.init_from.resolve()
        parent_sha = file_digest(parent_path)
        parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
        if not isinstance(parent_payload, dict) or parent_payload.get("architecture") != ARCHITECTURE:
            raise ValueError("--init-from must be a strict plate detector checkpoint")
        model.load_state_dict(parent_payload["state_dict"], strict=True)
        if not args.smoke and args.epochs < minimum_complete_epochs(stage_parent=True):
            raise ValueError("formal init-from stage requires at least10 complete epochs")
        release_path = OUT/"model_release.json"
        release = json.loads(release_path.read_text(encoding="utf-8-sig")) if release_path.is_file() else {}
        verified_parent = (release.get("epoch_completed") == 40
                           and release.get("model_sha256") == parent_sha)
        args._stage_parent = build_stage_parent(
            parent_path, parent_sha,
            parent_epochs=40 if verified_parent else None,
            parent_exposures=release.get("exposures") if verified_parent else None,
            parent_train_pool_sha256=release.get("train_pool_sha256") if verified_parent else None,
            release_manifest_sha256=file_digest(release_path) if verified_parent else None)
    elif not args.smoke and args.epochs < minimum_complete_epochs(stage_parent=False):
        raise ValueError("formal training requires at least40 complete epochs")
    model.train(); started = time.monotonic(); steps_this_run = 0
    for block in model.features[:args.freeze_backbone_blocks]:
        block.eval()
    args._wall_start = started
    args._phase_unix = int(time.time())
    args._profile = []
    args._finite_history = torch.ones((), dtype=torch.bool, device=args.device)
    completed_this_run = []
    io_threads = args.io_threads
    with ThreadPoolExecutor(max_workers=io_threads) as executor:
        for epoch in range(epoch_completed, args.epochs):
            order = np.random.default_rng(42+epoch).permutation(len(pool))
            coverage = hashlib.sha256(order.astype("uint32").tobytes()).hexdigest()
            epoch_start = time.monotonic(); start_cursor = cursor
            epoch_exposures, new_exposures = cursor, 0
            counts = Counter(restored_counts or {})
            if cursor and restored_counts is None:
                counts.update(pool.subsets[int(i)] for i in order[:cursor])
            restored_counts = None
            args._epoch_counts = counts
            args._epoch_start, args._epoch_before = epoch_start, epoch_before
            def schedule(offset):
                return [executor.submit(read_training_sample, pool, int(i), epoch, args.image_size,
                                        augment=not args.smoke,
                                        camera_augment=args.camera_augment)
                        for i in order[offset:offset+args.batch_size]]
            pending = schedule(cursor)
            for offset in range(cursor, len(pool), args.batch_size):
                prof_start = time.perf_counter() if args.profile else 0.
                indices = order[offset:offset+args.batch_size]
                samples = [future.result() for future in pending]
                prof_decode = time.perf_counter() if args.profile else 0.
                # At most one additional batch is decoded while the GPU works.
                pending = schedule(offset+len(indices)) if not args.smoke else []
                background = []
                for sample in samples:
                    negative = background_negative(sample)
                    if negative is not None:
                        background.append(negative)
                    if len(background) >= max(2, args.batch_size//16):
                        break
                samples.extend(background)
                prof_background = time.perf_counter() if args.profile else 0.
                if args.profile:
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                    events[0].record()
                host_images = torch.from_numpy(np.stack([s[0] for s in samples]))
                images = host_images.to(device=args.device, non_blocking=True).to(
                    dtype=torch.float32, memory_format=memory_format).div_(255.)
                targets = make_targets(samples, args.image_size, args.device)
                prof_targets = time.perf_counter() if args.profile else 0.
                if args.profile: events[1].record()
                progress = (epoch+offset/len(pool))/max(1, args.epochs)
                factor = .01+.99*(1+math.cos(math.pi*progress))/2
                warm = min(1., (epoch+offset/len(pool)+.01))
                for group, base_lr in zip(optimizer.param_groups, args._base_lrs):
                    group["lr"] = base_lr*factor*warm
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                    output = model(images)
                prof_forward = time.perf_counter() if args.profile else 0.
                if args.profile: events[2].record()
                loss, parts = detector_loss(output, targets)
                if args.profile: events[3].record()
                record_finite_loss(args._finite_history, loss)
                if args.device == "cpu":
                    check_finite_history(args._finite_history)
                prof_loss = time.perf_counter() if args.profile else 0.
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                prof_backward = time.perf_counter() if args.profile else 0.
                if args.profile: events[4].record()
                scaler.step(optimizer); scaler.update()
                if args.profile:
                    events[5].record()
                    prof_end = time.perf_counter()
                    args._profile.append(({"decode_wait": (prof_decode-prof_start)*1000,
                        "background": (prof_background-prof_decode)*1000,
                        "input_targets": (prof_targets-prof_background)*1000,
                        "forward_enqueue": (prof_forward-prof_targets)*1000,
                        "loss_finite_wait": (prof_loss-prof_forward)*1000,
                        "backward_clip_enqueue": (prof_backward-prof_loss)*1000,
                        "optimizer_enqueue": (prof_end-prof_backward)*1000,
                        "batch_total": (prof_end-prof_start)*1000}, events))
                exposures += len(indices); epoch_exposures += len(indices)
                new_exposures += len(indices); steps_this_run += 1
                counts.update(s[3] for s in samples)
                next_cursor = offset+len(indices)
                if steps_this_run % 50 == 0 or args.smoke:
                    check_finite_history(args._finite_history)
                    memory = process_memory()
                    vram = torch.cuda.max_memory_reserved() if args.device == "cuda" else 0
                    free = available_memory() if not args.smoke else None
                    record = {"epoch": epoch+1, "cursor": next_cursor, "pool_n": len(pool),
                              "loss": float(loss.detach()), "exposures": exposures,
                              "elapsed_seconds": time.monotonic()-epoch_start,
                              "phase_unix": args._phase_unix, "system_available_bytes": free,
                              "batch_size": args.batch_size, "image_size": args.image_size,
                              "working_set": memory["working_set"], "peak_working_set": memory["peak_working_set"],
                              "cuda_peak_reserved": vram}
                    with (args.out_dir/"steps.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record)+"\n")
                    if memory["working_set"] > hard_limit or vram > args.cuda_reserved_cap_mb*1_000_000:
                        _save_atomic(_checkpoint(model, optimizer, scaler, args, epoch, next_cursor, exposures, pool_sha), last)
                        raise MemoryError("own RAM/VRAM budget reached; saved last, reduce batch/workers")
                    if not args.smoke and free < 2*1024**3:
                        _save_atomic(_checkpoint(model, optimizer, scaler, args, epoch, next_cursor, exposures, pool_sha), last)
                        raise MemoryError("system free below2GiB with workers0; saved last and stopped safely")
                if args.smoke:
                    _save_atomic({"architecture": ARCHITECTURE, "input_size": [args.image_size]*2,
                                  "state_dict": model.cpu().state_dict(), "score_threshold": .25}, last)
                    restored = PlateDetector.from_file(last)
                    assert all(torch.isfinite(v).all() for v in restored.model.state_dict().values())
                    smoke = {"mode": "cpu_smoke", "trained_formal_epochs": 0,
                             "one_optimizer_step": True, "pool_images": len(indices),
                             "batch_images": len(samples), "background_negatives": len(background),
                             "loss": float(loss.detach()), "hard_working_set_cap_applied": hard_ws_applied,
                             "memory": process_memory(), "input_size": args.image_size,
                             "device": args.device, "deployment_safe_load": True,
                             "pretrained_backbone": not args.random_smoke}
                    (args.out_dir/"smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
                    print(json.dumps(smoke)); return smoke
                if args._wall_before+(time.monotonic()-started) > args.max_hours*3600:
                    _save_atomic(_checkpoint(model, optimizer, scaler, args, epoch, next_cursor, exposures, pool_sha), last)
                    raise RuntimeError("wall-clock budget reached before completion; last saved, cannot claim40 epochs")
                if args.stop_after_steps and steps_this_run >= args.stop_after_steps:
                    _save_atomic(_checkpoint(model, optimizer, scaler, args, epoch, next_cursor, exposures, pool_sha), last)
                    _dump_profile(args)
                    print(json.dumps({"phase_stopped": True, "epoch_completed": epoch,
                                      "cursor": next_cursor, "global_exposures": exposures}), flush=True)
                    return {"epoch_completed": epoch, "cursor": next_cursor, "exposures": exposures}
            if epoch_exposures != len(pool) or new_exposures != len(pool)-start_cursor:
                raise AssertionError("epoch did not cover every remaining training row")
            cursor = 0; epoch_completed = epoch+1
            _save_atomic(_checkpoint(model, optimizer, scaler, args, epoch_completed, 0, exposures, pool_sha), last)
            elapsed = epoch_before+time.monotonic()-epoch_start
            epoch_before = 0.
            total_hours_at_rate = elapsed*args.epochs/3600
            record = {"epoch_completed": epoch_completed, "pool_n": len(pool),
                      "epoch_exposures": epoch_exposures, "global_exposures": exposures,
                      "subsets": dict(counts), "coverage_sha256": coverage,
                      "background_unrecorded_upper_bound": args._background_unknown,
                      "elapsed_seconds": elapsed, "remaining_hours_at_this_rate":
                      elapsed*(args.epochs-epoch_completed)/3600,
                      "estimated_total_hours_at_this_rate": total_hours_at_rate,
                      "image_size": args.image_size, "batch_size": args.batch_size,
                      "training_wall_seconds": args._wall_before+time.monotonic()-started,
                      "memory": process_memory(), "cuda_peak_reserved": torch.cuda.max_memory_reserved()}
            with (args.out_dir/"epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True); completed_this_run.append(record)
            args._background_unknown = 0
            if (args.stop_after_epoch and epoch_completed >= args.stop_after_epoch) or (
                    epoch_completed == 1 and total_hours_at_rate > args.max_hours):
                break
    _dump_profile(args)
    return {"epoch_completed": epoch_completed, "exposures": exposures,
            "completed_this_run": completed_this_run}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT/"train-resnet")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--start-training", action="store_true", help="manual authorized GPU start; never auto-starts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--workers", type=int, choices=(0,), default=0)
    parser.add_argument("--io-threads", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="initialize a fresh fine-tuning stage from detector weights only")
    parser.add_argument("--camera-augment", action="store_true",
                        help="with 80%% probability transform samples onto a 1280x720 camera canvas")
    parser.add_argument("--allow-resize-resume", action="store_true")
    parser.add_argument("--allow-code-change-resume", action="store_true")
    parser.add_argument("--resume-change-reason")
    parser.add_argument("--ram-cap-mb", type=int, default=2800, help="decimal MB; total process headroom retained")
    parser.add_argument("--cuda-fraction", type=float, default=.42)
    parser.add_argument("--cuda-reserved-cap-mb", type=int, default=4000)
    parser.add_argument("--low-memory-start", action="store_true")
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="bounded CUDA-event numeric stage timings")
    parser.add_argument("--freeze-backbone-blocks", type=int, choices=(0, 5, 6), default=5,
                        help="5 freezes stem+layer1; 6 also freezes layer2")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=8.)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-smoke", action="store_true", help="only smoke may skip ImageNet download")
    args = parser.parse_args(argv)
    if args.prepare_data:
        return prepare_pool(fresh=args.fresh)
    if args.random_smoke and not args.smoke:
        parser.error("formal training always uses ImageNet initialization")
    if args.smoke and args.device != "cpu":
        parser.error("smoke runs use this worker's CPU interpreter")
    if args.profile and (args.device != "cuda" or not 0 < args.stop_after_steps <= 100):
        parser.error("profiling needs CUDA and stop-after-steps1..100")
    if args.batch_size < 1 or args.epochs < 1 or args.image_size < 128 or args.image_size % 32:
        parser.error("invalid batch/epoch/input size")
    if not 1400 <= args.ram_cap_mb <= 2800 or args.stop_after_steps < 0:
        parser.error("RAM cap must be1400..2800MB; stop-after-steps must be nonnegative")
    try:
        validate_init_resume_options(args.init_from, args.resume)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.cuda_fraction <= 1 or args.cuda_reserved_cap_mb < 1:
        parser.error("CUDA fraction must be in (0,1] and reserved cap must be positive")
    return run_training(args)


if __name__ == "__main__":
    main()
