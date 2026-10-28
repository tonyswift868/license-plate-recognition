"""Bounded prefetch preserves batch bytes and keeps GPU fallback on main thread."""
from __future__ import annotations

import threading

import cv2
import numpy as np
import torch

import lpr_tools.train.train_province_readers as trainer


GROUPS = ("base", "green", "db", "blur", "tilt", "rotate", "fn", "challenge", "weather")


class TinyDataset:
    def __init__(self, rows, *, gpu_fallback=False):
        self.rows = rows
        self.seed = 123
        self.epoch = 2
        self.synth_probability = .35
        self.gpu_fallback = gpu_fallback
        self.kind = "char"
        self.synthetic_gt_fallbacks = 0
        self.row_threads = []
        self.finish_threads = []

    def __len__(self):
        return len(self.rows)

    def row(self, index):
        self.row_threads.append(threading.get_ident())
        return dict(self.rows[index])

    def specification(self, index):
        row = self.row(index)
        rng = trainer.random.Random(self.seed + self.epoch * 1_000_003 + index)
        # Use a deterministic jitter-shaped box and consume the same RNG state
        # from the synthetic test finish on both serial and prefetched runs.
        return row, rng, (2, 2, 10, 10)

    def finish(self, image, row, index, rng, *, bbox, defer_gt_fallback=False):
        self.finish_threads.append(threading.get_ident())
        variants = [(row["label"], False), (row["label"], True)] if self.synthetic_targets[index] >= 0 else [
            (row["label"], False)]
        output = []
        for label, is_synthetic in variants:
            value = index + rng.random()
            tensor = torch.tensor([[[value]]], dtype=torch.float32)
            theta = torch.full((2, 3), value, dtype=torch.float32)
            if is_synthetic:
                from lpr.charset import PROVINCES
                label = PROVINCES[int(self.synthetic_targets[index])] + row["label"][1:]
            output.append((tensor, label, index, theta, row["subset"], is_synthetic))
        return output


def _rows_with_cache(tmp_path):
    rows = []
    pack = tmp_path / "tiny-jpegs.pack"
    offset = 0
    with pack.open("wb") as stream:
        for index, subset in enumerate(GROUPS):
            image = np.full((16, 16, 3), index * 13, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            payload = encoded.tobytes()
            stream.write(payload)
            rows.append({"path": f"fixture-{index}.jpg", "label": "皖A12345",
                         "subset": subset, "cache_pack": str(pack),
                         "pack_offset": str(offset), "pack_length": str(len(payload)),
                         "x0": "0", "y0": "0", "orig_w": "16", "orig_h": "16"})
            offset += len(payload)
    return rows


def _collect(loader):
    return list(loader)


def test_serial_and_single_thread_prefetch_batches_are_identical(tmp_path):
    rows = _rows_with_cache(tmp_path)
    serial_ds, prefetched_ds = TinyDataset(rows), TinyDataset(rows)
    serial = _collect(trainer.PackedEpoch(serial_ds, [r["label"] for r in rows],
        [r["subset"] for r in rows], batch_size=4, block_size=3, prefetch=False))
    prefetched = _collect(trainer.PackedEpoch(prefetched_ds, [r["label"] for r in rows],
        [r["subset"] for r in rows], batch_size=4, block_size=3, prefetch=True))
    assert len(serial) == len(prefetched)
    for left, right in zip(serial, prefetched):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            else:
                assert a == b
    assert len(serial_ds.row_threads) == len(prefetched_ds.row_threads) == len(rows)
    assert any(thread_id != threading.get_ident() for thread_id in prefetched_ds.row_threads)
    real_indices = [index for batch in prefetched for index, synthetic in zip(batch[4], batch[7])
                    if not synthetic]
    assert sorted(real_indices) == list(range(len(rows)))
    assert len(real_indices) == len(set(real_indices))


def test_gpu_fallback_marker_is_decoded_only_on_main_thread(tmp_path, monkeypatch):
    rows = [{"path": f"raw-{i}.jpg", "label": "皖A12345", "subset": group}
            for i, group in enumerate(GROUPS)]
    ds = TinyDataset(rows, gpu_fallback=True)
    main_thread = threading.get_ident()
    calls = []

    def stub_row_crop(row, bbox, *, gpu_fallback):
        calls.append((threading.get_ident(), row["path"], gpu_fallback))
        assert threading.get_ident() == main_thread
        assert gpu_fallback is True
        return np.ones((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(trainer, "row_crop", stub_row_crop)
    batches = _collect(trainer.PackedEpoch(ds, [r["label"] for r in rows],
        [r["subset"] for r in rows], batch_size=4, block_size=3, prefetch=True))
    assert batches
    assert len(calls) == len(rows)
    assert {thread_id for thread_id, _, _ in calls} == {main_thread}
    assert all(thread_id == main_thread for thread_id in ds.finish_threads)


def test_synthetic_gt_fallback_defers_worker_finish_and_reuses_photo_rng(monkeypatch):
    ds = trainer.ProvinceDataset.__new__(trainer.ProvinceDataset)
    ds.gpu_fallback = True
    ds.seed = 7
    ds.epoch = 3
    ds.kind = "char"
    ds.synthetic_targets = np.asarray([1], dtype=np.int16)
    ds.synthetic_gt_fallbacks = 0
    row = {"path": "fixture.jpg", "label": "皖A12345", "subset": "base", "quad": "fixture"}
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    bbox = (0, 0, 4, 4)
    monkeypatch.setattr(trainer, "local_quad", lambda *_: np.zeros((4, 2), dtype=np.float32))
    monkeypatch.setattr(trainer, "affine_target", lambda *_: torch.zeros((2, 3)))
    monkeypatch.setattr(trainer, "bbox_from_quad", lambda _quad: bbox)
    monkeypatch.setattr(trainer, "normalized_first_char",
                        lambda *_: np.zeros((32, 32), dtype=np.uint8))
    main_thread = threading.get_ident()
    row_crop_calls = []

    def stub_row_crop(_row, _bbox, *, gpu_fallback):
        row_crop_calls.append(threading.get_ident())
        assert threading.get_ident() == main_thread
        assert gpu_fallback is True
        return np.ones((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(trainer, "row_crop", stub_row_crop)
    replacement_calls = []

    def stub_replace(*args, **kwargs):
        replacement_calls.append(threading.get_ident())
        if len(replacement_calls) <= 2:
            raise ValueError("province glyph slot falls outside source image")
        return args[0].copy(), "沪A12345", {"fixture": True}

    monkeypatch.setattr(trainer, "_replace_province_locked", stub_replace)
    photo_rng = trainer.random.Random(99)
    rng_before = photo_rng.getstate()
    worker_errors = []

    def run_worker_finish():
        try:
            ds.finish(image, row, 0, photo_rng, bbox=bbox, defer_gt_fallback=True)
        except Exception as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=run_worker_finish)
    worker.start()
    worker.join()
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], trainer._DeferredFinish)
    assert row_crop_calls == []
    assert photo_rng.getstate() == rng_before

    finished = ds.finish(image, row, 0, photo_rng, bbox=bbox)
    assert len(row_crop_calls) == 1
    assert row_crop_calls[0] == main_thread
    assert ds.synthetic_gt_fallbacks == 1
    assert [item[1] for item in finished] == ["皖A12345", "沪A12345"]
