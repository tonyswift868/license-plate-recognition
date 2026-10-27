import io
import random
import tarfile

import cv2
import numpy as np
import pytest
import torch

from lpr_tools.train.packed_box_batches import PackedBoxBatches
from lpr_tools.train.train_box_reader import collate


def _memory_dataset(monkeypatch):
    import lpr_tools.train.packed_box_batches as module
    rows = []
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for index in range(9):
            ok, jpeg = cv2.imencode(".jpg", np.full((100, 200, 3), 90+index, np.uint8))
            assert ok
            offset = archive.tell()+512
            info = tarfile.TarInfo(f"{index}.jpg")
            info.size = len(jpeg)
            tar.addfile(info, io.BytesIO(jpeg.tobytes()))
            rows.append({"cache_pack": "memory", "pack_offset": str(offset),
                         "pack_length": str(len(jpeg)), "x0": "0", "y0": "0",
                         "orig_w": "200", "orig_h": "100", "path": f"source{index}"})
    contents = archive.getvalue()
    class MemoryPath:
        def __init__(self, path): pass
        def open(self, mode): return io.BytesIO(contents)
    monkeypatch.setattr(module, "Path", MemoryPath)
    class Dataset:
        seed, epoch, gpu_fallback = 42, 0, True
        def __len__(self): return len(rows)
        def row(self, index): return rows[index]
        def specification(self, index): return rows[index], random.Random(index), (20, 20, 80, 30)
        def finish(self, image, row, index, rng):
            assert image.size > 0
            return torch.zeros(3, 48, 160), "皖A12345", index
    return Dataset(), rows


@pytest.mark.parametrize("available", [1_500_000_000, 8_000_000_000])
def test_tar_blocks_and_low_memory_microblocks_cover_exactly_one_full_epoch(monkeypatch, available):
    import lpr_tools.train.train_box_reader as train
    monkeypatch.setattr(train, "free_ram_bytes", lambda: available)
    dataset, _ = _memory_dataset(monkeypatch)
    loader = PackedBoxBatches(dataset, 2, collate, block_size=4)
    batches = list(loader)
    assert sorted(i for batch in batches for i in batch[4]) == list(range(9))
    assert sum(len(batch[4]) for batch in batches) == 9
    assert loader.reduced_for_memory == (available < 2_000_000_000)
    assert loader.original_fallbacks == 0


def test_tar_offset_outside_archive_is_rejected(monkeypatch):
    dataset, rows = _memory_dataset(monkeypatch)
    rows[-1]["pack_length"] = "999999999"
    with pytest.raises(ValueError, match="offsets"):
        list(PackedBoxBatches(dataset, 2, collate, block_size=4))
