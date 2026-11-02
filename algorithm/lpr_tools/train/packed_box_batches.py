"""Read a standard JPEG TAR in shuffled small blocks, without extracting image files."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from lpr.recognize.box_reader import BOX_PAD


class PackedBoxBatches:
    """No workers or persistent decoded images: one compact block buffer at a time."""

    def __init__(self, dataset, batch_size: int, collate_fn, *, block_size: int = 256):
        if batch_size <= 0 or block_size % batch_size:
            raise ValueError("packed block size must be a positive multiple of batch size")
        self.dataset, self.batch_size, self.collate_fn = dataset, batch_size, collate_fn
        self.block_size = block_size
        self.original_fallbacks = 0
        self.reduced_for_memory = False  # Already zero workers / zero prefetch threads.

    def __iter__(self):
        from lpr_tools.train.train_box_reader import free_ram_bytes

        dataset = self.dataset
        rng = np.random.default_rng(dataset.seed + dataset.epoch)
        block_order = rng.permutation(math.ceil(len(dataset)/self.block_size))
        with Path(dataset.row(0)["cache_pack"]).open("rb") as stream:
            total_size = stream.seek(0, 2)
            for block in block_order:
                begin = int(block)*self.block_size
                end = min(len(dataset), begin+self.block_size)
                if free_ram_bytes() < 2_000_000_000:
                    self.reduced_for_memory = True
                # Workers are already zero. Under pressure reduce encoded prefetch
                # from 256 images to a single training batch; coverage is unchanged.
                quantum = self.batch_size if self.reduced_for_memory else self.block_size
                for part in range(begin, end, quantum):
                    yield from self._read_range(stream, total_size, part, min(end, part+quantum), rng)

    def _read_range(self, stream, total_size: int, begin: int, end: int, rng):
        from lpr_tools.train.train_box_reader import cache_contains, row_crop

        dataset = self.dataset
        rows = [dataset.row(index) for index in range(begin, end)]
        offsets = [int(row["pack_offset"]) for row in rows]
        lengths = [int(row["pack_length"]) for row in rows]
        if (any(length <= 0 for length in lengths) or offsets != sorted(offsets)
                or offsets[0] < 0 or offsets[-1]+lengths[-1] > total_size):
            raise ValueError("packed JPEG offsets are invalid")
        stream.seek(offsets[0])
        data = stream.read(offsets[-1]+lengths[-1]-offsets[0])
        batch = []
        for relative in rng.permutation(end-begin):
            relative = int(relative)
            index = begin+relative
            row, photo_rng, bbox = dataset.specification(index)
            encoded = np.frombuffer(data, dtype=np.uint8,
                                    offset=offsets[relative]-offsets[0], count=lengths[relative])
            cached = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if cached is None:
                raise ValueError(f"unreadable packed JPEG at source {row['path']}")
            if not cache_contains(row, bbox, BOX_PAD, cached.shape):
                self.original_fallbacks += 1
            image = row_crop(row, bbox, cached_image=cached, gpu_fallback=dataset.gpu_fallback)
            batch.append(dataset.finish(image, row, index, photo_rng))
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)
