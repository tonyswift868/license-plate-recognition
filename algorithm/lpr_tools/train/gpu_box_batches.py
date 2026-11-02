"""Batched CUDA JPEG decoding, with CPU crop_box applied to identical source pixels."""
from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from lpr.boxcrop import crop_box
from lpr.recognize.box_reader import BOX_PAD, box_tensor
from lpr_tools.train.box_data import bbox_from_quad, jitter_bbox, sample_jitter
from torchvision.io import ImageReadMode, decode_jpeg, read_file


def rgb_region_for_box(rgb: torch.Tensor, bbox: tuple, gt_bbox: tuple,
                       pad: tuple = BOX_PAD) -> tuple[np.ndarray, tuple]:
    """Copy enough original pixels to CPU, retaining integer origin for exact crop_box.

    Transfer at least the GT+50% region, enlarged to include any measured jitter.
    Translation by an integer preserves crop_box's floor/ceil and image clipping.
    No JPEG re-encoding, perspective transform or interpolation precedes crop_box.
    """
    ih, iw = rgb.shape[1:]
    x, y, w, h = bbox
    gx, gy, gw, gh = gt_bbox
    x0 = max(0, math.floor(min(gx-.5*gw, x-pad[0]*w)))
    y0 = max(0, math.floor(min(gy-.5*gh, y-pad[1]*h)))
    x1 = min(iw, math.ceil(max(gx+1.5*gw, x+(1+pad[0])*w)))
    y1 = min(ih, math.ceil(max(gy+1.5*gh, y+(1+pad[1])*h)))
    bgr = rgb[:, y0:y1, x0:x1].permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]
    return bgr, (x-x0, y-y0, w, h)


class GpuBoxBatches:
    """One Python process, at most three byte-read threads, no CUDA DataLoader workers."""

    def __init__(self, dataset, batch_size: int, collate_fn, *, read_threads: int = 3):
        self.dataset, self.batch_size, self.collate_fn = dataset, batch_size, collate_fn
        self.read_threads = read_threads
        self.reduced_for_memory = False
        self.decoded_images = 0
        self.decode_errors = 0

    def __iter__(self):
        from lpr_tools.train.train_box_reader import free_ram_bytes

        dataset = self.dataset
        order = np.random.default_rng(dataset.seed + dataset.epoch).permutation(len(dataset))
        with ThreadPoolExecutor(max_workers=self.read_threads) as pool:
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start+self.batch_size].tolist()
                rows = [dataset.row(index) for index in indices]
                if free_ram_bytes() < 2_000_000_000:
                    self.reduced_for_memory = True
                # Once reduced, keep serial reads for this run; no repeated scale-up.
                paths = [row["path"] for row in rows]
                encoded = ([read_file(path) for path in paths] if self.reduced_for_memory
                           else list(pool.map(read_file, paths)))
                decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
                batch = []
                for row, index, rgb in zip(rows, indices, decoded, strict=False):
                    rng = random.Random(dataset.seed + dataset.epoch*1_000_003 + index)
                    gt = bbox_from_quad(row["quad"])
                    bbox = gt
                    if rng.random() < .65:
                        bbox = jitter_bbox(gt, sample_jitter(rng, row["subset"], dataset.samples))
                    source, local_box = rgb_region_for_box(rgb, bbox, gt)
                    image = crop_box(source, local_box, *BOX_PAD)
                    if not image.size:
                        raise ValueError(f"measured jitter produced empty crop: {row['path']}")
                    if rng.random() < .4:
                        gamma = rng.uniform(.65, 1.5)
                        lut = np.asarray([255*(v/255.)**gamma for v in range(256)], np.uint8)
                        image = cv2.LUT(image, lut)
                    if rng.random() < .15:
                        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.2, 1.))
                    batch.append((box_tensor(image), row["label"], index))
                self.decoded_images += len(decoded)
                # Drop the full-resolution CUDA images before model activations allocate.
                del decoded, encoded, rgb, source
                yield self.collate_fn(batch)
