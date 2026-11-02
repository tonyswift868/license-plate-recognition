"""Full-epoch direct-box training, bounded crop cache and explicit CPU/GPU CLI.

CSV rows are mmap-indexed rather than copied into workers. Cache JPEGs retain
original spatial resolution; a jitter needing pixels outside the cache falls
back to the original image, never silently truncates the requested crop.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import mmap
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.boxcrop import crop_box
from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize import file_sha256
from lpr.recognize.box_reader import BOX_PAD, INPUT_SIZE, BoxNet, box_tensor, checkpoint
from lpr.recognize.crnn import _greedy_decode
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lpr_tools.train.box_data import bbox_from_quad, iter_rows, jitter_bbox, sample_jitter


def free_ram_bytes() -> int:
    if os.name != "nt":
        import psutil
        return psutil.virtual_memory().available
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(key, ctypes.c_ulonglong) for key in
                      ("total", "available", "page_total", "page_available", "virtual_total",
                       "virtual_available", "extended")]]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available)


def process_ram_bytes(*, peak: bool = False) -> int:
    # workers=0, so the training interpreter owns all substantial working memory.
    # The launcher also records its small redirector and PowerShell parent externally.
    if os.name != "nt":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    *[(key, ctypes.c_size_t) for key in
                      ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                       "pagefile", "peak_pagefile", "private")]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.peak_rss if peak else counters.rss)


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def prepare_cache(pool: Path, cache: Path, *, limit: int = 2_800_000_000) -> dict:
    """A single resumable temporary cache; no plate rectification or rescaling."""
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "index.csv"
    fields = ("path", "label", "subset", "brightness", "sha256", "quad", "cache_file",
              "x0", "y0", "orig_w", "orig_h")
    used = sum(p.stat().st_size for p in cache.rglob("*.jpg"))
    count = 0
    started = time.perf_counter()
    temporary = cache / "index.partial.csv"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in iter_rows(pool):
            source = read_image(row["path"])
            bbox = bbox_from_quad(row["quad"])
            x, y, width, height = bbox
            x0, y0 = max(0, math.floor(x - .5 * width)), max(0, math.floor(y - .5 * height))
            crop = crop_box(source, bbox, .5, .5)
            if not crop.size:
                raise ValueError(f"empty cache crop: {row['path']}")
            target = cache / row["sha256"][:2] / (row["sha256"] + ".jpg")
            if not target.exists():
                success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not success or used + encoded.nbytes > limit:
                    raise RuntimeError("crop cache would exceed 3 GB; shrink audited pool first")
                target.parent.mkdir(parents=True, exist_ok=True)
                encoded.tofile(target)
                used += encoded.nbytes
            record = {key: row[key] for key in fields[:6]}
            record.update(cache_file=str(target.resolve()), x0=x0, y0=y0,
                          orig_w=source.shape[1], orig_h=source.shape[0])
            writer.writerow(record)
            count += 1
            if count % 5000 == 0:
                stream.flush()
                print(json.dumps({"cache_count": count, "jpeg_bytes": used,
                                  "seconds": time.perf_counter() - started}), flush=True)
    temporary.replace(index)
    report = {"pool_sha256": file_sha256(pool), "cache_index_sha256": file_sha256(index),
              "cache_path": str(cache.resolve()), "samples": count, "jpeg_bytes": used,
              "seconds": time.perf_counter() - started, "quality": 90,
              "scale": "original resolution", "gt_bbox_pad": [.5, .5]}
    (cache / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def cache_contains(row: dict, bbox: tuple, pad: tuple, cached_shape: tuple) -> bool:
    x, y, w, h = bbox
    x0, y0 = int(row["x0"]), int(row["y0"])
    left, top = max(0, math.floor(x-pad[0]*w)), max(0, math.floor(y-pad[1]*h))
    right = min(int(row["orig_w"]), math.ceil(x+w+pad[0]*w))
    bottom = min(int(row["orig_h"]), math.ceil(y+h+pad[1]*h))
    return left >= x0 and top >= y0 and right <= x0+cached_shape[1] and bottom <= y0+cached_shape[0]


def row_crop(row: dict, bbox: tuple, pad: tuple = BOX_PAD, *,
             cached_image: np.ndarray | None = None, gpu_fallback: bool = False) -> np.ndarray:
    if cached_image is not None:
        if cache_contains(row, bbox, pad, cached_image.shape):
            x, y, w, h = bbox
            return crop_box(cached_image, (x-int(row["x0"]), y-int(row["y0"]), w, h), *pad)
    if row.get("cache_file"):
        source = read_image(row["cache_file"])
        if cache_contains(row, bbox, pad, source.shape):
            x, y, w, h = bbox
            return crop_box(source, (x-int(row["x0"]), y-int(row["y0"]), w, h), *pad)
    if gpu_fallback:
        from torchvision.io import ImageReadMode, decode_jpeg, read_file

        from lpr_tools.train.gpu_box_batches import rgb_region_for_box
        rgb = decode_jpeg(read_file(row["path"]), mode=ImageReadMode.RGB, device="cuda")
        source, local = rgb_region_for_box(rgb, bbox, bbox_from_quad(row["quad"]), pad)
        return crop_box(source, local, *pad)
    return crop_box(read_image(row["path"]), bbox, *pad)


class BoxDataset(Dataset):
    def __init__(self, index: Path, samples: dict, *, training: bool, seed: int = 42,
                 gpu_fallback: bool = False) -> None:
        self.path = Path(index)
        self.samples, self.training, self.seed, self.epoch = samples, training, seed, 0
        self.gpu_fallback = gpu_fallback
        offsets = []
        with self.path.open("rb") as stream:
            self.header = next(csv.reader([stream.readline().decode("utf-8-sig").strip()]))
            while True:
                position = stream.tell()
                if not stream.readline():
                    break
                offsets.append(position)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._file = self._map = None
        if not len(self.offsets):
            raise ValueError("empty box index")

    def __len__(self):
        return len(self.offsets)

    def row(self, index: int) -> dict:
        if self._map is None:
            self._file = self.path.open("rb")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._map.seek(int(self.offsets[index]))
        values = next(csv.reader([self._map.readline().decode("utf-8").strip()]))
        return dict(zip(self.header, values, strict=False))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = state["_map"] = None
        return state

    def specification(self, index: int):
        row = self.row(index)
        rng = random.Random(self.seed + self.epoch*1_000_003 + index)
        bbox = bbox_from_quad(row["quad"])
        if self.training and rng.random() < .65:
            bbox = jitter_bbox(bbox, sample_jitter(rng, row["subset"], self.samples))
        return row, rng, bbox

    def finish(self, image: np.ndarray, row: dict, index: int, rng: random.Random):
        if not image.size:
            raise ValueError(f"measured jitter produced empty crop: {row['path']}")
        if self.training:
            # Mild photometric variation leaves the actual dark/tilt geometry intact.
            if rng.random() < .4:
                gamma = rng.uniform(.65, 1.5)
                lut = np.asarray([255*(v/255.)**gamma for v in range(256)], np.uint8)
                image = cv2.LUT(image, lut)
            if rng.random() < .15:
                image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.2, 1.))
        return box_tensor(image), row["label"], index

    def __getitem__(self, index: int):
        row, rng, bbox = self.specification(index)
        image = row_crop(row, bbox, gpu_fallback=self.gpu_fallback)
        return self.finish(image, row, index, rng)


def collate(rows):
    images, labels, indices = zip(*rows, strict=False)
    targets = torch.tensor([CHAR_TO_INDEX[c] for label in labels for c in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, lengths, list(labels), list(indices)


def evaluate(model: BoxNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = count = 0
    with torch.inference_mode():
        for images, _, _, labels, _ in loader:
            logits = model(images.to(device)).cpu()
            for i, label in enumerate(labels):
                correct += _greedy_decode(logits[:, i])[0] == label
                count += 1
    return {"correct": correct, "n": count, "accuracy": correct/count}


def run(args) -> dict:
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.workers != 0:
        raise ValueError("resource-constrained release uses DataLoader workers=0")
    ram_limit = 1_000_000_000 if args.smoke and args.device == "cpu" else 2_700_000_000
    if args.epochs < 30 and not args.smoke:
        raise ValueError("release training requires at least 30 complete epochs")
    if not args.smoke:
        if args.audit is None:
            raise ValueError("release training requires --audit")
        audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
        hashes = audit["output_hashes"]
        if file_sha256(args.tune) != hashes["tune.csv"] or file_sha256(args.sampler) != hashes["sampler.json"]:
            raise ValueError("tune or sampler differs from the isolation audit")
        if args.decode_jpeg:
            if args.device != "cuda" or file_sha256(args.train) != hashes["pool.csv"]:
                raise ValueError("CUDA JPEG training requires the audited original pool")
        else:
            cache_audit_path = args.train.parent / "audit.json"
            cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8-sig"))
            if (cache_audit.get("pool_sha256") != hashes["pool.csv"]
                    or file_sha256(args.train) != cache_audit.get("cache_index_sha256")):
                raise ValueError("cache training index differs from the audited full pool")
        checks = audit["content_checks"]
        if any(checks[key] != 0 for key in ("train_tune_sha_overlap", "train_protected_sha_overlap",
                                           "tune_protected_sha_overlap")):
            raise ValueError("training data overlaps held-out source bytes")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        if free_ram_bytes() < 4_000_000_000:
            raise RuntimeError("less than 4 GB free RAM; wait before GPU training")
        torch.cuda.set_per_process_memory_fraction(.42)
        # .42 is the dispatch ceiling; also honor a strict decimal 5 GB allocator cap.
        strict_fraction = min(.42, 5_000_000_000/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.set_per_process_memory_fraction(strict_fraction)
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    metadata = json.loads(args.sampler.read_text(encoding="utf-8"))
    samples = metadata.get("samples_by_subset", metadata.get("subsets", metadata))
    dataset = BoxDataset(args.train, samples, training=True, seed=args.seed,
                         gpu_fallback=args.device == "cuda")
    tune = BoxDataset(args.tune, samples, training=False, seed=args.seed)
    if args.packed_cache:
        from lpr_tools.train.packed_box_batches import PackedBoxBatches
        loader = PackedBoxBatches(dataset, args.batch_size, collate)
    elif args.decode_jpeg:
        from lpr_tools.train.gpu_box_batches import GpuBoxBatches
        loader = GpuBoxBatches(dataset, args.batch_size, collate, read_threads=3)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            drop_last=False, collate_fn=collate,
                            generator=torch.Generator().manual_seed(args.seed))
    tune_loader = DataLoader(tune, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)
    model = BoxNet(imagenet=args.imagenet).to(device)
    init_source = None
    if args.init_checkpoint:
        init_source = {"path": str(args.init_checkpoint.resolve()), "sha256": file_sha256(args.init_checkpoint)}
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        from lpr.recognize.box_reader import ARCH
        if saved.get("architecture") != ARCH or saved.get("input_size") != list(INPUT_SIZE):
            raise ValueError("initial checkpoint architecture mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        init_source["optimizer_reset"] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    loss_fn = nn.CTCLoss(blank=NUM_CLASSES, zero_infinity=False)
    history = []
    best = -1.
    started = time.perf_counter()
    ram_peak = 0
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        model.train()
        loss_sum = 0.
        seen = np.zeros(len(dataset), dtype=np.bool_)
        epoch_started = time.perf_counter()
        for step, (images, targets, lengths, _, indices) in enumerate(loader, 1):
            if seen[indices].any():
                raise RuntimeError("duplicate sample within an epoch")
            seen[indices] = True
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits.log_softmax(-1), targets.to(device),
                           torch.full((len(images),), logits.shape[0], dtype=torch.long), lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            if step % 50 == 0 or step == 1:
                ram = process_ram_bytes()
                ram_peak = max(ram_peak, ram)
                if ram > ram_limit:
                    raise RuntimeError(f"own process RAM exceeds budget {ram_limit}: {ram}")
                print(json.dumps({"epoch": epoch+1, "step": step, "seen": int(seen.sum()),
                                  "loss": float(loss.detach()), "ram_bytes": ram,
                                  "seconds": time.perf_counter()-started}), flush=True)
        if not seen.all():
            raise RuntimeError("incomplete epoch")
        payload = checkpoint(model) if not args.smoke else None
        if not args.smoke:
            torch.save(payload, args.out / "last.pt")
        if args.device == "cuda":
            # All selection inference, like final evaluation, uses the office CPU venv.
            scored = subprocess.run([str(args.cpu_python), "-m", "lpr_tools.train.score_box_reader",
                                     "--model", str(args.out / "last.pt"), "--tune", str(args.tune)],
                                    check=True, text=True, capture_output=True, encoding="utf-8")
            score = json.loads(scored.stdout)
        else:
            score = evaluate(model, tune_loader, device)
        record = {"epoch": epoch+1, "samples": len(dataset), "unique_samples": int(seen.sum()),
                  "loss": loss_sum/len(dataset), "tune": score,
                  "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter()-epoch_started}
        history.append(record)
        if score["accuracy"] > best:
            best = score["accuracy"]
            if not args.smoke:
                torch.save(payload, args.out / "best.pt")
        scheduler.step()
        report = {"recipe": vars(args) | {"device": str(device)}, "history": history,
                  "best_accuracy": best, "best_epoch": max(history, key=lambda r:r["tune"]["accuracy"])["epoch"],
                  "total_exposures": len(dataset)*(epoch+1),
                  "ram_peak_bytes": max(ram_peak, process_ram_bytes(peak=True)),
                  "gpu_allocated_peak": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                  "gpu_reserved_peak": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
                  "io_reduced_for_low_memory": getattr(loader, "reduced_for_memory", False),
                  "original_gpu_fallbacks": getattr(loader, "original_fallbacks", 0),
                  "init_source": init_source,
                  "wall_seconds": time.perf_counter()-started,
                  "train_sha256": file_sha256(args.train), "tune_sha256": file_sha256(args.tune)}
        (args.out / "history.json").write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
        projection = {"completed_epochs": epoch+1, "latest_epoch_seconds": record["seconds"],
                      "mean_epoch_seconds": sum(r["seconds"] for r in history)/len(history),
                      "projected_total_seconds": sum(r["seconds"] for r in history)/len(history)*args.epochs,
                      "exceeds_eight_hours": sum(r["seconds"] for r in history)/len(history)*args.epochs > 28800}
        (args.out / "projection.json").write_text(json.dumps(projection, indent=2), encoding="utf-8")
        if epoch == 0:
            with Path("algorithm/out/receipts/第十批.md").open("a", encoding="utf-8") as receipt:
                receipt.write("\n首个正式epoch已完成："+json.dumps(projection, ensure_ascii=False)+"\n")
            if projection["exceeds_eight_hours"]:
                raise RuntimeError("first-epoch projection exceeds 8 hours; reduce scale before continuing")
        print(json.dumps(record), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--decode-jpeg", action="store_true", help="batch torchvision CUDA JPEG decode of original images")
    parser.add_argument("--packed-cache", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--cpu-python", type=Path, default=Path(".venv/Scripts/python.exe").resolve())
    parser.add_argument("--smoke", action="store_true", help="explicit non-release tiny CPU smoke")
    parser.add_argument("--prepare-cache", type=Path)
    args = parser.parse_args()
    if args.prepare_cache:
        cv2.setNumThreads(1)
        print(json.dumps(prepare_cache(args.train, args.prepare_cache), indent=2))
    else:
        print(json.dumps(run(args), default=str))


if __name__ == "__main__":
    main()
