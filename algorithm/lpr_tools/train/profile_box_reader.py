"""Bounded CPU read / GPU step probe before committing to a 30-epoch run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lpr.charset import CHAR_TO_INDEX, NUM_CLASSES
from lpr.recognize.box_reader import INPUT_SIZE, BoxNet, BoxPlateReader
from lpr_tools.train.train_box_reader import free_ram_bytes, process_ram_bytes


def profile(device: str, batch: int, steps: int) -> dict:
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if device == "cuda":
        if free_ram_bytes() < 4_000_000_000:
            raise RuntimeError("less than 4 GB free RAM before GPU probe")
        torch.cuda.set_per_process_memory_fraction(.42)
    model = BoxNet().to(device)
    if device == "cpu":
        reader = BoxPlateReader("throughput-probe-only", model)
        image = np.random.default_rng(42).integers(0, 256, (80, 240, 3), dtype=np.uint8)
        for _ in range(5):
            reader.read(image)
        times = []
        for _ in range(steps):
            started = time.perf_counter_ns()
            reader.read(image)
            times.append((time.perf_counter_ns()-started)/1e6)
        return {"device": device, "threads": 2, "n": steps, "mean_ms": float(np.mean(times)),
                "p95_ms": float(np.percentile(times, 95)), "max_ms": max(times),
                "over_10ms": sum(v > 10 for v in times), "ram_bytes": process_ram_bytes(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "note": "untrained synthetic crop throughput probe, not final accuracy or latency acceptance"}
    model.train()
    images = torch.randn(batch, 3, INPUT_SIZE[1], INPUT_SIZE[0], device=device)
    target = torch.tensor([CHAR_TO_INDEX[c] for c in "皖A12345"]*batch, device=device)
    lengths = torch.full((batch,), 7, dtype=torch.long)
    criterion = torch.nn.CTCLoss(blank=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    times = []
    for step in range(steps+5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits.log_softmax(-1), target,
                         torch.full((batch,), logits.shape[0], dtype=torch.long), lengths)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if step >= 5:
            times.append(elapsed)
        if process_ram_bytes() > 2_400_000_000:
            raise RuntimeError("GPU probe exceeds process RAM budget")
    return {"device": device, "threads": 2, "batch_size": batch, "steps": steps,
            "mean_step_seconds": float(np.mean(times)), "images_per_second": batch/np.mean(times),
            "ram_bytes": process_ram_bytes(), "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
            "note": "synthetic compute-only probe excludes real decode/crop I/O"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = profile(args.device, args.batch_size, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
