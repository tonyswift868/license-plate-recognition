"""Training-only GT-box checkpoint selection, executed by the office CPU environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from lpr.recognize.box_reader import BoxPlateReader
from lpr_tools.train.train_box_reader import BoxDataset, collate, evaluate, process_ram_bytes
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tune", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reader = BoxPlateReader.from_file(args.model, version="checkpoint-selection-only")
    dataset = BoxDataset(args.tune, {}, training=False)
    loader = DataLoader(dataset, batch_size=64, num_workers=0, collate_fn=collate)
    result = evaluate(reader.model, loader, torch.device("cpu"))
    result["cpu_peak_ram_bytes"] = process_ram_bytes(peak=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
