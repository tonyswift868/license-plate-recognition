"""Train CharCNN on normalized character directories (CPU, no pretrained weights)."""

import argparse
import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lpr.charset import NUM_CLASSES
from lpr.recognize import write_manifest_entry
from lpr.recognize.cnn_model import CharCNN
from .augment import augment_char
from .data import load_split


class CharacterDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, *, seed: int,
                 augment: bool) -> None:
        self.images, self.labels = images, labels
        self.seed, self.augment, self.epoch = seed, augment, 0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if self.augment:
            # Independent per epoch/index, also reproducible with Windows workers.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image = augment_char(image, rng)
        return torch.from_numpy(image.copy()).unsqueeze(0).float() / 255.0, int(self.labels[index])


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


def train(data: Path, out: Path, *, version: str, epochs: int = 15,
          batch_size: int = 256, lr: float = 1e-3, seed: int = 42,
          num_workers: int = 0, weight_decay: float = 1e-4,
          log_path: Path | None = None) -> list[dict]:
    if epochs < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0 or num_workers not in (0, 1, 2):
        raise ValueError("epochs/batch_size/lr must be positive; num_workers must be 0..2")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    train_images, train_labels = load_split(data, "train")
    val_images, val_labels = load_split(data, "val")
    train_data = CharacterDataset(train_images, train_labels, seed=seed, augment=True)
    val_data = CharacterDataset(val_images, val_labels, seed=seed, augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=num_workers, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    model = CharCNN()
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    weights = torch.tensor(len(train_labels) / (NUM_CLASSES * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path is not None else out.with_suffix(".train.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "val_char_acc", "lr"])
        writer.writeheader()
        for epoch in range(epochs):
            model.train()
            train_data.epoch = epoch
            loss_total = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(labels)
            model.eval()
            correct = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    correct += int((model(images).argmax(dim=1) == labels).sum())
            row = {"epoch": epoch + 1, "loss": loss_total / len(train_data),
                   "val_char_acc": correct / len(val_data), "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(row)
            handle.flush()
            history.append(row)
            print(f"epoch={epoch + 1}/{epochs} loss={row['loss']:.6f} "
                  f"val_char_acc={row['val_char_acc']:.6f} lr={row['lr']:.8f}", flush=True)
            scheduler.step()
    torch.save(model.state_dict(), out)
    write_manifest_entry(out, version=version, train_data=str(data),
                         val_char_acc=history[-1]["val_char_acc"])
    print(f"saved={out.as_posix()} version={version} log={log_path.as_posix()}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    train(args.data, args.out, version=args.version, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, seed=args.seed, num_workers=args.num_workers,
          weight_decay=args.weight_decay, log_path=args.log)


if __name__ == "__main__":
    main()
