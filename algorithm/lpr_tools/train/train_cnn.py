"""Train CharCNN on one or more normalized character directories."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image = augment_char(image, rng)
        return torch.from_numpy(image.copy()).unsqueeze(0).float() / 255.0, int(self.labels[index])


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


def _as_sources(data, data_weights=None):
    sources = [Path(data)] if isinstance(data, (str, Path)) else [Path(p) for p in data]
    if not sources:
        raise ValueError("at least one --data directory is required")
    weights = [1.0] * len(sources) if data_weights is None else [float(w) for w in data_weights]
    if len(weights) != len(sources) or any(not np.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("data_weights must contain one positive finite weight per data directory")
    return sources, weights


def _load_sources(sources, weights, split, *, seed, require_all):
    imgs, labs, sample_weights = [], [], []
    for source_id, (root, source_weight) in enumerate(zip(sources, weights)):
        images, labels = load_split(root, split, seed=seed + source_id, require_all=False)
        if not len(labels):
            if split == "val":
                continue
            raise ValueError(f"no {split} samples found in {root}")
        imgs.append(images)
        labs.append(labels)
        # The weight controls the source's total sampling probability, then samples
        # are uniform within that source. Normalize below in WeightedRandomSampler.
        sample_weights.extend([source_weight / len(labels)] * len(labels))
    if not imgs:
        raise ValueError("combined val data contains no samples")
    images, labels = np.concatenate(imgs), np.concatenate(labs)
    if require_all:
        missing = np.flatnonzero(np.bincount(labels, minlength=NUM_CLASSES) == 0)
        if len(missing):
            raise ValueError(f"combined {split} data is missing classes: {missing.tolist()}")
    return images, labels, np.asarray(sample_weights, dtype=np.double)


def _data_signature(sources):
    signature = []
    for root in sources:
        files = []
        for split in ("train", "val"):
            for path in sorted((root / split).glob("*/*.png")):
                stat = path.stat()
                files.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
        signature.append({"path": str(root.resolve()), "files": files})
    return signature


def _save_checkpoint(path, *, model, optimizer, scheduler, epoch, config, data_signature,
                     loader_generator, sampler_generator, train_seconds, val_char_acc):
    np_state = np.random.get_state()
    checkpoint = {
        "format": 1, "epoch": epoch, "config": config, "data_signature": data_signature,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
        "numpy_rng": {"name": np_state[0], "keys": torch.as_tensor(np_state[1].astype(np.int64)),
                      "pos": int(np_state[2]), "has_gauss": int(np_state[3]),
                      "cached_gaussian": float(np_state[4])},
        "torch_rng": torch.get_rng_state(),
        "loader_rng": loader_generator.get_state(), "sampler_rng": sampler_generator.get_state(),
        "train_seconds": train_seconds,
        "val_char_acc": val_char_acc,
    }
    torch.save(checkpoint, path)


def train(data, out: Path, *, version: str, epochs: int = 15,
          batch_size: int = 256, lr: float = 1e-3, seed: int = 42,
          num_workers: int = 0, weight_decay: float = 1e-4,
          log_path: Path | None = None, data_weights=None,
          resume: Path | None = None, checkpoint_path: Path | None = None,
          max_epochs_this_run: int | None = None) -> list[dict]:
    if (epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0
            or not np.isfinite(weight_decay) or weight_decay < 0
            or num_workers not in (0, 1, 2)):
        raise ValueError("epochs/batch_size/lr must be positive; num_workers must be 0..2")
    sources, weights_by_source = _as_sources(data, data_weights)
    # epochs means the desired total epoch count, including epochs already in checkpoint.
    config = {"epochs": epochs, "batch_size": batch_size, "lr": lr,
              "weight_decay": weight_decay, "seed": seed, "num_workers": num_workers,
              "data_weights": weights_by_source}
    signature = _data_signature(sources)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); cv2.setNumThreads(1)
    train_images, train_labels, sampling_weights = _load_sources(
        sources, weights_by_source, "train", seed=seed, require_all=True)
    val_images, val_labels, _ = _load_sources(sources, weights_by_source, "val", seed=seed, require_all=False)
    train_data = CharacterDataset(train_images, train_labels, seed=seed, augment=True)
    val_data = CharacterDataset(val_images, val_labels, seed=seed, augment=False)
    loader_generator = torch.Generator().manual_seed(seed)
    sampler_generator = torch.Generator().manual_seed(seed + 1)
    sampler = WeightedRandomSampler(torch.as_tensor(sampling_weights), len(train_data), replacement=True,
                                    generator=sampler_generator)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                              generator=loader_generator, num_workers=num_workers, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    model = CharCNN()
    # Class loss weights use raw merged counts; source weights independently set
    # source exposure through the sampler and are not included in this formula.
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = torch.tensor(len(train_labels) / (NUM_CLASSES * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else out.with_suffix(".checkpoint.pt")
    log_path = Path(log_path) if log_path is not None else out.with_suffix(".train.csv")
    sidecar_path = out.with_suffix(".training.json")
    protected = {p.resolve() for p in (out, checkpoint_path, log_path, sidecar_path)}
    if len(protected) != 4:
        raise ValueError("checkpoint, inference model, training log, and timing sidecar must use distinct paths")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    first_epoch, elapsed_total, last_accuracy = 0, 0.0, None
    if resume is not None:
        state = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if state.get("format") != 1 or state.get("data_signature") != signature:
            raise ValueError("checkpoint format or training data directories are incompatible")
        if state["config"] != config:
            raise ValueError("checkpoint hyperparameters/sampling weights are incompatible")
        first_epoch = int(state["epoch"])
        if epochs < first_epoch:
            raise ValueError("epochs is the target total and cannot be less than completed checkpoint epochs")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np_state = state["numpy_rng"]
        np.random.set_state((np_state["name"], np_state["keys"].cpu().numpy().astype(np.uint32),
                             int(np_state["pos"]), int(np_state["has_gauss"]),
                             float(np_state["cached_gaussian"])))
        torch.set_rng_state(state["torch_rng"])
        loader_generator.set_state(state["loader_rng"]); sampler_generator.set_state(state["sampler_rng"])
        elapsed_total = float(state.get("train_seconds", 0.0))
        last_accuracy = state.get("val_char_acc")

    run_end = epochs if max_epochs_this_run is None else min(epochs, first_epoch + max_epochs_this_run)
    if max_epochs_this_run is not None and max_epochs_this_run < 1:
        raise ValueError("max_epochs_this_run must be positive")

    fields = ["epoch", "loss", "val_char_acc", "lr", "epoch_seconds", "train_seconds"]
    mode = "a" if first_epoch and log_path.exists() and log_path.stat().st_size else "w"
    history = []
    with log_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w": writer.writeheader()
        for epoch in range(first_epoch, run_end):
            started = time.perf_counter()
            model.train(); train_data.epoch = epoch
            loss_total = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels); loss.backward(); optimizer.step()
                loss_total += loss.item() * len(labels)
            model.eval(); correct = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    correct += int((model(images).argmax(dim=1) == labels).sum())
            epoch_seconds = time.perf_counter() - started
            elapsed_total += epoch_seconds
            row = {"epoch": epoch + 1, "loss": loss_total / len(train_data),
                   "val_char_acc": correct / len(val_data), "lr": optimizer.param_groups[0]["lr"],
                   "epoch_seconds": epoch_seconds, "train_seconds": elapsed_total}
            writer.writerow(row); handle.flush()
            last_accuracy = row["val_char_acc"]
            # Keep the public history deterministic for existing callers; timings
            # live in the flushed CSV and the .training.json sidecar.
            history.append({key: row[key] for key in ("epoch", "loss", "val_char_acc", "lr")})
            print(f"epoch={epoch + 1}/{epochs} loss={row['loss']:.6f} "
                  f"val_char_acc={row['val_char_acc']:.6f} lr={row['lr']:.8f} "
                  f"epoch_seconds={epoch_seconds:.3f}", flush=True)
            scheduler.step()
            _save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler,
                             epoch=epoch + 1, config=config, data_signature=signature,
                             loader_generator=loader_generator, sampler_generator=sampler_generator,
                             train_seconds=elapsed_total, val_char_acc=last_accuracy)
    torch.save(model.state_dict(), out)
    source_manifest = json.dumps([{"path": str(p), "weight": w} for p, w in zip(sources, weights_by_source)],
                                 ensure_ascii=False)
    final_acc = history[-1]["val_char_acc"] if history else last_accuracy
    write_manifest_entry(out, version=version, train_data=source_manifest, val_char_acc=final_acc)
    sidecar_path.write_text(
        json.dumps({"train_seconds": elapsed_total}, indent=2) + "\n", encoding="utf-8")
    print(f"saved={out.as_posix()} version={version} log={log_path.as_posix()} "
          f"checkpoint={checkpoint_path.as_posix()} train_seconds={elapsed_total:.3f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="training data root; repeat for mixed sources")
    parser.add_argument("--data-weight", type=float, action="append",
                        help="source sampling weight, one per --data (default 1 each)")
    parser.add_argument("--epochs", type=int, default=15, help="target total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", type=Path, help="resume from a training checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="training checkpoint path (separate from inference weights)")
    parser.add_argument("--stop-after", type=int,
                        help="optional maximum epochs in this invocation; checkpoints at epoch boundaries")
    args = parser.parse_args()
    train(args.data, args.out, version=args.version, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, seed=args.seed, num_workers=args.num_workers,
          weight_decay=args.weight_decay, log_path=args.log, data_weights=args.data_weight,
          resume=args.resume, checkpoint_path=args.checkpoint, max_epochs_this_run=args.stop_after)


if __name__ == "__main__":
    main()
