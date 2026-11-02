"""Train a compact plate-level CRNN with CTC from plates/v1 labels.csv."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.charset import CHAR_TO_INDEX, CHARSET
from lpr.recognize import write_manifest_entry
from lpr.recognize.crnn import BLANK, INPUT_SIZE, CrnnNet
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _read_rows(root: Path, split: str, excluded_origins: set[str] | None = None) -> list[dict[str, str]]:
    root_abs = os.path.abspath(root)
    path = Path(root_abs) / "labels.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            file_path = Path(row["file"])
            if file_path.parts[0].casefold() != split:
                continue
            if split == "train" and excluded_origins:
                origin = os.path.normcase(os.path.normpath(row["orig_path"]))
                if origin in excluded_origins:
                    continue
            text = row["plate_no"]
            if (len(text) not in (7, 8) or int(row["layout"]) != len(text)
                    or any(char not in CHAR_TO_INDEX for char in text)):
                raise ValueError(f"invalid plate label in {path}: {text!r}")
            rows.append({"_path": os.path.normpath(os.path.join(root_abs, str(file_path))),
                         "plate_no": text, "source": row["source"], "layout": row["layout"]})
    return rows


class PlateDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], *, augment: bool, seed: int):
        self.rows, self.augment, self.seed = rows, augment, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = Path(row["_path"])
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable plate image: {path}")
        width = 276 if len(row["plate_no"]) == 8 else 252
        image = cv2.resize(image, (width, 80), interpolation=cv2.INTER_AREA)
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
            image = _augment(image, rng)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255).sub_(0.5).div_(0.5)
        target = torch.tensor([CHAR_TO_INDEX[c] for c in row["plate_no"]], dtype=torch.long)
        return tensor.unsqueeze(0), target, len(target), row["source"]


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    if rng.random() < .35:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(.2, 1.2)))
    if rng.random() < .55:
        gain, bias = float(rng.uniform(.7, 1.3)), float(rng.uniform(-20, 20))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < .45:
        sigma = float(rng.uniform(2, 12))
        noise = rng.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < .25:
        dx, dy = float(rng.uniform(-.012, .012) * w), float(rng.uniform(-.025, .025) * h)
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .15:
        x, y = int(rng.integers(0, w - 12)), int(rng.integers(4, h - 5))
        image[y:y + int(rng.integers(2, 7)), x:x + int(rng.integers(8, 20))] = image.mean(axis=(0, 1))
    return image


def _collate(batch):
    images, targets, lengths, sources = zip(*batch, strict=False)
    return torch.stack(images), torch.cat(targets), torch.tensor(lengths), sources


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def _record_draw_counts(counts: dict[str, int], sources, lengths) -> None:
    for source, length in zip(sources, lengths.tolist(), strict=False):
        group = "ground_truth" if source == "ground_truth" else (
            "pipeline_candidate" if source.startswith("pipeline_candidate") else source)
        key = f"{group}/layout{int(length)}"
        counts[key] = counts.get(key, 0) + 1


def _weights(rows: list[dict[str, str]], *, real_green_boost: float = 1.0) -> torch.DoubleTensor:
    counts = Counter(r["plate_no"][0] for r in rows)
    # Province-balanced plate draws, capped at 8x; optional factor applies only to real 8-char crops.
    values = []
    for row in rows:
        weight = min(8.0, np.sqrt(max(counts.values()) / counts[row["plate_no"][0]]))
        is_real_green = row["layout"] == "8" and (
            row["source"] == "ground_truth" or row["source"].startswith("pipeline_candidate"))
        values.append(weight * real_green_boost if is_real_green else weight)
    return torch.as_tensor(values, dtype=torch.double)


def _build_sampler(sample_weights: torch.Tensor, samples_per_epoch: int, seed: int):
    return WeightedRandomSampler(sample_weights, samples_per_epoch, replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def _sampling_summary(rows: list[dict[str, str]], *, real_green_boost: float = 1.0) -> dict[str, object]:
    weights_tensor = _weights(rows, real_green_boost=real_green_boost)
    weights = weights_tensor.tolist()
    total = sum(weights)
    group_totals: dict[str, float] = {}
    for row, weight in zip(rows, weights, strict=False):
        for group in ("source", "layout", "province"):
            key = str(row[group]) if group != "province" else row["plate_no"][0]
            group_totals[f"{group}:{key}"] = float(group_totals.get(f"{group}:{key}", 0.0) + weight / total)
    return {"method": "sqrt_inverse_province_frequency_capped_8_times_real_green_factor",
            "real_green_boost": float(real_green_boost), "expected_probability": group_totals}


def _normalize_checkpoint_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized = dict(metadata)
    normalized.setdefault("real_green_boost", 1.0)
    normalized.setdefault("initialization", None)
    normalized.setdefault("execution_lineage", [])
    sampling = normalized.get("sampling")
    if isinstance(sampling, dict) and sampling.get("method") == "sqrt_inverse_province_frequency_capped_8":
        if float(normalized["real_green_boost"]) != 1.0:
            raise ValueError("legacy sampler metadata cannot represent a non-default green boost")
        normalized_sampling = dict(sampling)
        normalized_sampling["method"] = "sqrt_inverse_province_frequency_capped_8_times_real_green_factor"
        normalized_sampling["real_green_boost"] = 1.0
        normalized["sampling"] = normalized_sampling
    return normalized


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval(model, rows, batch_size=256):
    model.eval()
    metrics = {"all": [0, 0], "ground_truth": [0, 0], "pipeline_candidate": [0, 0], "province": [0, 0], "characters": [0, 0]}
    dataset = PlateDataset(rows, augment=False, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    with torch.inference_mode():
        for images, targets, lengths, sources in loader:
            logits = model(images)
            indices = logits.argmax(-1).transpose(0, 1).tolist()
            offset = 0
            for pred_ids, length, source in zip(indices, lengths.tolist(), sources, strict=False):
                text_ids = targets[offset:offset + length].tolist()
                offset += length
                truth = "".join(CHARSET[i] for i in text_ids)
                pred = []
                previous = BLANK
                for label in pred_ids:
                    if label != BLANK and label != previous:
                        pred.append(CHARSET[label])
                    previous = label
                prediction = "".join(pred)
                metrics["all"][1] += 1
                metrics["all"][0] += prediction == truth
                matches = sum(a == b for a, b in zip(prediction, truth, strict=False))
                metrics["characters"][0] += matches
                metrics["characters"][1] += max(len(prediction), len(truth))
                key = "ground_truth" if source == "ground_truth" else ("pipeline_candidate" if source.startswith("pipeline_candidate") else None)
                if key:
                    metrics[key][1] += 1
                    metrics[key][0] += prediction == truth

                metrics["province"][1] += 1
                metrics["province"][0] += bool(prediction) and prediction[0] == truth[0]
    model.train()
    return {k: {"correct": int(v[0]), "total": int(v[1]), "accuracy": (v[0] / v[1] if v[1] else None)} for k, v in metrics.items()}


def train(data: list[Path], out: Path, *, version="crnn_v1", epochs=30, batch_size=128,
          samples_per_epoch=32768, seed=42, num_workers=0, lr=.001, resume=None, exclude_origins=None,
          stop_after=None, resume_execution_change=False, init_weights=None,
          init_training_run=None, real_green_boost=1.0):
    if (not 0 <= num_workers <= 4 or epochs < 1 or batch_size < 1 or samples_per_epoch < 1
            or (stop_after is not None and stop_after < 1)
            or not np.isfinite(real_green_boost) or real_green_boost < 1):
        raise ValueError("invalid training limits")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    excluded = set()
    if exclude_origins:
        with Path(exclude_origins).open(encoding="utf-8-sig") as handle:
            excluded = {os.path.normcase(os.path.normpath(line.strip()))
                        for line in handle if line.strip()}
    train_rows = [r for root in data for r in _read_rows(root, "train", excluded)]
    val_rows = _read_rows(data[0], "val")
    if not train_rows or not val_rows:
        raise ValueError("real train and validation rows are required")
    if init_weights is not None and resume is not None:
        raise ValueError("init_weights and resume are mutually exclusive")
    if (init_weights is None) != (init_training_run is None):
        raise ValueError("init_weights and init_training_run must be supplied together")
    init_path = Path(init_weights).resolve() if init_weights is not None else None
    if init_path is not None and init_path == out.resolve():
        raise ValueError("init weights and output model must use distinct paths")
    init_state = None
    init_info = None
    if init_path is not None:
        if not init_path.is_file():
            raise FileNotFoundError(f"initial weights not found: {init_path}")
        init_state = torch.load(init_path, map_location="cpu", weights_only=True)
        if not isinstance(init_state, dict) or not init_state or any(
                not isinstance(k, str) or not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                for k, v in init_state.items()):
            raise ValueError("initial weights must be a finite tensor state_dict")
        init_info = {"kind": "local_training_continuation", "path": str(init_path),
                     "sha256": _hash(init_path), "source_training_run": str(init_training_run)}
    checkpoint_path = out.with_suffix(".checkpoint.pt")
    log_path = out.with_suffix(".training.json")
    if resume is None and any(path.exists() for path in (out, checkpoint_path, log_path)):
        raise FileExistsError("training outputs already exist; choose a new path or resume checkpoint")
    if resume_execution_change and resume is None:
        raise ValueError("execution-change resume requires an explicit source checkpoint")
    if resume_execution_change and Path(resume).resolve() == checkpoint_path.resolve():
        raise ValueError("execution-change resume requires a distinct output checkpoint path")
    if resume_execution_change and any(path.exists() for path in (out, checkpoint_path, log_path)):
        raise FileExistsError("execution-change resume requires a new output directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"version": version, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                "samples_per_epoch": samples_per_epoch, "num_workers": num_workers, "lr": lr,
                "input_size": list(INPUT_SIZE),
                "data": [{"path": str(p.resolve()), "labels_sha256": _hash(p / "labels.csv")} for p in data],
                "validation": {"path": str(data[0].resolve()), "split": "val", "labels_sha256": _hash(data[0] / "labels.csv")},
                "train_rows": len(train_rows), "val_rows": len(val_rows),
                "source_counts": dict(Counter(r["source"] for r in train_rows)),
                "sampling": _sampling_summary(train_rows, real_green_boost=real_green_boost),
                "real_green_boost": float(real_green_boost), "initialization": init_info,
                "synthetic_train_only": [str(p.resolve()) for p in data[1:]],
                "excluded_origins_file": str(Path(exclude_origins).resolve()) if exclude_origins else None,
                "excluded_origins_sha256": _hash(Path(exclude_origins)) if exclude_origins else None,
                "excluded_origin_count": len(excluded), "trainer_sha256": _hash(Path(__file__)),
                "model_code_sha256": _hash(Path(__file__).resolve().parents[2] / "lpr" / "recognize" / "crnn.py"),
                "execution_lineage": [], "pretrained_weights": None}
    model = CrnnNet()
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start = 0
    history = []
    elapsed = 0.0
    best = -1.0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        old_metadata = _normalize_checkpoint_metadata(ckpt["metadata"])
        old_lineage = old_metadata.get("execution_lineage", [])
        if resume_execution_change:
            ignored = {"num_workers", "trainer_sha256", "execution_lineage"}
            old_core = {k: v for k, v in old_metadata.items() if k not in ignored}
            new_core = {k: v for k, v in metadata.items() if k not in ignored}
            if old_core != new_core or (old_metadata["num_workers"] == num_workers
                                        and old_metadata["trainer_sha256"] == metadata["trainer_sha256"]):
                raise ValueError("execution-only resume permits only worker count and trainer implementation changes")
            source_best = Path(resume).with_name(Path(resume).name.removesuffix(".checkpoint.pt") + ".pt")
            source_log = Path(resume).with_name(Path(resume).name.removesuffix(".checkpoint.pt") + ".training.json")
            if not source_best.is_file():
                raise FileNotFoundError(f"source best checkpoint model is missing: {source_best}")
            metadata["execution_lineage"] = old_lineage + [{
                "checkpoint_sha256": _hash(Path(resume)),
                "source_best_path": str(source_best.resolve()), "source_best_sha256": _hash(source_best),
                "source_log_sha256": _hash(source_log) if source_log.is_file() else None,
                "from_num_workers": old_metadata["num_workers"], "to_num_workers": num_workers,
                "from_trainer_sha256": old_metadata["trainer_sha256"],
                "to_trainer_sha256": metadata["trainer_sha256"],
            }]
            shutil.copy2(source_best, out)
            if source_log.is_file() and source_log.stat().st_size:
                shutil.copy2(source_log, log_path)
            else:
                with log_path.open("w", encoding="utf-8") as history_log:
                    for item in ckpt["history"]:
                        history_log.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            metadata["execution_lineage"] = old_lineage
            metadata["initialization"] = old_metadata["initialization"]
            if old_metadata != metadata:
                raise ValueError("resume training provenance/config differs")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, history, elapsed, best = ckpt["epoch"], ckpt["history"], ckpt["train_seconds"], ckpt["best"]
        torch.set_rng_state(ckpt["torch_rng"])
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    train_ds = PlateDataset(train_rows, augment=True, seed=seed)
    sample_weights = _weights(train_rows, real_green_boost=real_green_boost)
    with log_path.open("a" if resume else "w", encoding="utf-8") as log:
        for epoch in range(start, min(epochs, start + stop_after) if stop_after is not None else epochs):
            train_ds.epoch = epoch
            sampler = _build_sampler(sample_weights, samples_per_epoch, seed + epoch)
            loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                collate_fn=_collate, worker_init_fn=_worker_init if num_workers else None,
                                generator=torch.Generator().manual_seed(seed + epoch + 100_000),
                                persistent_workers=False)
            model.train()
            began = time.perf_counter()
            total = 0.0
            batches = 0
            actual_draw_counts: dict[str, int] = {}
            for images, targets, lengths, sources in loader:
                _record_draw_counts(actual_draw_counts, sources, lengths)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full((len(images),), logits.shape[0], dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, lengths)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite CTC loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.item())
                batches += 1
                if batches % 50 == 0:
                    print(f"epoch={epoch+1} batch={batches} loss={total/batches:.4f}", flush=True)
            if sum(actual_draw_counts.values()) != samples_per_epoch:
                raise RuntimeError("actual source/layout draw counts do not sum to samples_per_epoch")
            metrics = _eval(model, val_rows)
            elapsed += time.perf_counter() - began
            score = metrics["all"]["accuracy"] or 0.0
            if score > best:
                best = score
                torch.save(model.state_dict(), out)
            entry = {"epoch": epoch + 1, "loss": total / max(batches, 1), "metrics": metrics,
                     "actual_draw_counts": actual_draw_counts,
                     "epoch_seconds": time.perf_counter() - began, "train_seconds": elapsed}
            history.append(entry)
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                        "history": history, "train_seconds": elapsed, "best": best, "metadata": metadata,
                        "torch_rng": torch.get_rng_state()}, checkpoint_path)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
    best_metrics = max(history, key=lambda item: item["metrics"]["all"]["accuracy"] or 0)["metrics"]
    write_manifest_entry(out, version=version, train_data=json.dumps(metadata, ensure_ascii=False),
                         val_char_acc=best_metrics["characters"]["accuracy"], input_size=INPUT_SIZE)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("algorithm/models/crnn_v1.pt"))
    parser.add_argument("--version", default="crnn_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--exclude-origins", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-execution-change", action="store_true")
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--init-training-run")
    parser.add_argument("--real-green-boost", type=float, default=1.0)
    args = parser.parse_args()
    train(args.data, args.out, version=args.version, epochs=args.epochs, batch_size=args.batch_size,
          samples_per_epoch=args.samples_per_epoch, seed=args.seed, num_workers=args.num_workers,
          lr=args.lr, resume=args.resume, exclude_origins=args.exclude_origins, stop_after=args.stop_after,
          resume_execution_change=args.resume_execution_change, init_weights=args.init_weights,
          init_training_run=args.init_training_run, real_green_boost=args.real_green_boost)


if __name__ == "__main__":
    main()

















