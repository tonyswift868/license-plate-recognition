"""Continue our compact CRNN on audited, original-disjoint hard plate crops."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from lpr.charset import CHAR_TO_INDEX, CHARSET
from lpr.recognize import file_sha256, write_manifest_entry
from lpr.recognize.crnn import BLANK, INPUT_SIZE, CrnnNet, plate_tensor
from lpr_tools.train.train_crnn import _collate, _worker_init
from torch import nn
from torch.utils.data import DataLoader, Dataset

SUBSETS = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
MIXTURE = {**{key: .10 for key in SUBSETS}, "base7": .12, "real8": .10, "font": .08}


@dataclass(frozen=True, slots=True)
class Sample:
    path: str
    text: str
    group: str
    gt: bool
    brightness: int | None = None


def read_index(path: Path, partition: str) -> list[Sample]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            text = row["plate_no"]
            if row["partition"] != partition:
                raise ValueError("unexpected partition in audited index")
            if len(text) not in (7, 8) or row["layout"] != str(len(text)) or any(c not in CHAR_TO_INDEX for c in text):
                raise ValueError("invalid plate label/layout")
            group = row["subset"] or ("real8" if len(text) == 8 else "base7")
            if group not in MIXTURE:
                raise ValueError(f"unknown sampling group: {group}")
            rows.append(Sample(row["crop_path"], text, group,
                               row["source"] == "ground_truth" or row["source"].endswith(":ground_truth"),
                               int(row["brightness"]) if row["brightness"] else None))
    if not rows:
        raise ValueError("empty index")
    return rows


def read_fonts(root: Path) -> list[Sample]:
    rows = []
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["file"].replace("\\", "/").startswith("train/"):
                continue
            text = row["plate_no"]
            if row["source"] != "font_synthetic" or row["orig_path"]:
                raise ValueError("only independent font-synthetic data allowed here")
            if len(text) not in (7, 8) or any(c not in CHAR_TO_INDEX for c in text):
                raise ValueError("invalid synthetic label")
            rows.append(Sample(str(root / row["file"]), text, "font", False))
    return rows


def sampling_weights(rows: list[Sample]) -> torch.Tensor:
    # Normalize within each source stratum: province balancing must not change group shares.
    indices = defaultdict(list)
    for i, row in enumerate(rows):
        indices[(row.group, row.gt)].append(i)
    if set(group for group, _ in indices) != set(MIXTURE):
        raise ValueError("all seven hard groups, base7, real8 and fonts must be represented")
    weights = torch.zeros(len(rows), dtype=torch.double)
    for (group, gt), positions in indices.items():
        counts = Counter(rows[i].text[0] for i in positions)
        raw = torch.tensor([min(4., (max(counts.values()) / counts[rows[i].text[0]]) ** .5)
                            for i in positions], dtype=torch.double)
        source_share = (.65 if gt else .35) if (group, not gt) in indices else 1.
        weights[positions] = raw / raw.sum() * MIXTURE[group] * source_share
    return weights


def augment(image: np.ndarray, rng: np.random.Generator, *, geometry: bool = True) -> np.ndarray:
    """Own implementation of mild geometry, illumination and blur perturbations."""
    h, w = image.shape[:2]
    if geometry and rng.random() < .35:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.float32([.025 * w, .05 * h])
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .55:
        gamma = float(rng.uniform(.65, 1.6))
        lut = np.clip((np.arange(256) / 255.) ** gamma * 255., 0, 255).astype(np.uint8)
        image = cv2.LUT(image, lut).astype(np.float32)
        image = image * float(rng.uniform(.75, 1.25)) + float(rng.uniform(-12, 12))
        if rng.random() < .3:
            shade = np.linspace(float(rng.uniform(.65, 1.)), 1., w, dtype=np.float32)
            if rng.random() < .5:
                shade = shade[::-1]
            image *= shade[None, :, None]
        image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < .25:
        kind = int(rng.integers(3))
        if kind == 0:
            image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(.2, .9)))
        elif kind == 1:
            k = int(rng.choice([3, 5]))
            kernel = np.zeros((k, k), np.float32)
            if rng.random() < .8:
                kernel[k // 2, :] = 1. / k
            else:
                kernel[:, k // 2] = 1. / k
            image = cv2.filter2D(image, -1, kernel)
        else:
            scale = float(rng.uniform(.65, .9))
            small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < .2:
        image = np.clip(image.astype(np.float32) + rng.normal(0, rng.uniform(1, 6), image.shape), 0, 255).astype(np.uint8)
    return image


class Crops(Dataset):
    def __init__(self, rows, *, draws=None, epoch=0, seed=42):
        self.rows, self.draws, self.epoch, self.seed = rows, draws, epoch, seed

    def __len__(self):
        return len(self.rows) if self.draws is None else len(self.draws)

    def __getitem__(self, i):
        row = self.rows[i if self.draws is None else self.draws[i]]
        image = cv2.imdecode(np.fromfile(row.path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (80, 276 if len(row.text) == 8 else 252, 3):
            raise ValueError(f"bad native plate crop: {row.path}")
        if self.draws is not None:
            image = augment(image, np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i])))
        target = torch.tensor([CHAR_TO_INDEX[c] for c in row.text], dtype=torch.long)
        return plate_tensor(image), target, len(target), f"{row.group}/{'GT' if row.gt else 'pipeline'}/layout{len(row.text)}"


def evaluate(model, rows, batch_size=128):
    metrics = defaultdict(lambda: [0, 0])
    model.eval()
    offset = 0
    with torch.inference_mode():
        for images, _targets, _lengths, _ in DataLoader(Crops(rows), batch_size=batch_size, collate_fn=_collate):
            indices = model(images).argmax(-1).transpose(0, 1).tolist()
            for ids in indices:
                row = rows[offset]
                offset += 1
                chars, previous = [], BLANK
                for label in ids:
                    if label != BLANK and label != previous:
                        chars.append(CHARSET[label])
                    previous = label
                prediction = "".join(chars)
                keys = ["all", "GT" if row.gt else "pipeline", "layout" + str(len(row.text))]
                if row.gt and row.group in SUBSETS:
                    keys.append(row.group + "_GT")
                    if row.group == "db" and row.brightness is not None and row.brightness <= 19:
                        keys.append("db_dark_GT")
                for key in keys:
                    metrics[key][0] += int(prediction == row.text)
                    metrics[key][1] += 1
                metrics["characters"][0] += sum(a == b for a, b in zip(prediction, row.text, strict=False))
                metrics["characters"][1] += max(len(prediction), len(row.text))
    return {key: {"correct": a, "n": n, "accuracy": a / n} for key, (a, n) in metrics.items()}


def selection_score(tune, val):
    values = [tune[key + "_GT"]["accuracy"] for key in SUBSETS]
    return (min(values), sum(values) / len(values), val["all"]["accuracy"])


def stop_reason(history, *, patience=3, minimum_epochs=6):
    if len(history) < minimum_epochs:
        return None
    for subset in SUBSETS:
        values = [item["tune"][subset + "_GT"]["accuracy"] for item in history]
        # Strictly best count, not a rounded percentage; no extension when a hard subset stalls.
        best = max(values)
        last_improvement = next(i for i, value in enumerate(values) if value == best)
        if best < .90 and len(values) - 1 - last_improvement >= patience:
            return f"{subset}: best GT accuracy {best:.6f} < .90, no new best in {patience} epochs"
    return None


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(index_dir: Path, fonts: Path, init: Path, out: Path, *, epochs=12,
          samples_per_epoch=65536, batch_size=128, lr=.0003, seed=42,
          num_workers=1, resume=False, stop_after=None):
    if epochs < 1 or samples_per_epoch < 1 or batch_size < 1 or not 0 <= num_workers <= 2 or not 0 < lr <= .01:
        raise ValueError("invalid training limits")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if out.resolve() == init.resolve():
        raise ValueError("initializer must never be overwritten")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    audit_path = index_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    for key in ("train", "tune", "val"):
        if file_sha256(index_dir / (key + ".csv")) != audit["output_sha256"][key]:
            raise ValueError(f"audited {key} index changed")
    content_checks = audit["content_checks"]
    pairwise_overlap = content_checks.get("partition_pairwise_sha256_overlap")
    required_partitions = ("train_tune", "train_val", "tune_val")
    if (not isinstance(pairwise_overlap, dict)
            or any(key not in pairwise_overlap or pairwise_overlap[key] != 0
                   for key in required_partitions)
            or any(content_checks[key] != 0 for key in (
            "train_selected_sha_overlap", "fit_report_sha_overlap", "selected_sha_seen_by_v1",
            "protected_hash_overlap_train", "protected_hash_overlap_tune"))):
        raise ValueError("data leakage audit failed")
    train_rows = read_index(index_dir / "train.csv", "train") + read_fonts(fonts)
    tune_rows, val_rows = read_index(index_dir / "tune.csv", "tune"), read_index(index_dir / "val.csv", "val")
    weights = sampling_weights(train_rows)
    metadata = {"version": "crnn_v2_raw", "epochs": epochs, "samples_per_epoch": samples_per_epoch,
                "batch_size": batch_size, "lr": lr, "seed": seed, "num_workers": num_workers,
                "torch_threads": 4, "opencv_threads": 1, "loader_threads_each": 1,
                "initialization": {"path": str(init.resolve()), "sha256": file_sha256(init), "kind": "own_crnn_v1_continuation"},
                "data_audit_sha256": file_sha256(audit_path), "indices_sha256": audit["output_sha256"],
                "synthetic_labels_sha256": file_sha256(fonts / "labels.csv"), "synthetic_root": str(fonts),
                "trainer_sha256": file_sha256(Path(__file__)),
                "reader_sha256": file_sha256(Path(__file__).resolve().parents[2] / "lpr/recognize/crnn.py"),
                "helper_sha256": file_sha256(Path(__file__).with_name("train_crnn.py")),
                "selection": "lexicographic minimum hard GT, macro hard GT, filtered val exact plate",
                "stop_loss": "after >=6 epochs, any subset best<.90 and no new best for3 complete epochs",
                "mixture": MIXTURE, "gt_share_within_real_group": .65, "province_cap": 4,
                "train_rows": len(train_rows), "tune_rows": len(tune_rows), "val_rows": len(val_rows)}
    out.mkdir(parents=True, exist_ok=True)
    checkpoint, best_path = out / "checkpoint.pt", out / "best.pt"
    metadata_path = out / "metadata.json"
    if not resume and any(p.exists() for p in (checkpoint, best_path, metadata_path)):
        raise FileExistsError("training outputs exist; use exact resume or new output")
    model = CrnnNet()
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=False)
    history, start, best_score, best_epoch = [], 0, (-1., -1., -1.), 0
    if resume:
        old = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if old["metadata"] != metadata:
            raise ValueError("resume provenance differs")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        history, start, best_score, best_epoch = old["history"], old["epoch"], tuple(old["best_score"]), old["best_epoch"]
        torch.set_rng_state(old["torch_rng"])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    end = epochs if stop_after is None else min(epochs, start + stop_after)
    reason = stop_reason(history)
    for epoch in range(start, end):
        if reason:
            break
        began = time.perf_counter()
        # Fixed continuation LR, halve after epoch6; schedule is fixed before seeing final sets.
        current_lr = lr * (1. if epoch < 6 else .5)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        draws = torch.multinomial(weights, samples_per_epoch, replacement=True,
                                  generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        loader = DataLoader(Crops(train_rows, draws=draws, epoch=epoch, seed=seed),
                            batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
                            worker_init_fn=_worker_init if num_workers else None,
                            generator=torch.Generator().manual_seed(seed + epoch + 100000))
        counts, losses = Counter(), []
        model.train()
        for step, (images, targets, lengths, sources) in enumerate(loader, 1):
            counts.update(sources)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits.log_softmax(2), targets,
                             torch.full((len(images),), logits.shape[0], dtype=torch.long), lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite CTC loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": sum(losses) / len(losses),
                                  "seconds": time.perf_counter() - began}), flush=True)
        if sum(counts.values()) != samples_per_epoch:
            raise RuntimeError("draw count mismatch")
        tune, val = evaluate(model, tune_rows), evaluate(model, val_rows)
        score = selection_score(tune, val)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            _atomic_save(model.state_dict(), best_path)
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "lr": current_lr,
                  "tune": tune, "val": val, "actual_draw_counts": dict(counts),
                  "epoch_seconds": time.perf_counter() - began, "best_epoch": best_epoch}
        history.append(record)
        reason = stop_reason(history)
        if score[0] >= .95:
            reason = "all seven hard GT subsets reached .95"
        _atomic_save({"metadata": metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "history": history, "epoch": epoch + 1, "best_score": best_score,
                      "best_epoch": best_epoch, "torch_rng": torch.get_rng_state()}, checkpoint)
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result = {"best_epoch": best_epoch, "best_score": best_score, "epochs_completed": len(history),
              "stop_reason": reason or ("epoch budget reached" if len(history) == epochs else "requested checkpoint pause"),
              "best_sha256": file_sha256(best_path)}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    best_record = history[best_epoch - 1]
    write_manifest_entry(best_path, version="crnn_v2_raw", train_data=json.dumps(metadata, ensure_ascii=False),
                         val_char_acc=best_record["val"]["characters"]["accuracy"], input_size=INPUT_SIZE)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    train(args.index_dir, args.fonts, args.init, args.out, epochs=args.epochs,
          samples_per_epoch=args.samples_per_epoch, batch_size=args.batch_size, lr=args.lr,
          num_workers=args.num_workers, seed=args.seed, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
