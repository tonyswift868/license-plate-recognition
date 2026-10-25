"""Build one mean template per class, then register its validation accuracy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from lpr.charset import NUM_CLASSES
from lpr.recognize import write_manifest_entry
from lpr.recognize.template import TemplateClassifier
from .data import load_split


def build(data: Path, out: Path, *, version: str, temperature: float = 0.1) -> float:
    images, labels = load_split(data, "train", require_all=False)
    if not len(labels):
        raise ValueError("template training split is empty")
    present = np.bincount(labels, minlength=NUM_CLASSES) > 0
    start = time.perf_counter()
    templates = np.zeros((NUM_CLASSES, 32, 32), dtype=np.float32)
    for label in np.flatnonzero(present):
        templates[label] = images[labels == label].mean(axis=0, dtype=np.float32)
    train_seconds = time.perf_counter() - start
    classifier = TemplateClassifier(version, templates, temperature, present)
    val_images, val_labels = load_split(data, "val", require_all=False)
    if not len(val_labels):
        raise ValueError("template validation split is empty")
    accuracy = float(np.mean(classifier.predict_proba(val_images).argmax(axis=1) == val_labels))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A file handle preserves the caller's exact filename (no implicit .npz suffix).
    with out.open("wb") as handle:
        np.savez_compressed(handle, templates=templates, temperature=np.float32(temperature),
                            present_classes=present)
    write_manifest_entry(out, version=version, train_data=str(data), val_char_acc=accuracy)
    out.with_suffix(".training.json").write_text(
        json.dumps({"train_seconds": train_seconds,
                    "present_classes": np.flatnonzero(present).tolist()}, indent=2) + "\n",
        encoding="utf-8")
    print(f"template train_samples={len(labels)} val_samples={len(val_labels)} "
          f"temperature={temperature:g} val_char_acc={accuracy:.6f}")
    print(f"template present_classes={int(present.sum())} "
          f"missing_classes={np.flatnonzero(~present).tolist()} train_seconds={train_seconds:.3f}")
    print(f"saved={out.as_posix()} version={version}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    build(args.data, args.out, version=args.version, temperature=args.temperature)


if __name__ == "__main__":
    main()
