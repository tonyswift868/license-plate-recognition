"""Train capped, probability-calibrated RBF SVC on OpenCV HOG features."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from lpr.recognize import write_manifest_entry
from lpr.recognize.svm import SvmClassifier, hog_features
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from .data import load_split


def validation_accuracy(classifier, images: np.ndarray, labels: np.ndarray,
                        *, batch_size: int = 1024) -> float:
    """Accumulate exact validation correctness with a bounded prediction batch."""
    if batch_size < 1 or len(images) != len(labels) or not len(labels):
        raise ValueError("validation requires matching nonempty images/labels and batch_size > 0")
    correct = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        probs = classifier.predict_proba(images[start:stop])
        if probs.shape != (stop - start, 65):
            raise ValueError(f"classifier returned unexpected validation shape: {probs.shape}")
        correct += int((probs.argmax(axis=1) == labels[start:stop]).sum())
    return correct / len(labels)


def train(data: Path, out: Path, *, version: str, seed: int = 42,
          max_per_class: int = 500, calibrated: bool = False) -> float:
    if not 2 <= max_per_class <= 500:
        raise ValueError("max_per_class must be in 2..500")
    images, labels = load_split(data, "train", max_per_class=max_per_class, seed=seed,
                                require_all=False)
    if not len(labels):
        raise ValueError("SVM training split is empty")
    counts = np.bincount(labels, minlength=65)
    excluded = np.flatnonzero((counts > 0) & (counts < 2))
    source_counts = counts.copy()
    trainable = np.flatnonzero(counts >= 2)
    if len(trainable) < 2:
        raise ValueError("SVM training requires at least two classes with two or more samples")
    keep = np.isin(labels, trainable)
    images, labels = images[keep], labels[keep]
    counts = np.bincount(labels, minlength=65)
    features = hog_features(images)
    start = time.perf_counter()
    base = (SVC(kernel="rbf", C=10, random_state=seed) if calibrated else
            SVC(kernel="rbf", C=10, probability=True, random_state=seed))
    print(f"svm train_samples={len(labels)} features={features.shape[1]} kernel=rbf C=10 "
          f"calibrated={calibrated} max_per_class={max_per_class} "
          f"trainable_classes={trainable.tolist()} "
          f"excluded_classes={[(int(label), int(source_counts[label])) for label in excluded]}", flush=True)
    if calibrated:
        cv = min(5, int(counts[counts > 0].min()))
        model = CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False)
    else:
        model = base
    model.fit(features, labels)
    train_seconds = time.perf_counter() - start
    val_images, val_labels = load_split(data, "val", require_all=False)
    if not len(val_labels):
        raise ValueError("SVM validation split is empty")
    accuracy = validation_accuracy(SvmClassifier(version, model), val_images, val_labels)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    write_manifest_entry(out, version=version, train_data=str(data), val_char_acc=accuracy)
    out.with_suffix(".training.json").write_text(
        json.dumps({"train_seconds": train_seconds,
                    "trained_classes": trainable.tolist(),
                    "excluded_classes": [{"label": int(label), "samples": int(
                        source_counts[label])}
                        for label in excluded],
                    "calibrated": calibrated, "max_per_class": max_per_class}, indent=2) + "\n",
        encoding="utf-8")
    print(f"svm val_samples={len(val_labels)} val_char_acc={accuracy:.6f}")
    print(f"saved={out.as_posix()} version={version}")
    print(f"train_seconds={train_seconds:.3f}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--calibrated", action="store_true",
                        help="use CalibratedClassifierCV(SVC(), ensemble=False)")
    args = parser.parse_args()
    train(args.data, args.out, version=args.version, seed=args.seed,
          max_per_class=args.max_per_class, calibrated=args.calibrated)


if __name__ == "__main__":
    main()
