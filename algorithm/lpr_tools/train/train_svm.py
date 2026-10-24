"""Train capped, probability-calibrated RBF SVC on OpenCV HOG features."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC

from lpr.recognize import write_manifest_entry
from lpr.recognize.svm import SvmClassifier, hog_features
from .data import load_split


def train(data: Path, out: Path, *, version: str, seed: int = 42,
          max_per_class: int = 500) -> float:
    if not 2 <= max_per_class <= 500:
        raise ValueError("max_per_class must be in 2..500")
    images, labels = load_split(data, "train", max_per_class=max_per_class, seed=seed)
    if np.bincount(labels).min() < 2:
        raise ValueError("SVC probability calibration needs at least two samples per class")
    features = hog_features(images)
    print(f"svm train_samples={len(labels)} features={features.shape[1]} kernel=rbf C=10 "
          f"probability=True max_per_class={max_per_class}", flush=True)
    model = SVC(kernel="rbf", C=10, probability=True, random_state=seed)
    model.fit(features, labels)
    val_images, val_labels = load_split(data, "val")
    accuracy = float(np.mean(SvmClassifier(version, model).predict_proba(val_images).argmax(axis=1) == val_labels))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    write_manifest_entry(out, version=version, train_data=str(data), val_char_acc=accuracy)
    print(f"svm val_samples={len(val_labels)} val_char_acc={accuracy:.6f}")
    print(f"saved={out.as_posix()} version={version}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()
    train(args.data, args.out, version=args.version, seed=args.seed, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
