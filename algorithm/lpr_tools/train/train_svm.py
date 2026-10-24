"""Train capped, probability-calibrated RBF SVC on OpenCV HOG features."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from lpr.recognize import write_manifest_entry
from lpr.recognize.svm import SvmClassifier, hog_features
from .data import load_split


def train(data: Path, out: Path, *, version: str, seed: int = 42,
          max_per_class: int = 500, calibrated: bool = False) -> float:
    if not 2 <= max_per_class <= 500:
        raise ValueError("max_per_class must be in 2..500")
    images, labels = load_split(data, "train", max_per_class=max_per_class, seed=seed)
    counts = np.bincount(labels, minlength=65)
    if counts.min() < 2:
        raise ValueError("SVM training requires at least two samples per class for calibration")
    features = hog_features(images)
    start = time.perf_counter()
    base = (SVC(kernel="rbf", C=10, random_state=seed) if calibrated else
            SVC(kernel="rbf", C=10, probability=True, random_state=seed))
    print(f"svm train_samples={len(labels)} features={features.shape[1]} kernel=rbf C=10 "
          f"calibrated={calibrated} max_per_class={max_per_class}", flush=True)
    if calibrated:
        cv = min(5, int(counts.min()))
        model = CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False)
    else:
        model = base
    model.fit(features, labels)
    train_seconds = time.perf_counter() - start
    val_images, val_labels = load_split(data, "val")
    accuracy = float(np.mean(SvmClassifier(version, model).predict_proba(val_images).argmax(axis=1) == val_labels))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    write_manifest_entry(out, version=version, train_data=str(data), val_char_acc=accuracy)
    out.with_suffix(".training.json").write_text(
        json.dumps({"train_seconds": train_seconds}, indent=2) + "\n", encoding="utf-8")
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
