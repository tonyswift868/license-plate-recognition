"""Small, data-agnostic helpers for global CharCNN temperature calibration."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping

import numpy as np
import torch

from lpr.charset import NUM_CLASSES


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool):
        raise ValueError("temperature must be finite and positive")
    try:
        value = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    return value


def fit_temperature(logits_or_logprobs: np.ndarray, labels: np.ndarray,
                    bounds: tuple[float, float] = (0.1, 3.0),
                    iterations: int = 40) -> dict[str, float | int | list[float]]:
    """Fit one scalar temperature by minimizing mean stable cross-entropy NLL."""
    try:
        scores = np.asarray(logits_or_logprobs, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scores must be a finite N x C numeric array") from exc
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2:
        raise ValueError("scores must have shape N x C with N >= 1 and C >= 2")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    target = np.asarray(labels)
    if (target.shape != (scores.shape[0],) or not np.issubdtype(target.dtype, np.integer)
            or np.issubdtype(target.dtype, np.bool_)):
        raise ValueError("labels must be an integer vector with one label per score row")
    target = target.astype(np.int64, copy=False)
    if (target < 0).any() or (target >= scores.shape[1]).any():
        raise ValueError("labels must be valid score-column indices")

    try:
        lower, upper = bounds
        lower, upper = float(lower), float(upper)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounds must be two finite positive values in increasing order") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or lower >= upper:
        raise ValueError("bounds must be two finite positive values in increasing order")
    if (isinstance(iterations, bool) or not isinstance(iterations, Integral)
            or iterations < 1):
        raise ValueError("iterations must be a positive integer")

    rows = np.arange(scores.shape[0])

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = scores / temperature
            maximum = scaled.max(axis=1, keepdims=True)
            shifted = scaled - maximum
            losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[rows, target]
        result = float(losses.mean())
        if not math.isfinite(result):
            raise ValueError("scores exceed the numeric range for stable NLL")
        return result

    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(int(iterations)):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = nll(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = nll(x2)
    temperature = math.exp((left + right) / 2.0)
    return {
        "temperature": temperature,
        "raw_nll": nll(0.0),
        "calibrated_nll": nll(math.log(temperature)),
        "bounds": [lower, upper],
        "iterations": int(iterations),
        "samples": int(scores.shape[0]),
    }


def apply_temperature(state_dict: Mapping[str, torch.Tensor], temperature: float
                      ) -> dict[str, torch.Tensor]:
    """Return a new pure-tensor state_dict with the final CharCNN linear layer scaled."""
    value = _validate_temperature(temperature)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping of string keys to tensors")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
           for key, tensor in state_dict.items()):
        raise ValueError("state_dict must contain only string keys and tensors")
    if any(tensor.is_floating_point() and not torch.isfinite(tensor).all().item()
           for tensor in state_dict.values()):
        raise ValueError("state_dict floating tensors must be finite")

    weight_key, bias_key = "classifier.5.weight", "classifier.5.bias"
    if weight_key not in state_dict or bias_key not in state_dict:
        raise ValueError("state_dict is missing the final CharCNN linear layer")
    weight, bias = state_dict[weight_key], state_dict[bias_key]
    if (not weight.is_floating_point() or not bias.is_floating_point()
            or tuple(weight.shape) != (NUM_CLASSES, 256)
            or tuple(bias.shape) != (NUM_CLASSES,)):
        raise ValueError("final CharCNN linear weights have incompatible shapes or dtypes")

    calibrated = {key: tensor.clone() for key, tensor in state_dict.items()}
    calibrated[weight_key] = weight / value
    calibrated[bias_key] = bias / value
    if not torch.isfinite(calibrated[weight_key]).all().item() or not torch.isfinite(calibrated[bias_key]).all().item():
        raise ValueError("calibrated final layer must remain finite")
    return calibrated
