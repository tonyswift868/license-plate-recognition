"""Train an independent, bounded ROI corner refiner from audited CCPD positives."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpr.corner_net import (  # noqa: E402
    REFINE_INPUT_SIZE, REFINE_ARCHITECTURE, RefineModel, refine_coordinates,
    letterbox_image,
)
from lpr.io_utils import imread  # noqa: E402
from lpr_tools import train_corner as source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "out" / "eighth" / "data"
DIST_DEFAULT = ROOT / "out" / "tenth" / "proposal-error-distribution.json"
PADDING = .25
SEED = 42
BATCH_SIZE = 16
STEPS_PER_ROUND = 2000
MAX_STEPS = 4000
OUTPUT_STRIDE = 2
SIGMA_GRID_PX = 1.5
COORDINATE_WEIGHT = 20.
MAX_RETRIES = 20


def read_proposal_distribution(path: Path) -> tuple[np.ndarray, str]:
    """Read only the independent-holdout empirical bbox residual distribution."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "normalized_center_log_size_v1":
        raise ValueError("unsupported proposal error distribution schema")
    scope = str(payload.get("source_scope", "")).lower()
    if "independent holdout" not in scope or ("devhard" in scope and "never devhard" not in scope):
        raise ValueError("proposal distribution must come only from the independent holdout")
    traces = payload.get("source_traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("proposal distribution is missing trace provenance")
    for record in traces:
        trace = Path(record.get("path", ""))
        if "devhard" in str(trace).lower() or not trace.is_file() or hashlib.sha256(trace.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"proposal distribution trace hash mismatch: {trace}")
    samples = np.asarray(payload.get("samples"), dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 4 or len(samples) < 100:
        raise ValueError("proposal distribution needs at least 100 four-value samples")
    if not np.isfinite(samples).all():
        raise ValueError("proposal distribution contains non-finite residuals")
    # Bound tails so a malformed or extreme proposal cannot create unbounded crops.
    samples[:, :2] = np.clip(samples[:, :2], -.45, .45)
    samples[:, 2:] = np.clip(samples[:, 2:], -.65, .65)
    return samples, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_gt(row: dict[str, str], width: int, height: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Parse and explicitly validate source pixels before any ROI is constructed."""
    quad = source.parse_quad(row)
    if quad.shape != (4, 2) or np.any(quad < 0) or np.any(quad[:, 0] > width - 1) or np.any(quad[:, 1] > height - 1):
        raise source.SourceQuadOutOfBounds(f"source quad outside {width}x{height}: {row.get('path')}")
    bbox = source.parse_bbox(row)
    x, y, w, h = bbox
    if x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        raise ValueError(f"source bbox outside {width}x{height}: {row.get('path')}")
    return quad, bbox


def sample_proposal_bbox(gt_bbox: tuple[float, float, float, float],
                         residual: np.ndarray) -> tuple[float, float, float, float]:
    """Apply center/log-size residuals measured from independent real candidates."""
    values = np.asarray(residual, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("proposal residual must contain four finite values")
    x, y, width, height = map(float, gt_bbox)
    if width <= 0 or height <= 0:
        raise ValueError("ground-truth bbox dimensions must be positive")
    dx, dy = np.clip(values[:2], -.45, .45)
    log_w, log_h = np.clip(values[2:], -.65, .65)
    predicted_width = width * math.exp(float(log_w))
    predicted_height = height * math.exp(float(log_h))
    center_x = x + .5 * width + float(dx) * width
    center_y = y + .5 * height + float(dy) * height
    return (center_x - .5 * predicted_width, center_y - .5 * predicted_height,
            predicted_width, predicted_height)


def expanded_crop(shape: tuple[int, ...], bbox: tuple[float, float, float, float],
                  padding: float = PADDING) -> tuple[int, int, int, int]:
    if not math.isfinite(padding) or padding < 0 or padding > 1:
        raise ValueError("padding must be finite and between 0 and 1")
    return source._roi_bounds(shape, bbox, padding)


def crop_target(image: np.ndarray, bbox: tuple[float, float, float, float],
                quad: np.ndarray, *, padding: float = PADDING
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letterbox an expanded proposal and map GT/predicted bbox corners into it."""
    x0, y0, x1, y1 = expanded_crop(image.shape, bbox, padding)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("expanded proposal crop collapsed")
    if np.any(quad[:, 0] < x0) or np.any(quad[:, 0] >= x1) or np.any(quad[:, 1] < y0) or np.any(quad[:, 1] >= y1):
        raise ValueError("proposal crop does not contain all GT corners")
    canvas, transform = letterbox_image(image[y0:y1, x0:x1], REFINE_INPUT_SIZE)
    target = quad.copy().astype(np.float32)
    target[:, 0] = (target[:, 0] - x0) * transform.scale_x + transform.pad_left
    target[:, 1] = (target[:, 1] - y0) * transform.scale_y + transform.pad_top
    target /= np.asarray(REFINE_INPUT_SIZE, np.float32)
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
        raise ValueError("transformed GT corners are outside refiner canvas")
    x, y, width, height = bbox
    proposal_corners = np.asarray([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], np.float32)
    proposal_corners[:, 0] = (proposal_corners[:, 0] - x0) * transform.scale_x + transform.pad_left
    proposal_corners[:, 1] = (proposal_corners[:, 1] - y0) * transform.scale_y + transform.pad_top
    proposal_corners /= np.asarray(REFINE_INPUT_SIZE, np.float32)
    before = np.mean(np.linalg.norm(proposal_corners - target, axis=1))
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.)
    return tensor, torch.from_numpy(target), np.asarray(before, dtype=np.float32)


def make_corner_heatmaps(target: torch.Tensor, *, grid_height: int = REFINE_INPUT_SIZE[1] // OUTPUT_STRIDE,
                         grid_width: int = REFINE_INPUT_SIZE[0] // OUTPUT_STRIDE,
                         sigma: float = SIGMA_GRID_PX) -> torch.Tensor:
    if target.ndim != 3 or target.shape[1:] != (4, 2) or not torch.isfinite(target).all():
        raise ValueError("corner targets must be finite [B,4,2]")
    if torch.any(target < 0) or torch.any(target > 1) or grid_height <= 0 or grid_width <= 0 or sigma <= 0:
        raise ValueError("corner targets/grid/sigma are out of range")
    device, dtype = target.device, target.dtype
    ys = torch.arange(grid_height, device=device, dtype=dtype)
    xs = torch.arange(grid_width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = target[..., 0] * grid_width - .5
    cy = target[..., 1] * grid_height - .5
    dist = ((xx[None, None] - cx[..., None, None]) ** 2
            + (yy[None, None] - cy[..., None, None]) ** 2)
    heatmaps = torch.exp(-.5 * dist / (sigma * sigma))
    return heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def refine_loss(logits: torch.Tensor, target_xy: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected = (target_xy.shape[0], 4, REFINE_INPUT_SIZE[1] // 2, REFINE_INPUT_SIZE[0] // 2)
    if tuple(logits.shape) != expected:
        raise ValueError(f"unexpected refine output shape {tuple(logits.shape)}")
    target_heatmaps = make_corner_heatmaps(target_xy, grid_height=logits.shape[-2], grid_width=logits.shape[-1])
    log_probabilities = logits.flatten(2).log_softmax(dim=-1)
    heatmap = -(target_heatmaps.flatten(2) * log_probabilities).sum(dim=-1).mean()
    predicted_xy = refine_coordinates(logits)
    coordinates = F.smooth_l1_loss(predicted_xy, target_xy, beta=.02)
    return heatmap + COORDINATE_WEIGHT * coordinates, {"heatmap": heatmap.detach(), "coordinates": coordinates.detach()}


def _safe_training_rows(data_dir: Path, holdout_rows: int = 1200) -> tuple[list[dict], list[dict], dict[str, Any]]:
    train_rows, all_validation, _selected, audit = source._keypoint_safe_inputs(data_dir)
    positives = [r for r in train_rows if r.get("has_plate") == "1" and r.get("subset") != "ccpd_np"]
    if len(positives) != 20400:
        raise ValueError(f"expected 12,000 base/green + 8,400 hard positives, got {len(positives)}")
    counts = {subset: sum(r["subset"] == subset for r in positives) for subset in
              ("ccpd_base", "ccpd_green", "ccpd_db", "ccpd_blur", "ccpd_tilt", "ccpd_rotate", "ccpd_fn", "ccpd_challenge", "ccpd_weather")}
    if counts["ccpd_base"] + counts["ccpd_green"] != 12000 or any(counts[s] != 1200 for s in counts if s.startswith("ccpd_") and s not in ("ccpd_base", "ccpd_green")):
        raise ValueError(f"unexpected positive subset composition: {counts}")
    hard_holdout = [r for r in all_validation if r["subset"] in source.HARD_SUBSETS]
    if len(hard_holdout) != 2100:
        raise ValueError("expected the audited 2,100-image independent hard holdout")
    # Holdout rows are metadata-only; pixels are read later only for model choice.
    return positives, hard_holdout, {"positive_rows": len(positives), "subset_counts": counts,
        "hard_holdout_rows": len(hard_holdout), "parent_audit": audit}


class RefineDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], residual_samples: np.ndarray, *, seed: int = SEED):
        self.rows = rows
        self.samples = residual_samples
        self.seed = seed
        self.epoch = 0
        self.source_invalid: set[str] = set()
        self.counters = {"proposal_crop_retries": 0, "source_annotation_invalid_paths": set(),
                         "source_annotation_errors": {}, "decode_failure_paths": set(),
                         "retry_exhaustions": 0}

    def __len__(self): return len(self.rows)
    def set_epoch(self, epoch: int): self.epoch = epoch

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        for attempt in range(MAX_RETRIES):
            row = self.rows[index] if attempt == 0 else self.rows[int(rng.integers(0, len(self.rows)))]
            key = source._path_key(row["path"])
            if key in self.source_invalid:
                self.counters["proposal_crop_retries"] += 1
                continue
            image = imread(row["path"])
            if image is None:
                self.counters["decode_failure_paths"].add(key)
                raise OSError(f"safe training image no longer decodes: {row['path']}")
            height, width = image.shape[:2]
            try:
                quad, gt_bbox = parse_gt(row, width, height)
            except ValueError as exc:
                self.source_invalid.add(key)
                self.counters["source_annotation_invalid_paths"].add(key)
                self.counters["source_annotation_errors"][key] = str(exc)[:300]
                self.counters["proposal_crop_retries"] += 1
                continue
            residual = self.samples[int(rng.integers(0, len(self.samples)))]
            bbox = sample_proposal_bbox(gt_bbox, residual)
            try:
                tensor, target, before = crop_target(image, bbox, quad)
            except ValueError as exc:
                if "does not contain" in str(exc) or "collapsed" in str(exc):
                    self.counters["proposal_crop_retries"] += 1
                    continue
                raise
            return tensor, target, torch.tensor(float(before)), row["subset"]
        self.counters["retry_exhaustions"] += 1
        raise RuntimeError(f"failed to build a valid proposal ROI after {MAX_RETRIES} draws")


def _source_hashes() -> dict[str, str]:
    return {"corner_net.py": hashlib.sha256((ROOT / "lpr" / "corner_net.py").read_bytes()).hexdigest(),
            "train_corner.py": hashlib.sha256((ROOT / "lpr_tools" / "train_corner.py").read_bytes()).hexdigest(),
            "train_refine.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _save_deploy(path: Path, model: RefineModel, data_sha: str, dist_sha: str):
    payload = {"architecture": REFINE_ARCHITECTURE, "input_size": list(REFINE_INPUT_SIZE),
               "padding": PADDING, "state_dict": model.state_dict(),
               "data_audit_sha256": data_sha, "proposal_distribution_sha256": dist_sha,
               "source_hashes": _source_hashes(), "seed": SEED}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp); temp.replace(path)


def _save_resume(path: Path, model: RefineModel, optimizer: torch.optim.Optimizer,
                 *, step: int, round_number: int, epoch: int, data_sha: str, dist_sha: str,
                 initial_state_sha: str, exposures: dict[str, Any], history: list,
                 source_hashes: dict[str, str]):
    payload = {"architecture": REFINE_ARCHITECTURE, "input_size": list(REFINE_INPUT_SIZE),
               "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": step, "round": round_number, "epoch": epoch,
               "data_audit_sha256": data_sha, "proposal_distribution_sha256": dist_sha,
               "initial_state_sha256": initial_state_sha, "exposures": exposures,
               "source_hashes": source_hashes,
               "history": history,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "numpy_rng": np.random.get_state(), "saved_at": time.time()}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp); temp.replace(path)


def train(*, data_dir: Path, distribution_path: Path, out_dir: Path, round_number: int,
          steps: int, resume: Path | None = None, audit_only: bool = False):
    if round_number not in (1, 2) or steps < 1 or steps > STEPS_PER_ROUND:
        raise ValueError("refiner allows only rounds 1/2 and at most 2,000 steps each")
    if (round_number == 1) != (resume is None):
        raise ValueError("round 1 starts fresh and round 2 requires a resume checkpoint")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows, _holdout_metadata_only, pool_audit = _safe_training_rows(data_dir)
    proposal_samples, dist_sha = read_proposal_distribution(distribution_path)
    audit = {"architecture": REFINE_ARCHITECTURE, "input_size": list(REFINE_INPUT_SIZE),
             "seed": SEED, "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
             "batch_size": BATCH_SIZE, "round": round_number, "requested_steps": steps,
             "proposal_samples": len(proposal_samples), "proposal_distribution_sha256": dist_sha,
             "pool_audit": pool_audit, "source_hashes": _source_hashes(), "training_started": False,
             "holdout_pixels_read": False, "devhard_decoded_or_trained": False,
             "devhard_bytes_hashed_for_exclusion": True,
             "holdout_evaluation": "external eval_refine.py on actual independent candidates; training does not decode holdout pixels"}
    data_sha = hashlib.sha256(json.dumps(pool_audit, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    audit["data_audit_sha256"] = data_sha
    out_dir.mkdir(parents=True, exist_ok=True)
    if not audit_only and round_number == 1 and resume is None:
        initial_path = out_dir / "initial-random-state.pt"
        training_files = [initial_path, out_dir / "training-log.jsonl",
                          out_dir / "round1-summary.json", out_dir / "round2-summary.json",
                          *out_dir.glob("checkpoint-step-*.pt"), *out_dir.glob("corner-refine-step-*.pt")]
        if any(path.exists() for path in training_files):
            raise ValueError("fresh round-1 run directory already contains training artifacts")
    audit_path = out_dir / ("audit-only.json" if audit_only else f"round{round_number}-data-audit.json")
    if not audit_only:
        audit["training_started"] = True
    encoded_audit = json.dumps(audit, indent=2, ensure_ascii=False)
    audit_path.write_text(encoded_audit, encoding="utf-8")
    if not audit_only:
        (out_dir / "data-audit.json").write_text(encoded_audit, encoding="utf-8")
    if audit_only:
        print(json.dumps(audit, ensure_ascii=False)); return
    initial_path = out_dir / "initial-random-state.pt"
    model = RefineModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001 if round_number == 1 else .0003)
    start_step = 0; history = []
    exposures = {"round": 0, "total": 0, "subsets": {}}
    if resume is not None:
        expected_resume = out_dir / "checkpoint-step-2000.pt"
        if Path(resume).resolve() != expected_resume.resolve():
            raise ValueError("round 2 can resume only the locally generated step-2000 checkpoint")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if (checkpoint.get("architecture") != REFINE_ARCHITECTURE
                or tuple(checkpoint.get("input_size", ())) != REFINE_INPUT_SIZE
                or checkpoint.get("round") != 1
                or checkpoint.get("data_audit_sha256") != data_sha
                or checkpoint.get("proposal_distribution_sha256") != dist_sha
                or checkpoint.get("source_hashes") != _source_hashes()
                or checkpoint.get("step") != 2000
                or not initial_path.is_file()
                or hashlib.sha256(initial_path.read_bytes()).hexdigest() != checkpoint.get("initial_state_sha256")):
            raise ValueError("resume checkpoint does not match the local fixed round-1 state/data")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        for group in optimizer.param_groups: group["lr"] = .0003
        start_step = checkpoint["step"]; history = list(checkpoint["history"])
        exposures = checkpoint["exposures"]
        exposures["round"] = 0
    else:
        torch.save({"architecture": REFINE_ARCHITECTURE, "input_size": list(REFINE_INPUT_SIZE),
                    "state_dict": model.state_dict(), "seed": SEED}, initial_path)
    initial_state_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    optimizer.param_groups[0]["lr"] = .001 if round_number == 1 else .0003
    dataset = RefineDataset(rows, proposal_samples)
    batch_count = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    end_step = start_step + steps
    cap = 2000 if round_number == 1 else MAX_STEPS
    if end_step > cap: raise ValueError("requested steps exceed refiner round/global cap")
    model.train(); step = start_step; started = time.monotonic()
    # Global counts carry over from round 1; round counts reset at each phase.
    for epoch in range(start_step // batch_count, (end_step - 1) // batch_count + 1):
        dataset.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(SEED + epoch), drop_last=False)
        skip = start_step % batch_count if epoch == start_step // batch_count else 0
        for images, targets, _before, subsets in loader:
            if skip: skip -= 1; continue
            if step >= end_step: break
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, parts = refine_loss(logits, targets)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite refiner loss at step {step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); step += 1
            exposures["round"] += int(images.shape[0])
            exposures["total"] += int(images.shape[0])
            for subset, count in Counter(subsets).items():
                exposures["subsets"][subset] = exposures["subsets"].get(subset, 0) + int(count)
            if step % 100 == 0 or step == end_step:
                record = {"step": step, "loss": float(loss.detach()), "heatmap": float(parts["heatmap"]),
                          "coordinates": float(parts["coordinates"]), "lr": optimizer.param_groups[0]["lr"],
                          "batch_image_exposures": int(images.shape[0]), "batch_source_subsets": dict(Counter(subsets)),
                          "round_cumulative_exposures": exposures["round"],
                          "global_cumulative_exposures": exposures["total"],
                          "global_cumulative_source_subsets": dict(exposures["subsets"]),
                          "proposal_crop_retries": dataset.counters["proposal_crop_retries"],
                          "source_annotation_invalid_unique": len(dataset.source_invalid)}
                history.append(record)
                with (out_dir / "training-log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if step % 500 == 0 or step == end_step:
                _save_resume(out_dir / f"checkpoint-step-{step}.pt", model, optimizer, step=step,
                             round_number=round_number, epoch=epoch, data_sha=data_sha, dist_sha=dist_sha,
                             initial_state_sha=initial_state_sha, exposures=exposures, history=history,
                             source_hashes=_source_hashes())
                _save_deploy(out_dir / f"corner-refine-step-{step}.pt", model, data_sha, dist_sha)
        if step >= end_step: break
    elapsed = time.monotonic() - started
    summary = {"step": step, "requested_steps_this_round": steps, "round": round_number,
               "elapsed_seconds": elapsed, "data_audit_sha256": data_sha,
               "proposal_distribution_sha256": dist_sha, "round_image_exposures": exposures["round"],
               "global_image_exposures": exposures["total"],
               "global_exposures_by_subset": exposures["subsets"],
               "initial_state_sha256": initial_state_sha,
               "source_hashes": _source_hashes(),
               "source_annotation_invalid_paths": sorted(dataset.source_invalid),
               "source_annotation_invalid_count": len(dataset.source_invalid),
               "source_annotation_errors": dataset.counters["source_annotation_errors"],
               "proposal_crop_retry_count": dataset.counters["proposal_crop_retries"],
               "retry_exhaustions": dataset.counters["retry_exhaustions"],
               "holdout_metrics": "reported by eval_refine.py on actual independent candidates",
               "loss_milestones": history[-20:]}
    (out_dir / f"round{round_number}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--proposal-distribution", type=Path, default=DIST_DEFAULT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=STEPS_PER_ROUND)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        train(data_dir=args.data_dir, distribution_path=args.proposal_distribution, out_dir=args.out_dir,
              round_number=args.round, steps=args.steps, resume=args.resume, audit_only=args.audit_only)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__": main()
