"""Focused contracts for conditional camera fine-tuning setup."""
from __future__ import annotations

import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lpr_tools import train_det


class _FixedRng:
    def __init__(self, values=(0.0, 0.9, 0.9, 0.9), scale=0.6):
        self.values = iter(values)
        self.scale = scale

    def random(self):
        return next(self.values)

    def choice(self, _values):
        return self.scale

    def uniform(self, low, high, size=None):
        if size is not None:
            return np.zeros(size, dtype=np.float32)
        return (low + high) / 2


class Batch12TrainingSetupTests(unittest.TestCase):
    def test_init_from_and_resume_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train_det.validate_init_resume_options(Path("v1.pt"), Path("last.pt"))
        train_det.validate_init_resume_options(Path("v1.pt"), None)
        train_det.validate_init_resume_options(None, Path("last.pt"))

    def test_stage_epoch_floor_and_learning_rates_preserve_legacy_defaults(self):
        self.assertEqual(train_det.minimum_complete_epochs(stage_parent=False), 40)
        self.assertEqual(train_det.minimum_complete_epochs(stage_parent=True), 10)
        self.assertEqual(train_det.stage_learning_rates(stage_parent=False), (1e-4, 1e-3))
        self.assertEqual(train_det.stage_learning_rates(stage_parent=True), (2e-5, 2e-4))

    def test_stage_provenance_records_fresh_epoch_zero_state(self):
        parent = train_det.build_stage_parent(
            Path("algorithm/models/plate_det_v1.pt"), "a" * 64,
            parent_epochs=40, parent_exposures=8_521_720,
            parent_train_pool_sha256="b" * 64, release_manifest_sha256="c" * 64)
        self.assertEqual(Path(parent["model_path"]), Path("algorithm/models/plate_det_v1.pt"))
        self.assertEqual(parent["model_sha256"], "a" * 64)
        self.assertEqual(parent["parent_training"]["verified_epoch_completed"], 40)
        self.assertEqual(parent["stage_start_epoch"], 0)
        self.assertEqual(parent["stage_start_training_wall_seconds"], 0.)
        self.assertEqual(parent["optimizer_state"], "new")
        self.assertEqual(parent["rng_state"], "new")

    def test_old_checkpoint_without_camera_hash_remains_compatible(self):
        self.assertIsNone(train_det.validate_camera_source(
            {"source_hashes": {"trainer.py": "legacy"}}, current_sha="new-camera-hash",
            allow_change=False))
        with self.assertRaisesRegex(ValueError, "camera augmentation source changed"):
            train_det.validate_camera_source(
                {"camera_source_sha256": "old"}, current_sha="new",
                allow_change=False)

    def _pool(self, path: Path, *, positive: bool):
        ok, encoded = cv2.imencode(".jpg", np.full((1160, 720, 3), 96, np.uint8))
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())
        pool = train_det.CompactPool.__new__(train_det.CompactPool)
        pool.paths = [str(path)]
        pool.subsets = ["ccpd_base" if positive else "ccpd_np"]
        if positive:
            pool.labels = array("f", [1., 100., 200., 80., 40.,
                                      100., 200., 180., 200., 180., 240., 100., 240.,
                                      720., 1160.])
        else:
            pool.labels = array("f", [0.] + [0.]*12 + [720., 1160.])
        return pool

    def test_camera_augment_maps_positive_labels_and_keeps_np_targets_empty(self):
        from lpr_tools import eval_det
        temporary_root = Path(__file__).resolve().parents[1]/"out"/"twelfth"/"temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            folder = Path(temporary)
            fake_canvas = np.zeros((720, 1280, 3), np.uint8)
            mapped_box = (100., 200., 200., 80.)
            mapped_quad = [[100., 200.], [300., 200.], [300., 280.], [100., 280.]]
            camera_calls = []

            def canvas(image, bbox, quad, scale):
                camera_calls.append((bbox, np.asarray(quad), scale, image.shape[:2]))
                return fake_canvas, mapped_box, mapped_quad, {"gt_bbox_fully_visible": True}

            with patch("numpy.random.default_rng", return_value=_FixedRng()), \
                    patch.object(eval_det, "camera_canvas", side_effect=canvas):
                positive = train_det.read_training_sample(
                    self._pool(folder/"positive.jpg", positive=True), 0, 0, 256,
                    augment=True, camera_augment=True)
            self.assertEqual(len(camera_calls), 1)
            self.assertAlmostEqual(camera_calls[0][0][0], 50.)
            self.assertAlmostEqual(camera_calls[0][0][1], 100.)
            self.assertEqual(camera_calls[0][2], .6)
            self.assertEqual(positive[0].shape, (3, 256, 256))
            self.assertTrue(np.allclose(positive[1][0], [20., 96., 60., 112.]))
            self.assertTrue(np.allclose(positive[2][0], [[20., 96.], [60., 96.],
                                                         [60., 112.], [20., 112.]]))

            np_calls = []
            def negative_canvas(image, bbox, quad, scale):
                np_calls.append((bbox, np.asarray(quad), scale))
                return fake_canvas, mapped_box, mapped_quad, {"gt_bbox_fully_visible": True}

            with patch("numpy.random.default_rng", return_value=_FixedRng()), \
                    patch.object(eval_det, "camera_canvas", side_effect=negative_canvas):
                negative = train_det.read_training_sample(
                    self._pool(folder/"negative.jpg", positive=False), 0, 0, 256,
                    augment=True, camera_augment=True)
            self.assertEqual(len(np_calls), 1)
            self.assertAlmostEqual(np_calls[0][0][0], 179.5)
            self.assertAlmostEqual(np_calls[0][0][1], 289.5)
            self.assertEqual(negative[1].shape, (0, 4))
            self.assertEqual(negative[2].shape, (0, 4, 2))

    def test_camera_oversize_retries_smaller_scale_but_other_errors_propagate(self):
        from lpr_tools import eval_det
        root = Path(__file__).resolve().parents[1]/"out"/"twelfth"/"temp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            pool = self._pool(Path(temporary)/"source.jpg", positive=True)
            result = (np.zeros((720, 1280, 3), np.uint8), (100., 200., 200., 80.),
                      [[100., 200.], [300., 200.], [300., 280.], [100., 280.]], {})
            with patch("numpy.random.default_rng", return_value=_FixedRng(scale=1.6)), \
                    patch.object(eval_det, "camera_canvas", side_effect=[
                        ValueError("scaled GT bbox+quad union exceeds camera canvas"), result]) as transform:
                sample = train_det.read_training_sample(pool, 0, 0, 256, augment=False, camera_augment=True)
            self.assertEqual([c.args[3] for c in transform.call_args_list], [1.6, 1.3])
            self.assertEqual(sample[1].shape, (1, 4))
            with patch("numpy.random.default_rng", return_value=_FixedRng()), \
                    patch.object(eval_det, "camera_canvas", side_effect=ValueError("invalid label")):
                with self.assertRaisesRegex(ValueError, "invalid label"):
                    train_det.read_training_sample(pool, 0, 0, 256, augment=False, camera_augment=True)


if __name__ == "__main__":
    unittest.main()
