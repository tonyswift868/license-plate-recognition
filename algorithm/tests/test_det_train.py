"""CPU tests for target creation and detector training loss semantics."""
import unittest
from array import array
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lpr_tools.train_det import (CompactPool, detector_geometry_is_valid,
                                 background_negative, detector_loss, jpeg_size, make_targets,
                                 read_training_sample, reduced_to_original)


class DetectorTrainingLossTests(unittest.TestCase):
    def test_replacement_export_clears_stale_selected_ledger(self):
        import cv2
        from lpr_tools import train_det
        worker = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=worker) as temporary:
            folder = Path(temporary)
            image_path = folder/"source.jpg"
            ok, image = cv2.imencode(".jpg", np.zeros((80, 100, 3), np.uint8))
            self.assertTrue(ok)
            image_path.write_bytes(image.tobytes())
            row = {"path": str(image_path), "has_plate": "1", "bbox": "10 20 40 16",
                   "quad": "10 20 50 20 50 36 10 36", "skip_reason": "invalid_format",
                   "subset": "ccpd_base", "source_category": "base_green"}
            conn = train_det.init_db(folder/"ledger.sqlite3")
            try:
                sha = train_det.file_sha(str(image_path))
                key = train_det.path_key(str(image_path))
                conn.execute("INSERT INTO selected_sha VALUES (?, ?, ?)", (sha, key, "stale"))
                for seq in (1, 2):
                    conn.execute("INSERT INTO candidate VALUES (?, ?, ?, ?, ?)",
                                 (seq, "base_green", json.dumps(row), "", key))
                conn.commit()
                with patch.object(train_det, "DATA", folder):
                    report, train_file, _holdout = train_det.assemble(conn)
                self.assertEqual(report["selected"], 1)
                self.assertEqual(report["duplicate_content"], 1)
                self.assertEqual(len(list(train_det.csv_rows(Path(train_file[0])))), 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM selected_sha").fetchone()[0], 1)
            finally:
                conn.close()

    def _outputs(self):
        return {
            "heatmap": torch.zeros((1, 1, 16, 16), requires_grad=True),
            "box": torch.zeros((1, 4, 16, 16), requires_grad=True),
            "corners": torch.zeros((1, 8, 16, 16), requires_grad=True),
        }

    def test_no_plate_sample_trains_heatmap_only(self):
        no_box = np.zeros((0, 4), dtype=np.float32)
        no_corners = np.zeros((0, 4, 2), dtype=np.float32)
        sample = (np.zeros((3, 128, 128), dtype=np.uint8), no_box, no_corners, "ccpd_np")
        target = make_targets([sample], 128, "cpu")
        outputs = self._outputs()
        loss, parts = detector_loss(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(outputs["heatmap"].grad.abs().sum()), 0.0)
        self.assertEqual(float(outputs["box"].grad.abs().sum()), 0.0)
        self.assertEqual(float(outputs["corners"].grad.abs().sum()), 0.0)
        self.assertEqual(float(parts["box"]), 0.0)
        self.assertEqual(float(parts["corners"]), 0.0)

    def test_positive_sample_has_finite_nonzero_box_and_corner_loss(self):
        box = np.asarray([[32, 32, 96, 80]], dtype=np.float32)
        corners = np.asarray([[[32, 32], [96, 32], [96, 80], [32, 80]]], dtype=np.float32)
        sample = (np.zeros((3, 128, 128), dtype=np.uint8), box, corners, "ccpd_db")
        target = make_targets([sample], 128, "cpu")
        outputs = self._outputs()
        loss, parts = detector_loss(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["box"]), 0.0)
        self.assertGreater(float(parts["corners"]), 0.0)
        loss.backward()
        self.assertGreater(float(outputs["heatmap"].grad.abs().sum()), 0.0)
        self.assertGreater(float(outputs["box"].grad.abs().sum()), 0.0)
        self.assertGreater(float(outputs["corners"].grad.abs().sum()), 0.0)
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_bad_target_shapes_fail_early(self):
        bad_sample = (np.zeros((3, 128, 128), dtype=np.uint8),
                      np.asarray([[0, 0, 0, 1]], dtype=np.float32),
                      np.zeros((1, 4, 2), dtype=np.float32), "ccpd_db")
        with self.assertRaises(ValueError):
            make_targets([bad_sample], 128, "cpu")

    def test_detector_geometry_filter_ignores_ocr_skip_reason(self):
        valid_but_unreadable = {
            "has_plate": "1", "bbox": "10 20 40 16",
            "quad": "10 20 50 20 50 36 10 36",
            "skip_reason": "unsupported_layout",
        }
        self.assertTrue(detector_geometry_is_valid(valid_but_unreadable, 100, 80))
        independent_bbox = dict(valid_but_unreadable, bbox="12 22 36 12")
        self.assertTrue(detector_geometry_is_valid(independent_bbox, 100, 80))
        invalid_geometry = dict(valid_but_unreadable, bbox="10 20 0 16")
        self.assertFalse(detector_geometry_is_valid(invalid_geometry, 100, 80))

    def test_odd_jpeg_dimensions_use_axis_specific_reduced_scale(self):
        import cv2
        image = np.zeros((479, 641, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        worker = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=worker) as temp_dir:
            path = Path(temp_dir) / "odd.jpg"
            path.write_bytes(encoded.tobytes())
            original_w, original_h = jpeg_size(str(path))
            reduced = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        decoded_h, decoded_w = reduced.shape[:2]
        self.assertEqual((original_w, original_h), (641, 479))
        mapped_end = reduced_to_original(float(decoded_w), float(decoded_h), original_w, original_h,
                                         decoded_w, decoded_h)
        self.assertAlmostEqual(mapped_end[0], original_w)
        self.assertAlmostEqual(mapped_end[1], original_h)
        self.assertNotEqual(original_w / decoded_w, original_h / decoded_h)

    def test_odd_jpeg_training_box_maps_using_actual_reduced_shape(self):
        import cv2
        image = np.zeros((479, 641, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        worker = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=worker) as temp_dir:
            path = Path(temp_dir) / "odd.jpg"
            path.write_bytes(encoded.tobytes())
            pool = CompactPool.__new__(CompactPool)
            pool.paths = [str(path)]
            pool.subsets = ["ccpd_db"]
            pool.labels = array("f", [1., 100., 50., 200., 100.,
                                      100., 50., 300., 50., 300., 150., 100., 150.,
                                      641., 479.])
            sample = read_training_sample(pool, 0, 0, 128, augment=False)
        expected_box = np.asarray([[
            100 * 128 / 641, 50 * 96 / 479 + 16,
            300 * 128 / 641, 150 * 96 / 479 + 16,
        ]], dtype=np.float32)
        self.assertEqual(sample[0].shape, (3, 128, 128))
        self.assertTrue(np.allclose(sample[1], expected_box, atol=2e-4, rtol=0))

    def test_background_negative_excludes_ground_truth_and_rejects_np(self):
        plane = np.indices((160, 160)).sum(axis=0).astype(np.uint8) % 100
        rgb = np.stack((plane, plane, plane))
        rgb[:, 35:125, 35:125] = 250
        positive = (rgb, np.asarray([[40, 40, 120, 120]], np.float32),
                    np.asarray([[[35, 35], [125, 35], [125, 125], [35, 125]]], np.float32), "ccpd_db")
        negative = background_negative(positive)
        self.assertIsNotNone(negative)
        self.assertEqual(negative[1].shape, (0, 4))
        self.assertEqual(negative[2].shape, (0, 4, 2))
        self.assertEqual(negative[3], "background_negative")
        self.assertLessEqual(int(negative[0].max()), 99)

        np_sample = (rgb, np.zeros((0, 4), np.float32),
                     np.zeros((0, 4, 2), np.float32), "ccpd_np")
        self.assertIsNone(background_negative(np_sample))

    def test_resume_source_validation_accepts_unchanged_sources(self):
        from lpr_tools import train_det
        payload = {"source_hashes": {"model.py": "same", "trainer.py": "same"}}
        with patch.object(train_det, "source_hashes",
                          return_value={"model.py": "same", "trainer.py": "same"}):
            self.assertEqual(train_det.validate_resume_sources(payload, allow_change=False), {})

    def test_resume_source_validation_rejects_changes_without_authorization(self):
        from lpr_tools import train_det
        payload = {"source_hashes": {"model.py": "old"}}
        with patch.object(train_det, "source_hashes", return_value={"model.py": "new"}):
            with self.assertRaises(ValueError):
                train_det.validate_resume_sources(payload, allow_change=False)

    def test_resume_source_validation_requires_reason_for_override(self):
        from lpr_tools import train_det
        payload = {"source_hashes": {"model.py": "old"}}
        with patch.object(train_det, "source_hashes", return_value={"model.py": "new"}):
            for reason in (None, "", "   "):
                with self.subTest(reason=reason), self.assertRaises(ValueError):
                    train_det.validate_resume_sources(payload, allow_change=True, reason=reason)

    def test_resume_source_validation_reports_authorized_changes(self):
        from lpr_tools import train_det
        payload = {"source_hashes": {"model.py": "old"}}
        with patch.object(train_det, "source_hashes", return_value={"model.py": "new"}):
            self.assertEqual(
                train_det.validate_resume_sources(payload, allow_change=True,
                                                  reason="resume after loss-check optimization"),
                {"model.py": {"old": "old", "new": "new"}},
            )

    def test_finite_loss_history_stays_true_for_finite_losses(self):
        from lpr_tools import train_det
        history = torch.tensor(True)
        train_det.record_finite_loss(history, torch.tensor(3.5, requires_grad=True))
        train_det.record_finite_loss(history, torch.tensor(0.25))
        self.assertTrue(bool(history))
        train_det.check_finite_history(history)

    def test_nonfinite_loss_is_sticky_and_raises_at_check(self):
        from lpr_tools import train_det
        history = torch.tensor(True)
        train_det.record_finite_loss(history, torch.tensor(float("nan")))
        train_det.record_finite_loss(history, torch.tensor(1.0))
        self.assertFalse(bool(history))
        with self.assertRaises(FloatingPointError):
            train_det.check_finite_history(history)

    def test_finite_check_history_does_not_attach_to_or_modify_loss(self):
        from lpr_tools import train_det
        loss = torch.tensor(2.0, requires_grad=True)
        history = torch.tensor(True)
        train_det.record_finite_loss(history, loss)
        self.assertTrue(loss.requires_grad)
        self.assertIsNone(loss.grad_fn)
        self.assertFalse(history.requires_grad)


if __name__ == "__main__":
    unittest.main()
