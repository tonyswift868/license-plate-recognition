"""Small pure metric tests for the detector evaluation protocol."""
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from lpr_tools.eval_det import (bbox_iou, corner_error_px, first_ok_read,
                                BoxCrnnAdapter, summarize_np, topk_hits,
                                validate_purpose)


@dataclass
class Candidate:
    bbox: tuple


class DetectorEvaluationTests(unittest.TestCase):
    def test_bbox_iou_and_invalid_geometry(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3)
        self.assertEqual(bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_topk_respects_candidate_order_and_k(self):
        candidates = [Candidate((20, 20, 2, 2)), Candidate((0, 0, 10, 10))]
        self.assertFalse(topk_hits(candidates, (0, 0, 10, 10), k=1, threshold=.7))
        self.assertTrue(topk_hits(candidates, (0, 0, 10, 10), k=3, threshold=.7))

    def test_corner_error_uses_corresponding_points(self):
        truth = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shifted = [(1, 0), (11, 0), (11, 10), (1, 10)]
        self.assertAlmostEqual(corner_error_px(shifted, truth), 1.0)

    def test_reader_returns_first_operational_ok_without_truth(self):
        rows = [("bad", [0.1], "blue"), ("first", [0.9], "blue"),
                ("later", [1.0], "blue")]
        def decode(text, confidences, color):
            return SimpleNamespace(plate_no=text, status="OK" if text != "bad" else "LOW_CONFIDENCE",
                                   confidence=confidences[0])
        self.assertEqual(first_ok_read(rows, decode), ("first", "OK"))

    def test_np_image_and_proposal_counts(self):
        report = summarize_np([{"candidate_count": 0}, {"candidate_count": 2},
                               {"candidate_count": 1}])
        self.assertEqual(report["images_with_false_positive"], 2)
        self.assertEqual(report["false_positive_candidates"], 3)

    def test_fixed_manifests_require_frozen_checkpoint(self):
        with self.assertRaises(ValueError):
            validate_purpose("val", False)
        validate_purpose("val", True)
        validate_purpose("holdout", False)

    def test_box_crnn_adapter_processes_arbitrary_crop_without_shape_hypotheses(self):
        import numpy as np
        import torch
        from lpr.charset import NUM_CLASSES

        class FakeModel:
            def __init__(self):
                self.input_shape = None

            def __call__(self, tensor):
                self.input_shape = tuple(tensor.shape)
                logits = torch.zeros((1, 1, NUM_CLASSES + 1))
                logits[0, 0, NUM_CLASSES] = 20.0
                return logits

        model = FakeModel()
        reader = SimpleNamespace(name="fake_crnn", version="crnn_v2",
                                 model=model, temperature=1.0)
        adapter = BoxCrnnAdapter(reader)
        text, confidences = adapter.read(np.full((37, 93, 3), 127, dtype=np.uint8))
        self.assertEqual(model.input_shape, (1, 1, 40, 160))
        self.assertEqual((text, confidences), ("", []))
        self.assertEqual(adapter.version, "crnn_v2")

    def test_box_crnn_adapter_matches_existing_reader_on_canonical_crop(self):
        import numpy as np
        from lpr.recognize.crnn import CrnnNet, CrnnReader

        reader = CrnnReader("crnn_v2", CrnnNet().eval())
        adapter = BoxCrnnAdapter(reader)
        crop = np.random.default_rng(17).integers(0, 256, (80, 252, 3), dtype=np.uint8)
        self.assertEqual(adapter.read(crop), reader.read(crop))


if __name__ == "__main__":
    unittest.main()
