from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from lpr_tools.eval.common import letterbox
from lpr_tools.eval_det import (
    bbox_iou,
    camera_canvas,
    camera_canvas_metadata,
    indexed_dimensions,
    ranked_box_metrics,
    threshold_curve,
    width_distribution,
)


class Batch12ThresholdMetricsTests(unittest.TestCase):
    def test_threshold_curve_uses_image_level_false_positive_rate(self):
        rows = threshold_curve(
            positive_records=[[(0.8, 0.75)], [(0.4, 0.9)], []],
            negative_records=[[(0.9, 0.0), (0.8, 0.0)], [(0.3, 0.0)], []],
            thresholds=(0.25, 0.5, 0.85),
        )
        self.assertEqual([r["positive_top1_iou_0_7_recall"] for r in rows], [2 / 3, 1 / 3, 0])
        self.assertEqual([r["np_false_positive_image_rate"] for r in rows], [2 / 3, 1 / 3, 1 / 3])

    def test_bbox_iou_is_zero_for_invalid_or_nonoverlapping_boxes(self):
        self.assertEqual(bbox_iou((0, 0, 0, 2), (0, 0, 1, 1)), 0)
        self.assertEqual(bbox_iou((0, 0, 1, 1), (3, 3, 1, 1)), 0)

    def test_top1_uses_first_candidate_and_width_quantiles_are_linear(self):
        metrics = ranked_box_metrics([[{"iou": .2}, {"iou": .99}], []])
        self.assertEqual(metrics["topk_iou_0_7"]["1"]["hits"], 0)
        self.assertEqual(metrics["topk_iou_0_7"]["3"]["hits"], 1)
        self.assertEqual(metrics["mean_candidates"], 1.)
        self.assertEqual(width_distribution([1, 2, 3])["median"], 2)

    def test_metadata_crop_keeps_plate_inside_canvas_at_large_scale(self):
        meta = camera_canvas_metadata(720, 1160, (45, 462, 615, 171),
                                      ((45, 462), (637, 513), (660, 633), (68, 582)), 1.6)
        x, y, w, h = meta["bbox_xywh"]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 1280)
        self.assertLessEqual(y + h, 720)

    def test_scale_one_matches_benchmark_pixels_and_geometry(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[10:20, 20:40] = (0, 0, 255)
        bbox = (20, 10, 20, 10)
        quad = ((20, 10), (40, 10), (40, 20), (20, 20))
        actual, actual_bbox, actual_quad, _ = camera_canvas(image, bbox, quad, 1.0, (128, 72))
        expected = letterbox(image, (128, 72))
        meta = camera_canvas_metadata(80, 40, bbox, quad, 1.0, (128, 72))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual_bbox, meta["bbox_xywh"], atol=0, rtol=0)
        np.testing.assert_allclose(actual_quad, meta["quad_xy"], atol=0, rtol=0)
        self.assertEqual(actual[24, 45, 2], 255)

    def test_portrait_marker_tracks_geometry_at_all_six_scales(self):
        image = np.zeros((1160, 720, 3), np.uint8)
        image[600:660, 100:220] = (0, 0, 255)
        bbox = (100, 600, 120, 60)
        quad = ((100, 600), (220, 600), (220, 660), (100, 660))
        for scale in (.4, .6, .8, 1., 1.3, 1.6):
            with self.subTest(scale=scale):
                canvas, box, corners, meta = camera_canvas(image, bbox, quad, scale)
                x, y, w, h = box
                self.assertEqual(int(canvas[round(y+h/2), round(x+w/2), 2]), 255)
                self.assertTrue(meta['gt_quad_fully_visible'])
                np.testing.assert_allclose(corners, [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                if scale == 1.:
                    np.testing.assert_array_equal(canvas, letterbox(image, (1280, 720)))

    def test_invalid_geometry_is_rejected_before_decoding(self):
        quad = ((1, 1), (3, 1), (3, 3), (1, 3))
        for box, scale, canvas in (((float('nan'), 1, 2, 2), 1., (1280, 720)),
                                   ((1, 1, 2, 2), 0., (1280, 720)),
                                   ((1, 1, 2, 2), 1., (0, 720))):
            with self.subTest(box=box, scale=scale, canvas=canvas), self.assertRaises(ValueError):
                camera_canvas_metadata(720, 1160, box, quad, scale, canvas)

    def test_indexed_dimensions_falls_back_to_jpeg_sof_without_decoding(self):
        root = Path("algorithm/out/twelfth/temp")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            image = Path(temp) / "sample.jpg"
            image.write_bytes(bytes.fromhex("FFD8 FFC0 0007 08 0002 0003"))
            index = Path(temp) / "index.csv"
            with index.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("path",))
                writer.writeheader()
                writer.writerow({"path": str(image)})
            from lpr_tools.eval.index_samples import path_key
            self.assertEqual(indexed_dimensions(index, {path_key(image)}),
                             {path_key(image): (3, 2)})


if __name__ == "__main__":
    unittest.main()
