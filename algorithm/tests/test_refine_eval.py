from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from lpr.types import PlateCandidate
from lpr_tools.eval_refine import _corner_error, _ordered_reader_read, _summarize


def test_corner_error_uses_mean_corresponding_pixel_distance():
    candidate = PlateCandidate([(0, 0), (10, 0), (10, 10), (0, 10)],
                               (0, 0, 10, 10), "blue", 1.0, "test")
    assert _corner_error(candidate, candidate.corners) == 0
    shifted = PlateCandidate([(3, 4), (13, 4), (13, 14), (3, 14)],
                             (0, 0, 10, 10), "blue", 1.0, "test")
    assert _corner_error(shifted, candidate.corners) == 5


def test_ordered_crnn_stops_at_first_ok_without_ground_truth_selection():
    sample = SimpleNamespace(row=SimpleNamespace(plate_no="皖A12345"))
    reader = SimpleNamespace(read=lambda image: (image, [0.95] * 7))
    rects = [SimpleNamespace(image="first", color="blue"),
             SimpleNamespace(image="second", color="blue")]
    calls = []

    def decode(text, confidences, color, min_confidence):
        calls.append(text)
        status = "LOW_CONFIDENCE" if text == "first" else "OK"
        return SimpleNamespace(status=SimpleNamespace(value=status), confidence=.95,
                               plate_no=text)

    with patch("lpr_tools.eval_refine._decode_text", side_effect=decode):
        prediction, meta = _ordered_reader_read(rects, sample, reader, .8)
    assert prediction == "second"
    assert meta["status"] == "OK"
    assert calls == ["first", "second"]


def test_geometry_summary_uses_only_original_bbox_hits_and_all_image_accuracy():
    records = [
        {"path": "a", "baseline_correct": True, "refined_correct": False,
         "oracle_baseline_correct": True, "oracle_refined_correct": True,
         "candidates": [{"source": "network", "bbox_iou": .6, "bbox_hit": True, "before_corner_error_px": 10.,
                         "after_corner_error_px": 5.}]},
        {"path": "b", "baseline_correct": False, "refined_correct": True,
         "oracle_baseline_correct": False, "oracle_refined_correct": True,
         "candidates": [{"source": "classic", "bbox_iou": .2, "bbox_hit": False, "before_corner_error_px": 100.,
                         "after_corner_error_px": 2.}]},
    ]
    summary = _summarize(records)
    assert summary["baseline_crnn_exact_correct"] == 1
    assert summary["refined_crnn_exact_correct"] == 1
    assert summary["corner_geometry"]["bbox_hit_candidates"] == 1
    assert summary["corner_geometry"]["before_mean_corner_error_px"] == 10
    assert summary["corner_geometry"]["after_mean_corner_error_px"] == 5
    assert summary["corner_geometry"]["after_le_8px_ratio"] == 1
    assert summary["corner_geometry_by_source"]["network"]["bbox_hit_pairs"] == 1
    assert summary["best_iou_candidate_per_image"]["images_without_bbox_hit"] == 1
    assert summary["best_iou_candidate_per_image"]["bbox_hit_pairs"] == 1


def test_corner_error_rejects_non_finite_coordinates():
    candidate = PlateCandidate([(np.nan, 0), (1, 0), (1, 1), (0, 1)],
                               (0, 0, 1, 1), "blue", 1.0, "test")
    try:
        _corner_error(candidate, [(0, 0), (1, 0), (1, 1), (0, 1)])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite candidate corners must be rejected")
