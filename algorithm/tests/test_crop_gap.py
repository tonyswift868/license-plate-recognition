import json

import numpy as np
from lpr.config import load_config
from lpr_tools.eval.crop_gap import (
    _bbox_crop,
    _load_ceiling_cache,
    _make_thumbnail,
    _route_correct,
    _same_candidate_alt_rect_rescue,
    _summarize,
    classify_primary_bucket,
)


def test_primary_bucket_has_fixed_priority_and_keeps_unsupported_separate():
    assert classify_primary_bucket(unsupported=True, other_candidate=True,
                                   expanded=True, gt_correct=True) == "unsupported"
    assert classify_primary_bucket(other_candidate=True, expanded=True,
                                   same_candidate_route=True, gt_correct=True) == "alternate_candidate"
    assert classify_primary_bucket(same_candidate_route=True, expanded=True,
                                   gt_correct=True) == "same_candidate_route_selection"
    assert classify_primary_bucket(expanded=True, gt_correct=True) == "crop_expansion"
    assert classify_primary_bucket(gt_correct=True) == "gt_geometry_rescue"
    assert classify_primary_bucket() == "reader_or_segment_residual"


def test_persistent_decode_error_is_highest_bucket_priority():
    assert classify_primary_bucket(decode_error=True, unsupported=True,
                                   other_candidate=True) == "decode_error"


def test_unwarped_bbox_crop_expands_clips_and_scales_to_plate_size():
    cfg = load_config()
    image = np.zeros((100, 120, 3), np.uint8)
    image[10:90, 10:110] = (20, 40, 60)
    crop = _bbox_crop(image, (5, 20, 40, 20), .2, cfg, "blue", 7)
    assert crop.image.shape == (80, 252, 3)
    assert crop.candidate.bbox == (0, 16, 53, 28)
    assert np.array_equal(crop.homography, np.eye(3))


def test_route_correct_uses_any_exact_route_but_respects_route_fields():
    routes = {
        "segment_cnn": {"selected": {"correct": False}},
        "crnn": {"correct": True},
    }
    assert _route_correct(routes)
    routes["crnn"]["correct"] = False
    assert not _route_correct(routes)
    assert not _route_correct(None)


def test_summary_reads_full_records_with_nested_diagnostic():
    rows = [{"subset": "devhard_db", "path": "db/a.jpg",
             "diagnostic": {"primary_bucket": "same_candidate_route_selection",
                            "potential_rescues": {"same_candidate_route_selection": True}}}]
    summary = _summarize(rows)
    assert summary["buckets"]["same_candidate_route_selection"]["n"] == 1
    assert summary["potential_rescues"]["same_candidate_route_selection"]["n"] == 1
    assert summary["potential_rescues"]["same_candidate_route_selection"]["examples"][0]["path"] == "db/a.jpg"


def test_same_candidate_rect_rescue_uses_selected_crop_identity_not_rect_number():
    evidence = [{"index": 1, "rectifications": [
        {"rectification": 1, "selected_pipeline_crop": False,
         "routes": {"segment_cnn": {"selected": {"correct": True}}, "crnn": {"correct": False}}},
        {"rectification": 2, "selected_pipeline_crop": True,
         "routes": {"segment_cnn": {"selected": {"correct": False}}, "crnn": {"correct": False}}},
    ]}]
    assert _same_candidate_alt_rect_rescue(evidence, 1)
    evidence[0]["rectifications"][0]["selected_pipeline_crop"] = True
    assert not _same_candidate_alt_rect_rescue(evidence, 1)


def test_ceiling_cache_is_optional_and_cnn_crnn_changes_do_not_invalidate_top3(tmp_path, monkeypatch):
    import lpr_tools.eval.crop_gap as crop_gap

    cfg = load_config()
    cached_cfg = cfg.model_dump(mode="json")
    cached_cfg["runtime"]["model_path"] = "other-cnn.pt"
    cached_cfg["runtime"]["plate_reader_model_path"] = "other-crnn.pt"
    cached_cfg["runtime"]["torch_threads"] = 2
    cached_cfg["postprocess"]["min_confidence"] = 0.5
    cached_cfg["rectify"]["size_7"] = [250, 80]
    monkeypatch.setattr(crop_gap, "model_identity", lambda _cfg, _field: {"sha256": "corner-sha"})
    manifest = {"manifest_sha256": "manifest-sha"}
    assert _load_ceiling_cache(None, manifest, [], cfg) == (None, {})
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({
        "manifests": {"devhard": {"sha256": "manifest-sha"}},
        "config_and_models": {"effective_config": cached_cfg,
                              "models": {"corner": {"sha256": "corner-sha"}}},
        "records": [],
    }), encoding="utf-8")
    data, rows = _load_ceiling_cache(cache_path, manifest, [], cfg)
    assert data is not None and rows == {}


def test_ceiling_cache_rejects_changed_corner_locator_chain(tmp_path, monkeypatch):
    import lpr_tools.eval.crop_gap as crop_gap

    cfg = load_config()
    cached_cfg = cfg.model_dump(mode="json")
    monkeypatch.setattr(crop_gap, "model_identity", lambda _cfg, _field: {"sha256": "new-corner-sha"})
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({
        "manifests": {"devhard": {"sha256": "manifest-sha"}},
        "config_and_models": {"effective_config": cached_cfg,
                              "models": {"corner": {"sha256": "old-corner-sha"}}},
        "records": [],
    }), encoding="utf-8")
    try:
        _load_ceiling_cache(cache_path, {"manifest_sha256": "manifest-sha"}, [], cfg)
    except ValueError as exc:
        assert "corner model differs" in str(exc)
    else:
        raise AssertionError("changed corner model must invalidate cached Top-3")


def test_thumbnail_labels_are_above_crop_and_width_stays_within_400px():
    final = np.full((80, 252, 3), (10, 20, 30), np.uint8)
    gt = np.full((80, 252, 3), (40, 50, 60), np.uint8)
    alt = np.full((80, 252, 3), (70, 80, 90), np.uint8)
    two = _make_thumbnail({}, None, final, gt, {})
    three = _make_thumbnail({}, None, final, gt, {"best_alternative_crop": alt})
    assert two.shape[1] == 390
    assert three.shape[1] == 399
    assert max(two.shape[:2]) <= 400 and max(three.shape[:2]) <= 400
    assert np.array_equal(two[20, 30], final[5, 40])
    assert np.array_equal(two[20, 225], gt[5, 40])
