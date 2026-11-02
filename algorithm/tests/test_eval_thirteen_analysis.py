from lpr_tools.eval.thirteen_analysis import _category, _norm, _read_index, _xywh_iou


def test_mosaic_roundtrips_through_a_chinese_windows_filename(tmp_path):
    import cv2
    import numpy as np
    from lpr_tools.eval.thirteen_analysis import _write_mosaic
    cv2.setNumThreads(1)
    target = tmp_path / "框偏拼图.jpg"
    _write_mosaic(target, np.zeros((40, 80, 3), dtype=np.uint8))
    decoded = cv2.imdecode(np.frombuffer(target.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (40, 80, 3)


def test_index_keeps_only_selected_rows_and_parses_official_bbox(tmp_path):
    path = tmp_path / "index.csv"
    chosen = str(tmp_path / "chosen.jpg")
    other = str(tmp_path / "other.jpg")
    path.write_text("path,quad,bbox\n" + chosen + ",0 0 10 0 10 10 0 10,0 0 10 10\n" +
                    other + ",malformed-unused,malformed-unused\n", encoding="utf-8")
    result = _read_index(path, {_norm(chosen)})
    assert list(result) == [_norm(chosen)]
    assert result[_norm(chosen)]["bbox"] == [0, 0, 10, 10]


def test_diagnostic_crop_clamps_bounds_and_includes_both_boxes():
    import numpy as np
    from lpr_tools.eval.thirteen_analysis import _diagnostic_crop
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = _diagnostic_crop(image, [0, 0, 10, 10], [150, 70, 40, 20])
    assert cropped.shape == (100, 200, 3)
    local = _diagnostic_crop(image, [50, 40, 30, 10])
    assert 10 <= local.shape[0] < 100
    assert 30 <= local.shape[1] < 200


def test_axis_aligned_iou_for_xywh_boxes():
    assert _xywh_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert _xywh_iou((0, 0, 10, 10), (5, 0, 10, 10)) == 1 / 3


def test_missing_route_is_distinct_from_no_legal_output(tmp_path):
    image_path = str(tmp_path / "x.jpg")
    record = {"truth": "皖A12345", "candidate_count": 1, "path": image_path,
              "boxes": {"0": {"bbox": [0, 0, 10, 10], "plate": "沪B45678",
                                "valid": True, "confidence": .5}}, "rects": []}
    index = {_norm(image_path): {"bbox": [0, 0, 10, 10]}}
    assert _category(record, {"plate": "沪B45678"}, index) == "有路线缺读且其余全错"
    record["boxes"]["0"]["valid"] = False
    assert _category(record, {"plate": None}, index) == "无合法路线输出"


def test_error_buckets_use_selected_box_before_text_error(tmp_path):
    image_path = str(tmp_path / "x.jpg")
    record = {"truth": "皖A12345", "candidate_count": 1, "path": image_path,
              "boxes": {"0": {"bbox": [100, 100, 10, 10]}}, "rects": []}
    index = {_norm(image_path): {"bbox": [0, 0, 10, 10], "quad": [(0, 0), (10, 0), (10, 10), (0, 10)]}}
    assert _category(record, {"plate": "皖A12346", "candidate_index": 0}, index) == "框偏_IoU<0.7"


def test_error_buckets_distinguish_province_and_single_other_character(tmp_path):
    image_path = str(tmp_path / "x.jpg")
    record = {"truth": "皖A12345", "candidate_count": 1, "path": image_path,
              "boxes": {"0": {"bbox": [0, 0, 10, 10]}}, "rects": []}
    index = {_norm(image_path): {"bbox": [0, 0, 10, 10], "quad": [(0, 0), (10, 0), (10, 10), (0, 10)]}}
    assert _category(record, {"plate": "沪A12345", "candidate_index": 0}, index) == "只错省份字"
    assert _category(record, {"plate": "皖A12346", "candidate_index": 0}, index) == "只错其他一位"


def test_any_valid_route_correct_counts_even_when_route_envelope_prefers_wrong(tmp_path):
    image_path = str(tmp_path / "x.jpg")
    record = {"truth": "皖A12345", "candidate_count": 1, "path": image_path,
              "boxes": {"0": {"bbox": [0, 0, 10, 10], "plate": "皖A12347", "valid": True}},
              "rects": [{"candidate_index": 0, "rect_index": 0,
                         "segment": {"plate": "皖A12345", "confidence": .2, "valid": True},
                         "reader": {"plate": "皖A12346", "confidence": .9, "valid": True}},
                        {"candidate_index": 0, "rect_index": 1,
                         "segment": {"plate": "皖A12367", "confidence": .95, "valid": True}}]}
    index = {_norm(image_path): {"bbox": [0, 0, 10, 10]}}
    assert _category(record, {"plate": "皖A12367", "candidate_index": 0}, index) == "有路线读对但仲裁挑错"
