from lpr_tools.eval.three_route_expected import runtime_projection


def test_error_projection_preserves_ok_and_blocks_special_manual_color():
    record = {"candidate_count": 1, "rects": [{"candidate_index": 0, "color": "black"}]}
    auto = {"status": "OK", "plate": "A", "confidence": .99, "route": "box"}
    assert runtime_projection(record, auto) == auto
    manual = {**auto, "status": "LOW_CONFIDENCE"}
    assert runtime_projection(record, manual)["status"] == "UNSUPPORTED"
    assert runtime_projection(record, manual)["plate"] is None


def test_no_read_projection_keeps_best_invalid_route_and_stable_tie():
    record = {"candidate_count": 1, "rects": [
        {"candidate_index": 0, "rect_index": 0, "color": "blue",
         "segment": {"status": "INVALID_FORMAT", "plate": "X", "confidence": .5},
         "reader": {"status": "INVALID_FORMAT", "plate": "Y", "confidence": .5}},
        {"candidate_index": 0, "rect_index": 1,
         "reader": {"status": "INVALID_FORMAT", "plate": "Z", "confidence": .8}},
    ]}
    selected = runtime_projection(record, {"status": "NO_READ"})
    assert selected["plate"] == "Z" and selected["route"] == "reader"
    record["rects"].pop()
    assert runtime_projection(record, {"status": "NO_READ"})["plate"] == "X"
    assert runtime_projection({"candidate_count": 1}, {"status": "NO_READ"})["status"] == "SEG_FAILED"
    assert runtime_projection({"candidate_count": 1, "lead_color": "black"},
                              {"status": "NO_READ"})["status"] == "SEG_FAILED"
    assert runtime_projection({"candidate_count": 0}, {"status": "NO_READ"})["status"] == "NO_PLATE"
