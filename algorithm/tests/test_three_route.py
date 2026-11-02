from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import lpr_tools.eval.three_route as three_route
import pytest
from lpr_tools.eval.three_route import (
    _boundary_values,
    _fallback,
    _load_custom_factory,
    _model_manifest_identity,
    _normalize_verification_meta,
    _read_box_result,
    evaluate,
    exact_search,
    join_routes,
    reference_decision,
    replay_auto,
    select_box_pad,
    validate_records,
)


def _attempt(plate, conf=0.9, status="OK"):
    return {"plate": plate, "confidence": conf, "status": status,
            "valid": status == "OK"}


def _record(split, truth, rects, boxes, **extra):
    return {"path": f"/{split}/{truth}.jpg", "split": split, "truth": truth,
            "subset": "tiny", "category": "clear", "lead_color": "blue",
            "rects": rects, "boxes": boxes, **extra}


def _rect(ci, ri, segment=None, reader=None):
    return {"candidate_index": ci, "rect_index": ri,
            "segment": segment, "reader": reader}


def test_agreement_then_box_solo_preserves_candidate_rect_first_ok():
    row = _record("val", "A", [
        _rect(0, 0, _attempt("wrong", 0.99), _attempt("reader", 0.98)),
        _rect(0, 1, _attempt("A", 0.80), _attempt("A", 0.80)),
        _rect(1, 0, _attempt("A", 0.99), _attempt("A", 0.99)),
    ], {"0": _attempt("A", 0.91), "1": _attempt("A", 1.0)})

    # A high box threshold exposes the later same-rect agreement. The later
    # candidate cannot jump ahead of the first successful rect.
    out = replay_auto(row, tau_agree=0.8, tau_box=1.01)
    assert out["plate"] == "A"
    assert out["candidate_index"] == 0
    assert out["rect_index"] == 1

    # At a permissive B threshold, B is tested after that rect's pair checks.
    out = replay_auto(row, tau_agree=0.95, tau_box=0.91)
    assert out["plate"] == "A"
    assert out["selected_route"] == "box"
    assert out["candidate_index"] == 0
    assert out["rect_index"] is None


def test_invalid_routes_never_agree_or_box_accept():
    row = _record("val", "A", [_rect(0, 0,
                        _attempt("A", 0.99, "INVALID_FORMAT"), _attempt("A", 0.99))],
                  {"0": _attempt("A", 0.99, "INVALID_FORMAT")})
    assert replay_auto(row, 0.0, 0.0) is None


def test_box_only_candidate_without_rectification_can_pass_solo():
    row = _record("val", "A", [], {"0": _attempt("A", 0.96)}, candidate_count=1)
    out = replay_auto(row, tau_agree=0.0, tau_box=0.95)
    assert out["selected_route"] == "box"
    assert out["candidate_index"] == 0
    assert out["rect_index"] is None


def test_independent_reference_decision_returns_full_selected_result():
    row = _record("val", "A", [_rect(0, 0, _attempt("X", .99), _attempt("Y", .98)),
                                _rect(0, 1, _attempt("A", .85), _attempt("A", .84)),
                                _rect(1, 0, _attempt("A", .99), _attempt("A", .99))],
                  {"0": _attempt("A", .91), "1": _attempt("A", 1.0)}, candidate_count=2)
    decision = reference_decision(row, tau_agree=.8, tau_box=.91)
    assert decision == {"status": "OK", "plate": "A", "confidence": .85,
                        "route": "segment", "candidate_index": 0, "rect_index": 1}


def test_fallback_policies_are_deterministic_and_not_auto_pass():
    row = _record("val", "A", [_rect(0, 0, _attempt("S", 0.7), _attempt("R", 0.8))],
                  {"0": _attempt("B", 0.95)})
    assert _fallback(row, "box")["plate"] == "B"
    assert _fallback(row, "crnn")["plate"] == "R"
    assert _fallback(row, "agreement_then_reader")["plate"] == "R"
    assert _fallback(row, "majority")["plate"] == "B"
    assert replay_auto(row, 1.0, 1.0) is None


def test_fallback_replay_matches_independent_reference_fields():
    row = _record("val", "A", [_rect(0, 0, _attempt("A", .7), _attempt("A", .9))],
                  {"0": _attempt("A", .8)}, candidate_count=1)
    for policy in ("agreement_then_reader", "majority", "box", "crnn"):
        replayed = _fallback(row, policy)
        reference = reference_decision(row, tau_agree=2.0, tau_box=2.0, fallback=policy)
        assert {key: replayed[key] for key in
                ("status", "plate", "confidence", "route", "candidate_index", "rect_index")} == reference
    assert _fallback(row, "majority")["route"] == "segment"
    assert _fallback(row, "majority")["confidence"] == .7
    assert _fallback(row, "agreement_then_reader")["route"] == "reader"
    assert _fallback(row, "agreement_then_reader")["confidence"] == .7


def test_fallback_reference_agrees_on_pair_priority_and_pair_confidence():
    # All routes agree, but their confidences differ. Fixed pair priority is
    # S/R first, and a pair fallback reports the weaker route confidence.
    row = _record("devhard", "A", [_rect(2, 4, _attempt("A", .93), _attempt("A", .81))],
                  {"2": _attempt("A", .99)}, candidate_count=3)
    for policy in ("majority", "agreement_then_reader"):
        actual = _fallback(row, policy)
        reference = reference_decision(row, tau_agree=2.0, tau_box=2.0,
                                       fallback=policy)
        assert {key: actual[key] for key in
                ("status", "plate", "confidence", "route", "candidate_index", "rect_index")} == reference
    majority = _fallback(row, "majority")
    assert majority["route"] == "segment"
    assert majority["confidence"] == .81
    agreement = _fallback(row, "agreement_then_reader")
    assert agreement["route"] == "reader"
    assert agreement["confidence"] == .81


def test_exact_search_matches_bruteforce_on_multiple_rect_and_split_fixture():
    rows = [
        # An early candidate's wrong B can steal the next candidate's pair.
        _record("val", "A", [_rect(0, 0, _attempt("X", .97), _attempt("Y", .92)),
                              _rect(1, 0, _attempt("A", .85), _attempt("A", .84))],
                {"0": _attempt("Z", .93), "1": _attempt("A", .92)}),
        _record("val", "B", [_rect(0, 0, _attempt("B", .90), _attempt("B", .91))],
                {"0": _attempt("C", .99)}),
        _record("devhard", "C", [_rect(0, 0, _attempt("C", .88), _attempt("D", .98)),
                                  _rect(1, 0, _attempt("C", .95), _attempt("C", .94))],
                {"0": _attempt("C", .90), "1": _attempt("D", .96)}),
        _record("devhard", "D", [_rect(0, 0, _attempt("D", .92), _attempt("D", .91))],
                {"0": _attempt("D", .90)}),
    ]
    scan = exact_search(rows)
    assert scan["complete"] is True
    agree_values = _boundary_values(rows, "agree")
    box_values = _boundary_values(rows, "box")
    feasible = []
    for ta in agree_values:
        for tb in box_values:
            metrics = evaluate(rows, ta, tb)
            if all(metrics[s]["auto_pass"] > 0 and
                   metrics[s]["auto_pass_correct"] * 1000 >=
                   995 * metrics[s]["auto_pass"] for s in ("val", "devhard")):
                feasible.append((sum(metrics[s]["auto_pass"] for s in ("val", "devhard")), ta, tb))
    assert scan["best"]["accepted"] == max(row[0] for row in feasible)
    chosen = scan["best"]
    chosen_metrics = evaluate(rows, chosen["tau_agree"], chosen["tau_box"])
    assert chosen_metrics["val"]["auto_pass"] == chosen["threshold_counts"]["val"]["accepted"]
    assert chosen_metrics["devhard"]["auto_pass"] == chosen["threshold_counts"]["devhard"]["accepted"]


def test_subset_metrics_keep_dark_tilt_and_green_groups_separate():
    dark_green = _record("devhard", "A", [_rect(0, 0, _attempt("A"), _attempt("A"))],
                         {"0": _attempt("A")}, subset="ccpd_db", brightness=19,
                         plate_color="green")
    tilted = _record("devhard", "B", [_rect(0, 0, _attempt("B"), _attempt("B"))],
                      {"0": _attempt("B")}, subset="ccpd_tilt")
    metrics = evaluate([dark_green, tilted], 0.8, 0.8)
    assert metrics["devhard/ccpd_db_brightness_le_19"]["n"] == 1
    assert metrics["devhard/ccpd_tilt_plus_rotate"]["n"] == 1
    assert metrics["devhard/green"]["n"] == 1


def test_exact_search_keeps_a_99_5_percent_boundary_solution():
    rows = []
    for split in ("val", "devhard"):
        for index in range(200):
            truth = f"P{index}"
            predicted = "WRONG" if index == 0 else truth
            conf = 1.0 if index == 0 else 0.95
            rows.append(_record(split, truth,
                                [_rect(0, 0, _attempt(predicted, conf), _attempt(predicted, conf))],
                                {"0": _attempt("OTHER", 0.7)}))
    scan = exact_search(rows)
    assert scan["complete"] is True
    assert scan["best"]["accepted"] == 400
    for split in ("val", "devhard"):
        counts = scan["best"]["threshold_counts"][split]
        assert counts == {"accepted": 200, "correct": 199}


def test_box_join_converts_one_based_indices_and_checks_candidate_source():
    paired = [{"path": "D:/data/a.jpg", "truth": "A", "split": "val", "subset": "tiny",
               "category": "clear", "candidate_count": 2,
               "rects": [{**_rect(0, 0, _attempt("A"), _attempt("A")),
                          "candidate_source": "net"},
                         {**_rect(1, 0, _attempt("B"), _attempt("B")),
                          "candidate_source": "edge"}]}]
    raw_box = {"path": "D:/data/a.jpg", "truth": "A", "split": "val",
               "pads": [{"pad_order": 3, "pad_x": .1, "pad_y": .2,
        "candidates": [
            {"candidate_index": 1, "source": "net", "prediction": "A", "min_confidence": .8,
             "status": "OK", "legal": True},
            {"candidate_index": 2, "source": "edge", "prediction": "B", "min_confidence": .9,
             "status": "OK", "legal": True}]}]}
    selected = select_box_pad(raw_box, 3)
    joined = join_routes(paired, [selected])
    assert joined[0]["boxes"]["0"]["plate"] == "A"
    assert joined[0]["boxes"]["1"]["plate"] == "B"


def test_box_join_rejects_wrong_candidate_source():
    paired = [{"path": "D:/data/a.jpg", "truth": "A", "split": "val", "subset": "tiny",
               "category": "clear", "candidate_count": 1,
               "rects": [{**_rect(0, 0), "candidate_source": "net"}]}]
    selected = {"path": "D:/data/a.jpg", "truth": "A", "split": "val",
                "box_pad": {"pad_order": 3}, "candidates": [
        {"candidate_index": 0, "source": "color", "plate": "A", "confidence": .9,
         "status": "OK", "valid": True}]}
    try:
        join_routes(paired, [selected])
    except ValueError as exc:
        assert "source mismatch" in str(exc)
    else:
        raise AssertionError("candidate source mismatch must fail closed")


def test_join_uses_fixed_devhard_membership_independent_of_index_split():
    paired = [{"path": "D:/data/hard-val-index.jpg", "truth": "A", "split": "test",
               "subset": "ccpd_base", "category": "clear", "candidate_count": 1,
               "rects": [{**_rect(0, 0, _attempt("A"), _attempt("A")),
                          "candidate_source": "net"}]}]
    box = {"path": "D:/data/hard-val-index.jpg", "truth": "A", "split": "devhard",
           "index_split": "val", "box_pad": {"pad_order": 1},
           "candidates": [{"candidate_index": 0, "source": "net", "plate": "A",
                           "confidence": .9, "status": "OK", "valid": True}]}
    joined = join_routes(paired, [box])
    assert joined[0]["split"] == "devhard"
    assert joined[0]["index_split"] == "val"


def test_trace_validation_rejects_nonfinite_or_out_of_range_confidence():
    row = _record("val", "A", [_rect(0, 0, _attempt("A", float("nan")), None)],
                  {"0": _attempt("A")}, candidate_count=1)
    try:
        validate_records([row])
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("non-finite confidence must fail closed")

    row["rects"][0]["segment"]["confidence"] = 1.01
    try:
        validate_records([row])
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("confidence above one must fail closed")


def test_custom_factory_path_contract_and_manifest_sha(tmp_path, monkeypatch):
    model = tmp_path / "future_reader.bin"
    model.write_bytes(b"fixture-weights")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({model.name: {"sha256": digest, "version": "fixture"}}),
        encoding="utf-8")
    assert _model_manifest_identity(model)["sha256"] == digest

    received = []

    class Adapter:
        def read(self, crop):
            received.append(crop)
            return {"plate": "A", "confidence": .9, "status": "OK", "valid": True}

    module = ModuleType("test_three_route_factory")

    def factory(model_path):
        assert isinstance(model_path, Path)
        assert model_path == model
        return Adapter()

    module.make_adapter = factory
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    adapter = _load_custom_factory(f"{module.__name__}:make_adapter", model, "read")

    class NativeCrop:
        shape = (73, 211, 3)
        size = 73 * 211 * 3

    crop = NativeCrop()
    result = _read_box_result(adapter, crop, "blue", .8, False, None)
    assert received == [crop]
    assert received[0].shape == (73, 211, 3)
    assert result["plate"] == "A"


def test_verification_metadata_normalizes_joined_and_live_schemas_with_independent_box():
    cnn, detector, reader, box = ({"sha256": value} for value in ("c", "d", "r", "b"))
    effective = {"runtime": {"max_candidates": 3}, "locate": {"x": 1}}
    joined = _normalize_verification_meta({
        "schema": "three-route-v1",
        "models": {"model_path": cnn, "corner_model_path": detector,
                   "plate_reader_model_path": reader},
        "box_models": {"box_reader": box},
        "box_best_pad": {"pad_x": .1, "pad_y": .2},
        "effective_config": effective,
    })
    live = _normalize_verification_meta({
        "schema": "three-route-live-v1",
        "models": {"cnn": cnn, "detector": detector, "reader": reader, "box": box},
        "box_pad": {"pad_x": .1, "pad_y": .2},
        "config": {"effective_config": effective},
    })
    assert live == joined
    assert live["box_models"]["box_reader"]["sha256"] == "b"
    assert live["models"]["plate_reader_model_path"]["sha256"] == "r"


def test_live_cli_writes_complete_result_to_out(tmp_path, monkeypatch):
    output = tmp_path / "live-trace.jsonl.gz"
    report = tmp_path / "cli-report.json"
    fake = {"status": "complete", "rows": 3100, "verification_pass": True,
            "fresh_trace": str(output), "search": {"complete": True, "best": {}}}
    monkeypatch.setattr(three_route, "run_live", lambda **kwargs: fake)
    code = three_route.main([
        "--live-verify", str(tmp_path / "joined.jsonl.gz"),
        "--policy-report", str(tmp_path / "policy.json"), "--config", "config.toml",
        "--live-output", str(output), "--out", str(report),
    ])
    assert code == 0
    assert json.loads(report.read_text(encoding="utf-8")) == fake


@pytest.mark.parametrize("collision", ["out", "live_trace"])
def test_live_cli_rejects_existing_outputs_before_running_models(tmp_path, monkeypatch, collision):
    output = tmp_path / "live-trace.jsonl.gz"
    report = tmp_path / "cli-report.json"
    collided = report if collision == "out" else output
    collided.write_text("existing", encoding="utf-8")

    def unexpected_run_live(**kwargs):
        raise AssertionError("run_live must not start when an output already exists")

    monkeypatch.setattr(three_route, "run_live", unexpected_run_live)
    with pytest.raises(FileExistsError):
        three_route.main([
            "--live-verify", str(tmp_path / "joined.jsonl.gz"),
            "--policy-report", str(tmp_path / "policy.json"), "--config", "config.toml",
            "--live-output", str(output), "--out", str(report),
        ])
