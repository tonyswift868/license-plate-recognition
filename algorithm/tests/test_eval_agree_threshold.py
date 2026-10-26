import random
import json
from types import SimpleNamespace

import lpr_tools.eval.agree_threshold as agree_threshold_module
from lpr import pipeline
from lpr.pipeline import _Attempt
from lpr.types import DecodeResult, RecogStatus
from lpr_tools.eval.agree_threshold import (arbitrate_pair, exact_branch_bound,
                                            replay_record, threshold_boundaries,
                                            _compile_records, _fast_counts)


def _pred(plate, confidence, status="OK", valid=True):
    return {"plate": plate, "confidence": confidence, "status": status, "valid": valid}


def test_agreement_precedes_solo_and_uses_minimum_confidence():
    segment = _pred("皖A12345", 0.91)
    reader = _pred("皖A12345", 0.82)
    accepted = arbitrate_pair(segment, reader, 0.82, 1.01, 1.01, "reader")
    assert accepted == {"plate": "皖A12345", "confidence": 0.91, "status": "OK"}
    rejected = arbitrate_pair(segment, reader, 0.83, 1.01, 1.01, "reader")
    assert rejected["status"] == "LOW_CONFIDENCE"


def test_solo_preference_falls_back_to_other_route():
    segment = _pred("皖A12345", 0.9)
    reader = _pred("皖B12345", 0.8)
    assert arbitrate_pair(segment, reader, 0.95, 0.85, 0.85, "reader")["plate"] == "皖A12345"
    assert arbitrate_pair(segment, reader, 0.95, 0.85, 0.75, "reader")["plate"] == "皖B12345"


def test_invalid_format_is_excluded_from_valid_arbitration():
    invalid = _pred("bad", 0.99, "INVALID_FORMAT", valid=False)
    reader = _pred("皖A12345", 0.7)
    result = arbitrate_pair(invalid, reader, 0.8, 1.01, 1.01, "segment")
    assert result["plate"] == "皖A12345"
    assert result["status"] == "LOW_CONFIDENCE"


def test_first_ok_rectification_wins_and_later_rect_is_not_used():
    record = {"candidate_count": 1, "lead_color": "blue", "rects": [
        {"segment": _pred("皖B12345", 0.9), "reader": _pred("皖B12345", 0.9)},
        {"segment": _pred("皖A12345", 0.9), "reader": _pred("皖A12345", 0.9)},
    ]}
    result = replay_record(record, 0.8, 1.01, 1.01, "reader")
    assert result["plate"] == "皖B12345"


def test_pair_replay_matches_runtime_arbitrator_over_threshold_grid():
    pp_base = SimpleNamespace(topk=3)
    values = [
        ("皖A12345", 0.9, RecogStatus.OK),
        ("皖B12345", 0.7, RecogStatus.LOW_CONFIDENCE),
        ("bad", 0.99, RecogStatus.INVALID_FORMAT),
    ]
    for (sp, sc, ss) in values:
        for (rp, rc, rs) in values:
            seg_dec = DecodeResult(ss, sp, len(sp), "", sc, [], [], "unknown")
            rd_dec = DecodeResult(rs, rp, len(rp), "", rc, [], [], "unknown")
            for agree in (0.0, 0.7, 0.9, 1.01):
                for solo_seg in (0.7, 0.9, 1.01):
                    for solo_reader in (0.7, 0.9, 1.01):
                        for prefer in ("segment", "reader"):
                            pp = SimpleNamespace(agree_min_confidence=agree,
                                solo_min_confidence_segment=solo_seg,
                                solo_min_confidence_reader=solo_reader,
                                disagree_prefer=prefer, topk=3)
                            actual = pipeline._arbitrate(_Attempt(seg_dec, None, None),
                                                         _Attempt(rd_dec, None, None), pp)
                            expected = arbitrate_pair(_pred(sp, sc, ss.value, ss != RecogStatus.INVALID_FORMAT),
                                                      _pred(rp, rc, rs.value, rs != RecogStatus.INVALID_FORMAT),
                                                      agree, solo_seg, solo_reader, prefer)
                            assert actual.dec.plate_no == expected["plate"]
                            assert actual.dec.status.value == expected["status"]
                            assert actual.dec.confidence == expected["confidence"]


def test_branch_bound_matches_bruteforce_on_tiny_records():
    rows = []
    for split, truth, seg, reader in [
        ("val", "皖A12345", _pred("皖A12345", .9), _pred("皖B12345", .8)),
        ("val", "皖A12345", _pred("皖B12345", .7), _pred("皖A12345", .9)),
        ("devhard", "皖A12345", _pred("皖A12345", .8), _pred("皖A12345", .9)),
        ("devhard", "皖A12345", _pred("皖B12345", .9), _pred("皖B12345", .9)),
    ]:
        rows.append({"split": split, "truth": truth, "lead_color": "blue",
                     "candidate_count": 1, "rects": [{"segment": seg, "reader": reader}]})
    best = exact_branch_bound(rows, 500)
    values = (sorted({min(s["confidence"], r["confidence"])
                      for row in rows for rect in row["rects"]
                      for s, r in [(rect["segment"], rect["reader"])]
                      if s["plate"] == r["plate"]} | {0.0, 1.0000000000000002}),
              threshold_boundaries(rows, "segment"), threshold_boundaries(rows, "reader"))
    brute = 0
    for prefer in ("segment", "reader"):
        for a in values[0]:
            for s in values[1]:
                for r in values[2]:
                    grouped = {"val": [x for x in rows if x["split"] == "val"],
                               "devhard": [x for x in rows if x["split"] == "devhard"]}
                    metrics = []
                    for items in grouped.values():
                        accepted = correct = 0
                        for item in items:
                            out = replay_record(item, a, s, r, prefer)
                            accepted += out["status"] == "OK"
                            correct += out["status"] == "OK" and out["plate"] == item["truth"]
                        metrics.append((accepted, correct))
                    if all(n and c * 1000 >= 500 * n for n, c in metrics):
                        brute = max(brute, sum(n for n, _ in metrics))
    assert best["search_complete"]
    assert best["best"]["total_auto_pass"] == brute


def test_branch_bound_matches_bruteforce_on_seeded_multirect_traces():
    for seed in (7, 19, 43):
        rng = random.Random(seed)
        rows = []
        for idx in range(8):
            truth = "皖A12345"
            rects = []
            for _ in range(rng.randint(2, 4)):
                def route():
                    if rng.random() < .18:
                        return None
                    confidence = rng.choice((.5, .7, .9))
                    plate = truth if rng.random() < .55 else "皖B12345"
                    valid = rng.random() >= .12
                    return _pred(plate if valid else "bad", confidence,
                                 "OK" if valid else "INVALID_FORMAT", valid)
                rects.append({"segment": route(), "reader": route()})
            rows.append({"split": "val" if idx < 4 else "devhard", "truth": truth,
                         "lead_color": "blue", "candidate_count": 2, "rects": rects})
        vector = _fast_counts(_compile_records(rows), {
            "agree_min": .7, "solo_seg": .7, "solo_reader": .9, "prefer": "reader"})
        replayed = {"val": {"accepted": 0, "correct": 0},
                    "devhard": {"accepted": 0, "correct": 0}}
        for row in rows:
            group = "val" if row["split"] == "val" else "devhard"
            out = replay_record(row, .7, .7, .9, "reader")
            replayed[group]["accepted"] += out["status"] == "OK"
            replayed[group]["correct"] += out["status"] == "OK" and out["plate"] == row["truth"]
        assert {k: vector[k] for k in replayed} == replayed
        av = sorted({min(s["confidence"], r["confidence"])
                     for row in rows for rect in row["rects"]
                     for s, r in [(rect["segment"], rect["reader"])]
                     if s and r and s["valid"] and r["valid"] and s["plate"] == r["plate"]}
                    | {0.0, 1.0000000000000002})
        sv = threshold_boundaries(rows, "segment")
        rv = threshold_boundaries(rows, "reader")
        for permille in (500, 990, 995):
            exact = exact_branch_bound(rows, permille)
            brute = 0
            for prefer in ("segment", "reader"):
                for a in av:
                    for s in sv:
                        for r in rv:
                            valid = True
                            total = 0
                            for split in ("val", "devhard"):
                                accepted = correct = 0
                                for item in rows:
                                    if ("val" if item["split"] == "val" else "devhard") != split:
                                        continue
                                    out = replay_record(item, a, s, r, prefer)
                                    accepted += out["status"] == "OK"
                                    correct += out["status"] == "OK" and out["plate"] == item["truth"]
                                valid &= accepted > 0 and correct * 1000 >= permille * accepted
                                total += accepted
                            if valid:
                                brute = max(brute, total)
            assert exact["search_complete"], (seed, permille)
            assert (exact["best"]["total_auto_pass"] if exact["best"] else 0) == brute, (seed, permille)


def test_branch_bound_checkpoint_resumes_same_frontier(tmp_path):
    rows = [
        {"split": "val", "truth": "皖A12345", "subset": "ccpd_base", "lead_color": "blue",
         "candidate_count": 2, "rects": [
             {"segment": _pred("皖B12345", .9), "reader": _pred("皖B12345", .9)},
             {"segment": _pred("皖A12345", .8), "reader": _pred("皖A12345", .8)}]},
        {"split": "devhard", "truth": "皖A12345", "subset": "ccpd_db", "lead_color": "blue",
         "candidate_count": 2, "rects": [
             {"segment": _pred("皖A12345", .8), "reader": _pred("皖B12345", .7)},
             {"segment": _pred("皖A12345", .7), "reader": _pred("皖A12345", .9)}]},
    ]
    checkpoint = tmp_path / "checkpoint.json"
    first = exact_branch_bound(rows, 500, max_nodes=1, checkpoint_path=checkpoint,
                               trace_sha256="tiny-trace")
    assert not first["search_complete"]
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["trace_sha256"] == "tiny-trace" and state["frontier"]
    resumed = exact_branch_bound(rows, 500, max_nodes=10000, checkpoint_path=checkpoint,
                                 trace_sha256="tiny-trace")
    fresh = exact_branch_bound(rows, 500, max_nodes=10000, trace_sha256="tiny-trace",
                               checkpoint_path=tmp_path / "fresh.json")
    assert resumed["search_complete"]
    assert resumed["best"] == fresh["best"]
    assert resumed["search_nodes"] == fresh["search_nodes"]


def test_periodic_checkpoint_keeps_unprocessed_node_and_resume_matches_fresh(tmp_path, monkeypatch):
    rows = [
        {"split": "val", "truth": "皖A12345", "subset": "ccpd_base", "lead_color": "blue",
         "candidate_count": 2, "rects": [
             {"segment": _pred("皖B12345", .9), "reader": _pred("皖B12345", .9)},
             {"segment": _pred("皖A12345", .8), "reader": _pred("皖A12345", .8)}]},
        {"split": "devhard", "truth": "皖A12345", "subset": "ccpd_db", "lead_color": "blue",
         "candidate_count": 2, "rects": [
             {"segment": _pred("皖A12345", .8), "reader": _pred("皖B12345", .7)},
             {"segment": _pred("皖A12345", .7), "reader": _pred("皖A12345", .9)}]},
    ]
    checkpoint = tmp_path / "periodic.json"
    original_bounds = agree_threshold_module.DecisionBits.bounds
    raised = False

    def fail_after_periodic_save(self, low, high):
        nonlocal raised
        if not raised:
            raised = True
            raise RuntimeError("simulated interruption after periodic checkpoint")
        return original_bounds(self, low, high)

    monkeypatch.setattr(agree_threshold_module.DecisionBits, "bounds", fail_after_periodic_save)
    try:
        exact_branch_bound(rows, 500, max_nodes=100, checkpoint_path=checkpoint,
                           trace_sha256="periodic-trace", checkpoint_interval=1)
    except RuntimeError as exc:
        assert "simulated interruption" in str(exc)
    else:
        raise AssertionError("expected simulated interruption")
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["nodes"] == 0
    assert len(state["frontier"]) == 2

    monkeypatch.setattr(agree_threshold_module.DecisionBits, "bounds", original_bounds)
    resumed = exact_branch_bound(rows, 500, max_nodes=10000, checkpoint_path=checkpoint,
                                 trace_sha256="periodic-trace", checkpoint_interval=1)
    fresh = exact_branch_bound(rows, 500, max_nodes=10000, trace_sha256="periodic-trace",
                               checkpoint_interval=1)
    assert resumed["search_complete"]
    assert resumed["best"] == fresh["best"]
    assert resumed["search_nodes"] == fresh["search_nodes"]
