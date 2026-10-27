import json

import pytest

from lpr_tools.eval import three_route_workflow as workflow


def test_fresh_workflow_has_serial_phases_and_freezes_fallback_on_baseline(tmp_path, monkeypatch):
    calls = []
    def fresh(output, options):
        calls.append(("fresh", options["box_model"], options["detector_loader"]))
        return output, output.with_suffix(".report.json")
    def auto(trace, base_report, target, split_path=None):
        calls.append(("auto",))
        assert split_path and split_path.exists()
        return {"selected_policy": "011", "selected_total_accepted": 9,
                "policies": {"000": {"val": {"accepted": 3}, "devhard": {"accepted": 4}}}}
    def features(trace, target, options):
        calls.append(("features",))
        return target
    def split(trace, feature, target):
        calls.append(("split",))
        target.write_text("{}", encoding="utf-8")
        return target
    def refine(trace, feature, base, final, directory):
        calls.append(("refine",))
        baseline = json.loads(base.read_text(encoding="utf-8"))
        assert baseline["selected_policy"] == "000"
        assert baseline["selected_total_accepted"] == 7
        return {"baseline_rule": "G8_L", "final_rule": "G8_L"}
    monkeypatch.setattr(workflow, "_fresh_trace", fresh)
    monkeypatch.setattr(workflow, "_auto_review", auto)
    monkeypatch.setattr(workflow, "_features", features)
    monkeypatch.setattr(workflow, "_split", split)
    monkeypatch.setattr(workflow, "_refine", refine)
    result = workflow.run_workflow(tmp_path / "new", fresh=True, pad_x=.08, pad_y=.12,
                                   box_model="v2.pt", detector_loader="detector:factory")
    assert [call[0] for call in calls] == ["fresh", "features", "split", "auto", "refine"]
    assert result["final_rule"] == "G8_L"
    assert calls[0][1:] == ("v2.pt", "detector:factory")


def test_auto_only_skips_feature_model_and_rejects_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_auto_review", lambda *args: {
        "selected_policy": "000", "selected_total_accepted": 4})
    def forbidden(*args):
        raise AssertionError("auto-only replay must not start feature/model work")
    monkeypatch.setattr(workflow, "_features", forbidden)
    target = tmp_path / "new"
    assert workflow.run_workflow(target, mode="auto", trace="t.gz", base_report="r.json")["mode"] == "auto"
    with pytest.raises(FileExistsError):
        workflow.run_workflow(target, mode="auto", trace="t.gz", base_report="r.json")


def test_refine_helper_exports_the_frozen_application_with_runtime_projection(tmp_path, monkeypatch):
    from lpr_tools.eval import fallback_refine, three_route_expected

    calls = []
    def fit(*args, **kwargs):
        calls.append("fit")
        return {"selection_failed": False, "selected_rule": "D0_M"}
    def frozen(trace, features, policy, baseline, directory):
        calls.append("frozen")
        assert baseline == tmp_path / "fallback-refine.json"
        return {"selected_rule": "D0_M", "decisions_file": "frozen-decisions.gz"}
    def export(trace, decisions, target):
        calls.append("export")
        assert decisions == "frozen-decisions.gz"
        assert target == tmp_path / "expected-runtime.jsonl.gz"
        return {"rows": 3100}
    monkeypatch.setattr(fallback_refine, "run_refine", fit)
    monkeypatch.setattr(fallback_refine, "apply_frozen", frozen)
    monkeypatch.setattr(three_route_expected, "export_expected", export)
    result = workflow._refine("trace.gz", "features.gz", "baseline.json", "final.json", tmp_path)
    assert calls == ["fit", "frozen", "export"]
    assert result["expected_runtime"]["rows"] == 3100
