"""Small tensor-contract tests for province model calibration and export."""
from __future__ import annotations

import pytest
import torch
import hashlib
import json

from lpr_tools.train.calibrate_province_readers import (
    _char_province_only, _temperature_export, _unwrap_model,
    _validate_training_history,
)
import lpr_tools.train.calibrate_province_readers as calibration


def test_unwraps_only_matching_province_training_checkpoint():
    state = {"w": torch.ones(2)}
    payload = {"format": "province-reader-training-v1", "kind": "crnn",
               "model_state": state, "epoch": 15}
    actual, wrapper = _unwrap_model(payload, "crnn")
    assert actual is state
    assert wrapper["epoch"] == 15
    with pytest.raises(ValueError, match="kind mismatch"):
        _unwrap_model(payload, "box")


def test_temperature_exports_change_metadata_only_and_add_scalar():
    box = {"architecture": "mobilenet_v3_small_stem_stn_gn_bigru160_ctc_v2",
           "state_dict": {"weight": torch.arange(4)}, "temperature": 1.0,
           "box_pad": [0.08, 0.12]}
    box_out = _temperature_export("box", box, 1.25)
    assert box_out["temperature"] == 1.25
    assert box_out["state_dict"] is box["state_dict"]
    assert box["temperature"] == 1.0

    crnn = {"weight": torch.arange(4)}
    crnn_out = _temperature_export("crnn", crnn, 0.75)
    assert torch.equal(crnn_out["weight"], crnn["weight"])
    assert crnn_out["ctc_temperature"].shape == torch.Size([])
    assert crnn_out["ctc_temperature"].item() == 0.75
    assert "ctc_temperature" not in crnn


def test_char_validation_allows_only_province_output_rows_to_change():
    old = {"features.weight": torch.arange(3),
           "classifier.5.weight": torch.zeros(65, 4),
           "classifier.5.bias": torch.zeros(65)}
    new = {key: value.clone() for key, value in old.items()}
    new["classifier.5.weight"][0] = 1
    proof = _char_province_only(old, new)
    assert proof["all_non_head_tensors_identical"]
    assert proof["non_province_classifier_rows_identical"]
    assert proof["province_rows_with_changes"] == 1

    both_changed = {key: value.clone() for key, value in old.items()}
    both_changed["classifier.5.weight"][:31] = 1
    both_changed["classifier.5.bias"][:31] = 1
    assert _char_province_only(old, both_changed)["province_rows_with_changes"] == 31

    bad_row = {key: value.clone() for key, value in new.items()}
    bad_row["classifier.5.bias"][31] = 1
    with pytest.raises(ValueError, match="non-province classifier bias"):
        _char_province_only(old, bad_row)

    bad_feature = {key: value.clone() for key, value in new.items()}
    bad_feature["features.weight"][0] = 1
    with pytest.raises(ValueError, match="non-head tensor"):
        _char_province_only(old, bad_feature)


def test_history_proof_checks_full_epochs_provenance_and_stable_best(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "EXPECTED_TRAIN_ROWS", 2)
    data = tmp_path / "data"
    data.mkdir()
    index = data / "combined_train.csv"
    index.write_text("path,label\na,皖A12345\nb,陕A12345\n", encoding="utf-8")
    init = tmp_path / "init.pt"
    sampler = tmp_path / "sampler.json"
    tune = tmp_path / "tune.csv"
    audit_path = data / "province_audit.json"
    independent = data / "independent_audit.json"
    for path in (init, sampler, tune):
        path.write_bytes(path.name.encode())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {"schema": "province-coverage-data-v1", "outputs": {
        index.name: {"sha256": digest(index)},
    }, "source_hashes": {"batch11_tune_sha256": digest(tune)}}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    independent.write_text(json.dumps({"passed": True,
                                       "combined_train": {"by_province": {"皖": 1, "陕": 1}}}),
                            encoding="utf-8")
    provenance = {"init": str(init), "init_sha256": digest(init),
                  "index_sha256": digest(index), "audit_sha256": digest(audit_path),
                  "sampler_sha256": digest(sampler), "sampler": str(sampler)}
    history = [{"epoch": i, "real_unique": 2, "real_physical": 2,
                "synthetic_physical": 2, "synthetic_provinces": {"陕": 2},
                "selection": {"selection_score": 0.8 if i in (3, 5) else 0.7}}
               for i in range(1, 21)]
    report = {"kind": "box", "epochs_completed": 20, "best_epoch": 3, "best_score": 0.8,
              "history": history, "provenance": provenance}
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(report), encoding="utf-8")
    wrapper = {"kind": "box", "epoch": 3, "cursor": 0, "best_epoch": 3, "best_score": 0.8,
               "history": history[:3], "provenance": provenance}
    proof = _validate_training_history("box", wrapper, data, history_path, tune)
    assert proof["best_epoch"] == 3
    assert proof["first_maximum_tie_policy"] == "earliest_epoch"
    bad_wrapper = {**wrapper, "cursor": 1}
    with pytest.raises(ValueError, match="cursor"):
        _validate_training_history("box", bad_wrapper, data, history_path, tune)


def test_history_proof_accepts_real_batch12_metadata_and_checkpoint_contract(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    data = root / "algorithm" / "out" / "data" / "batch12"
    tune = root / "algorithm" / "out" / "data" / "batch11" / "tune.csv"
    index = data / "combined_train.csv"
    audit = data / "province_audit.json"
    sampler = root / "algorithm" / "out" / "data" / "batch11" / "sampler.json"
    init = root / "algorithm" / "models" / "box_reader_v2.pt"
    augmentation_source = root / "algorithm" / "lpr_tools" / "train" / "province_synth.py"
    font = Path(r"C:\Windows\Fonts\simhei.ttf")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {"init": str(init.resolve()), "init_sha256": sha(init),
                  "index_sha256": sha(index), "audit_sha256": sha(audit),
                  "sampler_sha256": sha(sampler),
                  "augmentation_source_sha256": sha(augmentation_source),
                  "font_sha256": sha(font)}
    history = [{"epoch": epoch, "real_unique": 210564, "real_physical": 210564,
                "synthetic_physical": 0, "synthetic_provinces": {},
                "selection": {"selection_score": 0.91 if epoch == 1 else 0.9}}
               for epoch in range(1, 21)]
    report = {"kind": "box", "epochs_completed": 20, "best_epoch": 1,
              "best_score": 0.91, "history": history, "provenance": provenance}
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(report), encoding="utf-8")
    wrapper = {"kind": "box", "epoch": 1, "cursor": 0, "best_epoch": 1,
               "best_score": 0.91, "history": history[:1], "provenance": provenance}
    proof = _validate_training_history("box", wrapper, data, history_path, tune)
    assert proof["physical_counts_by_province_match_audit"] is True
    assert sum(proof["physical_counts_by_province"].values()) == 210564
