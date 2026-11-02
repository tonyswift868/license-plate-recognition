"""Small real-model coverage for the character ablation report."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lpr.io_utils import imread, imwrite
from lpr.recognize import load_classifier, write_manifest_entry
from lpr.recognize.cnn_model import CharCNN
from lpr_tools.eval.ablation import (
    _training_seconds,
    evaluate,
    fill_plate_accuracy,
    load_chars,
    render_markdown,
)
from lpr_tools.train.build_templates import build
from lpr_tools.train.synth_chars import generate_dataset
from lpr_tools.train.train_svm import train


@pytest.fixture(scope="module")
def trained_models(default_font, tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation-models")
    data = root / "data"
    generate_dataset(data, per_class=8)
    template_path = root / "template.npz"
    svm_path = root / "svm.joblib"
    build(data, template_path, version="ablation-template")
    train(data, svm_path, version="ablation-svm", max_per_class=6)

    torch.manual_seed(123)
    cnn_path = root / "cnn.pt"
    torch.save(CharCNN().state_dict(), cnn_path)
    write_manifest_entry(cnn_path, version="ablation-cnn", train_data="synthetic test fixture",
                         val_char_acc=None)

    # Exercise the A7 sidecar convention with an explicit recorded value.
    cnn_path.with_suffix(".training.json").write_text(
        json.dumps({"train_seconds": 2.5}), encoding="utf-8")
    svm_path.with_suffix(".training.json").write_text(
        json.dumps({"train_seconds": 7.75}), encoding="utf-8")
    # Exercise CSV fallback for the template model.
    template_path.with_suffix(".training.json").unlink(missing_ok=True)
    template_path.with_suffix(".train.csv").write_text(
        "epoch,epoch_seconds\n1,1.25\n2,2.75\n", encoding="utf-8")
    return data, {"cnn": cnn_path, "template": template_path, "svm": svm_path}


def write_samples(source_root: Path, target: Path, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    images, truths = [], []
    for label in labels:
        files = sorted((source_root / "val" / str(label)).glob("*.png"))
        assert files, f"fixture lacks class {label}"
        image = imread(files[0], 0)
        out = target / str(label) / f"sample_{len(images)}.png"
        imwrite(out, image)
        assert out.is_file()
        images.append(image)
        truths.append(label)
    return np.stack(images), np.asarray(truths)


def test_ablation_real_classifiers_metrics_sparse_groups_and_report(trained_models, tmp_path):
    data, models = trained_models
    chars = tmp_path / "only-alnum"
    images, labels = write_samples(data, chars, [31, 32, 41])
    report = evaluate(chars, models)
    assert torch.get_num_threads() == 2
    assert report["sample_count"] == 3
    for metric in report["models"]:
        classifier = load_classifier(metric["kind"], models[metric["kind"]])
        expected = np.concatenate([classifier.predict_proba(image[None]) for image in images]).argmax(1)
        assert metric["overall_accuracy"] == pytest.approx(float(np.mean(expected == labels)))
        assert metric["overall_n"] == 3
        assert metric["province_accuracy"] is None and metric["province_n"] == 0
        assert metric["alnum_n"] == 3 and metric["alnum_accuracy"] is not None
        assert metric["mean_ms"] >= 0 and metric["model_bytes"] > 0
        if metric["kind"] == "cnn":
            assert metric["train_seconds"] == 2.5
        elif metric["kind"] == "template":
            assert metric["train_seconds"] == 4.0
        else:
            assert metric["train_seconds"] == 7.75
    markdown = render_markdown(report)
    assert "| 方法 | 总体准确率 | 省份字准确率 | 字母数字准确率 |" in markdown
    assert "N/A (n=0)" in markdown
    assert "## 整牌准确率" in markdown and "eval_accuracy" in markdown


def test_ablation_province_only_and_empty_input_errors(trained_models, tmp_path):
    data, models = trained_models
    chars = tmp_path / "province-only"
    write_samples(data, chars, [0, 1])
    metrics = evaluate(chars, models)["models"]
    assert all(m["province_n"] == 2 and m["province_accuracy"] is not None for m in metrics)
    assert all(m["alnum_n"] == 0 and m["alnum_accuracy"] is None for m in metrics)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no character"):
        load_chars(empty)


def test_load_chars_parallel_reader_keeps_lexical_path_order_and_validation(tmp_path):
    root = tmp_path / "ordered"
    imwrite(root / "2" / "b.png", np.full((32, 32), 22, dtype=np.uint8))
    imwrite(root / "10" / "a.png", np.full((32, 32), 10, dtype=np.uint8))
    imwrite(root / "metadata" / "ignored.png", np.zeros((32, 32), dtype=np.uint8))
    images, labels = load_chars(root)
    np.testing.assert_array_equal(labels, [10, 2])
    assert [int(image[0, 0]) for image in images] == [10, 22]

    invalid_shape = tmp_path / "bad-shape"
    imwrite(invalid_shape / "0" / "bad.png", np.zeros((31, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="32 x 32 uint8"):
        load_chars(invalid_shape)

    invalid_class = tmp_path / "bad-class"
    imwrite(invalid_class / "65" / "bad.png", np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="class index"):
        load_chars(invalid_class)


def test_training_time_metadata_precedence_and_missing(tmp_path):
    model = tmp_path / "old.pt"
    model.touch()
    model.with_suffix(".training.json").write_text('{"train_seconds": 8}', encoding="utf-8")
    assert _training_seconds(model, {"train_seconds": 3, "training_seconds": 4}) == 3
    assert _training_seconds(model, {}) == 8
    model.with_suffix(".training.json").unlink()
    model.with_suffix(".train.csv").write_text(
        "epoch,epoch_seconds\n1,1.5\n2,2.0\n", encoding="utf-8")
    assert _training_seconds(model, {}) == 3.5
    model.with_suffix(".train.csv").unlink()
    assert _training_seconds(model, {}) is None


def test_fill_plate_accuracy_integration_hook(trained_models, tmp_path):
    data, models = trained_models
    chars = tmp_path / "chars"
    write_samples(data, chars, [31])
    path = tmp_path / "ablation.md"
    path.write_text(render_markdown(evaluate(chars, models)), encoding="utf-8")
    fill_plate_accuracy(path, {"clear": 0.95, "night": 0.8})
    content = path.read_text(encoding="utf-8")
    assert "clear: 95.00%" in content and "night: 80.00%" in content
    missing_section = tmp_path / "missing.md"
    missing_section.write_text("# report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="section"):
        fill_plate_accuracy(missing_section, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_fill_plate_accuracy_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="plate accuracy"):
        fill_plate_accuracy(tmp_path / "unused.md", {"clear": value})
