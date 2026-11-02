"""契约测试（架构师）：字符表、字符图规格、数据结构、错误码、配置键、阶段签名、坏输入不抛异常。

执行者不得修改本文件；这里失败说明契约被破坏。
"""

from __future__ import annotations

import inspect
import json

import lpr
import numpy as np
import pytest
from lpr import charset
from lpr.charnorm import CHAR_SIZE, normalize_char, unify_polarity
from lpr.config import load_config
from lpr.errors import DEFAULT_MESSAGES, ERROR_CODES, ConfigError, ModelLoadError, failure_result
from lpr.io_utils import decode_image, encode_jpg, imread, imwrite
from lpr.pipeline import PlateRecognizer
from lpr.recognize import load_classifier, write_manifest_entry
from lpr.testing import FakeRecognizer
from lpr.types import RecognitionResult, RecogStatus

S = RecogStatus

# ---- 字符表 ----


def test_charset_frozen():
    assert charset.NUM_CLASSES == 65
    assert len(set(charset.CHARSET)) == 65
    assert (len(charset.PROVINCES), len(charset.DIGITS), len(charset.LETTERS)) == (31, 10, 24)
    assert "I" not in charset.LETTERS and "O" not in charset.LETTERS
    assert charset.CHARSET[0] == "皖" and charset.CHARSET[30] == "新"
    assert charset.CHARSET_SHA == "ea8a61c65117"


def test_charset_groups():
    assert [charset.CHARSET[i] for i in charset.PROVINCE_IDX] == charset.PROVINCES
    assert [charset.CHARSET[i] for i in charset.DIGIT_IDX] == charset.DIGITS
    assert [charset.CHARSET[i] for i in charset.LETTER_IDX] == charset.LETTERS
    assert charset.index_of("陕") == 26


# ---- 字符图规格 ----


def test_normalize_char_spec():
    crop = np.full((60, 20), 30, np.uint8)
    crop[:, 5:15] = 200                                   # 窄高字符，类似“1”，上下贴满裁剪框
    out = normalize_char(crop)
    assert out.shape == (CHAR_SIZE, CHAR_SIZE) and out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0
    cols = np.where(out.max(axis=0) > 0)[0]
    rows = np.where(out.max(axis=1) > 0)[0]
    assert rows.max() - rows.min() + 1 == 28              # 长边缩到 28
    assert cols.max() - cols.min() + 1 < 28               # 保持宽高比
    assert out[0].max() == 0 and out[-1].max() == 0       # 上下是黑边


def test_normalize_char_accepts_bgr_and_constant():
    bgr = np.full((40, 30, 3), 77, np.uint8)
    out = normalize_char(bgr)
    assert out.shape == (32, 32) and out.max() == 0


def test_normalize_char_rejects_empty():
    with pytest.raises(ValueError):
        normalize_char(np.zeros((0, 10), np.uint8))


def test_unify_polarity():
    g = np.array([[10, 200]], np.uint8)
    assert (unify_polarity(g, "blue") == g).all()
    assert (unify_polarity(g, "yellow") == 255 - g).all()
    assert (unify_polarity(g, "green") == 255 - g).all()


# ---- 数据结构与错误码 ----


def test_error_codes_table():
    assert set(ERROR_CODES) == set(S) and set(DEFAULT_MESSAGES) == set(S)
    assert ERROR_CODES == {S.OK: 0, S.NO_PLATE: 3001, S.LOW_CONFIDENCE: 3002, S.SEG_FAILED: 3003,
                           S.INVALID_FORMAT: 3004, S.BAD_INPUT: 3005, S.UNSUPPORTED: 3006,
                           S.INTERNAL_ERROR: 3099}


def _full_result() -> RecognitionResult:
    return RecognitionResult(
        status=S.OK, error_code=0, message="识别成功", plate_no="陕A12345", plate_color="blue", layout=7,
        confidence=np.float32(0.91), char_confidences=[np.float32(0.95)] * 7,
        candidates=[("陕A12345", np.float64(0.91))], bbox=(1, 2, 3, 4),
        corners=[(np.float32(1), 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
        plate_image=np.zeros((80, 252, 3), np.uint8), elapsed_ms=12.3, stage_ms={"locate": np.float64(5.0)},
        debug={"x": np.zeros((2, 2), np.uint8)})


def test_to_dict_is_json_serializable():
    d = _full_result().to_dict()
    text = json.dumps(d, ensure_ascii=False)
    assert "陕A12345" in text
    assert "plate_image" not in d and "debug" not in d
    assert d["status"] == "OK" and d["bbox"] == [1, 2, 3, 4]
    json.dumps(failure_result(S.NO_PLATE).to_dict())


def test_public_exports():
    for name in ("PlateRecognizer", "RecognitionResult", "RecogStatus", "DuplicateFilter", "PlateVoteSession"):
        assert hasattr(lpr, name)


# ---- 配置 ----

FROZEN_KEYS = {
    "runtime": {"torch_threads", "model_path", "classifier", "max_candidates",
                "corner_model_path", "plate_reader_model_path", "box_reader_model_path", "detector_input_size"},
    "preprocess": {"min_side", "work_width", "clahe_clip", "clahe_grid", "blur_ksize"},
    "locate": {"aspect_min", "aspect_max", "area_ratio_min", "area_ratio_max", "max_tilt_deg",
               "close_kernel_rel", "min_row_transitions", "nms_iou", "blue_hsv", "yellow_hsv", "green_hsv"},
    "rectify": {"size_7", "size_8", "expand_ratio", "deskew_max_deg"},
    "segment": {"binarize", "layout7_centers", "layout7_char_w", "layout8_centers", "layout8_char_w", "char_size"},
    "postprocess": {"min_confidence", "topk", "province_prior", "arbitration", "agree_min_confidence",
                    "solo_min_confidence_segment", "solo_min_confidence_reader", "disagree_prefer",
                    "nonok_fallback", "three_route_agree_min_confidence",
                    "three_route_box_solo_min_confidence", "three_route_legacy_sr_agreement",
                    "three_route_segment_solo", "three_route_reader_solo",
                    "three_route_province_support_min_routes", "fallback_predicted_tilt_min_deg"},
    "voting": {"min_frames", "agree_frames", "window_s"},
    "dedup": {"window_s", "dhash_max_hamming"},
}


def test_config_keys_frozen(cfg):
    for section, keys in FROZEN_KEYS.items():
        assert set(type(getattr(cfg, section)).model_fields) == keys, section
    assert cfg.rectify.size_7 == (252, 80) and cfg.rectify.size_8 == (276, 80)
    assert cfg.segment.char_size == 32


def test_config_overrides_and_errors(tmp_path):
    cfg = load_config(runtime__classifier="svm", postprocess={"min_confidence": 0.9})
    assert cfg.runtime.classifier == "svm" and cfg.postprocess.min_confidence == 0.9
    with pytest.raises(ConfigError):
        load_config(runtime__no_such_key=1)
    with pytest.raises(ConfigError):
        load_config(rectify__size_7=[300, 80])
    partial = tmp_path / "my.toml"
    partial.write_text('[lanes]\n"1" = [10, 20, 300, 200]\n', encoding="utf-8")
    cfg = load_config(partial)
    assert cfg.lane_roi(1) == (10, 20, 300, 200) and cfg.lane_roi(2) is None
    assert cfg.preprocess.work_width == 800                # 其余键沿用 default.toml


def test_config_env_var(tmp_path, monkeypatch):
    p = tmp_path / "env.toml"
    p.write_text("[runtime]\nmax_candidates = 2\n", encoding="utf-8")
    monkeypatch.setenv("LPR_CONFIG", str(p))
    assert load_config().runtime.max_candidates == 2


# ---- 阶段函数签名（执行者不得改动） ----

STAGE_SIGNATURES = {
    "lpr.preprocess.preprocess": ["image", "cfg", "roi"],
    "lpr.locate.locate": ["pre", "cfg", "max_candidates"],
    "lpr.rectify.rectify": ["image", "cand", "cfg"],
    "lpr.segment.segment": ["rect", "layout", "cfg"],
    "lpr.postprocess.decode": ["probs", "layout", "color", "cfg"],
    "lpr.voting.vote": ["results", "min_frames", "min_confidence"],
    "lpr.voting.PlateVoteSession.__init__": ["self", "agree_frames", "window_s", "min_frames", "min_confidence"],
    "lpr.voting.PlateVoteSession.feed": ["self", "result", "ts"],
    "lpr.dedup.DuplicateFilter.__init__": ["self", "window_s", "dhash_max_hamming"],
    "lpr.dedup.DuplicateFilter.check": ["self", "lane_id", "result", "ts"],
    "lpr.recognize.cnn.CnnClassifier.from_file": ["path", "version"],
    "lpr.recognize.template.TemplateClassifier.from_file": ["path", "version"],
    "lpr.recognize.svm.SvmClassifier.from_file": ["path", "version"],
    "lpr.corner_net.CornerNet.from_file": ["path", "version", "score_threshold_override"],  # 02 第十二批：可选、默认不改行为
    "lpr.corner_net.CornerNet.detect": ["self", "image", "max_candidates"],
    "lpr.recognize.crnn.CrnnReader.from_file": ["path", "version"],
    "lpr.recognize.crnn.CrnnReader.read": ["self", "plate_image"],
}


@pytest.mark.parametrize("qualname,params", STAGE_SIGNATURES.items())
def test_stage_signatures(qualname, params):
    import importlib

    parts = qualname.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            break
        except ModuleNotFoundError:
            continue
    for attr in parts[i:]:
        obj = getattr(obj, attr)
    assert list(inspect.signature(obj).parameters) == params


def test_optional_rectify_candidates_signature():
    """阶段三可选的多假设函数：存在时签名必须与 rectify 一致，返回 list[RectifiedPlate]（首个为首选）。"""
    import lpr.rectify as rect_mod

    fn = getattr(rect_mod, "rectify_candidates", None)
    if fn is None:
        pytest.skip("rectify_candidates 尚未实现")
    assert list(inspect.signature(fn).parameters) == ["image", "cand", "cfg"]


def test_classifiers_have_predict_proba():
    from lpr.recognize.cnn import CnnClassifier
    from lpr.recognize.svm import SvmClassifier
    from lpr.recognize.template import TemplateClassifier

    for cls in (CnnClassifier, TemplateClassifier, SvmClassifier):
        assert list(inspect.signature(cls.predict_proba).parameters) == ["self", "chars"]


# ---- recognize* 永不抛异常 ----

BAD_ARRAYS = [None, "x.jpg", np.zeros((0, 0, 3), np.uint8), np.zeros((100, 100, 3), np.float32),
              np.zeros((10, 10, 4), np.uint8), np.zeros((2, 2, 2, 2), np.uint8)]


@pytest.mark.parametrize("bad", BAD_ARRAYS, ids=lambda b: type(b).__name__)
def test_bad_array_input_returns_bad_input(recognizer, bad):
    r = recognizer.recognize(bad)
    assert r.status is S.BAD_INPUT and r.error_code == 3005 and r.plate_no is None


@pytest.mark.parametrize("data", [b"", b"not an image", bytes(100), None, "abc"])
def test_bad_bytes_returns_bad_input(recognizer, data):
    r = recognizer.recognize_bytes(data)
    assert r.status is S.BAD_INPUT


@pytest.mark.parametrize("image", [np.zeros((720, 1280, 3), np.uint8), np.zeros((50, 50), np.uint8),
                                   np.full((480, 640, 1), 255, np.uint8)])
def test_valid_input_never_raises(recognizer, image):
    r = recognizer.recognize(image, lane_id=1, debug=True)
    assert isinstance(r, RecognitionResult)
    assert r.error_code == ERROR_CODES[r.status]
    assert r.model_version.startswith("lpr-")
    json.dumps(r.to_dict(), ensure_ascii=False)


def test_recognize_many_never_raises(recognizer, blank_image):
    assert recognizer.recognize_many([]).status is S.BAD_INPUT
    r = recognizer.recognize_many([blank_image, encode_jpg(blank_image), b"junk"])
    assert isinstance(r, RecognitionResult) and r.error_code == ERROR_CODES[r.status]


def test_warmup_never_raises(recognizer):
    recognizer.warmup()


# ---- 模型加载 ----


def test_from_config_missing_model_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        PlateRecognizer.from_config(runtime__model_path=str(tmp_path / "nope.pt"))


def test_manifest_charset_mismatch_raises(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    with pytest.raises(ModelLoadError, match="登记"):
        load_classifier("cnn", model)
    write_manifest_entry(model, version="m_v0", train_data="test", val_char_acc=None)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["m.pt"]["charset_sha"] = "000000000000"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="字符表"):
        load_classifier("cnn", model)


def test_manifest_file_hash_mismatch_raises(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    write_manifest_entry(model, version="m_v0", train_data="test", val_char_acc=None)
    model.write_bytes(b"other weights")
    with pytest.raises(ModelLoadError, match="sha256"):
        load_classifier("cnn", model)


# ---- FakeRecognizer ----


@pytest.mark.parametrize("method", ["recognize", "recognize_bytes", "recognize_many", "warmup", "from_config"])
def test_fake_signature_matches_real(method):
    real = inspect.signature(getattr(PlateRecognizer, method))
    fake = inspect.signature(getattr(FakeRecognizer, method))
    assert list(real.parameters) == list(fake.parameters)


def test_fake_rotates_ok(blank_image):
    fake = FakeRecognizer()
    plates = [fake.recognize(blank_image).plate_no for _ in range(6)]
    assert plates[0] == "陕A12345" and plates[5] == plates[0]
    r = fake.recognize(blank_image)
    assert r.status is S.OK and r.plate_image is not None
    json.dumps(r.to_dict(), ensure_ascii=False)


def test_fake_script(blank_image):
    fake = FakeRecognizer(script=["OK", "LOW_CONFIDENCE", "NO_PLATE"])
    got = [fake.recognize(blank_image).status for _ in range(4)]
    assert got == [S.OK, S.LOW_CONFIDENCE, S.NO_PLATE, S.OK]
    assert fake.recognize_bytes(b"junk").status is S.BAD_INPUT


def test_fake_samples_dir(tmp_path):
    a = np.full((100, 200, 3), 10, np.uint8)
    b = np.full((100, 200, 3), 20, np.uint8)
    c = np.full((100, 200, 3), 30, np.uint8)
    imwrite(tmp_path / "陕A99999__LOW.png", a)
    imwrite(tmp_path / "NO_PLATE_1.png", b)
    imwrite(tmp_path / "陕AD12345.png", c)
    fake = FakeRecognizer(samples_dir=tmp_path)
    r = fake.recognize_bytes((tmp_path / "陕A99999__LOW.png").read_bytes())
    assert (r.status, r.plate_no) == (S.LOW_CONFIDENCE, "陕A99999")
    assert fake.recognize(imread(tmp_path / "NO_PLATE_1.png")).status is S.NO_PLATE
    r = fake.recognize(c)
    assert (r.plate_no, r.plate_color, r.layout, r.vehicle_type_hint) == ("陕AD12345", "green", 8, "small")
    assert fake.recognize_many([c, a]).frames_used == 2


# ---- 中文路径读写 ----


def test_io_utils_chinese_path(tmp_path):
    img = np.random.default_rng(42).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    path = tmp_path / "中文目录" / "车牌.png"
    imwrite(path, img)
    assert (imread(path) == img).all()
    assert decode_image(path.read_bytes()).shape == (40, 60, 3)
    assert decode_image(b"") is None and decode_image(b"garbage") is None
