import cv2
import numpy as np
import pytest

from lpr import charnorm
from lpr.locate import locate, box_iou
from lpr.preprocess import preprocess
from lpr.rectify import rectify
from lpr.segment import segment
from lpr.types import RectifiedPlate, PlateCandidate
from lpr_tools.synth_scenes import draw_plate, make_scene
from .test_locate import truth_bbox


def plate_rect(color='blue', layout=7):
    text = '皖AB1234' if layout == 7 else '皖AD12345'
    width = 252 if layout == 7 else 276
    image = cv2.resize(draw_plate(text, color), (width, 80))
    return RectifiedPlate(image, layout, color, np.eye(3), None)


@pytest.mark.parametrize('color,layout', [('blue', 7), ('yellow', 7), ('green', 8), ('yellow', 8)])
@pytest.mark.parametrize('binarize', ['otsu', 'adaptive'])
def test_segment_clean_polarity_boxes_and_normalization(cfg, color, layout, binarize, monkeypatch):
    rect = plate_rect(color, layout)
    scfg = cfg.segment.model_copy(update={'binarize': binarize})
    calls = []
    unify, normalize = charnorm.unify_polarity, charnorm.normalize_char
    def tracked_unify(g, c):
        calls.append(('polarity', c)); return unify(g, c)
    def tracked_normalize(crop):
        calls.append(('normalize', crop.shape)); return normalize(crop)
    monkeypatch.setattr(charnorm, 'unify_polarity', tracked_unify)
    monkeypatch.setattr(charnorm, 'normalize_char', tracked_normalize)
    result = segment(rect, layout, scfg)
    assert result.ok and len(result.boxes) == layout
    assert result.chars.shape == (layout, 32, 32) and result.chars.dtype == np.uint8
    assert calls[0] == ('polarity', color) and len(calls) == layout+1
    assert 0 < result.quality <= 1
    g = unify(cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY), color)
    for box, char in zip(result.boxes, result.chars):
        x, y, w, h = box
        assert 0 <= x < x+w <= rect.image.shape[1] and 0 <= y < y+h <= 80
        assert np.array_equal(char, normalize(g[y:y+h, x:x+w]))
        assert not char[0].any() and char.max() == 255


@pytest.mark.parametrize('index', [0, 1, 2, 4, 5, 6, 8, 12, 14, 25, 31, 32, 34, 35])
def test_segment_located_scenes(cfg, index):
    image, row = make_scene(index)
    cs = locate(preprocess(image, cfg.preprocess), cfg.locate)
    cand = max(cs, key=lambda c: box_iou(c.bbox, truth_bbox(row)))
    rect = rectify(image, cand, cfg.rectify)
    seg = segment(rect, rect.layout, cfg.segment)
    assert seg.ok and len(seg.chars) == row['layout']


@pytest.mark.parametrize('image', [None, np.empty((0, 0, 3), np.uint8), np.zeros((4, 12, 3), np.uint8),
    np.zeros((80, 252, 3), np.uint8), np.full((80, 252, 3), 255, np.uint8), np.zeros((80, 252), np.float32)])
def test_segment_failure_shape(cfg, image):
    rect = RectifiedPlate(image, 7, 'blue', np.eye(3), None)
    result = segment(rect, 7, cfg.segment)
    assert not result.ok and result.boxes == [] and result.chars.shape == (0, 32, 32)
    assert result.chars.dtype == np.uint8


def test_segment_invalid_layout_missing_slot_wrong_count(cfg):
    rect = plate_rect()
    assert not segment(rect, 6, cfg.segment).ok
    assert not segment(None, 7, cfg.segment).ok
    assert not segment(rect, 8, cfg.segment).ok
    rect.image[10:70, 150:180] = (180, 65, 15)
    assert not segment(rect, 7, cfg.segment).ok


def test_segment_offset_scale_and_fragmented_province(cfg):
    rect = plate_rect()
    # Divide the square into disconnected strokes; the slot must keep them together.
    rect.image[30:40, 8:35] = (180, 65, 15)
    rect.image = cv2.warpAffine(rect.image, np.float32([[.95, 0, 7], [0, 1, 0]]), (252, 80), borderValue=(180, 65, 15))
    assert segment(rect, 7, cfg.segment).ok
