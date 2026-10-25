import cv2
import numpy as np
import pytest

from lpr import charnorm
from lpr.locate import locate, box_iou
from lpr.preprocess import preprocess
from lpr.rectify import rectify
from lpr.segment import _aligned_slots, _crossing_edge_ink, _fallback_binary_from_gray, segment
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


def _legacy_aligned_slots(binary, centers, char_w):
    h, w = binary.shape
    projection = np.count_nonzero(binary, axis=0)
    prefix = np.r_[0, projection.cumsum()]
    relative = np.asarray(centers)-.5
    best_score, best = -np.inf, []
    for scale in np.linspace(.90, 1.08, 10):
        radius = w*char_w*scale*.65
        for offset in range(-round(.06*w), round(.06*w)+1, 2):
            xs = (.5+relative*scale)*w + offset
            left = np.clip(np.floor(xs-radius).astype(int), 0, w)
            right = np.clip(np.ceil(xs+radius).astype(int), 0, w)
            midpoints = np.round((xs[:-1]+xs[1:])/2).astype(int)
            right[:-1] = np.minimum(right[:-1], midpoints)
            left[1:] = np.maximum(left[1:], midpoints)
            amounts = prefix[right]-prefix[left]
            total = amounts.sum()
            boundary_ink = (projection[np.minimum(left, w-1)]+
                            projection[np.maximum(right-1, 0)]).sum()
            score = (total + .1*len(centers)*amounts.min() - .5*boundary_ink
                     - .05*abs(offset)*h - .02*abs(scale-1)*w*h)
            if score > best_score:
                best_score, best = score, list(zip(left.tolist(), right.tolist()))
    return best


@pytest.mark.parametrize("layout,width", [(7, 252), (8, 276)])
def test_vectorized_alignment_matches_scalar_search(layout, width):
    rng = np.random.default_rng(321 + layout)
    binary = (rng.random((80, width)) < .05).astype(np.uint8) * 255
    centers = ([.086, .216, .395, .525, .655, .784, .914] if layout == 7 else
               [.079167, .1875, .379167, .4875, .595833, .704167, .8125, .920833])
    char_w = .102 if layout == 7 else .09375
    assert _aligned_slots(binary, centers, char_w) == _legacy_aligned_slots(binary, centers, char_w)


def test_raw_gray_fallback_removes_long_perimeter_ink(cfg):
    gray = np.zeros((80, 252), np.uint8)
    gray[78, 1:251] = 255
    gray[20:60, 8:24] = 255
    gray[20:60, 50:63] = 255
    cleaned = _fallback_binary_from_gray(gray, cfg.segment)
    assert not cleaned[78, 1:251].any()
    assert cleaned[20:60, 8:24].all()
    assert cleaned[20:60, 50:63].all()


@pytest.mark.parametrize('layout,width', [(7, 252), (8, 276)])
def test_fallback_removes_only_detached_compact_separator(layout, width, cfg):
    gray = np.zeros((80, width), np.uint8)
    gray[15:56, 67:76] = 255  # Tall character stroke in the separator region.
    gray[35:43, 80:89] = 255  # Detached compact dot / emblem.
    cleaned = _fallback_binary_from_gray(gray, cfg.segment)
    assert cleaned[15:56, 67:76].all()
    assert not cleaned[35:43, 80:89].any()


def test_valley_adjustment_requires_low_central_ink():
    centers = [.08, .20, .38, .51, .64, .77, .90]
    binary = np.zeros((80, 252), np.uint8)
    # Dense ink throughout every candidate interval is not safe evidence for a cut.
    binary[:, :] = 255
    plain = _aligned_slots(binary, centers, .10)
    adjusted = _aligned_slots(binary, centers, .10, projection_valleys=True)
    assert adjusted == plain


def test_slot_edge_counts_only_ink_crossing_cut():
    binary = np.zeros((80, 252), np.uint8)
    binary[20:60, 40:50] = 255
    # A complete glyph ending exactly at the slot's right edge has no crossing.
    assert _crossing_edge_ink(binary, 20, 60, 30, 50) == 0
    # A glyph continuing through the same cut is genuinely severed.
    binary[20:60, 50:54] = 255
    assert _crossing_edge_ink(binary, 20, 60, 30, 50) == 40
