import csv
import json

import numpy as np
import pytest

from lpr.io_utils import imread
from lpr_tools.synth_scenes import generate, make_scene


def test_synth_repeatability_and_coverage():
    colors, layouts, angles, widths = set(), set(), [], []
    for i in range(15):
        image, row = make_scene(i)
        again, label = make_scene(i)
        assert np.array_equal(image, again) and row == label
        assert image.shape == (720, 1280, 3) and image.dtype == np.uint8
        q = np.array(row['quad'])
        assert (q >= 0).all() and (q[:, 0] < 1280).all() and (q[:, 1] < 720).all()
        a, b = q[1]-q[0], q[2]-q[1]
        assert a[0]*b[1] - a[1]*b[0] > 0
        colors.add(row['color']); layouts.add(row['layout'])
        assert len(row['plate_text']) == row['layout']
        angles.append(np.degrees(np.arctan2(*(q[1]-q[0])[::-1])))
        widths.append(np.linalg.norm(q[1]-q[0]))
    assert colors == {'blue', 'yellow', 'green'} and layouts == {7, 8}
    assert min(angles) < -20 and max(angles) > 20 and max(widths) / min(widths) > 1.5


def test_synth_csv_chinese_path(tmp_path):
    out = tmp_path / '合成场景'
    rows = generate(out, 2)
    with (out / 'labels.csv').open(encoding='utf-8', newline='') as f:
        saved = list(csv.DictReader(f))
    assert json.loads(saved[0]['quad']) == rows[0]['quad']
    assert imread(out / saved[0]['file']).shape == (720, 1280, 3)
    assert generate(out / 'empty', 0) == []
    with pytest.raises(ValueError):
        generate(out, -1)

from lpr.locate import box_iou, locate
from lpr.preprocess import preprocess
from lpr.types import PreprocessOut


def truth_bbox(row):
    q = np.array(row['quad'])
    lo, hi = q.min(axis=0), q.max(axis=0)
    return (*lo, *(hi-lo))


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5, 6, 8, 11, 14, 25, 32])
def test_locate_synthetic_color_tilt_and_noise(cfg, index):
    image, row = make_scene(index)
    candidates = locate(preprocess(image, cfg.preprocess), cfg.locate)
    hits = [c for c in candidates if box_iou(c.bbox, truth_bbox(row)) >= .5]
    assert hits, (index, [(c.bbox, c.color) for c in candidates])
    assert hits[0].color == row['color']
    assert len(candidates) <= 3
    assert [c.score for c in candidates] == sorted((c.score for c in candidates), reverse=True)
    for c in candidates:
        q = np.array(c.corners)
        assert q[0, 0] < q[1, 0] and q[3, 0] < q[2, 0]
        assert q[0, 1] < q[3, 1] and q[1, 1] < q[2, 1]
        assert 0 <= c.score <= 1
    assert all(box_iou(a.bbox, b.bbox) <= cfg.locate.nms_iou for i, a in enumerate(candidates) for b in candidates[i+1:])


def test_locate_roi_coordinates(cfg):
    image, row = make_scene(2)
    pre = preprocess(image, cfg.preprocess, (200, 250, 900, 450))
    assert pre.scale < 1 and pre.offset == (200, 250)
    assert any(box_iou(c.bbox, truth_bbox(row)) >= .5 for c in locate(pre, cfg.locate))


def test_locate_edge_fallback_and_limit(cfg, monkeypatch):
    image = np.full((400, 800, 3), 80, np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(image, (200, 200), (500, 295), (180, 180, 180), -1)
    cv2.putText(image, 'AB12345', (205, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
    from lpr import locate as locate_module
    seen = []
    nms = locate_module._nms
    def capture(candidates, config, limit):
        seen.extend(c.source for c in candidates)
        return nms(candidates, config, limit)
    monkeypatch.setattr(locate_module, '_nms', capture)
    pre = preprocess(image, cfg.preprocess)
    assert 'edge_tight' in locate_module.detection_masks(pre, cfg.locate)
    cs = locate(pre, cfg.locate, 1)
    assert 'edge' in seen
    assert len(cs) == 1 and cs[0].source == 'edge' and cs[0].color == 'unknown'
    assert box_iou(cs[0].bbox, (200, 200, 300, 95)) > .5
    assert locate(pre, cfg.locate, 0) == []


def test_locate_empty_black_tiny(cfg):
    assert locate(None, cfg.locate) == []
    empty = np.empty((0, 0, 3), np.uint8)
    assert locate(PreprocessOut(empty, empty[..., 0], empty, 1, (0, 0)), cfg.locate) == []
    image = np.zeros((720, 1280, 3), np.uint8)
    assert locate(preprocess(image, cfg.preprocess), cfg.locate) == []
    image[300:303, 300:310] = (180, 65, 15)
    assert locate(preprocess(image, cfg.preprocess), cfg.locate) == []


def test_nms_keeps_higher_ranked_geometry_and_suppresses_overlaps(cfg):
    from lpr.locate import _nms
    from lpr.types import PlateCandidate
    def candidate(x, width, score, source):
        return PlateCandidate([(x, 0), (x+width, 0), (x+width, 40), (x, 40)],
                              (x, 0, width, 40), 'blue', score, source)
    proposals = [candidate(50, 100, .95, 'edge'), candidate(0, 60, .92, 'color'),
                 candidate(0, 180, .90, 'color')]
    result = _nms(proposals, cfg.locate, 3)
    assert [candidate.bbox for candidate in result] == [(50, 0, 100, 40), (0, 0, 60, 40)]
    assert result[0].source == 'merged' and result[0].score == .95


def test_nms_color_body_evidence_does_not_expand_plate_geometry(cfg):
    """A broad color region can corroborate a plate without replacing its box."""
    from lpr.locate import _nms
    from lpr.types import PlateCandidate

    def candidate(box, score, source):
        x, y, width, height = box
        return PlateCandidate([(x, y), (x+width, y), (x+width, y+height), (x, y+height)],
                              box, 'blue', score, source)

    plate = candidate((100, 100, 160, 50), .95, 'edge')
    vehicle_color = candidate((70, 90, 250, 80), .90, 'color')
    other = candidate((450, 100, 100, 40), .70, 'color')
    result = _nms([vehicle_color, plate, other], cfg.locate, 3)

    assert result[0].bbox == plate.bbox
    assert result[0].source == 'merged' and result[0].score == plate.score
    assert [c.score for c in result] == sorted((c.score for c in result), reverse=True)
    assert len(result) <= 3
    assert all(box_iou(a.bbox, b.bbox) <= cfg.locate.nms_iou
               for i, a in enumerate(result) for b in result[i+1:])
