import cv2
import numpy as np
import pytest

from lpr.errors import BadInputError
from lpr.locate import locate, order_points
from lpr.preprocess import preprocess
import lpr.rectify as rectify_module
from lpr.rectify import (_quad_candidates, _rank_quad_candidates,
                         _green_boundary_quad, _linear_percentile_sorted,
                         _quad_glyph_evidence, _quad_iou, _quad_pixel_quality,
                         _quad_rank_score, classify_color, refine_quad, rectify,
                         rectify_candidates)
from lpr.types import PlateCandidate, RectifiedPlate
from lpr_tools.synth_scenes import make_scene
from .test_locate import truth_bbox
from lpr.locate import box_iou


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5, 7, 8])
def test_rectify_scene_original_homography(cfg, index):
    image, row = make_scene(index)
    cs = locate(preprocess(image, cfg.preprocess), cfg.locate)
    cand = max(cs, key=lambda c: box_iou(c.bbox, truth_bbox(row)))
    result = rectify(image, cand, cfg.rectify)
    assert result.layout == row['layout'] and result.color == row['color']
    width = 276 if result.layout == 8 else 252
    assert result.image.shape == (80, width, 3) and result.image.dtype == np.uint8
    assert np.array_equal(result.image, cv2.warpPerspective(image, result.homography, (width, 80)))
    mapped = cv2.perspectiveTransform(np.float32(row['quad'])[None], result.homography)[0]
    target = np.float32([[0, 0], [width-1, 0], [width-1, 79], [0, 79]])
    assert np.linalg.norm(mapped-target, axis=1).max() < 12


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5, 7, 8])
def test_rectify_candidates_first_is_exact_rectify_and_bounded(cfg, index):
    image, row = make_scene(index)
    candidates = locate(preprocess(image, cfg.preprocess), cfg.locate)
    cand = max(candidates, key=lambda c: box_iou(c.bbox, truth_bbox(row)))

    expected = rectify(image, cand, cfg.rectify)
    results = rectify_candidates(image, cand, cfg.rectify)

    assert 1 <= len(results) <= 4
    assert results[0].layout == expected.layout
    assert results[0].color == expected.color
    assert np.array_equal(results[0].image, expected.image)
    assert np.array_equal(results[0].homography, expected.homography)
    assert results[0].candidate is expected.candidate or results[0].candidate == expected.candidate


def test_rectify_candidates_truncates_sorted_pool_to_four_stably(monkeypatch, cfg):
    image = np.zeros((120, 240, 3), np.uint8)
    cand = PlateCandidate([(20, 20), (120, 20), (120, 50), (20, 50)],
                          (20, 20, 100, 30), 'blue', .8, 'color')
    quads = [np.float32([[20+i, 20], [120+i, 20], [120+i, 50], [20+i, 50]])
             for i in range(6)]
    rects = [RectifiedPlate(np.full((80, 252, 3), i, np.uint8), 7, 'blue',
                            np.eye(3), cand) for i in range(6)]
    # All scores tie. The shared ranker must retain contour enumeration order.
    monkeypatch.setattr(rectify_module, '_quad_candidates',
                        lambda *_args: (quads[0], quads))
    monkeypatch.setattr(rectify_module, '_warp_quad',
                        lambda _image, _cand, quad: rects[next(
                            i for i, value in enumerate(quads) if value is quad)])
    monkeypatch.setattr(rectify_module, '_quad_pixel_quality', lambda *_args: 1.)
    monkeypatch.setattr(rectify_module, '_quad_iou', lambda *_args: 1.)
    ranked = _rank_quad_candidates(image, cand, cfg.rectify.expand_ratio)
    assert [item[1] for item in ranked] == rects

    monkeypatch.setattr(rectify_module, '_rank_quad_candidates',
                        lambda *_args, **_kwargs: ranked)
    result = rectify_candidates(image, cand, cfg.rectify)
    assert len(result) == 4
    assert [item.image[0, 0, 0] for item in result] == [0, 1, 2, 3]


def test_quad_candidate_pool_deduplicates_color_and_edge_contours():
    image = np.zeros((140, 260, 3), np.uint8)
    cv2.rectangle(image, (50, 40), (210, 95), (180, 65, 15), -1)
    cand = PlateCandidate([(50, 40), (210, 40), (210, 95), (50, 95)],
                          (50, 40, 160, 55), 'blue', .8, 'color')
    _, quads = _quad_candidates(image, cand, .10, max_candidates=40)
    assert all(_quad_iou(a, b) < .995 for i, a in enumerate(quads)
               for b in quads[i+1:])


def test_green_mask_fit_recovers_supported_trapezoid_and_requires_text():
    image = np.zeros((240, 460, 3), np.uint8)
    green = (25, 180, 20)
    truth = np.float32([[55, 65], [390, 92], [370, 185], [38, 150]])
    cv2.fillConvexPoly(image, truth.astype(np.int32), green)
    for i in range(7):
        x = int(76+i*43)
        y_top = int(67+i*3.4)
        y_bottom = int(144+i*3.7)
        cv2.rectangle(image, (x, y_top), (x+14, y_bottom), (15, 15, 15), -1)
    candidate = PlateCandidate(truth.tolist(), (38, 65, 352, 121),
                               'green', .8, 'color')
    fitted, evidence = _green_boundary_quad(image, candidate)
    assert fitted is not None and evidence['reason'] == 'accepted'
    assert cv2.isContourConvex(fitted)
    assert evidence['fit_support'] >= .45

    # A broad green vehicle panel has color edges, but no repeated text row;
    # it cannot add an alternative geometry to the rectifier pool.
    blank = np.zeros_like(image)
    cv2.fillConvexPoly(blank, truth.astype(np.int32), green)
    regular = _rank_quad_candidates(blank, candidate, .10)
    with_fit = _rank_quad_candidates(blank, candidate, .10, include_green_fit=True)
    assert len(with_fit) == len(regular)


def test_green_fit_quad_matching_existing_alternate_is_deduplicated(monkeypatch):
    image = np.zeros((120, 240, 3), np.uint8)
    cand = PlateCandidate([(20, 20), (120, 20), (120, 50), (20, 50)],
                          (20, 20, 100, 30), 'green', .8, 'color')
    primary = np.float32([[20, 20], [120, 20], [120, 50], [20, 50]])
    alternate = np.float32([[21, 20], [121, 20], [121, 50], [21, 50]])
    rect = RectifiedPlate(np.full((80, 276, 3), (50, 160, 60), np.uint8),
                          8, 'green', np.eye(3), cand)
    monkeypatch.setattr(rectify_module, '_quad_candidates',
                        lambda *_args: (primary, [primary, alternate]))
    monkeypatch.setattr(rectify_module, '_warp_quad', lambda *_args: rect)
    monkeypatch.setattr(rectify_module, '_quad_pixel_quality', lambda *_args: 5.)
    monkeypatch.setattr(rectify_module, '_green_boundary_quad',
                        lambda *_args: (alternate, {'fit_support': .8,
                            'top_residual': 1., 'bottom_residual': 1.}))
    monkeypatch.setattr(rectify_module, '_quad_glyph_evidence',
                        lambda *_args: {'glyph_count': 7, 'edge_touch_count': 0,
                            'gap_cv': .3, 'coverage': .7})
    ranked = _rank_quad_candidates(image, cand, .1, include_green_fit=True)
    assert len(ranked) == 2


@pytest.mark.parametrize('n', [1, 2, 7, 30, 83])
@pytest.mark.parametrize('quantile', [.08, .92])
def test_sorted_linear_percentile_matches_numpy(n, quantile):
    values = np.sort(np.random.default_rng(n).integers(0, 80, size=n))
    assert _linear_percentile_sorted(values, quantile) == pytest.approx(
        np.percentile(values, quantile*100), abs=1e-12)


@pytest.mark.parametrize('channels', [1, 3])
def test_rectify_candidates_accepts_grayscale_and_single_channel(cfg, channels):
    gray = np.full((120, 240), 90, np.uint8)
    image = gray if channels == 1 else gray[..., None]
    cand = PlateCandidate([(20, 20), (120, 20), (120, 50), (20, 50)],
                          (20, 20, 100, 30), 'blue', .8, 'color')
    result = rectify_candidates(image, cand, cfg.rectify)
    assert 1 <= len(result) <= 4
    assert result[0].image.shape == (80, 252, 3)


def test_rectify_candidates_rejects_invalid_inputs(cfg):
    with pytest.raises(BadInputError):
        rectify_candidates(np.empty((0, 0, 3), np.uint8), None, cfg.rectify)


def test_rectify_fallback_black_unknown_and_tiny(cfg):
    image = np.zeros((120, 240, 3), np.uint8)
    cand = PlateCandidate([(20, 20), (100, 20), (100, 45), (20, 45)], (20, 20, 81, 26), 'unknown', .5, 'edge')
    result = rectify(image, cand, cfg.rectify)
    assert result.layout == 0 and result.color == 'black' and not result.image.any()
    mapped = cv2.perspectiveTransform(np.float32(cand.corners)[None], result.homography)[0]
    assert np.allclose(mapped, [[0, 0], [251, 0], [251, 79], [0, 79]], atol=1e-3)
    cand = PlateCandidate([(20, 20), (24, 20), (24, 22), (20, 22)], (20, 20, 5, 3), 'blue', .5, 'color')
    assert rectify(image, cand, cfg.rectify).image.shape == (80, 252, 3)


@pytest.mark.parametrize('image', [None, np.empty((0, 0, 3), np.uint8), np.zeros((10, 10), np.float32)])
def test_rectify_bad_input(cfg, image):
    with pytest.raises(BadInputError):
        rectify(image, None, cfg.rectify)


def test_rectify_invalid_quad_and_outside(cfg):
    image = np.zeros((120, 240, 3), np.uint8)
    for points, box in [([(1, 1)]*4, (1, 1, 1, 1)), ([(300, 0), (400, 0), (400, 30), (300, 30)], (300, 0, 100, 30))]:
        with pytest.raises(BadInputError):
            rectify(image, PlateCandidate(points, box, 'blue', .5, 'color'), cfg.rectify)


@pytest.mark.parametrize('bgr,color', [((180, 65, 15), 'blue'), ((20, 205, 235), 'yellow'),
    ((110, 210, 100), 'green'), ((230, 230, 230), 'white'), ((80, 80, 80), 'unknown')])
def test_rectify_color_vote(bgr, color):
    assert classify_color(np.full((80, 252, 3), bgr, np.uint8)) == color


@pytest.mark.parametrize('bgr', [((27, 10, 2)), ((5, 25, 10))])
def test_rectify_color_vote_keeps_underexposed_chromatic_plates_uncertain(bgr):
    """Hue is not strong enough at very low value to claim a plate layout."""
    assert classify_color(np.full((80, 252, 3), bgr, np.uint8)) == 'unknown'


@pytest.mark.parametrize('left,right', [((180, 65, 15), (110, 210, 100)),
                                        ((20, 205, 235), (110, 210, 100))])
def test_rectify_color_vote_preserves_uncertainty_on_competing_hues(left, right):
    plate = np.empty((80, 252, 3), np.uint8)
    plate[:, :126] = left
    plate[:, 126:] = right
    assert classify_color(plate) == 'unknown'


def test_refine_quad_rejects_contour_that_expands_candidate_substantially():
    image = np.zeros((120, 240, 3), np.uint8)
    blue = (180, 65, 15)
    # A nearby body-colored patch overlaps the candidate but is larger than the
    # plate proposal, so its contour must not replace the candidate geometry.
    cv2.rectangle(image, (65, 39), (170, 71), blue, -1)
    corners = [(70, 40), (170, 40), (170, 70), (70, 70)]
    candidate = PlateCandidate(corners, (70, 40, 100, 30), 'blue', .8, 'color')
    refined = refine_quad(image, candidate, .10)
    fallback = order_points(cv2.boxPoints(cv2.minAreaRect(np.float32(corners))))
    assert _quad_iou(fallback, refined) >= .90
    assert np.allclose(refined, fallback)


def test_quad_pixel_quality_penalizes_glyph_components_touching_crop_edge():
    edge = np.full((80, 252, 3), (180, 65, 15), np.uint8)
    inset = edge.copy()
    for x in range(18, 230, 31):
        cv2.rectangle(edge, (x, 0), (x+9, 48), (245, 245, 245), -1)
        cv2.rectangle(inset, (x, 10), (x+9, 58), (245, 245, 245), -1)

    assert _quad_glyph_evidence(edge)['edge_touch_count'] == 7
    assert _quad_glyph_evidence(inset)['edge_touch_count'] == 0
    assert _quad_pixel_quality(inset, 7) > _quad_pixel_quality(edge, 7)


def test_quad_pixel_quality_does_not_reward_textless_body_color_block():
    textless = np.full((80, 252, 3), (180, 65, 15), np.uint8)
    plate_like = textless.copy()
    for x in range(18, 230, 31):
        cv2.rectangle(plate_like, (x, 12), (x+9, 60), (245, 245, 245), -1)

    assert _quad_pixel_quality(textless, 7) < _quad_pixel_quality(plate_like, 7)
    # With equal pixel evidence, a remote body contour loses to the current
    # geometric proposal instead of taking over solely through its color area.
    assert _quad_rank_score(0., .70) < _quad_rank_score(0., 1.)


def test_quad_edge_evidence_uses_plate_boundary_not_central_window():
    plate = np.full((80, 252, 3), (180, 65, 15), np.uint8)
    for x in range(18, 230, 31):
        cv2.rectangle(plate, (x, 4), (x+9, 70), (245, 245, 245), -1)
    # These glyphs cross the central measurement window at y=8, but all their
    # strokes remain inside the real plate image.
    evidence = _quad_glyph_evidence(plate)
    assert evidence['glyph_count'] == 7
    assert evidence['edge_touch_count'] == 0


def test_unknown_neutral_body_texture_is_not_accepted_as_white_plate(cfg):
    image = np.full((160, 320, 3), (225, 225, 225), np.uint8)
    # Two plate-sized fragments touch opposite crop edges, as in the audited
    # vehicle-texture false-white case; they do not form a repeated text row.
    cv2.rectangle(image, (60, 40), (82, 70), (35, 35, 35), -1)
    cv2.rectangle(image, (250, 90), (272, 119), (35, 35, 35), -1)
    candidate = PlateCandidate([(40, 40), (280, 40), (280, 120), (40, 120)],
                               (40, 40, 240, 80), 'unknown', .8, 'edge')
    result = rectify(image, candidate, cfg.rectify)
    alternatives = rectify_candidates(image, candidate, cfg.rectify)

    assert classify_color(result.image) == 'white'
    assert result.color == 'unknown' and result.layout == 0
    assert all(r.color == 'unknown' and r.layout == 0 for r in alternatives)


def test_unknown_candidate_with_repeated_glyphs_remains_white(cfg):
    image = np.full((160, 320, 3), (230, 230, 230), np.uint8)
    for x in range(58, 255, 31):
        cv2.rectangle(image, (x, 54), (x+9, 108), (25, 25, 25), -1)
    candidate = PlateCandidate([(40, 40), (280, 40), (280, 120), (40, 120)],
                               (40, 40, 240, 80), 'unknown', .8, 'edge')
    result = rectify(image, candidate, cfg.rectify)
    alternatives = rectify_candidates(image, candidate, cfg.rectify)

    evidence = _quad_glyph_evidence(result.image)
    assert evidence['glyph_count'] >= 5
    assert result.color == 'white' and result.layout == 0
    assert all(r.color == 'white' and r.layout == 0 for r in alternatives)
