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
