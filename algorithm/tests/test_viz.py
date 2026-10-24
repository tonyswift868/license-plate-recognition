from pathlib import Path

import numpy as np
import pytest

from lpr.errors import BadInputError
from lpr.io_utils import imread, imwrite
from lpr_tools.synth_scenes import make_scene
from lpr_tools.viz import export_image, main, stage_images


def test_viz_production_images_and_chinese_paths(tmp_path, cfg, monkeypatch):
    image, _ = make_scene(2)
    path = tmp_path / '中文' / '车牌.png'
    imwrite(path, image)
    out = tmp_path / '中间图'
    texts = []
    import cv2
    real = cv2.putText
    def checked(img, text, *args, **kwargs):
        assert text.isascii()
        texts.append(text)
        return real(img, text, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', checked)
    count = export_image(path, out, cfg)
    assert len(list(out.glob('*.png'))) == count and texts
    for key in ['03_mask_blue', '03_mask_yellow', '03_mask_green', '03_mask_edge',
                '04_candidates', '05_plate_0', '06_binary_0', '07_boxes_0_L8', '08_chars_0_L8']:
        assert imread(out / (key+'.png')).size > 0
    assert imread(out / '08_chars_0_L8.png').shape == (32, 256, 3)


@pytest.mark.parametrize('image', [None, np.empty((0, 0, 3), np.uint8), np.zeros((5, 20, 3), np.uint8)])
def test_viz_invalid_input(cfg, image):
    with pytest.raises(BadInputError):
        stage_images(image, cfg)


def test_viz_black_tiny_and_gray(cfg):
    image = np.zeros((120, 200, 3), np.uint8)
    for array in [image, image[..., 0], image[..., :1]]:
        images = stage_images(array, cfg)
        assert '04_candidates' in images and '05_plate_0' not in images
    image[60:63, 60:70] = (180, 65, 15)
    assert '05_plate_0' not in stage_images(image, cfg)


def test_viz_cli_directory_empty_and_bad_file(tmp_path, capsys):
    assert main([str(tmp_path), '--out', str(tmp_path / 'out')]) == 1
    imwrite(tmp_path / 'black.png', np.zeros((120, 200, 3), np.uint8))
    (tmp_path / 'bad.png').write_bytes(b'corrupt')
    assert main([str(tmp_path), '--out', str(tmp_path / 'out')]) == 1
    captured = capsys.readouterr()
    assert 'Visualized 1/2' in captured.out and 'Failed:' in captured.err
    assert (tmp_path / 'out' / '0001_black' / '04_candidates.png').exists()


def test_viz_same_stem_inputs_are_separate(tmp_path):
    paths = []
    for folder in ('a', 'b'):
        path = tmp_path / folder / 'frame.png'
        imwrite(path, np.zeros((120, 200, 3), np.uint8))
        paths.append(str(path))
    out = tmp_path / 'out'
    assert main([*paths, '--out', str(out)]) == 0
    assert len(list(out.glob('*/04_candidates.png'))) == 2
