import unittest

import numpy as np

from lpr_tools.train.measure_crop_errors import _box, _iou, _quad, pathkey, select_sources


class CropErrorMeasurementTests(unittest.TestCase):
    def test_box_and_iou_use_xywh(self):
        q = np.asarray([[10, 20], [30, 20], [30, 40], [10, 40]])
        self.assertEqual(_box(q), (10., 20., 20., 20.))
        self.assertEqual(_iou((10, 20, 20, 20), (10, 20, 20, 20)), 1.)
        self.assertEqual(_iou((0, 0, 1, 1), (3, 3, 1, 1)), 0.)

    def test_quad_rejects_wrong_point_count(self):
        with self.assertRaises(ValueError):
            _quad("1 2 3 4")

    def test_select_sources_uses_only_train_gt_and_subset_index(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        test_tmp = Path(__file__).resolve().parents[1] / "out" / "tmp-test-crop-errors"
        test_tmp.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=test_tmp) as td:
            root = Path(td)
            train = root / "train.csv"
            index = root / "index.csv"
            names = ("db", "blur", "tilt", "rotate", "fn", "challenge", "weather")
            train_rows = ["file,orig_path,source,layout,plate_no,crop_path,subset,brightness,partition,calibration_fold"]
            index_rows = ["path,subset,quad"]
            for i, subset in enumerate(names):
                path = f"C:/x/{subset}.jpg"
                train_rows.append(f"{i}.jpg,{path},hard_train:{subset}:ground_truth,7,皖A12345,crop,{subset},50,train,")
                index_rows.append(f"{path},ccpd_{subset},1 1 5 1 5 3 1 3")
            train_rows.extend(("tune.jpg,C:/x/tune.jpg,ground_truth,7,皖A12345,crop,blur,50,tune,",
                               "candidate.jpg,C:/x/db.jpg,pipeline_candidate:1:0,7,皖A12345,crop,db,50,train,"))
            train.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
            index.write_text("\n".join(index_rows) + "\n", encoding="utf-8")
            rows = select_sources(train, index, 1, 42)
            self.assertEqual(len(rows), 7)
            self.assertEqual({row[0] for row in rows}, set(names))
            self.assertEqual(pathkey("C:/x/db.jpg"), pathkey("C:/x/db.jpg"))


if __name__ == "__main__":
    unittest.main()


def test_measure_refuses_existing_evidence_before_loading_data(tmp_path):
    import pytest
    from lpr_tools.train.measure_crop_errors import measure
    evidence=tmp_path/'geometry';evidence.mkdir()
    saved=evidence/'train_sampler.json';saved.write_text('retained')
    with pytest.raises(FileExistsError):
        measure(tmp_path/'missing_train.csv',tmp_path/'missing_index.csv',evidence)
    assert saved.read_text()=='retained'
