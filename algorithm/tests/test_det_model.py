"""CPU unit tests for plate detector output decoding and model heads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from lpr.corner_net import CornerNet
from lpr.plate_det import ARCHITECTURE, PlateDetector, PlateDetModel, decode_predictions


class IdentityTransform:
    def to_source(self, points):
        return np.asarray(points, dtype=np.float32)


def synthetic_outputs(corner_value=0.0):
    heat = torch.full((1, 1, 4, 4), -20.0)
    heat[0, 0, 1, 1] = 8.0
    boxes = torch.zeros((1, 4, 4, 4))
    boxes[0, 2:, 1, 1] = float(np.log(2.0))
    corners = torch.full((1, 8, 4, 4), corner_value)
    return {"heatmap": heat, "box": boxes, "corners": corners}


class PlateDetectorModelTests(unittest.TestCase):
    def test_loader_rejects_finite_negative_batchnorm_variance(self):
        from lpr.errors import ModelLoadError
        model = PlateDetModel(pretrained=False)
        state = model.state_dict()
        state["features.1.running_var"].fill_(-1.)
        payload = {"architecture": ARCHITECTURE, "input_size": [128, 128],
                   "state_dict": state}
        with self.assertRaises(ModelLoadError):
            PlateDetector.from_checkpoint(payload, version="plate_det_v1")

    def test_model_output_shapes_and_finiteness(self):
        model = PlateDetModel(pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
        self.assertEqual(output["heatmap"].shape, (1, 1, 16, 16))
        self.assertEqual(output["box"].shape, (1, 4, 16, 16))
        self.assertEqual(output["corners"].shape, (1, 8, 16, 16))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_bbox_decodes_independently_of_corner_values(self):
        transform = IdentityTransform()
        ordinary = decode_predictions(synthetic_outputs(0.0), (32, 32, 3), transform,
                                      max_candidates=3, score_threshold=.9)
        malformed_aux = decode_predictions(synthetic_outputs(1.0), (32, 32, 3), transform,
                                           max_candidates=3, score_threshold=.9)
        self.assertEqual(len(ordinary), 1)
        self.assertEqual(len(malformed_aux), 1)
        self.assertEqual(ordinary[0].bbox, (4, 4, 16, 16))
        self.assertEqual(malformed_aux[0].bbox, ordinary[0].bbox)

    def test_nonfinite_auxiliary_corner_does_not_drop_valid_bbox(self):
        outputs = synthetic_outputs()
        outputs["corners"][0, 0, 1, 1] = float("nan")
        candidates = decode_predictions(outputs, (32, 32, 3), IdentityTransform(),
                                        max_candidates=3, score_threshold=.9)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bbox, (4, 4, 16, 16))
        self.assertEqual(len(candidates[0].corners), 4)

    def test_invalid_output_shapes_raise(self):
        outputs = synthetic_outputs()
        outputs["box"] = torch.zeros((1, 3, 4, 4))
        with self.assertRaises(ValueError):
            decode_predictions(outputs, (32, 32, 3), IdentityTransform(), max_candidates=1)

    def test_corner_factory_roundtrip_does_not_fetch_imagenet_weights(self):
        model = PlateDetModel(pretrained=False)
        payload = {"architecture": ARCHITECTURE, "input_size": [128, 128],
                   "score_threshold": .25, "state_dict": model.state_dict()}
        expected_state = {name: value.clone() for name, value in model.state_dict().items()}
        worker = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=worker) as temp_dir:
            checkpoint = Path(temp_dir) / "plate_det.pt"
            torch.save(payload, checkpoint)
            del model, payload
            original_detect = PlateDetector.detect
            with patch("torch.hub.load_state_dict_from_url",
                       side_effect=AssertionError("checkpoint loading must not download weights")), \
                    patch.object(PlateDetector, "detect", autospec=True,
                                 side_effect=original_detect) as initialize:
                loaded = CornerNet.from_file(checkpoint, version="plate_det_v1")
            initialize.assert_called_once()
            self.assertEqual(initialize.call_args.args[1].shape, (720, 1280, 3))
            self.assertEqual(initialize.call_args.args[1].dtype, np.uint8)
            self.assertEqual(initialize.call_args.args[2], 3)
        self.assertIsInstance(loaded, PlateDetector)
        self.assertEqual(loaded.version, "plate_det_v1")
        self.assertEqual(loaded.image_size, 128)
        self.assertTrue(all(torch.isfinite(value).all() for value in loaded.model.state_dict().values()))
        self.assertTrue(all(torch.equal(value, expected_state[name])
                            for name, value in loaded.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
