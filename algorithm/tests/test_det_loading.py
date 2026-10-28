"""Loading overrides preserve the checkpoint and the default inference policy."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from lpr.corner_net import CornerNet
from lpr.errors import ModelLoadError
from lpr.plate_det import ARCHITECTURE, PlateDetModel, PlateDetector, load_plate_detector


class LoadingThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = {"architecture": ARCHITECTURE, "input_size": [128, 128],
                       "score_threshold": .25, "state_dict": PlateDetModel().state_dict()}

    def test_default_and_override_preserve_weights_and_metadata(self):
        default = PlateDetector.from_checkpoint(self.payload, version="test")
        override = PlateDetector.from_checkpoint(self.payload, version="test", score_threshold=.7)
        self.assertEqual(default.score_threshold, .25)
        self.assertEqual(override.score_threshold, .7)
        self.assertEqual(self.payload["score_threshold"], .25)
        self.assertTrue(all(torch.equal(t, override.model.state_dict()[k])
                            for k, t in default.model.state_dict().items()))

    def test_invalid_overrides_are_rejected(self):
        for value in (False, float("nan"), float("inf"), 0., 1., -.1):
            with self.subTest(value=value), self.assertRaises(ModelLoadError):
                PlateDetector.from_checkpoint(self.payload, version="test", score_threshold=value)

    def test_file_and_dispatcher_forward_override(self):
        root = Path(__file__).resolve().parents[1]/"out"/"twelfth"/"temp"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as directory:
            path = Path(directory)/"model.pt"
            torch.save(self.payload, path)
            self.assertEqual(PlateDetector.from_file(path, score_threshold=.6).score_threshold, .6)
            self.assertEqual(CornerNet.from_file(path, version="test", score_threshold_override=.6).score_threshold, .6)

    def test_verified_factory_checks_manifest_and_forwards_threshold(self):
        detector = object.__new__(PlateDetector)
        path = Path("model.pt")
        with patch("lpr.recognize.verify_model", return_value={"version": "v2"}) as verify, \
                patch.object(CornerNet, "from_file", return_value=detector) as load:
            self.assertIs(load_plate_detector(path, score_threshold=.8), detector)
        verify.assert_called_once_with(path)
        load.assert_called_once_with(path, version="v2", score_threshold_override=.8)

    def test_override_on_legacy_checkpoint_is_not_silently_ignored(self):
        with patch("torch.load", return_value={"architecture": "corner_net_v1"}):
            with self.assertRaisesRegex(ModelLoadError, "only supported"):
                CornerNet.from_file(Path("legacy.pt"), version="old", score_threshold_override=.7)
