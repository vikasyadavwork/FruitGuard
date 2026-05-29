"""Detector contracts using local temporary weights and fake model outputs."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.vision.detector import AppleDetector


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def fake_box(class_id, confidence, bbox):
    return SimpleNamespace(cls=FakeTensor(class_id), conf=FakeTensor(confidence),
                           xyxy=[FakeTensor(list(bbox))])


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'local.pt').write_bytes(b'fake weights for a mocked loader')
        self.config = {
            'model_path': 'local.pt', 'device': 'cpu', 'input_size': 320,
            'detection_threshold': 0.35, 'apple_class_name': 'apple',
            'apple_class_id': None,
        }
        self.model = Mock()
        self.model.names = {2: 'banana', 9: 'apple', 47: 'person'}
        self.loader = patch('src.vision.detector.YOLO', return_value=self.model).start()
        self.addCleanup(patch.stopall)

    def make_detector(self, **changes):
        config = dict(self.config)
        config.update(changes)
        return AppleDetector(config, self.root)

    def test_missing_local_model_fails_without_loading_or_downloading(self):
        with self.assertRaisesRegex(ValueError, 'Model weights missing'):
            self.make_detector(model_path='absent.pt')
        self.loader.assert_not_called()

    def test_unavailable_cuda_does_not_load_model_or_fall_back(self):
        with patch('src.vision.detector.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Explicitly select --device cpu'):
                self.make_detector(device='cuda:0')
        self.loader.assert_not_called()

    def test_unavailable_cuda_index_does_not_load_model(self):
        with patch('src.vision.detector.torch.cuda.is_available', return_value=True), \
                patch('src.vision.detector.torch.cuda.device_count', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'CUDA device 1 unavailable'):
                self.make_detector(device='1')
        self.loader.assert_not_called()

    def test_explicit_cpu_mode_does_not_query_cuda_availability(self):
        with patch('src.vision.detector.torch.cuda.is_available') as availability:
            detector = self.make_detector()
        availability.assert_not_called()
        self.assertEqual(detector.device, 'cpu')
        self.assertIsNone(detector.cuda_index)

    def test_available_cuda_is_selected_explicitly(self):
        with patch('src.vision.detector.torch.cuda.is_available', return_value=True), \
                patch('src.vision.detector.torch.cuda.device_count', return_value=2):
            detector = self.make_detector(device='cuda:1')
        self.assertEqual(detector.device, 1)
        self.assertEqual(detector.cuda_index, 1)

    def test_resolves_apple_from_reordered_model_names(self):
        detector = self.make_detector()
        self.assertEqual(detector.class_id, 9)
        self.assertEqual(detector.class_name, 'apple')

    def test_custom_class_name_matches_case_insensitively(self):
        self.model.names = {0: 'background', 3: 'Orchard Apple'}
        detector = self.make_detector(apple_class_name='orchard apple')
        self.assertEqual(detector.class_id, 3)
        self.assertEqual(detector.class_name, 'Orchard Apple')

    def test_list_class_names_supported(self):
        self.model.names = ['pear', 'APPLE', 'person']
        self.assertEqual(self.make_detector().class_id, 1)

    def test_explicit_class_override_supported(self):
        detector = self.make_detector(apple_class_id=2)
        self.assertEqual(detector.class_id, 2)
        self.assertEqual(detector.class_name, 'banana')

    def test_invalid_override_rejected(self):
        with self.assertRaisesRegex(ValueError, 'apple_class_id does not exist'):
            self.make_detector(apple_class_id=100)

    def test_missing_or_ambiguous_apple_class_rejected(self):
        for names in ({1: 'pear'}, {1: 'apple', 2: 'APPLE'}):
            with self.subTest(names=names):
                self.model.names = names
                with self.assertRaisesRegex(ValueError, 'configured apple class name'):
                    self.make_detector()

    def test_detect_returns_bbox_class_confidence_and_no_manufactured_mask(self):
        detector = self.make_detector()
        self.model.predict.return_value = [SimpleNamespace(
            boxes=[fake_box(9, 0.91, (2.5, 3.0, 11.0, 12.0)),
                   fake_box(47, 0.99, (0, 0, 8, 8))], masks=None)]
        frame = np.zeros((16, 20, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (2.5, 3.0, 11.0, 12.0))
        self.assertEqual(detections[0].class_name, 'apple')
        self.assertAlmostEqual(detections[0].confidence, 0.91)
        self.assertIsNone(detections[0].mask)
        args, kwargs = self.model.predict.call_args
        self.assertIs(args[0], frame)
        self.assertEqual(kwargs['device'], 'cpu')
        self.assertEqual(kwargs['imgsz'], 320)
        self.assertEqual(kwargs['conf'], 0.35)
        self.assertEqual(kwargs['classes'], [9])

    def test_public_mask_polygon_is_aligned_to_original_frame(self):
        detector = self.make_detector()
        polygon = np.array([[6, 3], [10, 3], [10, 7], [6, 7]], dtype=np.float32)
        self.model.predict.return_value = [SimpleNamespace(
            boxes=[fake_box(9, 0.8, (4, 1, 12, 9))],
            masks=SimpleNamespace(xy=[polygon]))]
        mask = detector.detect(np.zeros((12, 18, 3), dtype=np.uint8))[0].mask
        self.assertEqual(mask.shape, (12, 18))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask.sum()), 25)
        self.assertEqual(int(mask[5, 8]), 1)
        self.assertEqual(int(mask[1, 1]), 0)
        self.assertEqual(int(mask[9, 8]), 0)

    def test_missing_boxes_and_empty_results_clear_detection_output(self):
        detector = self.make_detector()
        for results in ([], [SimpleNamespace(boxes=None, masks=None)]):
            with self.subTest(results=results):
                self.model.predict.return_value = results
                self.assertEqual(detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)), [])


if __name__ == '__main__':
    unittest.main()
