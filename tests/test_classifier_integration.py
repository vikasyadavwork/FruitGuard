"""Exercise the reliability gate before interchangeable CUDA measurements."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.inspection import inspect_frame
from src.utils.config import DEFAULTS, validate_config
from src.vision.quality_analyzer import analyze, measure_features_cpu


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.last_timing = {'kernel_ms': 0.01, 'h2d_ms': 0.02, 'd2h_ms': 0.03}

    def measure(self, roi, mask, config):
        self.calls.append((roi.copy(), mask.copy()))
        return measure_features_cpu(roi, mask, config)


class ClassifierIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((160, 180, 3), 235, dtype=np.uint8)
        cv2.circle(self.frame, (90, 80), 45, (30, 40, 210), -1)
        self.detection = SimpleNamespace(bbox=(35, 25, 145, 135), confidence=.95,
                                         class_name='apple', mask=None)

    def test_validated_foreground_passed_to_measurement_backend(self):
        backend = RecordingBackend()
        result = analyze(self.frame, self.detection, DEFAULTS['quality'], backend)
        self.assertEqual(len(backend.calls), 1)
        roi, mask = backend.calls[0]
        self.assertEqual(roi.shape[:2], mask.shape)
        self.assertLess(np.count_nonzero(mask), mask.size)
        self.assertEqual(result['measurement_backend'], 'cuda')
        self.assertEqual(result['measurement_timing']['kernel_ms'], .01)
        self.assertIsNotNone(result['mask_cpu_ms'])
        self.assertEqual(result['status'], 'GOOD')

    def test_unreliable_mask_does_not_run_kernel_or_create_measurements(self):
        self.frame[:] = (30, 40, 210)
        backend = RecordingBackend()
        result = analyze(self.frame, self.detection, DEFAULTS['quality'], backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(result['status'], 'REVIEW')
        self.assertIsNone(result['measurement_timing'])
        self.assertTrue(all(value is None for value in result['features'].values()))

    def test_backend_failure_does_not_silently_use_cpu(self):
        backend = RecordingBackend()
        with patch.object(backend, 'measure', side_effect=RuntimeError('CUDA execution failed')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA execution failed'):
                analyze(self.frame, self.detection, DEFAULTS['quality'], backend)

    def test_quality_commands_and_detection_disappearance_share_current_frame(self):
        backend = RecordingBackend()
        good = inspect_frame(self.frame, [self.detection], DEFAULTS['quality'], backend)
        self.assertEqual(good['command'], 'GREEN')
        cv2.circle(self.frame, (90, 80), 20, (10, 10, 10), -1)
        bad = inspect_frame(self.frame, [self.detection], DEFAULTS['quality'], backend)
        self.assertEqual(bad['command'], 'RED')
        self.assertEqual(inspect_frame(self.frame, [], DEFAULTS['quality'], backend)['command'], 'OFF')
        self.assertEqual(inspect_frame(self.frame, [self.detection] * 2, DEFAULTS['quality'], backend)['command'], 'YELLOW')
        self.assertEqual(len(backend.calls), 2)

    def test_classifier_config_rejects_automatic_or_invalid_device_modes(self):
        for key, value in [('backend', 'auto'), ('device_index', -1), ('device_index', True)]:
            config = copy.deepcopy(DEFAULTS)
            config['classifier'][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
