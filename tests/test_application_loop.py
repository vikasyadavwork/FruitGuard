"""Hardware-free loop tests validate state flow, not camera/LED integration."""

import copy
import pickle
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src import application
from src.utils.config import DEFAULTS
from src.vision.detector import Detection


def good_result(frame):
    """Tag a synthetic decision with the frame that produced it."""
    return {
        'system_state': 'INSPECTING', 'status': 'GOOD', 'command': 'GREEN',
        'reasons': ['Synthetic valid measurements'],
        'features': {'frame_marker': int(frame[0, 0, 0])},
        'validity': {'measurements': True}, 'bbox': [1, 1, 10, 10],
    }


class ApplicationLoopTests(unittest.TestCase):
    def setUp(self):
        self.frames = [np.full((24, 24, 3), value, dtype=np.uint8)
                       for value in (1, 2, 3)]
        self.camera = MagicMock()
        self.camera.isOpened.return_value = True
        self.camera.read.side_effect = [(True, frame) for frame in self.frames]
        self.detector = MagicMock()
        self.detector.class_name = 'apple'
        self.detector.path = Path('synthetic-local-model.pt')
        self.detector.detect.return_value = []
        self.hardware = MagicMock()
        self.hardware.ready = True
        self.hardware.connected = True
        self.hardware.error = None
        self.hardware.generation = 1
        self.hardware.poll.side_effect = [[], [30.0], [30.0], [30.0]]
        self.hardware.send.return_value = True
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.load = self.stack.enter_context(patch.object(
            application, 'load_config', return_value=copy.deepcopy(DEFAULTS)))
        self.load.return_value['classifier']['backend'] = 'cpu'
        self.stack.enter_context(patch.object(application.cv2, 'VideoCapture',
                                             return_value=self.camera))
        self.arduino_constructor = self.stack.enter_context(patch.object(
            application, 'ArduinoSerial', return_value=self.hardware))
        self.detector_constructor = self.stack.enter_context(patch.object(
            application, 'AppleDetector', return_value=self.detector))
        self.imshow = self.stack.enter_context(patch.object(application.cv2, 'imshow'))
        self.wait_key = self.stack.enter_context(patch.object(
            application.cv2, 'waitKey', return_value=ord('q')))
        self.destroy_windows = self.stack.enter_context(patch.object(
            application.cv2, 'destroyAllWindows'))
        self.render = self.stack.enter_context(patch.object(
            application, 'render', side_effect=lambda frame, *args: frame.copy()))

    def test_explicit_manual_mode_never_constructs_serial_and_cleans_up(self):
        result = application.main(['--no-arduino', '--device', 'cpu'])
        self.assertEqual(result, 0)
        self.arduino_constructor.assert_not_called()
        self.hardware.poll.assert_not_called()
        self.hardware.send.assert_not_called()
        self.hardware.close.assert_not_called()
        self.camera.release.assert_called_once_with()
        self.destroy_windows.assert_called_once_with()
        record = self.render.call_args[0][2]
        self.assertEqual(record['mode'], 'Hardware disabled / manual inspection')
        self.assertEqual(record['serial_state'], 'disabled')
        self.assertIsNone(record['distance']['distance_cm'])
        self.assertEqual(record['distance']['reading_state'], 'unavailable')
        self.assertEqual(record['inspection']['command'], 'OFF')

    def test_normal_hardware_exit_requests_off_cleanup(self):
        self.assertEqual(application.main(['--device', 'cpu']), 0)
        self.hardware.close.assert_called_once_with(normal=True)
        self.camera.release.assert_called_once_with()
        self.destroy_windows.assert_called_once_with()

    def test_save_uses_current_raw_frame_and_its_matching_metadata(self):
        detection = Detection((1, 1, 10, 10), 'apple', 0.9)
        self.detector.detect.return_value = [detection]
        self.wait_key.side_effect = [-1, ord('s'), ord('q')]
        # The UI output differs from the raw camera frame; save must use raw.
        self.render.side_effect = lambda frame, *args: np.full_like(frame, 99)
        saved = []

        def capture_sample(frame, record, directory):
            saved.append((frame, frame.copy(), copy.deepcopy(record), directory))
            return Path('saved-current-frame.json')

        with patch.object(application, 'inspect_frame',
                          side_effect=lambda frame, *args: good_result(frame)), \
                patch.object(application, 'save_sample', side_effect=capture_sample):
            self.assertEqual(application.main(['--no-arduino', '--device', 'cpu']), 0)
        self.assertEqual(len(saved), 1)
        raw_frame, image_snapshot, record, directory = saved[0]
        self.assertIs(raw_frame, self.frames[1])
        np.testing.assert_array_equal(image_snapshot, np.full_like(self.frames[1], 2))
        self.assertEqual(record['frame_index'], 1)
        self.assertEqual(record['inspection']['features']['frame_marker'], 2)
        self.assertEqual(record['detections'][0]['confidence'], 0.9)
        self.assertTrue(record['timestamp'])
        self.assertEqual(directory, application.ROOT / DEFAULTS['storage']['directory'])

    def test_model_failure_requests_fatal_blue_cleanup_without_normal_off(self):
        self.detector_constructor.side_effect = RuntimeError('Synthetic model failure')
        with self.assertLogs(application.LOG, level='ERROR') as logged:
            self.assertEqual(application.main(['--device', 'cpu']), 1)
        self.hardware.close.assert_called_once_with(normal=False)
        self.hardware.send.assert_not_called()
        self.camera.release.assert_called_once_with()
        self.destroy_windows.assert_called_once_with()
        self.assertIn('no subsequent OFF', '\n'.join(logged.output))

    def test_camera_failure_requests_fatal_cleanup(self):
        self.camera.read.side_effect = [(False, None)]
        with self.assertLogs(application.LOG, level='ERROR'):
            self.assertEqual(application.main(['--device', 'cpu']), 1)
        self.hardware.close.assert_called_once_with(normal=False)
        self.hardware.send.assert_not_called()
        self.camera.release.assert_called_once_with()
        self.destroy_windows.assert_called_once_with()

    def test_unexpected_corrupt_weights_are_fatal_and_do_not_request_off(self):
        self.detector_constructor.side_effect = pickle.UnpicklingError('Synthetic corrupt weights')
        with self.assertLogs(application.LOG, level='ERROR'):
            self.assertEqual(application.main(['--device', 'cpu']), 1)
        self.hardware.close.assert_called_once_with(normal=False)
        self.hardware.send.assert_not_called()
        self.camera.release.assert_called_once_with()

    def test_reading_that_expires_during_detection_clears_quality_before_send(self):
        clock = [100.0]
        detection = Detection((1, 1, 10, 10), 'apple', 0.95)

        def slow_detection(frame):
            clock[0] += DEFAULTS['serial']['reading_timeout_s'] + 0.5
            return [detection]

        self.detector.detect.side_effect = slow_detection
        with patch.object(application.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(application, 'inspect_frame',
                             side_effect=lambda frame, *args: good_result(frame)):
            self.assertEqual(application.main(['--device', 'cpu']), 0)
        self.hardware.send.assert_called_once_with('BLUE')
        record = self.render.call_args[0][2]
        self.assertEqual(self.render.call_args[0][1], [])
        self.assertEqual(record['detections'], [])
        self.assertEqual(record['distance']['reading_state'], 'stale')
        self.assertFalse(record['distance']['active'])
        self.assertEqual(record['inspection']['system_state'], 'ERROR')
        self.assertIsNone(record['inspection']['status'])
        self.assertIsNone(record['inspection']['bbox'])
        self.assertEqual(record['inspection']['features'], {})
        self.assertIn('stale', ' '.join(record['inspection']['reasons']))
        self.assertIsNotNone(record['timing']['inference_latency_ms'])

    def test_missing_cuda_classifier_is_fatal_without_cpu_fallback(self):
        self.load.return_value['classifier']['backend'] = 'cuda'
        with patch.object(application, 'create_measurement_backend',
                          side_effect=RuntimeError('CUDA classifier DLL missing')), \
                self.assertLogs(application.LOG, level='ERROR'):
            self.assertEqual(application.main(['--device', 'cpu']), 1)
        self.hardware.close.assert_called_once_with(normal=False)
        self.hardware.send.assert_not_called()
        self.detector.detect.assert_not_called()

    def test_cuda_classifier_context_is_closed_on_normal_exit(self):
        self.load.return_value['classifier']['backend'] = 'cuda'
        backend = MagicMock()
        with patch.object(application, 'create_measurement_backend', return_value=backend):
            self.assertEqual(application.main(['--no-arduino', '--device', 'cpu']), 0)
        backend.close.assert_called_once_with()
        self.assertEqual(self.render.call_args[0][2]['classifier_backend'], 'cuda')


if __name__ == '__main__':
    unittest.main()
