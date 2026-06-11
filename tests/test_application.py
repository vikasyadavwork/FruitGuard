import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.inspection import idle_result, inspect_frame
from src.utils.config import DEFAULTS, ROOT, load_config, validate_config
from src.utils.samples import save_sample
from src.vision.detector import Detection


class ConfigTests(unittest.TestCase):
    def test_repository_config_and_paths(self):
        config = load_config()
        self.assertEqual(config['serial']['baudrate'], 9600)
        self.assertTrue((ROOT / config['detector']['model_path']).is_file())

    def test_invalid_values(self):
        for section, key, value in [('serial', 'enter_cm', 46),
                                    ('serial', 'reading_timeout_s', 0),
                                    ('serial', 'baudrate', 115200),
                                    ('detector', 'device', 'auto'),
                                    ('detector', 'input_size', 641),
                                    ('detector', 'detection_threshold', float('nan')),
                                    ('quality', 'dark_weight', 0),
                                    ('quality', 'strong_dark_percentage', 1),
                                    ('quality', 'good_symmetry', 2),
                                    ('storage', 'directory', '../outside'),
                                    ('storage', 'directory', '\\outside'),
                                    ('storage', 'directory', 'C:outside')]:
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(DEFAULTS)
                config[section][key] = value
                if key == 'dark_weight':
                    config['quality']['color_weight'] = 0
                    config['quality']['symmetry_weight'] = 0
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_missing_unknown_or_wrong_types(self):
        for change in ('missing', 'unknown', 'boolean'):
            config = copy.deepcopy(DEFAULTS)
            if change == 'missing':
                del config['serial']['port']
            elif change == 'unknown':
                config['serial']['password'] = 'not-a-real-secret'
            else:
                config['camera']['index'] = True
            with self.assertRaises(ValueError):
                validate_config(config)


class FrameAndSampleTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((100, 100, 3), 230, np.uint8)
        self.apple = Detection((10., 10., 90., 90.), 'apple', .95)

    def test_disappearing_detection_clears_previous_good_and_defective(self):
        for status in ('GOOD', 'DEFECTIVE'):
            with patch('src.inspection.analyze', return_value={'status': status, 'reasons': [],
                                                               'features': {}, 'validity': {}, 'bbox': (10, 10, 90, 90)}):
                previous = inspect_frame(self.frame, [self.apple], DEFAULTS['quality'])
            self.assertEqual(previous['status'], status)
            current = inspect_frame(self.frame, [], DEFAULTS['quality'])
            self.assertIsNone(current['status'])
            self.assertEqual(current['command'], 'OFF')
            self.assertIsNone(current['bbox'])
            self.assertEqual(current['features'], {})

    def test_multiple_apples_review_without_single_fruit_measurements(self):
        result = inspect_frame(self.frame, [self.apple, self.apple], DEFAULTS['quality'])
        self.assertEqual(result['status'], 'REVIEW')
        self.assertEqual(result['command'], 'YELLOW')
        self.assertEqual(result['reasons'], ['Present one apple'])
        self.assertEqual(result['features'], {})

    def test_system_error_and_waiting_are_not_fruit_decisions(self):
        self.assertIsNone(idle_result('Waiting')['status'])
        self.assertEqual(idle_result('Camera error', True)['command'], 'BLUE')

    def _record(self, frame_number=7):
        return {'timestamp': '2026-09-27T00:00:00+00:00', 'frame_index': frame_number,
                'inspection': {'bbox': [10, 10, 90, 90], 'status': 'REVIEW',
                               'features': {}, 'validity': {}, 'reasons': ['test']},
                'configuration': copy.deepcopy(DEFAULTS)}

    def test_saves_original_roi_current_metadata_and_unique_names(self):
        with tempfile.TemporaryDirectory() as temp:
            path = save_sample(self.frame, self._record(), Path(temp))
            second = save_sample(self.frame, self._record(8), Path(temp))
            self.assertNotEqual(path, second)
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(record['frame_index'], 7)
            np.testing.assert_array_equal(cv2.imread(str(path.parent / record['files']['original'])), self.frame)
            np.testing.assert_array_equal(cv2.imread(str(path.parent / record['files']['roi'])), self.frame[10:90, 10:90])

    def test_image_failure_is_visible_and_no_json_record_written(self):
        with tempfile.TemporaryDirectory() as temp, patch('src.utils.samples.cv2.imwrite', return_value=False):
            with self.assertRaises(OSError):
                save_sample(self.frame, self._record(), Path(temp))
            self.assertEqual(list(Path(temp).glob('*.json')), [])

    def test_json_failure_propagates(self):
        record = self._record()
        record['nonfinite'] = float('nan')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                save_sample(self.frame, record, Path(temp))


if __name__ == '__main__':
    unittest.main()
