"""CUDA contract tests and opt-in measured GPU/CPU parity (no accuracy claims)."""
import ctypes
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.vision.cuda_quality import CudaQualityMeasurements
from src.vision.quality_analyzer import QUALITY_DEFAULTS


def cpu_reference(roi, mask, cfg):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[:, :, i] for i in range(3)]
    mask = mask != 0
    dark = v < cfg["dark_value_max"]
    brown = ((h >= cfg["brown_hue_min"]) & (h <= cfg["brown_hue_max"]) &
             (s >= cfg["brown_saturation_min"]) & (s <= cfg["brown_saturation_max"]) &
             (v >= cfg["brown_value_min"]) & (v <= cfg["brown_value_max"]))
    desaturated = ((s <= cfg["desaturated_saturation_max"]) &
                   (v >= cfg["desaturated_value_min"]) & (v <= cfg["desaturated_value_max"]))
    ys, xs = np.nonzero(mask)
    tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    reflected = np.fliplr(tight)
    count = np.count_nonzero(mask)
    result = {"dark_spot_percentage": 100.0 * np.count_nonzero(dark & mask) / count,
              "color_anomaly_score": float(np.count_nonzero((brown | desaturated) & mask)) / count,
              "symmetry_score": float(np.count_nonzero(tight & reflected)) / np.count_nonzero(tight | reflected)}
    result["visual_defect_score"] = (
        cfg["dark_weight"] * result["dark_spot_percentage"] / 100.0 +
        cfg["color_weight"] * result["color_anomaly_score"] +
        cfg["symmetry_weight"] * (1.0 - result["symmetry_score"])) / (
        cfg["dark_weight"] + cfg["color_weight"] + cfg["symmetry_weight"])
    return result


class CudaQualityContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.library_path = Path(self.directory.name) / "fake.dll"
        self.library_path.write_bytes(b"Mock library; never executed")
        self.library = Mock()
        self.library.fg_quality_create.return_value = 1
        self.library.fg_quality_analyze.side_effect = self.fake_analyze
        self.loader = patch("src.vision.cuda_quality.ctypes.CDLL", return_value=self.library).start()
        self.addCleanup(patch.stopall)
        self.roi = np.full((7, 9, 3), 230, dtype=np.uint8)
        self.mask = np.ones((7, 9), dtype=bool)
        self.cfg = dict(QUALITY_DEFAULTS)

    def fake_analyze(self, context, bgr, mask, width, height, tx, ty, tw, th,
                     thresholds, counts, timings, error, capacity):
        for i, count in enumerate((63, 0, 0, 63, 63)):
            counts[i] = count
        for i, timing in enumerate((.1, .2, .3, .4)):
            timings[i] = timing
        return 0

    def create(self, **kwargs):
        result = CudaQualityMeasurements(str(self.library_path), **kwargs)
        self.addCleanup(result.close)
        return result

    def test_missing_library_is_explicit_no_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "library missing"):
            CudaQualityMeasurements(str(self.library_path.with_name("missing.dll")))
        self.loader.assert_not_called()

    def test_invalid_device_rejected_before_loading(self):
        for value in (-1, True, 0.0, "0"):
            with self.assertRaisesRegex(ValueError, "device_index"):
                self.create(device_index=value)
        self.loader.assert_not_called()

    def test_failed_initialization_diagnostic_no_fallback(self):
        def failed_create(device, error, capacity):
            error.value = b"CUDA unavailable"
            return None
        self.library.fg_quality_create.side_effect = failed_create
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            self.create()

    def test_failed_measurement_keeps_timing_empty(self):
        engine = self.create()
        def failed(*args):
            args[-2].value = b"kernel failed"
            return -1
        self.library.fg_quality_analyze.side_effect = failed
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            engine.measure(self.roi, self.mask, self.cfg)
        self.assertEqual(engine.last_timing, {})
        self.assertEqual(engine.last_counts, {})

    def test_invalid_roi_and_mask_rejected_before_cuda(self):
        engine = self.create()
        for roi in (self.roi.astype(np.float32), self.roi[:, :, 0], np.empty((0, 2, 3), np.uint8)):
            with self.assertRaisesRegex(ValueError, "BGR ROI"):
                engine.measure(roi, self.mask, self.cfg)
        for mask in (np.ones((3, 4), bool), self.mask.astype(np.float32)):
            with self.assertRaisesRegex(ValueError, "mask matching"):
                engine.measure(self.roi, mask, self.cfg)
        with self.assertRaisesRegex(ValueError, "empty"):
            engine.measure(self.roi, np.zeros_like(self.mask), self.cfg)
        self.library.fg_quality_analyze.assert_not_called()

    def test_invalid_thresholds_rejected_before_cuda(self):
        engine = self.create()
        for key, value in (("dark_value_max", 300), ("brown_hue_min", float("nan")), ("dark_weight", -1)):
            config = dict(self.cfg)
            config[key] = value
            with self.assertRaises(ValueError):
                engine.measure(self.roi, self.mask, config)
        self.library.fg_quality_analyze.assert_not_called()

    def test_counts_normalization_timing_and_close(self):
        engine = self.create()
        measured = engine.measure(self.roi, self.mask, self.cfg)
        self.assertEqual(measured, {"dark_spot_percentage": 0.0, "color_anomaly_score": 0.0,
                                    "symmetry_score": 1.0, "visual_defect_score": 0.0})
        self.assertEqual(engine.last_counts["foreground"], 63)
        self.assertEqual(engine.last_timing["kernel_ms"], .2)
        self.assertGreater(engine.last_timing["wall_ms"], 0)
        engine.close()
        engine.close()
        self.library.fg_quality_destroy.assert_called_once_with(1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.measure(self.roi, self.mask, self.cfg)

    def test_noncontiguous_roi_and_tight_bounds(self):
        engine = self.create()
        mask = np.zeros((7, 9), np.uint8)
        mask[2:6, 3:8] = 255
        engine.measure(self.roi[:, ::-1, :], mask[:, ::-1], self.cfg)
        args = self.library.fg_quality_analyze.call_args[0]
        self.assertEqual(args[3:9], (9, 7, 1, 2, 5, 4))


@unittest.skipUnless(os.environ.get("FG_RUN_CUDA_TESTS") == "1", "Set FG_RUN_CUDA_TESTS=1 for actual custom CUDA checks")
class CudaQualityDeviceTests(unittest.TestCase):
    def setUp(self):
        self.engine = CudaQualityMeasurements()
        self.addCleanup(self.engine.close)
        self.cfg = dict(QUALITY_DEFAULTS)

    def assert_parity(self, roi, mask, cfg=None):
        cfg = cfg or self.cfg
        measured = self.engine.measure(roi, mask, cfg)
        expected = cpu_reference(roi, mask, cfg)
        for feature in expected:
            self.assertAlmostEqual(measured[feature], expected[feature], places=12, msg=feature)

    def test_random_image_tail_blocks_and_asymmetry(self):
        random = np.random.RandomState(491)
        for height, width in ((1, 1), (7, 9), (31, 257), (127, 263)):
            roi = random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            mask = random.rand(height, width) > .2
            mask[0, 0] = True
            self.assert_parity(roi, mask)

    def test_threshold_boundaries_and_bgr_ties(self):
        # All grayscale values exercise zero hue/saturation and V boundaries.
        gray = np.arange(256, dtype=np.uint8)
        roi = np.tile(gray[None, :, None], (3, 1, 3))
        self.assert_parity(roi, np.ones(roi.shape[:2], bool))
        hsv = np.zeros((180, 256, 3), np.uint8)
        hsv[:, :, 0] = np.arange(180, dtype=np.uint8)[:, None]
        hsv[:, :, 1] = np.arange(256, dtype=np.uint8)[None, :]
        for value in (59, 60, 61, 169, 170, 171, 255):
            hsv[:, :, 2] = value
            self.assert_parity(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.ones(hsv.shape[:2], bool))

    def test_dark_pixels_remain_foreground_and_allocation_reused(self):
        roi = np.full((65, 67, 3), (30, 160, 30), np.uint8)
        mask = np.zeros(roi.shape[:2], bool)
        mask[5:60, 7:62] = True
        roi[20:35, 20:35] = 0
        self.assert_parity(roi, mask)
        self.assertEqual(self.engine.last_counts["dark"], 225)
        self.assertGreater(self.engine.last_timing["allocation_ms"], 0)
        self.assert_parity(roi, mask)
        self.assertEqual(self.engine.last_timing["allocation_ms"], 0)

    def test_requested_unavailable_device_has_no_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            CudaQualityMeasurements(device_index=9999)


if __name__ == "__main__":
    unittest.main()
