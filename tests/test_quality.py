"""Synthetic checks verify rule/mask behavior, never real apple accuracy."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.vision.quality_analyzer import (QUALITY_DEFAULTS, analyze, clip_roi,
                                       decide_quality, validate_quality_config)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((160, 180, 3), 235, dtype=np.uint8)
        cv2.circle(self.frame, (90, 80), 45, (30, 40, 210), -1)
        self.detection = SimpleNamespace(bbox=(35, 25, 145, 135), confidence=.90,
                                         class_name="apple", mask=None)
        self.features = {"color_anomaly_score": .0, "dark_spot_percentage": .0,
                         "symmetry_score": 1.0, "visual_defect_score": .0}
        self.validity = {name: True for name in self.features}
        self.validity["mask_reliable"] = True

    def decide(self, **kwargs):
        return decide_quality(kwargs.get("confidence", .90), self.features,
                              self.validity, kwargs.get("clipped_fraction", 0.0), {})

    def test_roi_clips_and_reports_lost_area(self):
        roi, bounds, clipped = clip_roi(self.frame, (-20, 20, 80, 120))
        self.assertEqual(bounds, (0, 20, 80, 120))
        self.assertEqual(roi.shape[:2], (100, 80))
        self.assertAlmostEqual(clipped, .20)

    def test_invalid_roi(self):
        for box in [(float("nan"), 0, 100, 100), (0, 0, float("inf"), 10),
                    (10, 10, 0, 0), (200, 0, 230, 50), (1, 2, 3), None]:
            self.assertIsNone(clip_roi(self.frame, box)[0])
        roi, bounds, _ = clip_roi(self.frame, (0, 0, 10, 10))
        self.assertIsNone(roi)
        self.assertEqual(bounds, (0, 0, 10, 10))

    def test_plain_circle_is_valid_and_good(self):
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["status"], "GOOD", result["reasons"])
        self.assertTrue(result["validity"]["mask_reliable"])
        self.assertEqual(result["features"]["symmetry_score"], 1.0)
        self.assertLess(result["validity"]["foreground_pixels"], 110 * 110)

    def test_dark_patch_is_included_and_uses_foreground_denominator(self):
        baseline = analyze(self.frame, self.detection, {})
        cv2.circle(self.frame, (90, 80), 20, (15, 15, 15), -1)
        result = analyze(self.frame, self.detection, {})
        self.assertTrue(result["validity"]["mask_reliable"])
        self.assertEqual(result["validity"]["foreground_pixels"], baseline["validity"]["foreground_pixels"])
        self.assertGreater(result["features"]["dark_spot_percentage"], 19.0)
        self.assertEqual(result["status"], "DEFECTIVE")

    def test_dark_patch_same_as_background_is_filled(self):
        self.frame[:] = (15, 15, 15)
        cv2.circle(self.frame, (90, 80), 45, (30, 40, 210), -1)
        cv2.circle(self.frame, (90, 80), 20, (15, 15, 15), -1)
        result = analyze(self.frame, self.detection, {})
        self.assertTrue(result["validity"]["mask_reliable"])
        self.assertGreater(result["features"]["dark_spot_percentage"], 19.0)

    def test_bright_natural_red_green_yellow_not_color_defects(self):
        for color in [(30, 40, 210), (20, 210, 30), (20, 210, 230)]:
            cv2.circle(self.frame, (90, 80), 45, color, -1)
            result = analyze(self.frame, self.detection, {})
            self.assertEqual(result["features"]["color_anomaly_score"], 0.0)

    def test_varied_background_is_review_with_na_measurements(self):
        self.frame[25:38, 35:48] = (220, 10, 10)
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["status"], "REVIEW")
        self.assertFalse(result["validity"]["mask_reliable"])
        self.assertIsNone(result["features"]["symmetry_score"])

    def test_rectangle_is_not_assumed_to_be_apple_silhouette(self):
        self.frame[:] = (30, 40, 210)
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["features"]["visual_defect_score"])

    def test_supplied_mask_used_even_with_varied_background(self):
        self.frame[25:38, 35:48] = (220, 10, 10)
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (90, 80), 45, 1, -1)
        self.detection.mask = mask
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["status"], "GOOD")
        self.assertEqual(result["validity"]["mask_source"], "model")

    def test_bad_supplied_mask_is_review_not_fallback(self):
        for mask in [np.ones((20, 20), dtype=np.uint8),
                     np.full(self.frame.shape[:2], np.nan),
                     np.ones(self.frame.shape[:2], dtype=np.uint8)]:
            self.detection.mask = mask
            self.assertEqual(analyze(self.frame, self.detection, {})["status"], "REVIEW")

    def test_good_boundaries_are_inclusive(self):
        self.features.update({"dark_spot_percentage": QUALITY_DEFAULTS["good_dark_percentage"],
                              "color_anomaly_score": QUALITY_DEFAULTS["good_color_anomaly"],
                              "symmetry_score": QUALITY_DEFAULTS["good_symmetry"],
                              "visual_defect_score": QUALITY_DEFAULTS["good_defect_score"]})
        self.assertEqual(self.decide(confidence=.60, clipped_fraction=.08)["status"], "GOOD")

    def test_intermediate_and_strong_rule_boundaries(self):
        pairs = [("dark_spot_percentage", 3.01, 12.0),
                 ("color_anomaly_score", .061, .25),
                 ("symmetry_score", .849, .65),
                 ("visual_defect_score", .101, .25)]
        for name, intermediate, strong in pairs:
            original = self.features[name]
            self.features[name] = intermediate
            self.assertEqual(self.decide()["status"], "REVIEW", name)
            self.features[name] = strong
            self.assertEqual(self.decide()["status"], "DEFECTIVE", name)
            self.features[name] = original

    def test_uncertain_measurement_precedes_strong_defect(self):
        self.features["dark_spot_percentage"] = 80.0
        self.assertEqual(self.decide(confidence=.59)["status"], "REVIEW")
        self.assertEqual(self.decide(clipped_fraction=.081)["status"], "REVIEW")
        self.validity["symmetry_score"] = False
        self.assertEqual(self.decide()["status"], "REVIEW")
        self.validity["symmetry_score"] = True
        self.validity["mask_reliable"] = False
        self.assertEqual(self.decide()["status"], "REVIEW")

    def test_measurements_have_stated_ranges(self):
        result = analyze(self.frame, self.detection, {})
        for name, value in result["features"].items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100 if name == "dark_spot_percentage" else 1)

    def test_nonfinite_or_out_of_range_measurement_is_review(self):
        for value in (float("nan"), float("inf"), -1.0, 1.1):
            self.features["color_anomaly_score"] = value
            self.assertEqual(self.decide()["status"], "REVIEW")

    def test_occluded_contour_is_review(self):
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (90, 80), 45, 1, -1)
        mask[45:85, 90:145] = 0
        self.detection.mask = mask
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["status"], "REVIEW")
        self.assertFalse(result["validity"]["mask_reliable"])

    def test_already_clipped_detector_box_at_frame_edge_is_review(self):
        self.detection.bbox = (0, 25, 145, 135)
        result = analyze(self.frame, self.detection, {})
        self.assertEqual(result["clipped_fraction"], 0.0)
        self.assertEqual(result["status"], "REVIEW")
        self.assertFalse(result["validity"]["mask_reliable"])
        self.assertTrue(any("frame boundary" in reason for reason in result["reasons"]))

    def test_configuration_validation(self):
        validate_quality_config(dict(QUALITY_DEFAULTS))
        for name, value in (("good_dark_percentage", 101),
                            ("good_color_anomaly", .5),
                            ("strong_symmetry", .9),
                            ("foreground_lab_distance", 0),
                            ("max_foreground_fraction", .1),
                            ("background_corner_fraction", .5),
                            ("background_variation_percentile", 101),
                            ("frame_border_margin_pixels", -1),
                            ("brown_hue_max", 180),
                            ("min_roi_size", 4.5),
                            ("dark_value_max", True),
                            ("dark_weight", float("nan"))):
            config = dict(QUALITY_DEFAULTS)
            config[name] = value
            with self.assertRaises(ValueError, msg=name):
                validate_quality_config(config)
        config = dict(QUALITY_DEFAULTS)
        config.update(dark_weight=0, color_weight=0, symmetry_weight=0)
        with self.assertRaises(ValueError):
            validate_quality_config(config)


if __name__ == "__main__":
    unittest.main()
