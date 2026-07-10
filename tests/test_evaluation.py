"""Numerical/provenance tests; synthetic images establish no apple accuracy."""
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluation.benchmark_quality import benchmark, synthetic_frames
from evaluation.evaluate import evaluate, save_report
from evaluation.manifest import read_foreground_mask, read_manifest
from evaluation.metrics import classification_metrics, segmentation_metrics
from src.utils.config import DEFAULTS
from src.vision.quality_analyzer import measure_features_cpu


class MetricTests(unittest.TestCase):
    def test_confusion_axes_and_numerical_rates(self):
        metrics = classification_metrics(["GOOD", "GOOD", "DEFECTIVE", "REVIEW"],
                                         ["GOOD", "REVIEW", "DEFECTIVE", "GOOD"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(metrics["per_class"]["GOOD"]["precision"], .5)
        self.assertEqual(metrics["per_class"]["GOOD"]["recall"], .5)
        self.assertEqual(metrics["per_class"]["GOOD"]["f1"], .5)
        self.assertEqual(metrics["macro"]["f1"], .5)
        self.assertEqual(metrics["weighted"]["f1"], .5)
        self.assertEqual(metrics["review_fraction"], .25)
        self.assertEqual(metrics["decisive_coverage"], .75)

    def test_absent_class_has_defined_zero_metrics(self):
        result = classification_metrics(["GOOD"], ["GOOD"])
        self.assertEqual(result["per_class"]["DEFECTIVE"]["precision"], 0)
        self.assertEqual(result["per_class"]["DEFECTIVE"]["recall"], 0)

    def test_bad_classification_inputs_rejected(self):
        for truth, predicted in (([], []), (["GOOD"], []), (["fresh"], ["GOOD"])):
            with self.assertRaises(ValueError):
                classification_metrics(truth, predicted)

    def test_supplied_mask_iou_dice(self):
        truth = np.array([[1, 1], [0, 0]])
        predicted = np.array([[1, 0], [1, 0]])
        metrics = segmentation_metrics(truth, predicted)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["dice"], .5)

    def test_empty_mask_policy(self):
        empty = np.zeros((2, 2))
        self.assertEqual(segmentation_metrics(empty, empty), {"iou": 1, "dice": 1})
        self.assertEqual(segmentation_metrics(empty, np.ones((2, 2))), {"iou": 0, "dice": 0})

    def test_mask_invalid_dimensions_values_and_thresholds(self):
        valid = np.ones((2, 2))
        for invalid in (np.ones((3, 2)), np.ones((2, 2, 1)), np.zeros((0, 0)),
                        np.full((2, 2), np.nan), np.full((2, 2), 2), np.full((2, 2), -1)):
            with self.assertRaises(ValueError):
                segmentation_metrics(valid, invalid)
        for threshold in (0, -1, 2, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                segmentation_metrics(valid, valid, threshold)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "samples.csv"
        self.frame = np.full((160, 180, 3), 235, dtype=np.uint8)
        cv2.circle(self.frame, (90, 80), 45, (30, 40, 210), -1)
        cv2.imwrite(str(self.root / "one.png"), self.frame)
        second = self.frame.copy()
        second[0, 0] = (1, 2, 3)
        cv2.imwrite(str(self.root / "two.png"), second)
        self.row = {"image": "one.png", "label": "GOOD", "fruit_id": "apple-1",
                    "split": "test", "x1": 35, "y1": 25, "x2": 145,
                    "y2": 135, "confidence": .9, "mask": ""}
        self.config = copy.deepcopy(DEFAULTS)

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_manifest_resolves_relative_paths(self):
        self.write_rows([self.row])
        sample = read_manifest(self.manifest)[0]
        self.assertEqual(sample.image, self.root / "one.png")
        self.assertEqual(sample.bbox, (35, 25, 145, 135))

    def test_fruit_id_cross_split_rejected_before_filtering_test(self):
        second = dict(self.row, image="two.png", split="train")
        self.write_rows([self.row, second])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            read_manifest(self.manifest)

    def test_duplicate_content_different_fruit_id_rejected(self):
        (self.root / "two.png").write_bytes((self.root / "one.png").read_bytes())
        self.write_rows([self.row, dict(self.row, image="two.png", fruit_id="other", split="train")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_manifest(self.manifest)

    def test_missing_corrupt_image_unknown_label_or_detection_rejected(self):
        for update in ({"image": "missing.png"}, {"label": "FRESH"}, {"confidence": "nan"},
                       {"confidence": 1.1}, {"x1": 145}, {"split": "holdout"}, {"fruit_id": ""}):
            self.write_rows([dict(self.row, **update)])
            with self.assertRaises(ValueError):
                read_manifest(self.manifest)

    def test_mask_shape_and_binary_values_validated(self):
        cv2.imwrite(str(self.root / "mask.png"), np.zeros((10, 10), dtype=np.uint8))
        self.write_rows([dict(self.row, mask="mask.png")])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            read_manifest(self.manifest)
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        mask[1, 1] = 30
        cv2.imwrite(str(self.root / "mask.png"), mask)
        with self.assertRaisesRegex(ValueError, "binary"):
            read_manifest(self.manifest)

    def test_evaluation_test_only_and_synthetic_marking(self):
        second = dict(self.row, image="two.png", fruit_id="apple-2", split="train")
        self.write_rows([self.row, second])
        report = evaluate(self.manifest, self.config, synthetic_smoke=True)
        self.assertEqual(report["metrics"]["sample_count"], 1)
        self.assertEqual(report["audited_rows_all_splits"], 2)
        self.assertEqual(report["samples"][0]["result"]["status"], "GOOD")
        self.assertEqual(report["evaluation_kind"], "synthetic_behavior_check")

    def test_no_test_split_no_metrics(self):
        self.write_rows([dict(self.row, split="train")])
        with self.assertRaisesRegex(ValueError, "no test"):
            evaluate(self.manifest, self.config)

    def test_output_overwrite_refused(self):
        output = self.root / "report.json"
        save_report({"test": True}, output)
        with self.assertRaises(FileExistsError):
            save_report({"test": False}, output)
        self.assertEqual(json.loads(output.read_text())["test"], True)

    def test_actual_cli_synthetic_smoke(self):
        self.write_rows([self.row])
        output = self.root / "smoke.json"
        completed = subprocess.run([sys.executable, "-m", "evaluation.evaluate",
                                    "--manifest", str(self.manifest), "--backend", "cpu",
                                    "--synthetic-smoke", "--output", str(output)],
                                   cwd=str(Path(__file__).resolve().parents[1]),
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["evaluation_kind"], "synthetic_behavior_check")
        self.assertEqual(report["measurement_backend"], "cpu")


class BenchmarkTests(unittest.TestCase):
    def test_parity_reporting_with_reference_backend(self):
        class ReferenceBackend:
            last_timing = {"wall_ms": 1.0}

            def measure(self, roi, mask, config):
                return measure_features_cpu(roi, mask, config)

        report = benchmark(synthetic_frames()[:1], copy.deepcopy(DEFAULTS),
                           ReferenceBackend(), warmup=1, iterations=2, tolerance=0)
        self.assertEqual(report["comparison_failures"], 0)
        self.assertEqual(report["samples"][0]["cpu_quality_wall_ms"]["count"], 2)
        self.assertIn("mask_cpu_ms", report["samples"][0]["cuda_measurement_diagnostics"])


if __name__ == "__main__":
    unittest.main()
