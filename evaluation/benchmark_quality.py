"""Compare measured CPU/CUDA quality paths; synthetic inputs measure no accuracy."""
import argparse
import hashlib
import math
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

from evaluation.evaluate import new_report_path, project_path, quality_backend, save_report
from evaluation.manifest import read_foreground_mask, read_frame, read_manifest
from src.utils.config import load_config
from src.vision.quality_analyzer import FEATURE_NAMES, analyze


def synthetic_frames():
    """Controlled mask fixtures; neither dataset nor real-world accuracy evidence."""
    pairs = []
    for side in (128, 256, 512):
        for color, dark_radius in (((25, 35, 210), 0), ((20, 200, 25), side // 12),
                                   ((20, 200, 230), side // 7)):
            frame = np.full((side, side, 3), 235, dtype=np.uint8)
            center = side // 2
            radius = side // 3
            mask = np.zeros((side, side), dtype=np.uint8)
            cv2.circle(frame, (center, center), radius, color, -1)
            cv2.circle(mask, (center, center), radius, 1, -1)
            if dark_radius:
                cv2.circle(frame, (center, center), dark_radius, (15, 15, 15), -1)
            bounds = (center - radius - 7, center - radius - 7,
                      center + radius + 8, center + radius + 8)
            pairs.append(("synthetic_{}_{}".format(side, len(pairs)), frame,
                          SimpleNamespace(bbox=bounds, confidence=.95, mask=mask)))
            if not dark_radius:
                pairs.append(("synthetic_plain_background_{}".format(side), frame,
                              SimpleNamespace(bbox=bounds, confidence=.95, mask=None)))
    return pairs


def summarize(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "mean": float(array.mean()),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}


def benchmark(pairs, config, backend, warmup=10, iterations=100, tolerance=1e-9):
    if warmup < 1 or iterations < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Positive warmup/iterations and finite nonnegative tolerance required")
    if not pairs:
        raise ValueError("No benchmark frames")
    records = []
    mismatch_count = 0
    for name, frame, detection in pairs:
        reference = analyze(frame, detection, config)
        if not reference["validity"]["mask_reliable"]:
            raise ValueError("Benchmark requires reliable foreground: {}: {}".format(name, reference["reasons"]))
        for _ in range(warmup):
            analyze(frame, detection, config)
            analyze(frame, detection, config, measurement_backend=backend)
        cpu_ms = []
        cuda_ms = []
        diagnostic_values = {}
        results = None
        for _ in range(iterations):
            start = perf_counter()
            cpu_result = analyze(frame, detection, config)
            cpu_ms.append(1000 * (perf_counter() - start))
            start = perf_counter()
            cuda_result = analyze(frame, detection, config, measurement_backend=backend)
            cuda_ms.append(1000 * (perf_counter() - start))
            diagnostic = getattr(backend, "last_timing", {})
            for key, value in diagnostic.items():
                if (not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and value >= 0):
                    diagnostic_values.setdefault(key, []).append(value)
            mask_timing = cuda_result.get("mask_cpu_ms")
            if isinstance(mask_timing, (int, float)) and math.isfinite(mask_timing):
                diagnostic_values.setdefault("mask_cpu_ms", []).append(mask_timing)
            errors = {key: abs(cpu_result["features"][key] - cuda_result["features"][key])
                      for key in FEATURE_NAMES}
            matching = (max(errors.values()) <= tolerance and
                        cpu_result["status"] == cuda_result["status"] and
                        cpu_result["validity"] == cuda_result["validity"])
            if not matching:
                mismatch_count += 1
            results = {"cpu_features": cpu_result["features"], "cuda_features": cuda_result["features"],
                       "absolute_errors": errors, "cpu_status": cpu_result["status"],
                       "cuda_status": cuda_result["status"]}
        records.append({"source": name, "frame_shape": list(frame.shape), "bbox": list(detection.bbox),
                        "cpu_quality_wall_ms": summarize(cpu_ms),
                        "cuda_quality_wall_ms": summarize(cuda_ms),
                        "cuda_measurement_diagnostics": {key: summarize(values)
                                                         for key, values in diagnostic_values.items()},
                        "foreground_source": "external_supplied_mask" if detection.mask is not None else "plain_background_estimate",
                        "last_comparison": results})
    return {"application": "FruitGuard Vision", "warmup_per_frame": warmup,
            "iterations_per_frame": iterations, "feature_absolute_tolerance": tolerance,
            "comparison_failures": mismatch_count, "config_snapshot": config, "samples": records,
            "timing_boundaries": {
                "cpu_quality_wall_ms": "CPU ROI validation, foreground estimation/contour, features and decision",
                "cuda_quality_wall_ms": "Same ROI/CPU foreground work plus CUDA transfer, allocation, measurement and decision; bridge synchronizes",
                "cuda_measurement_diagnostics": "Bridge-reported timing boundaries/units; absent metrics are unavailable, not zero",
                "h2d_ms": "CUDA events around host-to-device ROI BGR and foreground mask copies",
                "kernel_ms": "CUDA events around BGR-to-HSV, dark/color tests, mirrored-mask comparison and GPU block reduction",
                "d2h_ms": "CUDA events around GPU partial-count copy to host",
                "allocation_ms": "Host wall time for CUDA buffer allocation/growth; zero if buffers reused",
                "wall_ms": "Bridge wall time including host contiguous copies, allocation, transfers, kernels, synchronization and final host reduction; excludes context initialization",
                "mask_cpu_ms": "CPU wall time around foreground estimate/contour/reliability validation; excludes ROI clipping",
                "excluded": "File decode, YOLO, camera, serial, OpenCV display, sample saving"},
            "claim": "Only measured quality-path timings. No real-apple accuracy or unmeasured acceleration claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cuda-device", type=int, help="Override config classifier.device_index")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.warmup <= 10000 or not 1 <= args.iterations <= 10000:
            raise ValueError("warmup and iterations must be in 1..10000")
        output = project_path(args.output) if args.output else new_report_path("cuda_quality_benchmark")
        if output.exists():
            raise ValueError("Output exists; refusing overwrite: {}".format(output))
        config = load_config(args.config)
        if args.synthetic:
            pairs = synthetic_frames()
        else:
            samples = [sample for sample in read_manifest(project_path(args.manifest)) if sample.split == "test"]
            pairs = []
            for sample in samples:
                frame = read_frame(sample.image)
                mask = read_foreground_mask(sample.mask, frame.shape[:2]) if sample.mask else None
                pairs.append((str(sample.image), frame, SimpleNamespace(
                    bbox=sample.bbox, confidence=sample.confidence, mask=mask)))
        if args.cuda_device is not None and args.cuda_device < 0:
            raise ValueError("CUDA device index must not be negative")
        backend = quality_backend("cuda", args.cuda_device, config)
        try:
            report = benchmark(pairs, config, backend, args.warmup, args.iterations, args.tolerance)
            report["cuda_diagnostic_schema"] = getattr(backend, "timing_description", None)
            report["cuda_library"] = str(backend.library_path)
            report["cuda_library_sha256"] = hashlib.sha256(backend.library_path.read_bytes()).hexdigest()
        finally:
            backend.close()
        report["cuda_device"] = args.cuda_device if args.cuda_device is not None else config["classifier"]["device_index"]
        report["input_kind"] = "synthetic_numerical_and_timing_only" if args.synthetic else "recorded_test_frames"
        save_report(report, output)
        print("Saved benchmark: {}".format(output.resolve()))
        print("Comparison failures: {}".format(report["comparison_failures"]))
        return 1 if report["comparison_failures"] else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print("Benchmark failed: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
