"""Own CUDA pixel measurements with explicit errors and no CPU fallback.

Foreground extraction and reliability checks are performed by quality_analyzer.
This module never infers a silhouette from darkness or claims food safety.
"""
import ctypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_NAMES = (
    "dark_value_max", "brown_hue_min", "brown_hue_max",
    "brown_saturation_min", "brown_saturation_max", "brown_value_min",
    "brown_value_max", "desaturated_saturation_max",
    "desaturated_value_min", "desaturated_value_max",
)


class CudaQualityMeasurements:
    """Reusable CUDA context. Not thread-safe; one owner should call measure.

    kernel_ms covers BGR to HSV, dark/color tests, mirrored mask comparison and
    GPU block reduction. h2d_ms is the stream event interval for ROI/mask copies;
    pageable host staging can extend beyond that interval. d2h_ms transfers
    partial counts to pinned host memory. wall_ms includes full host staging,
    contiguous host copies, allocation (when needed),
    CUDA work/synchronization, final host reduction and score conversion.
    Context/library initialization is outside each measurement's wall_ms.
    """
    backend = "custom_cuda_heuristics"

    def __init__(self, library_path: Optional[str] = None, device_index: int = 0):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("CUDA quality device_index must be a non-negative integer")
        path = Path(library_path) if library_path is not None else ROOT / "build" / "fruitguard_preprocess.dll"
        if not path.is_absolute():
            path = ROOT / path
        self.library_path = path.resolve()
        self.device_index = device_index
        self.last_timing = {}  # type: Dict[str, float]
        self.last_counts = {}  # type: Dict[str, int]
        self._context = None
        self._dll_directories = []
        if not self.library_path.is_file():
            raise RuntimeError("CUDA quality library missing: {}. Run scripts/build_cuda.ps1 or explicitly select CPU quality mode".format(self.library_path))
        # Runtime lookup does not install, upgrade or select a different toolkit.
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (self.library_path.parent,
                              Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin")):
                if directory.is_dir():
                    self._dll_directories.append(os.add_dll_directory(str(directory)))
        try:
            self._library = ctypes.CDLL(str(self.library_path))
            self._configure_abi()
            error = ctypes.create_string_buffer(1024)
            self._context = self._library.fg_quality_create(device_index, error, len(error))
            if not self._context:
                raise RuntimeError("CUDA quality initialization failed: {}".format(error.value.decode("utf-8", "replace")))
        except (OSError, AttributeError) as exc:
            self.close()
            raise RuntimeError("Cannot load custom CUDA quality library {}: {}. Rebuild with DefectAnalysis.cu; there is no silent CPU fallback".format(self.library_path, exc)) from exc
        except RuntimeError:
            self.close()
            raise

    def _configure_abi(self):
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        self._library.fg_quality_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self._library.fg_quality_create.restype = ctypes.c_void_p
        self._library.fg_quality_analyze.argtypes = [
            ctypes.c_void_p, byte_pointer, byte_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.fg_quality_analyze.restype = ctypes.c_int
        self._library.fg_quality_destroy.argtypes = [ctypes.c_void_p]
        self._library.fg_quality_destroy.restype = None

    def measure(self, roi: np.ndarray, mask: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        started = time.perf_counter()
        self.last_timing = {}
        self.last_counts = {}
        if not self._context:
            raise RuntimeError("CUDA quality context is closed")
        if not isinstance(roi, np.ndarray) or roi.dtype != np.uint8 or roi.ndim != 3 or roi.shape[2] != 3 or min(roi.shape[:2]) < 1:
            raise ValueError("Expected a nonempty uint8 BGR ROI")
        if (not isinstance(mask, np.ndarray) or mask.shape != roi.shape[:2]
                or mask.dtype not in (np.dtype("bool"), np.dtype("uint8"))):
            raise ValueError("Expected a boolean or uint8 foreground mask matching ROI dimensions")
        if roi.shape[0] * roi.shape[1] > (2 ** 31 - 1) // 3:
            raise ValueError("ROI too large for CUDA quality indexing")
        from src.vision.quality_analyzer import QUALITY_DEFAULTS, validate_quality_config
        cfg = dict(QUALITY_DEFAULTS)
        cfg.update(config.get("quality", config))
        validate_quality_config(cfg)
        bgr = np.ascontiguousarray(roi)
        foreground = np.ascontiguousarray(mask != 0, dtype=np.uint8)
        tx, ty, tw, th = cv2.boundingRect(foreground)
        if not tw or not th:
            raise ValueError("Foreground mask is empty")
        thresholds = (ctypes.c_int * len(THRESHOLD_NAMES))(*[cfg[name] for name in THRESHOLD_NAMES])
        counts = (ctypes.c_uint64 * 5)()
        timings = (ctypes.c_double * 4)()
        error = ctypes.create_string_buffer(1024)
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        status = self._library.fg_quality_analyze(
            self._context, bgr.ctypes.data_as(byte_pointer), foreground.ctypes.data_as(byte_pointer),
            bgr.shape[1], bgr.shape[0], tx, ty, tw, th, thresholds, counts, timings, error, len(error))
        if status:
            raise RuntimeError("CUDA quality analysis failed: {}".format(error.value.decode("utf-8", "replace")))
        total, dark, color, intersection, union = [int(value) for value in counts]
        if total <= 0 or union <= 0:
            raise RuntimeError("CUDA quality returned invalid empty foreground counts")
        features = {"dark_spot_percentage": 100.0 * dark / total,
                    "color_anomaly_score": float(color) / total,
                    "symmetry_score": float(intersection) / union}
        weighted = (cfg["dark_weight"] * features["dark_spot_percentage"] / 100.0 +
                    cfg["color_weight"] * features["color_anomaly_score"] +
                    cfg["symmetry_weight"] * (1.0 - features["symmetry_score"]))
        weight_sum = cfg["dark_weight"] + cfg["color_weight"] + cfg["symmetry_weight"]
        features["visual_defect_score"] = float(min(1.0, max(0.0, weighted / weight_sum)))
        self.last_counts = dict(zip(("foreground", "dark", "color", "symmetry_intersection", "symmetry_union"),
                                   [total, dark, color, intersection, union]))
        self.last_timing = dict(zip(("h2d_ms", "kernel_ms", "d2h_ms", "allocation_ms"),
                                   [float(value) for value in timings]))
        self.last_timing["wall_ms"] = (time.perf_counter() - started) * 1000.0
        return features

    def close(self):
        if self._context:
            self._library.fg_quality_destroy(self._context)
            self._context = None
        for directory in self._dll_directories:
            directory.close()
        self._dll_directories = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
