"""Uncalibrated visible-quality heuristics for a controlled, plain background.

The foreground estimate uses corner background colour, not brightness, then
fills the selected contour. Dark patches inside the apple therefore remain in
the measurement denominator. Neither these rules nor detector confidence
describe freshness, infection, contamination, or food safety.
"""

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


QUALITY_DEFAULTS = {
    "min_roi_size": 32,
    "min_foreground_pixels": 300,
    "min_foreground_fraction": 0.20,
    "max_foreground_fraction": 0.94,
    "background_corner_fraction": 0.12,
    "background_max_variation": 18.0,
    "background_variation_percentile": 85.0,
    "frame_border_margin_pixels": 2,
    "foreground_lab_distance": 22.0,
    "min_component_dominance": 0.85,
    "min_background_separation": 24.0,
    "max_boundary_fraction": 0.18,
    "min_contour_solidity": 0.85,
    "max_contour_aspect_ratio": 1.90,
    "dark_value_max": 60,
    "brown_hue_min": 8,
    "brown_hue_max": 25,
    "brown_saturation_min": 35,
    "brown_saturation_max": 220,
    "brown_value_min": 60,
    "brown_value_max": 170,
    "desaturated_saturation_max": 25,
    "desaturated_value_min": 60,
    "desaturated_value_max": 170,
    "dark_weight": 0.45,
    "color_weight": 0.30,
    "symmetry_weight": 0.25,
    "good_detection_confidence": 0.60,
    "max_clipped_fraction": 0.08,
    "good_dark_percentage": 3.0,
    "strong_dark_percentage": 12.0,
    "good_color_anomaly": 0.06,
    "strong_color_anomaly": 0.25,
    "good_symmetry": 0.85,
    "strong_symmetry": 0.65,
    "good_defect_score": 0.10,
    "strong_defect_score": 0.25,
}

FEATURE_NAMES = (
    "color_anomaly_score", "dark_spot_percentage", "symmetry_score",
    "visual_defect_score",
)


def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    settings = dict(QUALITY_DEFAULTS)
    settings.update(config.get("quality", config))
    return settings


def validate_quality_config(config: Dict[str, Any]) -> None:
    """Reject invalid ranges/orderings before any camera or hardware is opened."""
    if not isinstance(config, dict) or set(config) != set(QUALITY_DEFAULTS):
        raise ValueError("quality configuration has missing or unknown keys")
    for name, default in QUALITY_DEFAULTS.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("quality.{} must be a finite number".format(name))
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError("quality.{} must be an integer".format(name))
        if value < 0:
            raise ValueError("quality.{} must not be negative".format(name))
    unit_keys = ("min_foreground_fraction", "max_foreground_fraction",
                 "min_component_dominance", "max_boundary_fraction", "min_contour_solidity",
                 "good_detection_confidence", "max_clipped_fraction", "good_color_anomaly",
                 "strong_color_anomaly", "good_symmetry", "strong_symmetry",
                 "good_defect_score", "strong_defect_score")
    for name in unit_keys:
        if not 0 <= config[name] <= 1:
            raise ValueError("quality.{} must be in 0..1".format(name))
    for name in ("good_dark_percentage", "strong_dark_percentage"):
        if config[name] > 100:
            raise ValueError("quality.{} must be in 0..100".format(name))
    for name in ("brown_hue_min", "brown_hue_max"):
        if config[name] > 179:
            raise ValueError("quality.{} must be in OpenCV HSV hue range 0..179".format(name))
    hsv_keys = ("dark_value_max", "brown_saturation_min", "brown_saturation_max",
                "brown_value_min", "brown_value_max", "desaturated_saturation_max",
                "desaturated_value_min", "desaturated_value_max")
    for name in hsv_keys:
        if config[name] > 255:
            raise ValueError("quality.{} must be in 0..255".format(name))
    if config["min_roi_size"] < 4 or config["min_foreground_pixels"] < 1:
        raise ValueError("quality ROI dimensions and minimum foreground pixels must be positive")
    if not 0 < config["background_variation_percentile"] <= 100:
        raise ValueError("quality.background_variation_percentile must be in (0, 100]")
    if not 0 < config["background_corner_fraction"] <= .25:
        raise ValueError("quality.background_corner_fraction must be in (0, 0.25]")
    for name in ("background_max_variation", "foreground_lab_distance", "min_background_separation"):
        if not 0 < config[name] <= math.sqrt(3) * 255:
            raise ValueError("quality.{} must be a positive Lab distance up to sqrt(3)*255".format(name))
    if not 0 < config["min_foreground_fraction"] < config["max_foreground_fraction"] < 1:
        raise ValueError("quality foreground fraction bounds must satisfy 0 < min < max < 1")
    if config["min_component_dominance"] <= .5:
        raise ValueError("quality.min_component_dominance must exceed 0.5")
    if config["min_contour_solidity"] <= 0 or config["max_contour_aspect_ratio"] < 1:
        raise ValueError("quality contour solidity/aspect bounds are invalid")
    if sum(config[name] for name in ("dark_weight", "color_weight", "symmetry_weight")) <= 0:
        raise ValueError("quality weights must have a positive total")
    ascending_pairs = (("good_dark_percentage", "strong_dark_percentage"),
                       ("good_color_anomaly", "strong_color_anomaly"),
                       ("good_defect_score", "strong_defect_score"),
                       ("brown_hue_min", "brown_hue_max"),
                       ("brown_saturation_min", "brown_saturation_max"),
                       ("brown_value_min", "brown_value_max"),
                       ("desaturated_value_min", "desaturated_value_max"))
    for lower, upper in ascending_pairs:
        if config[lower] >= config[upper]:
            raise ValueError("quality.{} must be less than {}".format(lower, upper))
    if config["strong_symmetry"] >= config["good_symmetry"]:
        raise ValueError("quality.strong_symmetry must be less than good_symmetry")


def clip_roi(frame: np.ndarray, bbox: Sequence[float], min_size: int = 32
             ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    """Clip xyxy coordinates; return fraction of requested box outside frame.

    Invalid boxes return (None, None, 1). Boxes too small for inspection still
    return their clipped coordinates for UI/metadata, but have no valid ROI.
    """
    if frame is None or frame.ndim < 2 or min(frame.shape[:2]) < 1:
        return None, None, 1.0
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return None, None, 1.0
    if len(coordinates) != 4 or not all(math.isfinite(x) for x in coordinates):
        return None, None, 1.0
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None, None, 1.0
    height, width = frame.shape[:2]
    cx1, cy1 = max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1))
    cx2, cy2 = max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None, 1.0
    retained_fraction = ((cx2 - cx1) / (x2 - x1)) * ((cy2 - cy1) / (y2 - y1))
    clipped_fraction = min(1.0, max(0.0, 1.0 - retained_fraction))
    bounds = (int(math.floor(cx1)), int(math.floor(cy1)),
              int(math.ceil(cx2)), int(math.ceil(cy2)))
    ix1, iy1, ix2, iy2 = bounds
    if ix2 - ix1 < min_size or iy2 - iy1 < min_size:
        return None, bounds, clipped_fraction
    return frame[iy1:iy2, ix1:ix2], bounds, clipped_fraction


def _filled_component(raw_mask: np.ndarray, cfg: Dict[str, Any]
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    raw = np.ascontiguousarray(raw_mask.astype(np.uint8))
    count = int(np.count_nonzero(raw))
    if count < cfg["min_foreground_pixels"]:
        return None, None, "Insufficient foreground pixels"
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if nlabels <= 1:
        return None, None, "Foreground mask unavailable"
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if float(stats[index, cv2.CC_STAT_AREA]) / count < cfg["min_component_dominance"]:
        return None, None, "Foreground has multiple substantial components"
    component = (labels == index).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    # Fill holes rather than discarding dark or background-coloured apple spots.
    return filled.astype(bool), contour, None


def _foreground(roi: np.ndarray, supplied: Optional[np.ndarray], cfg: Dict[str, Any]
                ) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[str]]:
    info = {"mask_reliable": False, "mask_source": "model" if supplied is not None
            else "plain_background_estimate", "foreground_pixels": 0}
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    separation = None
    if supplied is None:
        height, width = roi.shape[:2]
        patch = max(1, int(min(height, width) * cfg["background_corner_fraction"]))
        corners = np.concatenate([lab[:patch, :patch].reshape(-1, 3),
                                  lab[:patch, -patch:].reshape(-1, 3),
                                  lab[-patch:, :patch].reshape(-1, 3),
                                  lab[-patch:, -patch:].reshape(-1, 3)])
        background = np.median(corners, axis=0)
        corner_distance = np.linalg.norm(corners - background, axis=1)
        if float(np.percentile(corner_distance, cfg["background_variation_percentile"])) > cfg["background_max_variation"]:
            return None, info, "Foreground mask unreliable: background corners vary"
        separation = np.linalg.norm(lab - background, axis=2)
        raw = separation >= cfg["foreground_lab_distance"]
    else:
        raw = supplied
    mask, contour, error = _filled_component(raw, cfg)
    if error:
        return None, info, "Foreground mask unreliable: " + error
    count = int(np.count_nonzero(mask))
    info["foreground_pixels"] = count
    fraction = float(count) / mask.size
    if count < cfg["min_foreground_pixels"] or not (
            cfg["min_foreground_fraction"] <= fraction <= cfg["max_foreground_fraction"]):
        return None, info, "Foreground mask unreliable: implausible foreground area"
    if separation is not None and float(np.median(separation[mask])) < cfg["min_background_separation"]:
        return None, info, "Foreground mask unreliable: insufficient background separation"
    # Border contact is a contour measurement, not a fraction of box pixels.
    # Use full contour points so CHAIN_APPROX_SIMPLE does not bias this ratio.
    full_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
    full_points = max(full_contours, key=cv2.contourArea).reshape(-1, 2)
    on_border = ((full_points[:, 0] == 0) | (full_points[:, 0] == mask.shape[1] - 1) |
                 (full_points[:, 1] == 0) | (full_points[:, 1] == mask.shape[0] - 1))
    if float(np.mean(on_border)) > cfg["max_boundary_fraction"]:
        return None, info, "Foreground mask unreliable: contour reaches ROI boundary"
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0.0
    _, _, width, height = cv2.boundingRect(contour)
    aspect = max(width, height) / float(min(width, height))
    if solidity < cfg["min_contour_solidity"] or aspect > cfg["max_contour_aspect_ratio"]:
        return None, info, "Foreground mask unreliable: shape suggests occlusion or contamination"
    info["mask_reliable"] = True
    return mask, info, None


def decide_quality(confidence: float, features: Dict[str, Optional[float]],
                   validity: Dict[str, Any], clipped_fraction: float,
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Reliability gates take priority over any visual defect threshold."""
    cfg = _settings(config)
    reasons = []
    if not math.isfinite(confidence) or confidence < cfg["good_detection_confidence"]:
        reasons.append("Low detection confidence")
    if not math.isfinite(clipped_fraction) or clipped_fraction > cfg["max_clipped_fraction"]:
        reasons.append("Apple partly outside frame")
    if not validity.get("mask_reliable", False):
        reasons.append("Foreground mask unreliable")
    for name in FEATURE_NAMES:
        value = features.get(name)
        maximum = 100.0 if name == "dark_spot_percentage" else 1.0
        if (not validity.get(name, False) or value is None or not math.isfinite(value)
                or not 0 <= value <= maximum):
            reasons.append("Measurement unavailable: " + name)
    if reasons:
        return {"status": "REVIEW", "reasons": reasons}
    strong = []
    if features["dark_spot_percentage"] >= cfg["strong_dark_percentage"]:
        strong.append("Dark-region percentage exceeds strong-defect threshold")
    if features["color_anomaly_score"] >= cfg["strong_color_anomaly"]:
        strong.append("Color anomaly exceeds strong-defect threshold")
    if features["symmetry_score"] <= cfg["strong_symmetry"]:
        strong.append("Shape asymmetry exceeds strong-defect threshold")
    if features["visual_defect_score"] >= cfg["strong_defect_score"]:
        strong.append("Combined visual defect score exceeds strong-defect threshold")
    if strong:
        return {"status": "DEFECTIVE", "reasons": strong}
    if features["dark_spot_percentage"] > cfg["good_dark_percentage"]:
        reasons.append("Dark-region percentage exceeds GOOD threshold")
    if features["color_anomaly_score"] > cfg["good_color_anomaly"]:
        reasons.append("Color anomaly exceeds GOOD threshold")
    if features["symmetry_score"] < cfg["good_symmetry"]:
        reasons.append("Shape symmetry below GOOD threshold")
    if features["visual_defect_score"] > cfg["good_defect_score"]:
        reasons.append("Combined visual defect score exceeds GOOD threshold")
    if reasons:
        return {"status": "REVIEW", "reasons": reasons}
    return {"status": "GOOD", "reasons": ["Visible anomalies below configured limits (uncalibrated)"]}


def measure_features_cpu(roi: np.ndarray, mask: np.ndarray,
                         config: Dict[str, Any]) -> Dict[str, float]:
    """CPU reference for the custom CUDA classifier's numerical validation."""
    cfg = _settings(config)
    if roi.dtype != np.uint8 or roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError('Expected a uint8 BGR ROI')
    if mask.shape != roi.shape[:2] or not np.any(mask):
        raise ValueError('A matching nonempty foreground mask is required')
    mask = np.asarray(mask, dtype=bool)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = [hsv[:, :, index] for index in range(3)]
    dark = value < cfg['dark_value_max']
    brown = ((hue >= cfg['brown_hue_min']) & (hue <= cfg['brown_hue_max']) &
             (saturation >= cfg['brown_saturation_min']) &
             (saturation <= cfg['brown_saturation_max']) &
             (value >= cfg['brown_value_min']) & (value <= cfg['brown_value_max']))
    desaturated = ((saturation <= cfg['desaturated_saturation_max']) &
                   (value >= cfg['desaturated_value_min']) &
                   (value <= cfg['desaturated_value_max']))
    count = float(np.count_nonzero(mask))
    dark_percentage = 100.0 * float(np.count_nonzero(dark & mask)) / count
    color_anomaly = float(np.count_nonzero((brown | desaturated) & mask)) / count
    ys, xs = np.nonzero(mask)
    tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    reflected = np.fliplr(tight)
    symmetry = float(np.count_nonzero(tight & reflected)) / float(np.count_nonzero(tight | reflected))
    weights = cfg['dark_weight'] + cfg['color_weight'] + cfg['symmetry_weight']
    combined = (cfg['dark_weight'] * dark_percentage / 100.0 +
                cfg['color_weight'] * color_anomaly + cfg['symmetry_weight'] * (1.0 - symmetry)) / weights
    return {'dark_spot_percentage': dark_percentage, 'color_anomaly_score': color_anomaly,
            'symmetry_score': symmetry, 'visual_defect_score': min(1.0, max(0.0, combined))}


def analyze(frame: np.ndarray, detection: Any, config: Dict[str, Any],
            measurement_backend: Optional[Any] = None) -> Dict[str, Any]:
    """Measure only the current detection and frame; never retain fruit state."""
    cfg = _settings(config)
    features = {name: None for name in FEATURE_NAMES}
    validity = {name: False for name in FEATURE_NAMES}
    validity.update({"mask_reliable": False, "mask_source": "unavailable", "foreground_pixels": 0})
    roi, bounds, clipped_fraction = clip_roi(frame, detection.bbox, cfg["min_roi_size"])
    error = None
    supplied = None
    model_mask = getattr(detection, "mask", None)
    measurement_timing = None
    mask_cpu_ms = None
    if roi is None:
        error = "Apple ROI invalid or too small"
    elif frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        error = "Expected uint8 BGR camera frame"
    elif (bounds[0] <= cfg["frame_border_margin_pixels"] or
          bounds[1] <= cfg["frame_border_margin_pixels"] or
          bounds[2] >= frame.shape[1] - cfg["frame_border_margin_pixels"] or
          bounds[3] >= frame.shape[0] - cfg["frame_border_margin_pixels"]):
        # Detectors commonly return boxes already clipped to the frame, so the
        # requested-box area loss alone cannot identify a cut-off apple.
        error = "Apple at frame boundary; possible clipping"
    elif model_mask is not None:
        mask_array = np.asarray(model_mask)
        numeric_or_bool = (np.issubdtype(mask_array.dtype, np.number) or mask_array.dtype == np.bool_)
        if mask_array.shape != frame.shape[:2] or not numeric_or_bool:
            error = "Foreground mask unreliable: model mask must match frame dimensions"
        elif not np.all(np.isfinite(mask_array)):
            error = "Foreground mask unreliable: model mask contains non-finite values"
        else:
            x1, y1, x2, y2 = bounds
            supplied = mask_array[y1:y2, x1:x2] > 0
    if error is None:
        mask_start = time.perf_counter()
        mask, mask_info, error = _foreground(roi, supplied, cfg)
        mask_cpu_ms = (time.perf_counter() - mask_start) * 1000.0
        validity.update(mask_info)
        if mask is not None:
            start = time.perf_counter()
            features = measure_features_cpu(roi, mask, cfg) if measurement_backend is None else measurement_backend.measure(roi, mask, cfg)
            measurement_timing = {'wall_ms': (time.perf_counter() - start) * 1000.0}
            if measurement_backend is not None:
                measurement_timing.update(getattr(measurement_backend, 'last_timing', {}))
            validity.update({name: True for name in FEATURE_NAMES})
    result = decide_quality(float(detection.confidence), features, validity, clipped_fraction, cfg)
    if error:
        result["reasons"].insert(0, error)
    result.update({"features": features, "validity": validity, "bbox": bounds,
                   "clipped_fraction": clipped_fraction,
                   "measurement_backend": 'cpu' if measurement_backend is None else 'cuda',
                   "mask_cpu_ms": mask_cpu_ms,
                   "measurement_timing": measurement_timing})
    return result
