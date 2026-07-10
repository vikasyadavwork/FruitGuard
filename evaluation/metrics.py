"""Small auditable classification and supplied-mask metrics, without a model."""
import math
from typing import Dict, List, Sequence

import numpy as np

LABELS = ("GOOD", "REVIEW", "DEFECTIVE")


def classification_metrics(truth: Sequence[str], predicted: Sequence[str]) -> Dict:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Truth and predictions must be equally sized and nonempty")
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for expected, actual in zip(truth, predicted):
        if expected not in LABELS or actual not in LABELS:
            raise ValueError("Unknown visible-quality label")
        matrix[LABELS.index(expected), LABELS.index(actual)] += 1
    per_class = {}
    for index, name in enumerate(LABELS):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": support, "predicted_count": predicted_count}
    n = len(truth)
    macro = {key: sum(row[key] for row in per_class.values()) / len(LABELS)
             for key in ("precision", "recall", "f1")}
    weighted = {key: sum(row[key] * row["support"] for row in per_class.values()) / n
                for key in ("precision", "recall", "f1")}
    review_count = sum(label == "REVIEW" for label in predicted)
    return {"labels": list(LABELS), "confusion_matrix": matrix.tolist(),
            "confusion_matrix_axes": {"rows": "human truth", "columns": "prediction"},
            "per_class": per_class, "macro": macro, "weighted": weighted,
            "sample_count": n, "review_fraction": review_count / n,
            "decisive_coverage": (n - review_count) / n,
            "zero_denominator_policy": "precision, recall and F1 are zero"}


def segmentation_metrics(truth: np.ndarray, predicted: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate actual binary/probability masks in 0..1; both empty scores 1.

    A foreground pixel has value >= threshold. One empty mask scores zero.
    No generated or estimated mask is assumed to be ground truth.
    """
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Mask threshold must be finite and in (0, 1]")
    arrays = [np.asarray(truth), np.asarray(predicted)]
    if arrays[0].shape != arrays[1].shape or arrays[0].ndim != 2 or arrays[0].size == 0:
        raise ValueError("Masks must be equally sized nonempty 2D arrays")
    for array in arrays:
        if (not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_)
                or np.issubdtype(array.dtype, np.complexfloating)):
            raise ValueError("Masks must contain numeric values")
        if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
            raise ValueError("Mask values must be finite and in 0..1")
    actual, estimate = [array >= threshold for array in arrays]
    intersection = int(np.count_nonzero(actual & estimate))
    union = int(np.count_nonzero(actual | estimate))
    total = int(np.count_nonzero(actual) + np.count_nonzero(estimate))
    return {"iou": intersection / union if union else 1.0,
            "dice": 2 * intersection / total if total else 1.0}
