"""Recorded frame manifests with split-leakage and provenance validation."""
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from evaluation.metrics import LABELS


@dataclass
class Sample:
    image: Path
    label: str
    fruit_id: str
    split: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    mask: Optional[Path]
    sha256: str
    pixels_sha256: str


def read_frame(path: Path) -> np.ndarray:
    # imdecode supports Windows Unicode paths that cv2.imread may reject.
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError("Cannot read image: {}".format(path)) from exc
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("Image cannot be decoded as BGR: {}".format(path))
    return frame


def read_foreground_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError("Cannot read foreground mask: {}".format(path)) from exc
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if mask is None or mask.shape != shape or mask.ndim != 2:
        raise ValueError("Foreground mask must be single-channel and match frame dimensions")
    if not np.all(np.isfinite(mask)) or not np.all(np.isin(mask, [0, 1, 255])):
        raise ValueError("Foreground mask must be binary (0/1 or 0/255)")
    return (mask > 0).astype(np.uint8)


def _path(value: str, parent: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def read_manifest(path: Path) -> List[Sample]:
    """Audit every split before selecting test; never infer boxes/confidence."""
    path = Path(path).resolve()
    required = {"image", "label", "fruit_id", "split", "x1", "y1", "x2", "y2", "confidence"}
    samples = []
    fruits = {}
    paths = set()
    hashes = set()
    pixels_hashes = set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest requires columns: {}".format(", ".join(sorted(required))))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Manifest has duplicate column names")
        if set(reader.fieldnames) - required - {"mask"}:
            raise ValueError("Manifest has unknown columns")
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
                raise ValueError("Missing or malformed value at line {}".format(line))
            row = {key: (value or "").strip() for key, value in row.items()}
            if row["label"] not in LABELS or row["split"] not in ("train", "val", "test"):
                raise ValueError("Unknown label or split at line {}".format(line))
            fruit_id, split = row["fruit_id"], row["split"]
            if fruit_id in fruits and fruits[fruit_id] != split:
                raise ValueError("fruit_id {} crosses splits".format(fruit_id))
            fruits[fruit_id] = split
            image = _path(row["image"], path.parent)
            frame = read_frame(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            pixels_digest = hashlib.sha256(str(frame.shape).encode("ascii") + frame.tobytes()).hexdigest()
            if image in paths or digest in hashes or pixels_digest in pixels_hashes:
                raise ValueError("Duplicate frame/path/content in manifest: {}".format(image))
            paths.add(image)
            hashes.add(digest)
            pixels_hashes.add(pixels_digest)
            try:
                bbox = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
                confidence = float(row["confidence"])
            except ValueError as exc:
                raise ValueError("Invalid recorded detection at line {}".format(line)) from exc
            if not all(math.isfinite(value) for value in bbox) or not math.isfinite(confidence):
                raise ValueError("Recorded detection must be finite")
            x1, y1, x2, y2 = bbox
            if not 0 <= confidence <= 1 or x1 >= x2 or y1 >= y2:
                raise ValueError("Invalid bounding box or confidence")
            height, width = frame.shape[:2]
            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
                raise ValueError("Recorded bounding box does not intersect frame")
            mask = _path(row["mask"], path.parent) if row.get("mask") else None
            if mask is not None:
                read_foreground_mask(mask, frame.shape[:2])
            samples.append(Sample(image, row["label"], fruit_id, split, bbox,
                                  confidence, mask, digest, pixels_digest))
    if not samples:
        raise ValueError("Manifest has no samples; no empirical metrics can be reported")
    return samples
