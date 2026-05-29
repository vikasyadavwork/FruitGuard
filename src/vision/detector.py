"""Small interchangeable detection interface; no defect classification here."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None


class AppleDetector:
    def __init__(self, config, root: Path):
        self.config = config
        self.path = (root / config['model_path']).resolve()
        if not self.path.is_file():
            raise ValueError('Model weights missing: {}. Configure existing local weights.'.format(self.path))
        device = str(config['device']).lower()
        self.cuda_index = None
        if device != 'cpu':
            index = int(device.replace('cuda:', ''))
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError('Configured CUDA device {} unavailable. Explicitly select --device cpu to use CPU.'.format(device))
            self.cuda_index = index
            self.device = index
        else:
            self.device = 'cpu'
        self.model = YOLO(str(self.path))
        names = self.model.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        override = config['apple_class_id']
        if override is not None:
            if override not in names:
                raise ValueError('apple_class_id does not exist in model class names')
            self.class_id = override
        else:
            matches = [key for key, value in names.items()
                       if value.casefold() == config['apple_class_name'].casefold()]
            if len(matches) != 1:
                raise ValueError('Model must contain the configured apple class name, or set apple_class_id')
            self.class_id = matches[0]
        self.class_name = names[self.class_id]

    def synchronize(self):
        if self.cuda_index is not None:
            torch.cuda.synchronize(self.cuda_index)

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(frame, device=self.device,
                                     imgsz=self.config['input_size'],
                                     conf=self.config['detection_threshold'],
                                     classes=[self.class_id], verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            # Ultralytics' public masks.xy coordinates undo letterbox padding.
            polygons = result.masks.xy if result.masks is not None else None
            for index, box in enumerate(result.boxes):
                if int(box.cls.item()) != self.class_id:
                    continue
                mask = None
                if polygons is not None and len(polygons[index]) >= 3:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.rint(polygons[index]).astype(np.int32)], 1)
                detections.append(Detection(tuple(float(v) for v in box.xyxy[0].cpu().tolist()),
                                            self.class_name, float(box.conf.item()), mask))
        return detections
