"""Save the unannotated current frame and its own immutable metadata snapshot."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from src.vision.quality_analyzer import clip_roi


def save_sample(frame, metadata, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex
    image_path = directory / (stem + '.png')
    if not cv2.imwrite(str(image_path), frame):
        raise OSError('Original image save failed: {}'.format(image_path))
    paths = {'original': image_path.name, 'roi': None}
    bbox = metadata['inspection'].get('bbox')
    if bbox is not None:
        roi, _, _ = clip_roi(frame, bbox, metadata['configuration']['quality']['min_roi_size'])
        if roi is not None:
            roi_path = directory / (stem + '_roi.png')
            if not cv2.imwrite(str(roi_path), roi):
                raise OSError('ROI save failed: {}'.format(roi_path))
            paths['roi'] = roi_path.name
    record = dict(metadata, files=paths)
    json_path = directory / (stem + '.json')
    serialized = json.dumps(record, indent=2, allow_nan=False)
    # JSON appears last so a metadata record never promises an unwritten image.
    with json_path.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    return json_path
