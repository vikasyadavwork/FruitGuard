"""OpenCV preview and readable side panel; originals remain unmodified."""
import textwrap

import cv2
import numpy as np

TITLE = 'FruitGuard Vision'
COLORS = {'GREEN': (0, 190, 0), 'YELLOW': (0, 220, 255),
          'RED': (0, 0, 255), 'BLUE': (255, 150, 0), 'OFF': (180, 180, 180)}


def _value(value, suffix=''):
    return 'N/A' if value is None else '{:.3f}{}'.format(value, suffix)


def render(frame, detections, record, notification=''):
    result = record['inspection']
    color = COLORS[result['command']]
    view = frame.copy()
    height, width = view.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        a = (int(max(0, min(width - 1, x1))), int(max(0, min(height - 1, y1))))
        b = (int(max(0, min(width - 1, x2))), int(max(0, min(height - 1, y2))))
        cv2.rectangle(view, a, b, color, 2)
        cv2.putText(view, '{} {:.2f}'.format(detection.class_name, detection.confidence),
                    (a[0], max(20, a[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    distance, timing = record['distance'], record['timing']
    features = result.get('features', {})
    classifier_timing = result.get('measurement_timing') or {}
    confidences = ', '.join('{:.3f}'.format(d.confidence) for d in detections) or 'N/A'
    lines = [TITLE, record['mode'],
             'System: ' + result['system_state'],
             'Visible quality estimate: ' + (result['status'] or 'N/A'),
             'Detector confidence: ' + confidences,
             'Distance: ' + _value(distance['distance_cm'], ' cm'),
             'Reading: ' + distance['reading_state'],
             'Visual defect score: ' + _value(features.get('visual_defect_score')),
             'Color anomaly: ' + _value(features.get('color_anomaly_score')),
             'Dark regions: ' + _value(features.get('dark_spot_percentage'), '%'),
             'Symmetry: ' + _value(features.get('symmetry_score')),
             'Classifier: custom heuristics / ' + record.get('classifier_backend', 'cpu'),
             'FPS: ' + _value(timing['fps']),
             'Inference pipeline: ' + _value(timing['inference_latency_ms'], ' ms'),
             'Quality pipeline: ' + _value(timing.get('quality_latency_ms'), ' ms'),
             'Classifier GPU kernel: ' + _value(classifier_timing.get('kernel_ms'), ' ms'),
             'RGB intended: ' + result['command'],
             'Serial: ' + record['serial_state'],
             'Reasons: ' + '; '.join(result['reasons']),
             'Requires calibration', 'Q: quit    S: save current sample']
    if notification:
        lines.append(notification)
    wrapped_lines = [(line, wrapped) for line in lines
                     for wrapped in (textwrap.wrap(line, width=53) or [''])]
    canvas = np.zeros((max(height, 700, 45 + 25 * len(wrapped_lines)), width + 500, 3), dtype=np.uint8)
    canvas[:height, :width] = view
    canvas[:, width:] = (28, 28, 28)
    y = 28
    for line, wrapped in wrapped_lines:
        cv2.putText(canvas, wrapped, (width + 14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .49, color if line.startswith('Visible quality') else (230, 230, 230), 1)
        y += 25
    return canvas
