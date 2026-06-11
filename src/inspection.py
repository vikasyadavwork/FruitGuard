"""Per-frame decisions are rebuilt; there is no previous-fruit result cache."""
from src.vision.quality_analyzer import analyze

COMMANDS = {'GOOD': 'GREEN', 'REVIEW': 'YELLOW', 'DEFECTIVE': 'RED'}


def inspect_frame(frame, detections, quality_config, measurement_backend=None):
    if not detections:
        return {'system_state': 'WAITING', 'status': None, 'command': 'OFF',
                'reasons': ['No apple detected'], 'features': {}, 'validity': {}, 'bbox': None}
    if len(detections) > 1:
        return {'system_state': 'INSPECTING', 'status': 'REVIEW', 'command': 'YELLOW',
                'reasons': ['Present one apple'], 'features': {}, 'validity': {}, 'bbox': None}
    result = analyze(frame, detections[0], quality_config, measurement_backend=measurement_backend)
    result.update(system_state='INSPECTING', command=COMMANDS[result['status']])
    return result


def idle_result(reason, error=False):
    return {'system_state': 'ERROR' if error else 'WAITING', 'status': None,
            'command': 'BLUE' if error else 'OFF', 'reasons': [reason],
            'features': {}, 'validity': {}, 'bbox': None}
