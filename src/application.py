"""FruitGuard Vision orchestration; serial failure never enables manual mode."""
import argparse
import logging
import time
from datetime import datetime, timezone

import cv2

from src.display import TITLE, render
from src.hardware.arduino_serial import ArduinoSerial, DistanceTrigger
from src.inspection import idle_result, inspect_frame
from src.utils.config import ROOT, load_config, validate_config
from src.utils.samples import save_sample
from src.vision.detector import AppleDetector

LOG = logging.getLogger(__name__)


def create_measurement_backend(config):
    if config['backend'] == 'cpu':
        return None
    from src.vision.cuda_quality import CudaQualityMeasurements
    return CudaQualityMeasurements(library_path=ROOT / config['library_path'],
                                   device_index=config['device_index'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='FruitGuard Vision: visible quality estimate')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--no-arduino', action='store_true', help='Explicit manual inspection; no serial access')
    parser.add_argument('--device', help='Explicit cpu or cuda:N override; never automatic CPU fallback')
    parser.add_argument('--quality-backend', choices=('cuda', 'cpu'), help='Custom CUDA classifier or explicit CPU reference')
    parser.add_argument('--port', help='Arduino COM port override')
    parser.add_argument('--camera', type=int, help='Webcam index override')
    parser.add_argument('--max-frames', type=int, help='Exit normally after N frames for a smoke check')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    try:
        config = load_config(args.config)
        if args.device is not None:
            config['detector']['device'] = args.device
        if args.quality_backend is not None:
            config['classifier']['backend'] = args.quality_backend
        if args.port is not None:
            config['serial']['port'] = args.port
        if args.camera is not None:
            config['camera']['index'] = args.camera
        validate_config(config)
    except (OSError, ValueError) as exc:
        LOG.error('Configuration error: %s', exc)
        return 1
    hardware = None if args.no_arduino else ArduinoSerial(config['serial'])
    trigger = DistanceTrigger(config['serial'])
    camera = None
    measurement_backend = None
    fatal = False
    intended_command = None
    try:
        if hardware:
            hardware.poll()
        camera = cv2.VideoCapture(config['camera']['index'])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
        if not camera.isOpened():
            raise RuntimeError('Camera could not open; check camera.index and other camera applications')
        LOG.info('Loading local detector on %s; first inference may take longer.', config['detector']['device'])
        detector = AppleDetector(config['detector'], ROOT)
        measurement_backend = create_measurement_backend(config['classifier'])
        LOG.info('Detector class %s; custom visual heuristic classifier backend: %s.',
                 detector.class_name, config['classifier']['backend'])
        frame_count, previous_frame_time = 0, None
        generation = hardware.generation if hardware else 0
        notification, notification_until = '', 0.0
        while True:
            loop_start = time.perf_counter()
            samples = hardware.poll() if hardware else []
            if hardware and hardware.generation != generation:
                trigger.reset()
                generation = hardware.generation
            now = time.monotonic()
            for distance in samples:
                trigger.update(distance, now)
            distance_state = trigger.state(now)
            if args.no_arduino:
                distance_state = {'distance_cm': None, 'last_valid_age_s': None,
                                  'reading_state': 'unavailable', 'active': True}
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError('Camera frame read failed')
            timestamp = datetime.now(timezone.utc).isoformat()
            detections, latency, quality_latency = [], None, None
            if hardware and hardware.error:
                result = idle_result(hardware.error, error=True)
            elif hardware and not hardware.ready:
                result = idle_result('Arduino starting; waiting for a fresh distance')
            elif not distance_state['active']:
                if distance_state['reading_state'] != 'fresh':
                    result = idle_result('Distance reading ' + distance_state['reading_state'], error=True)
                else:
                    result = idle_result('Waiting for apple below enter distance')
            else:
                detector.synchronize()
                start = time.perf_counter()
                detections = detector.detect(frame)
                detector.synchronize()
                latency = (time.perf_counter() - start) * 1000.0
                quality_start = time.perf_counter()
                result = inspect_frame(frame, detections, config['quality'], measurement_backend)
                quality_latency = (time.perf_counter() - quality_start) * 1000.0
            if hardware and hardware.ready and not hardware.error:
                # A slow capture/inference must not extend an old sensor reading.
                distance_state = trigger.state(time.monotonic())
                if not distance_state['active']:
                    detections = []
                    result = idle_result('Distance reading became stale during frame processing', error=True) if distance_state['reading_state'] != 'fresh' else idle_result('Inspection trigger released')
            if hardware:
                hardware.send(result['command'])
                if hardware.generation != generation:
                    trigger.reset()
                    generation = hardware.generation
                    distance_state = trigger.state(time.monotonic())
                    detections = []
                    result = idle_result(hardware.error or 'Serial connection changed', error=True)
            elif result['command'] != intended_command:
                LOG.info('Hardware disabled: intended RGB %s (not transmitted)', result['command'])
                intended_command = result['command']
            frame_time = time.perf_counter()
            fps = None if previous_frame_time is None else 1.0 / max(frame_time - previous_frame_time, 1e-9)
            previous_frame_time = frame_time
            record = {
                'timestamp': timestamp, 'frame_index': frame_count,
                'inspection': result,
                'detections': [{'bbox': list(d.bbox), 'class_name': d.class_name,
                                'confidence': d.confidence, 'segmentation_available': d.mask is not None}
                               for d in detections],
                'distance': distance_state,
                'mode': 'Hardware disabled / manual inspection' if args.no_arduino else 'Arduino / ultrasonic trigger',
                'serial_state': 'disabled' if hardware is None else
                                ('connected, ready' if hardware.ready else hardware.error or 'starting'),
                'model_identifier': str(detector.path), 'configuration': config,
                'classifier_identifier': 'fruitguard_visual_heuristics_v1',
                'classifier_backend': config['classifier']['backend'],
                'timing': {'fps': fps, 'inference_latency_ms': latency,
                           'quality_latency_ms': quality_latency,
                           'processing_ms': (frame_time - loop_start) * 1000.0,
                           'inference_boundary': 'Synchronized detector.predict: preprocessing, transfer, model, postprocessing, extraction'},
            }
            cv2.imshow(TITLE, render(frame, detections, record,
                                    notification if time.monotonic() < notification_until else ''))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                try:
                    saved = save_sample(frame, record, ROOT / config['storage']['directory'])
                    notification = 'Saved sample: ' + saved.name
                    LOG.info('Saved sample %s', saved)
                except (OSError, ValueError, TypeError, cv2.error) as exc:
                    notification = 'Sample save failed: ' + str(exc)
                    LOG.error('%s', notification)
                notification_until = time.monotonic() + 5.0
            frame_count += 1
            if key == ord('q') or (args.max_frames and frame_count >= args.max_frames):
                break
    except KeyboardInterrupt:
        LOG.info('Stopped by user')
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        fatal = True
        trigger.reset()
        LOG.exception('System error: %s. Attempting BLUE if Arduino is ready; no subsequent OFF on fatal shutdown.', exc)
    except Exception as exc:
        # Final application boundary: report unexpected loader/runtime failures
        # and preserve the error indication rather than treating them as success.
        fatal = True
        trigger.reset()
        LOG.exception('Unexpected system failure: %s. Attempting BLUE if ready; no subsequent OFF.', exc)
    finally:
        if measurement_backend is not None:
            try:
                measurement_backend.close()
            except (RuntimeError, OSError) as exc:
                LOG.error('CUDA classifier cleanup failed: %s', exc)
                fatal = True
        if hardware:
            hardware.close(normal=not fatal)
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
    return 1 if fatal else 0
