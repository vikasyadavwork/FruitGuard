"""Validated YAML configuration with paths anchored at the repository root."""
import math
from pathlib import Path

import yaml

from src.vision.quality_analyzer import QUALITY_DEFAULTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    'detector': {'model_path': 'yolo11n.pt', 'device': 'cuda:0', 'input_size': 640,
                 'detection_threshold': 0.35, 'apple_class_name': 'apple', 'apple_class_id': None},
    'camera': {'index': 0, 'width': 640, 'height': 480},
    'serial': {'port': 'COM5', 'baudrate': 9600, 'startup_delay_s': 2.0,
               'retry_interval_s': 3.0, 'write_timeout_s': 0.1,
               'max_read_bytes': 4096, 'max_line_bytes': 128,
               'min_distance_cm': 0.1, 'max_distance_cm': 400.0,
               'enter_cm': 40.0, 'exit_cm': 45.0, 'reading_timeout_s': 1.5},
    'quality': QUALITY_DEFAULTS,
    'classifier': {'backend': 'cuda', 'device_index': 0, 'library_path': 'build/fruitguard_preprocess.dll'},
    'storage': {'directory': 'data/inspections'},
}


def _number(value, name, minimum=0, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('{} must be a finite number'.format(name))
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError('{} is outside its permitted range'.format(name))
    if integer and not isinstance(value, int):
        raise ValueError('{} must be an integer'.format(name))


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULTS):
        raise ValueError('Configuration must contain exactly: {}'.format(', '.join(DEFAULTS)))
    for section, defaults in DEFAULTS.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != set(defaults):
            raise ValueError('{} has missing or unknown keys'.format(section))
        for name, default in defaults.items():
            value = values[name]
            label = section + '.' + name
            if default is None:
                if value is not None:
                    _number(value, label, integer=True)
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(label + ' must be boolean')
            elif isinstance(default, (int, float)):
                _number(value, label, integer=isinstance(default, int))
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(label + ' must be a nonempty string')
    d, s, q = config['detector'], config['serial'], config['quality']
    device = str(d['device']).lower()
    if device != 'cpu' and not (device.isdigit() or
                                (device.startswith('cuda:') and device[5:].isdigit())):
        raise ValueError('detector.device must be cpu, cuda:N, or an explicit CUDA index string')
    _number(d['detection_threshold'], 'detection_threshold', 0.001, 1)
    _number(d['input_size'], 'input_size', 32, 1280, True)
    if d['input_size'] % 32:
        raise ValueError('input_size must be a multiple of 32')
    if s['baudrate'] != 9600:
        raise ValueError('Existing Arduino protocol requires baudrate 9600')
    if not 0 < s['min_distance_cm'] < s['enter_cm'] < s['exit_cm'] < s['max_distance_cm']:
        raise ValueError('Require 0 < min_distance_cm < enter_cm < exit_cm < max_distance_cm')
    for key in ('retry_interval_s', 'write_timeout_s', 'reading_timeout_s'):
        _number(s[key], key, 0.001)
    for key in ('max_read_bytes', 'max_line_bytes'):
        _number(s[key], key, 16, 65536, True)
    for key in ('width', 'height'):
        _number(config['camera'][key], key, 32, 8192, True)
    # Analyzer exports validation for all measurement thresholds and their ordering.
    from src.vision.quality_analyzer import validate_quality_config
    validate_quality_config(q)
    if config['classifier']['backend'] not in ('cpu', 'cuda'):
        raise ValueError('classifier.backend must be an explicitly selected cpu or cuda')
    storage = Path(config['storage']['directory'])
    resolved_storage = (ROOT / storage).resolve()
    if storage.drive or storage.root or '..' in storage.parts or ROOT not in resolved_storage.parents:
        raise ValueError('storage.directory must stay inside the repository')
    return config


def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs' / 'config.yaml'
    if not path.is_absolute():
        path = ROOT / path
    try:
        with path.open('r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError('Invalid YAML configuration: {}'.format(exc)) from exc
    return validate_config(config)
