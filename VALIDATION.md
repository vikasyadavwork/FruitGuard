# FruitGuard Vision — Milestone 1 validation

Executed on 2026-09-27 in `D:\AurdinoCudaProject`, using only the existing
`.venv\Scripts\python.exe` for Python execution. No installations, downloads,
firmware changes, CUDA builds, or changes to pre-existing source files occurred.

## Results

- Final complete unittest suite: **59 tests passed**, no failures.
- Python compile/import checks: passed, including all new src/tests modules.
- `pip check`: no broken requirements found before or after implementation.
- Existing PyTorch CUDA tensor operation: passed on GTX 1050 Ti (6.1, 4 GB).
- Existing local `yolo11n.pt`: loaded; apple resolved by name to 47; synthetic
  black-frame GPU inference passed with zero detections. No accuracy conclusion.
- Real webcam 0: captured 640x480 BGR image and released successfully.
- Existing `build\CudaHello.exe`: exit 0, doubled 1..8 to 2..16.
- Explicit CUDA 12.4 nvcc version query: reports 12.4.131; no source was built.
- Real MVP manual-mode run: exit 0 after 12 frames, real CUDA/webcam, intended
  OFF logged without serial access; no apple detected.
- Real MVP hardware-mode run: exit 0 after 180 frames, COM5 opened, BLUE and
  OFF writes completed, serial closed/camera released by normal cleanup.
- Additional 120-frame hardware run with an in-process render observer: exit 0;
  actual reading 16.52 cm expired during first inference, yielded system ERROR,
  BLUE, cleared fruit result. A later 13.46 cm fresh reading activated inspection;
  no apple detected, WAITING/OFF. Numbers are observations from that run, not
  fabricated distances or sensor accuracy measurements.
- Mocked application-loop tests also verify manual mode never constructs serial,
  exact current raw frame/metadata on S, camera/model failure cleanup, corrupt
  weights fatal indication, and stale-during-inference clearing.

The first baseline GPU prediction took approximately 12 seconds including
initial setup. This is not a steady-state benchmark or a performance claim.
No hardware acknowledgement exists in the current serial protocol.

## Commands actually executed

Discovery/read commands included `rg --files` for instructions/source/README/
requirements/config/build/firmware/weights, `Get-ChildItem` on the repository,
src/build, `.venv`, and CUDA installations, `Get-Content` on the existing Python
and CUDA sources and `.venv\pyvenv.cfg`, and `git status --short`.
Git reported that this directory is not a Git repository. No AGENTS.md or .ino
files were found. The Python package inventory used:

```powershell
Set-Location D:\AurdinoCudaProject
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip list
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src
```

The read-only environment/model/camera probes executed these Python bodies with
the existing interpreter using `-c` (shown as here-strings for easy reuse):

```powershell
@'
import sys, importlib
print(sys.version, sys.executable)
for name in ['torch', 'torchvision', 'cv2', 'ultralytics', 'yaml', 'serial']:
    print(name, getattr(importlib.import_module(name), '__version__', 'N/A'))
import torch
print(torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
      torch.cuda.get_device_properties(0).total_memory)
print((torch.tensor([1., 2.], device='cuda:0') * 2).cpu().tolist())
import serial.tools.list_ports
print([(p.device, p.description) for p in serial.tools.list_ports.comports()])
'@ | .\.venv\Scripts\python.exe -

@'
import time, numpy as np, torch
from ultralytics import YOLO
model = YOLO('D:/AurdinoCudaProject/yolo11n.pt')
print([(key, value) for key, value in model.names.items() if value.lower() == 'apple'])
start = time.perf_counter()
result = model.predict(source=np.zeros((480,640,3), dtype=np.uint8),
                       device=0, imgsz=640, verbose=False)[0]
torch.cuda.synchronize()
print(len(result.boxes), time.perf_counter()-start, result.speed)
'@ | .\.venv\Scripts\python.exe -

@'
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())
ok, frame = cap.read() if cap.isOpened() else (False, None)
print(ok, None if frame is None else frame.shape)
cap.release()
'@ | .\.venv\Scripts\python.exe -

.\build\CudaHello.exe
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe' --version
```

Focused test commands during implementation and final checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_serial.py -v
.\.venv\Scripts\python.exe -m unittest tests.test_detector -v
.\.venv\Scripts\python.exe -m unittest tests.test_quality -v
.\.venv\Scripts\python.exe -m py_compile src\vision\quality_analyzer.py tests\test_quality.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_application_loop.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m src.main --help
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.main --no-arduino --max-frames 12
.\.venv\Scripts\python.exe -m src.main --max-frames 180
```

The additional actual hardware observation command executed was:

```powershell
@'
import sys
from src import application
original_render = application.render
seen = set()
def observed_render(frame, detections, record, notification=''):
    distance = record['distance']
    key = (record['inspection']['system_state'], record['inspection']['command'],
           distance['reading_state'], distance['active'])
    if key not in seen:
        seen.add(key)
        print('OBSERVED_STATE', key, 'DISTANCE_CM', distance['distance_cm'],
              'REASONS', record['inspection']['reasons'], flush=True)
    return original_render(frame, detections, record, notification)
application.render = observed_render
sys.exit(application.main(['--max-frames', '120']))
'@ | .\.venv\Scripts\python.exe -
```

## Added files / preserved files

Added application/config/documentation:

- `README.md`, `VALIDATION.md`, `requirements.txt`, `configs\config.yaml`.
- `src\__init__.py`, `src\main.py`, `src\application.py`, `src\display.py`,
  `src\inspection.py`.
- `src\vision\__init__.py`, `src\vision\detector.py`,
  `src\vision\quality_analyzer.py`.
- `src\hardware\__init__.py`, `src\hardware\arduino_serial.py`.
- `src\utils\__init__.py`, `src\utils\config.py`, `src\utils\samples.py`.
- `tests\__init__.py`, `tests\test_serial.py`, `tests\test_detector.py`,
  `tests\test_quality.py`, `tests\test_application.py`,
  `tests\test_application_loop.py`.
- `data\inspections\.gitkeep`.

Pre-existing files modified: **none**. Python compile/import checks generated
ordinary `__pycache__` files. Existing firmware/models/build outputs and `.venv`
were not deleted, replaced, or upgraded. No separate repository/root was made.

## Manual checks still required

- Present a real apple against a controlled background and verify detection/ROI
  reliability for the actual setup; test no apple and two apples physically.
- Collect labeled examples to calibrate heuristic thresholds and quantify false
  results. Synthetic images validate behavior only, not quality accuracy.
- Observe GREEN/YELLOW/RED/BLUE/OFF on the actual LED. Only BLUE/OFF serial-driver
  writes were exercised physically; no LED state was observed or acknowledged.
- Physically unplug/reconnect serial, test failed/stopped sensor readings and
  camera interruption; software recovery paths passed fake-serial tests.
- Press S/Q manually and inspect saved original/ROI/JSON associations. Real file
  saving/current-frame associations passed synthetic and mocked-loop tests;
  keyboard interactions with a real apple were not manually verified.
- Fatal-error BLUE persistence after serial close depends on existing firmware;
  firmware was unavailable for inspection and not reflashed.

Milestone 1 is the stopping point. Custom CUDA preprocessing, datasets, model
training, and segmentation research were not undertaken.
