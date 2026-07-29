# FruitGuard Vision

Webcam -> YOLO apple detection -> validated ROI/foreground -> custom CUDA color,
dark-region and symmetry measurements -> configurable GOOD / REVIEW / DEFECTIVE
decision -> OpenCV display -> Arduino RGB command.

Milestones 1–3 are implemented with the requested **custom CUDA heuristic
classifier**. It uses your configurable visual rules, not a trained neural
defect model. A CPU reference, independent CUDA preprocessing benchmark, labeled
sample evaluation tools, and numerical parity tests are included.

**This MVP uses uncalibrated visual defect heuristics.
It does not determine food safety or scientifically measure freshness.**

## Existing project and environment

The filesystem root remains `D:\AurdinoCudaProject`. The original
`src\SerialTest.py`, `src\YoloArduino.py`, `src\CudaTest.py`, local `yolo11n.pt`,
existing CUDA sources, and `build\CudaHello.*` were inspected before changes.
`CudaHello.cu` and its build outputs remain unchanged. The original
`ImagePreprocess.cu` is preserved byte-for-byte as `ImagePreprocessLegacy.cu`;
the active file now provides the validated reusable CHW preprocessing kernel.
The serial module reuses the inspected protocol and the detector uses
the same installed Ultralytics/PyTorch prediction path and local weights.
The legacy scripts remain runnable from the root, with their original behavior
(including the old person detector); use the new entry point for FruitGuard Vision.
No Arduino sketch, README, requirements file, build script, AGENTS.md, or Git
repository was present. Firmware was not changed or reflashed.

The existing interpreter is `.venv\Scripts\python.exe` (Python **3.8.0**).
Installed versions verified during implementation:

| Component | Existing version |
|---|---|
| PyTorch | 2.3.1+cu118 |
| torchvision | 0.18.1+cu118 |
| Ultralytics | 8.4.163 |
| OpenCV package | opencv-python 5.0.0.93 (cv2 reports 5.0.0) |
| NumPy | 1.24.4 |
| PySerial | 3.5 |
| PyYAML | 6.0.3 |

PyTorch CUDA runtime 11.8 sees the NVIDIA GTX 1050 Ti, 4 GB VRAM, compute
capability 6.1. Toolkit 12.4.131 is installed separately; a v13.4 installation
also exists. The build script explicitly uses Toolkit 12.4 and MSVC 14.38,
with `-gencode arch=compute_61,code=sm_61`; it never chooses v13.4 silently.
The native DLL uses CUDA independently; no PyTorch C++/CUDA extension is added.
There were no mismatches with the requested Python/PyTorch/torchvision/GPU
versions. No environment was recreated and no package was installed or upgraded.

`requirements.txt` records the existing direct dependency pins. No installation
is needed in this working environment. If rebuilding an environment later,
install the pinned torch/torchvision wheels from the cu118 index as specified in
that file, then install the remaining pinned packages. Do not replace the CUDA
PyTorch wheels when adding unrelated dependencies; do not blindly upgrade YOLO.

## Start from PowerShell

Use the existing environment explicitly; activation is optional:

```powershell
Set-Location D:\AurdinoCudaProject
# Rebuild the native library when CUDA source changes (already built here):
powershell -ExecutionPolicy Bypass -File scripts\build_cuda.ps1
.\.venv\Scripts\python.exe -m src.main
```

To activate that same environment:

```powershell
Set-Location D:\AurdinoCudaProject
.\.venv\Scripts\Activate.ps1
python -m src.main
```

Explicit hardware-free mode opens no serial port, deliberately bypasses the
ultrasonic trigger, shows `Hardware disabled / manual inspection`, and leaves
distance unavailable. Intended LED commands are logged, never transmitted:

```powershell
.\.venv\Scripts\python.exe -m src.main --no-arduino
```

CUDA is the configured default and is never silently replaced by CPU. To choose
CPU deliberately, or override a hardware setting:

```powershell
.\.venv\Scripts\python.exe -m src.main --no-arduino --device cpu --quality-backend cpu
.\.venv\Scripts\python.exe -m src.main --port COM5 --camera 0 --device cuda:0
.\.venv\Scripts\python.exe -m src.main --config configs/config.yaml
```

`--max-frames N` exits normally after N displayed frames for a bounded smoke
check. Relative config/model/storage paths are resolved against this repository,
not an accidental working directory. The module startup command itself is run
from the repository root. The window title is **FruitGuard Vision**.

`--device` selects YOLO's PyTorch device; `--quality-backend` independently selects
your classifier's CUDA or CPU measurement implementation. Defaults use CUDA for
both. CUDA-library/device failure is a system error, never an automatic CPU
fallback. Build with Visual Studio C++ tools and MSVC 14.38 installed;
`scripts\build_cuda.ps1 -CompilerVersion 14.38` selects the compatible compiler.

## Configuration and inspection

Edit `configs\config.yaml`. Set `serial.port` to your actual Arduino port (COM5
was enumerated here), `camera.index` to the correct webcam (0 captured here),
and `detector.model_path` to local weights. No weights or dataset were downloaded.
The existing `yolo11n.pt` is a small pretrained object detector, not a trained
apple defect classifier. Apple is resolved from model names (47 in these weights);
`apple_class_name` or an explicit `apple_class_id` supports a future replacement.
`AppleDetector.detect()` returns bounding box, class, confidence, and a mask only
when the model actually supplies segmentation. Future detectors can implement
this same small interface; the current weights supply boxes only.

Defaults: `device: cuda:0`, input size 640, detection threshold 0.35,
GOOD confidence requirement 0.60. Detector confidence is object-detection
confidence, not the probability that an apple is healthy.

`classifier.backend: cuda`, `classifier.device_index: 0`, and
`classifier.library_path: build/fruitguard_preprocess.dll` select the native
custom classifier. Foreground extraction/reliability remains on CPU. A fused
CUDA kernel computes 8-bit HSV, dark/brown/desaturated pixel counts and mirrored
foreground intersection/union, with shared-memory block reduction. A small host
reduction combines block counts and applies configured weights/rules. Device
buffers and a pinned partial-count buffer are reused as image sizes permit.
There are no neural defect-model weights to train or download.

The HC-SR04 trigger enters below 40 cm and releases above 45 cm; between those
values it retains its current state. Only valid readings in 0.1..400 cm refresh
the trigger. Missing, malformed, non-finite, zero, negative, or out-of-range
readings do not invent a distance. Readings become stale at 1.5 seconds and
disable inspection. Freshness is **time since receipt**, because `DIST:` has no
sensor acquisition timestamp. Queued readings cannot prove acquisition time.
Preview remains active while waiting; detection and analysis are skipped.
A slow inference cannot extend the freshness deadline: expired results are
discarded before sending a quality command. There is no temporal quality cache.

The input protocol remains `DIST:42.35` followed by a newline. Output commands
remain ASCII `GREEN\n`, `YELLOW\n`, `RED\n`, `BLUE\n`, `OFF\n` at 9600 baud.
Incoming CRLF is also accepted. Startup readiness waits 2 seconds without
sleeping in the frame loop, matching the original Arduino-reset accommodation.
Reads are bounded, partial lines retained, writes use a 0.1-second timeout, and
reconnection is retried at a 3-second interval. Only successful complete writes
are cached; reconnect resets that cache and clears the old trigger/fruit state.
Serial failure never silently selects hardware-free mode.

| Current condition | Intended command |
|---|---|
| GOOD apple | GREEN |
| REVIEW apple / multiple apples | YELLOW |
| DEFECTIVE visible measurements | RED |
| Camera/model/serial/sensor-reading error | BLUE, if transport is ready |
| Normal waiting / no apple | OFF |

More than one detected apple shows `Present one apple`, REVIEW/YELLOW, and no
individual measurements. No detection shows `No apple detected`, clears the
previous fruit, and uses OFF. Waiting/errors have no fruit quality status.
Successful writes establish only serial-driver delivery; the protocol has no
LED acknowledgement. A disconnected cable cannot receive BLUE, so errors are
shown locally/logged. Normal Q/Ctrl+C shutdown attempts OFF, closes serial,
releases the camera, and destroys windows. Fatal shutdown attempts BLUE and
does **not** immediately send OFF. If startup or a disconnect prevents writing,
the log states this; firmware behavior after close is not guaranteed.

## Features, rules, and controlled setup

Use one sufficiently large apple (ROI at least 32 pixels in each dimension), a
plain contrasting background, reasonably consistent diffuse lighting, minimal
occlusion, and the entire apple inside the frame. A rectangular detection box
is never assumed to be the apple silhouette.

With no segmentation, four small ROI corner patches estimate background in
OpenCV Lab coordinates. Variation/separation, component dominance, filled area,
contour boundary contact, solidity, and aspect ratio gate reliability. The
largest foreground contour is filled to preserve enclosed dark or
background-coloured spots. Foreground is selected by colour separation, not by
excluding dark pixels. Supplied segmentation masks are also checked. Frames
touching the image edge are conservatively REVIEW because detector boxes may
already be clipped. This estimate is conditional on the controlled setup;
clutter or indistinguishable fruit/background can still fool it.

All measurement/decision thresholds are in the YAML file. Unreliable masks or
ROIs produce unavailable measurements (`N/A` on screen, `null` in JSON), REVIEW,
and specific reasons. No trustworthy contour means no confident shape result.

| Measurement | Range / interpretation | GOOD boundary | Strong-defect boundary |
|---|---|---|---|
| Dark-region percentage | 0..100, foreground pixels with HSV V < 60 | <= 3% | >= 12% |
| Color anomaly score | 0..1, foreground fraction in configured dim brown/desaturated bands | <= 0.06 | >= 0.25 |
| Symmetry score | 0..1, higher is more symmetric; mask vs horizontal reflection IoU | >= 0.85 | <= 0.65 |
| Visual defect score | 0..1, higher is more abnormal | <= 0.10 | >= 0.25 |

Combined score = normalized weighted mean of dark fraction (weight 0.45), color
anomaly (0.30), and `1 - symmetry` (0.25). Natural bright red/green/yellow areas
are not inherently defect pixels. GOOD requires confident detection, valid
measurements, and every GOOD limit satisfied. Valid strong measurements produce
DEFECTIVE; intermediate values produce REVIEW. Confidence below 0.60, unreliable
measurements, likely occlusion, or excessive clipping take priority and produce
REVIEW even when a computed score is large. Thresholds are **uncalibrated
starting values**, not validated scientific boundaries.

Shadows, stems, reflections, dim natural colour variation, russeting, hands,
background, and viewpoint changes can cause false results. Only the visible
surface is inspected. The app detects neither infection nor contamination and
does not decide whether an apple is safe to eat. Calibrate against labeled
examples in your actual lighting before using results operationally.

## Display, controls, and saved samples

The preview includes boxes, class/confidence, distance freshness, mode, system
state, visible quality, decision reasons, features, FPS, and inference latency.
Unavailable values are N/A. Q quits; S saves the current displayed inspection.
Original unannotated frame, clipped apple ROI when valid, and JSON are saved in
`data\inspections\`, using UTC time plus a UUID. Saving while waiting/multiple
apples has no single ROI. The JSON contains the actual frame's capture timestamp,
frame index, detections, features, validity, status/reasons, distance receipt age,
model identifier, configuration snapshot including command-line overrides,
system/mode/serial state, timings, and output filenames. Save errors are visible
in the UI and log; a failed operation can leave an image without a JSON record.

Inference latency is measured wall time around synchronized `detector.predict`,
including Ultralytics preprocessing, transfers, inference, postprocessing, and
structured result extraction. It excludes quality analysis and display. It is
not kernel-only timing. FPS measures time between processed frames, including
the previous UI/keyboard/save work. `processing_ms` covers current serial poll,
capture, inference if active, analysis, and command attempt, before rendering.
The first frame has no FPS; inactive frames have no inference latency. Quality
pipeline latency includes ROI/mask validation, measurement and decision. The
separate classifier GPU-kernel value includes HSV/counting/symmetry/block
reduction only; H2D/D2H/host work are excluded from that kernel value. Saved
measurement timing includes those transfer and wall boundaries separately.
Initial model/CUDA startup can take substantially longer. Measured benchmark
reports are included; they do not establish real-apple accuracy.

## Validation and troubleshooting

Run with the existing interpreter:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

Tests use fake serial objects and synthetic images. They verify parsing/partial
lines, trigger hysteresis/staleness, failed writes/reconnect/deduplication, ROI
clipping, mask reliability and dark inclusion, rule boundaries, device/class
configuration, stale-result clearing, current-frame sample saving, and cleanup.
They do not establish real-world defect accuracy or LED operation. See
`VALIDATION.md` for the original Milestone 1 baseline, and `VALIDATION_CUDA.md`
for current CUDA/classifier checks and remaining manual validation.

Enable the actual native GPU tests after building the DLL:

```powershell
$env:FRUITGUARD_TEST_CUDA='1'
$env:FG_RUN_CUDA_TESTS='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The final run passed **100 tests**, including actual CUDA preprocessing and
custom-classifier comparisons. Tests requiring a native GPU are opt-in when
the above flags are absent.

- CUDA unavailable: verify this exact `.venv` interpreter and CUDA-enabled torch;
  rebuild the native DLL with Toolkit 12.4 if missing. Explicitly choose
  `--device cpu --quality-backend cpu` for full CPU mode. No fallback occurs.
- Missing weights / unknown apple class: set a real local model path and correct
  class name/override. The app refuses to auto-download missing weights.
- Camera failure: close other camera apps and set the correct webcam index.
- Serial error: check the port, cable, drivers, and that Serial Monitor/legacy
  scripts are closed. App retries while keeping preview responsive.
- BLUE/stale/unavailable distance: confirm firmware emits valid newline-delimited
  `DIST:` readings and sensor wiring/range; invalid readings do not activate.
- Frequent REVIEW: improve the plain background, lighting, framing, image size,
  and occlusion. Inspect saved samples before changing mask/quality thresholds.
- Saving fails: check destination permissions and free disk space; storage must
  resolve inside this repository. Partial image output may remain after failure.

## CUDA preprocessing and measured benchmark

`src\cuda\ImagePreprocess.cu/.cuh` implements BGR uint8 HWC -> RGB float32 CHW
normalized to 0..1, with bounds/error checks and reusable native contexts.
`src\cuda\BenchmarkPreprocess.cu` compares the CPU reference and CUDA result at
1x1, 17x19 (non-block-aligned), 640x640, and 1920x1080. The old program remains
in `ImagePreprocessLegacy.cu`. `CudaHello.cu` was not changed.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_cuda.ps1
powershell -ExecutionPolicy Bypass -File scripts\benchmark_cuda.ps1 -Runs 100 -Warmup 10 -OutputPath evaluation\cuda_benchmark_new.json
```

Reports refuse overwrite. The included `evaluation\cuda_benchmark.json` was
measured on the actual GTX 1050 Ti with CUDA 12.4, 10 warmups and 100 runs per
case. Maximum CPU/CUDA error was **0**, below absolute tolerance 1e-6.

| Input | CPU reference | CUDA kernel only | Complete CUDA to host output |
|---|---:|---:|---:|
| 640x640 | 0.997 ms | 0.103 ms | 1.567 ms |
| 1920x1080 | 5.509 ms | 0.396 ms | 8.048 ms |

These measured complete CUDA preprocessing paths were **slower** than the CPU
reference when returning all floats to host memory. Kernel time alone does not
establish acceleration. Complete time includes H2D, launch, and synchronous D2H
with reusable allocations; context setup/allocation costs and pageable transfer
timings are recorded separately. Kernel events are synchronized; complete time
is measured separately without intermediate timing instrumentation. The helper
`src.vision.preprocessing.preprocess_bgr` and reusable `CudaPreprocessor` provide
the same host-output transform for numerical experiments. Neither output is
inserted into YOLO, which receives the original BGR frame and preprocesses it
once using its working PyTorch path.

## Custom classifier and labeled evaluation

`src\cuda\DefectAnalysis.cu/.cuh` contains the custom CUDA classifier's fused
measurement kernel. `src\vision\cuda_quality.py` is its small ctypes interface;
`quality_analyzer.measure_features_cpu` is the OpenCV reference. Both feed the
same reliability-first rule decision. The integer HSV calculation follows the
[OpenCV reference algorithm](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_hsv.simd.hpp),
and GPU parity tests exercise configured threshold boundaries. The custom CUDA
classifier never consumes YOLO's detector confidence as a defect probability.

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmark_quality --synthetic --warmup 10 --iterations 100 --tolerance 1e-9
```

This benchmark measures synthetic controlled foregrounds, including plain
background estimates, and checks CPU/CUDA features, validity and decisions.
It records full quality wall time, CPU mask time, ROI/mask H2D, fused kernel,
partial-count D2H, reusable allocation cost and measurement wall time. Synthetic
timing and numerical agreement do not establish real-apple accuracy.

The included final report
`evaluation\reports\cuda_quality_benchmark_8ad6d954fcc44c22a915faf6835a3353.json`
has zero numerical/decision comparison failures across 12 inputs × 100 runs
after 10 warmups, at feature tolerance 1e-9. Mean full quality-path times for
plain-background synthetic frames were:

| Full frame | CPU quality pipeline | Custom CUDA quality pipeline |
|---|---:|---:|
| 128x128 | 1.431 ms | 1.433 ms |
| 256x256 | 3.087 ms | 2.854 ms |
| 512x512 | 9.416 ms | 7.725 ms |

The 512x512 CUDA pipeline included about 6.679 ms of CPU foreground work and
0.105 ms of fused GPU-kernel work. These results support reduced quality-path
time for these larger synthetic inputs, not universal acceleration or a measured
whole webcam/YOLO/Arduino application speedup. Small inputs had little benefit.

For real evaluation, manually label recorded original frames and create a CSV
with `image,label,fruit_id,split,x1,y1,x2,y2,confidence,mask` (mask optional).
Use GOOD/REVIEW/DEFECTIVE under documented visible-quality definitions. Boxes and
confidence must belong to that frame; the tool does not invent detections.
Paths are relative to the CSV. Keep every view of the same physical fruit in one
split, and keep near-duplicate frames together. Calibrate using train/calibration
and validation groups before freezing thresholds for the held-out test split.
The audit rejects fruit groups crossing splits and exact duplicate file/pixel
content; humans must identify near-duplicates. No labeled data was supplied, so
no empirical real-apple classification accuracy is reported.

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --manifest data\labeled\manifest.csv --backend cuda
.\.venv\Scripts\python.exe -m evaluation.evaluate --manifest data\labeled\manifest.csv --backend cpu
```

The tool evaluates only test rows, writes a new JSON report with configuration,
per-frame features/reasons, per-class precision/recall/F1, macro/weighted metrics,
confusion matrix and review coverage. It evaluates agreement with labels, not
food safety or the YOLO detector. `evaluation.evaluate_masks` provides IoU/Dice
only for actually supplied ground-truth/predicted binary masks; there is no
fabricated segmentation model. See `evaluation\README.md` for the exact mask
schema, zero-denominator policy, and commands. `data\README.md` describes label
definitions and dataset limitations; `models\README.md` explains assets.

As requested, no CNN, fake trained weights, dataset download, or environment
upgrade was introduced. A learned classifier or defect-segmentation model would
require a separately labeled dataset and held-out evaluation.

## Repository import history

The repository was imported and verified on September 27, 2026. Its initial
commits are grouped retrospectively by component and dated May–July 2026 at the
owner's request. Those dates organize the import; they are not evidence that
implementation or testing occurred on those dates. Actual validation dates and
measured results are retained in the validation documents and benchmark JSON.
