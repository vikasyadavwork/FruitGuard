# FruitGuard Vision — custom CUDA classifier validation

Implementation stays in `D:\AurdinoCudaProject`. This extends the Milestone 1
baseline documented in `VALIDATION.md`, following the user's clarified scope:
YOLO fruit detection plus an own CUDA color/dark/symmetry heuristic classifier,
rather than a trained neural defect classifier. No packages, Python environment,
PyTorch wheels, models, datasets, or firmware were replaced or upgraded.

## Executed checks

- Native build succeeded with **CUDA Toolkit 12.4**, **MSVC 14.38**, explicit
  `-gencode arch=compute_61,code=sm_61`, on the actual GTX 1050 Ti.
- Final complete suite: **100 tests passed, no skips or failures**, with both
  GPU opt-in flags set. One earlier full run skipped four quality GPU tests
  because only the preprocessing flag was set; the final run corrected that.
- Six preprocessing tests include actual CUDA transforms/reuse/resizing and
  unavailable-device diagnostics followed by successful valid-device recovery.
- Twelve custom CUDA quality tests include actual GPU comparisons to the CPU
  reference to 12 decimal places, random non-block-aligned/asymmetric masks,
  grayscale values, HSV threshold grids, dark foreground inclusion, and buffer
  reuse. Eight of these are bridge contracts and four use the actual GPU.
- Sixteen evaluation tests cover classification metrics, IoU/Dice, undefined
  denominator policies, mask validation, missing/corrupt images, fruit split
  leakage, duplicate pixels, held-out test selection, overwrite refusal, and
  synthetic report labeling.
- Existing serial, detector, ROI, conservative-mask, stale-result, saving, and
  shutdown tests remain present. New integration tests verify measurement
  backends receive only reliable foreground, never fall back silently, and
  missing native classifiers use fatal error cleanup.
- `compileall` for src/tests/evaluation passed. `pip check` reports no broken
  requirements. CLI help matches documented options.
- Real webcam + PyTorch CUDA + native classifier initialization smoke run
  completed normally after 12 frames in explicit no-Arduino mode; no apple
  detected, intended OFF logged without serial access. This verifies startup
  coexistence, not real-apple classification accuracy.
- The current application also completed a 120-frame real webcam/COM5 run
  after the final native rebuild. Both YOLO/native CUDA backends initialized;
  an OFF command was successfully written and normal cleanup completed.
  LED state was not observed or acknowledged.
- Actual CUDA evaluation CLI completed on temporary synthetic frame/mask
  fixtures; its report was marked synthetic and fixtures were cleaned. No
  real labeled evaluation was possible because no labeled data was supplied.

## Actual measured benchmarks

Preprocessing report: `evaluation/cuda_benchmark.json`. Ten warmups and 100
measurements per case; sizes 1x1,17x19,640x640,1920x1080; maximum absolute output
error **0**, tolerance 1e-6. Setup/allocation, CPU reference, pageable H2D,
synchronized kernel-only, D2H, and separately measured complete CUDA time are
reported. At 640x640 the CPU reference was 0.997 ms versus 1.567 ms complete
CUDA-to-host; at 1920x1080, 5.509 ms versus 8.048 ms. Returning every normalized
float to host made these complete CUDA paths slower. No whole-path acceleration
is claimed from their faster kernel-only figures.

Final custom-classifier report:
`evaluation/reports/cuda_quality_benchmark_8ad6d954fcc44c22a915faf6835a3353.json`.
Twelve synthetic inputs × 100 measured iterations after 10 warmups; tolerance
1e-9; **zero** feature, validity, or decision comparison failures. This final
run used the current DLL without competing application/build workloads.

Mean full quality-analysis timings for plain-background synthetic frames:

| Frame | CPU | Custom CUDA | CUDA kernel | CPU mask within CUDA path |
|---|---:|---:|---:|---:|
| 128x128 | 1.431 ms | 1.433 ms | 0.019 ms | 1.085 ms |
| 256x256 | 3.087 ms | 2.854 ms | 0.025 ms | 2.261 ms |
| 512x512 | 9.416 ms | 7.725 ms | 0.105 ms | 6.679 ms |

The fused custom kernel and compact partial-count transfer reduced quality-path
time for these larger synthetic inputs; this is not a measured end-to-end
webcam/YOLO/serial speedup. CPU foreground work remains material. The initial
report `cuda_quality_benchmark_0e11a35d577841ac8a531ed3137dceb6.json` is retained,
but the final uncontended report above is used for the documented timings.
Neither report is real-apple accuracy evidence.

## Commands actually executed

All Python commands use the existing interpreter, from the unchanged root:

```powershell
Set-Location D:\AurdinoCudaProject
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
.\scripts\build_cuda.ps1
.\scripts\benchmark_cuda.ps1 -Runs 100 -Warmup 10
$env:FRUITGUARD_TEST_CUDA='1'
.\.venv\Scripts\python.exe -m unittest tests.test_preprocessing -v
$env:FG_RUN_CUDA_TESTS='1'
.\.venv\Scripts\python.exe -m unittest tests.test_cuda_quality -v
.\.venv\Scripts\python.exe -m unittest tests.test_classifier_integration -v
.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v
.\.venv\Scripts\python.exe -m evaluation.benchmark_quality --synthetic --warmup 10 --iterations 100 --tolerance 1e-9
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m compileall -q src tests evaluation
.\.venv\Scripts\python.exe -m src.main --help
.\.venv\Scripts\python.exe -m src.main --no-arduino --max-frames 12
.\.venv\Scripts\python.exe -m src.main --max-frames 120
Get-FileHash src\cuda\ImagePreprocessLegacy.cu -Algorithm SHA256
```

Additional focused import/syntax checks and native rebuilds were executed after
fixing a real stale CUDA-error recovery problem and improving cleanup reporting.
The build script was rerun after adding the missing-classifier-source guard.
The synthetic CUDA evaluator invoked `evaluation.evaluate.main` with a
temporary recorded-frame manifest and `--backend cuda --synthetic-smoke`;
no fake model weights or real quality metrics were created.

## Files changed

Modified: `README.md`, `configs/config.yaml`, `src/application.py`,
`src/display.py`, `src/inspection.py`, `src/utils/config.py`,
`src/vision/quality_analyzer.py`, `src/cuda/ImagePreprocess.cu`,
`tests/test_application_loop.py`.

Added:

- `src/cuda/ImagePreprocess.cuh`, `ImagePreprocessLegacy.cu`,
  `BenchmarkPreprocess.cu`, `DefectAnalysis.cu`, `DefectAnalysis.cuh`.
- `src/vision/preprocessing.py`, `src/vision/cuda_quality.py`.
- `scripts/build_cuda.ps1`, `scripts/benchmark_cuda.ps1`.
- `evaluation/__init__.py`, `metrics.py`, `manifest.py`, `evaluate.py`,
  `evaluate_masks.py`, `benchmark_quality.py`, `README.md`, measured JSON reports.
- `tests/test_preprocessing.py`, `test_cuda_quality.py`, `test_evaluation.py`,
  `test_classifier_integration.py`.
- `data/README.md`, `models/README.md`, `VALIDATION_CUDA.md`.
- New native outputs in build: `fruitguard_preprocess.dll/.lib/.exp` and
  `BenchmarkPreprocess.exe/.lib/.exp`. Existing CudaHello outputs remain intact.

Original ImagePreprocess source is preserved exactly in ImagePreprocessLegacy:
SHA256 `A19FEFEB1D9908D17D7AE4265710F17D4BC802231B51F2A54C7C51D2134603DC`.
Original serial/webcam test scripts, `CudaHello.cu`, local `yolo11n.pt`, firmware
and environment were not modified. No CNN/training/checkpoint files were added.

## Manual validation still required

Present real apples in the controlled lighting/background to check detection,
foreground validity, natural color variations, shadows, occlusions and viewing
angles. Calibrate thresholds with a human-labeled validation set, then measure
precision/recall/F1/confusion matrix on separate held-out fruit groups.

Physically verify each LED command and cable/sensor/camera disconnect behavior;
the existing protocol has no LED acknowledgement. Synthetic/mocked tests verify
software behavior only. Actual defect classification and model accuracy have
not been established. The system still estimates visible quality only and does
not determine food safety or scientifically measure freshness.
