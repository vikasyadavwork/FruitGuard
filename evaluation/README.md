# FruitGuard Vision evaluation

These tools measure agreement with human **visible-quality** labels and compare CPU/custom CUDA numerical outputs and real timing. They do not evaluate freshness, infection, contamination or food safety. No real labeled apple dataset was supplied. Synthetic tests and benchmarks establish code behavior and timing only; no real-apple accuracy is reported.

Use the existing Python 3.8 environment without installing another model or dependency:

```powershell
Set-Location D:\AurdinoCudaProject
.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v
.\.venv\Scripts\python.exe -m evaluation.benchmark_quality --synthetic --warmup 10 --iterations 100 --tolerance 1e-9
```

The benchmark uses actual custom CUDA quality measurements and the CPU reference on matching controlled synthetic frames. It includes externally supplied masks and plain-background foreground estimates, warm-up, synchronized measurements, separate transfer/kernel/allocation/measurement/mask timings and full quality-path wall time. It excludes YOLO, webcam, serial, display, file decode and sample writing. It refuses unavailable CUDA rather than switching to CPU. CUDA context creation is outside per-frame timings; reusable-buffer allocation/growth is recorded. A CUDA kernel time alone does not demonstrate an end-to-end speedup. A unique JSON report is saved under `evaluation/reports/`; `--output` must name a new file.

## Recorded labeled frames

Prepare a UTF-8 CSV with these exact required fields and the optional `mask` column:

```csv
image,label,fruit_id,split,x1,y1,x2,y2,confidence,mask
images/apple_001.png,GOOD,fruit_001,test,35,25,145,135,0.91,
```

`image` must be the full original frame. `x1,y1,x2,y2` and `confidence` must be actual recorded apple detection values for that frame, not invented detector outputs. Bounding boxes use xyxy image coordinates. Finite confidence must be in 0..1; it is object-detection confidence, not probability of health. The image cannot be a rectangular apple crop implicitly treated as its silhouette. Optional `mask` is an actual binary externally supplied **foreground** mask, single-channel 0/1 or 0/255 and matching the full frame; it is not a defect mask or an output of a trained defect segmentation model. If omitted, the current conservative plain-background estimate is used, and unreliable measurements return REVIEW.

Labels are exactly `GOOD`, `REVIEW` and `DEFECTIVE`. Define them for visible appearance: GOOD means no apparent defect within the chosen inspection protocol; DEFECTIVE means clear visible damage/abnormality; REVIEW means ambiguous appearance, poor lighting, occlusion or insufficient inspection evidence. Write a consistent human labeling policy and retain limitations. Never use these labels as food-safety labels. Each physical apple has a stable `fruit_id`, including its different views and near-duplicate frames.

`split` is exactly `train`, `val` or `test`. These names support future data management; **no neural model training is implemented**. Use train/val for future heuristic calibration and reserve test for held-out evaluation. Every split is audited before selecting test: a fruit cannot cross splits, duplicate paths/file hashes/decoded pixel content are rejected, images/masks must decode, labels/boxes/confidence must be valid. The audit cannot detect every near-duplicate image or incorrect human fruit ID; keep capture sessions and views of one fruit in one split manually. At least one test row is required; empty datasets produce no metrics.

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --manifest data\labeled\manifest.csv --backend cpu
.\.venv\Scripts\python.exe -m evaluation.evaluate --manifest data\labeled\manifest.csv --backend cuda
.\.venv\Scripts\python.exe -m evaluation.benchmark_quality --manifest data\labeled\manifest.csv --warmup 10 --iterations 100
```

The CUDA path reads `classifier.library_path` and `classifier.device_index` from `configs/config.yaml`; override the index with `--cuda-device 0`. `--config` can select another validated configuration. All relative CLI paths are anchored at `D:\AurdinoCudaProject`, and manifest image/mask paths are relative to that manifest. Existing report files are never overwritten. The benchmark requires reliable foreground for each selected test sample; empirical evaluation records REVIEW when a mask is unreliable.

Reports include per-class precision/recall/F1, macro and support-weighted means, confusion matrix (truth rows, prediction columns), review fraction and decisive coverage (`1 - review_fraction`). Precision/recall/F1 are zero when their denominator is zero; absent classes therefore affect the macro mean and their support is explicit. Scores describe agreement with supplied human labels on the supplied test set, not universal quality accuracy. Detector performance is not evaluated because boxes/confidence are recorded inputs. Configuration, manifest/image hashes, reasons, validity, features and quality timing accompany each result. Synthetic fixtures must use `--synthetic-smoke` so their reports explicitly deny a real-apple accuracy claim.

## Optional supplied mask metrics

This utility evaluates actual ground-truth/predicted mask pairs supplied by the user. It does not create fake segmentation predictions or train a segmentation model. CSV columns must be exactly `ground_truth,predicted`:

```csv
ground_truth,predicted
masks/apple_truth.png,masks/apple_prediction.png
```

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_masks --manifest data\mask_pairs.csv --threshold 0.5
```

Image masks must be single-channel binary 0/1 or 0/255. NumPy `.npy` files may contain finite real 0..1 probabilities; pickle loading is disabled. Foreground is `value >= threshold`, with `0 < threshold <= 1`. Dimensions must match. IoU is intersection/union and Dice is twice intersection divided by total foreground. Both-empty pairs score 1; only one empty mask scores 0. Reports list per-pair values and unweighted pair means so empty pairs and sample counts remain visible. No report is generated without actual supplied pairs.
