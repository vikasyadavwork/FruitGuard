# FruitGuard Vision model assets

The existing object detector remains `../yolo11n.pt`; it detects apples using
model class names. Set `detector.model_path` in `configs/config.yaml` to switch
to compatible local weights. No download is performed for missing weights.

The custom color/dark/symmetry classifier has CUDA source code and configurable
rules, not neural model weights. Its compiled native library is under `build/`.
YOLO confidence describes apple detection only; it is not defect probability.
