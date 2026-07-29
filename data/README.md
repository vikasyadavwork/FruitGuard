# FruitGuard Vision samples and labels

`inspections/` holds captured original frames, optional apple ROIs, and same-frame
JSON metadata from S. These images are unlabeled; inferred status is not ground
truth. No dataset is downloaded automatically.

For empirical evaluation, copy the manifest schema from `evaluation/README.md`
and label samples manually as GOOD, REVIEW, or DEFECTIVE according to recorded
visible-quality definitions. Record all views/frames of the same physical fruit
with the same `fruit_id`. Keep train/calibration, validation, and held-out test
groups separate; do not distribute near-duplicate views between splits.

GOOD means visibly acceptable under your label policy, DEFECTIVE means a visible
defect under that policy, and REVIEW means uncertain visibility/labeling. None
of these labels describes food safety, infection, contamination, or scientific
freshness. Record variety, lighting, background, viewpoint, severity, uncertain
labels, and dataset limitations alongside the manifest. Brown natural russeting,
shadows, stems and reflections need deliberate label handling.

Calibrate YAML thresholds using calibration/validation groups and freeze them
before held-out test evaluation. The custom CUDA classifier is a deterministic
heuristic classifier; there are no trained defect-model weights or invented
accuracy figures. Future neural training would require a separately defined
labeled dataset and model validation.
