"""Evaluate recorded detections on held-out labeled frames, never freshness."""
import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from evaluation.manifest import read_foreground_mask, read_frame, read_manifest
from evaluation.metrics import classification_metrics
from src.utils.config import ROOT, load_config
from src.vision.quality_analyzer import analyze


def new_report_path(prefix: str) -> Path:
    return ROOT / "evaluation" / "reports" / (prefix + "_" + uuid.uuid4().hex + ".json")


def project_path(path: Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def save_report(report, path: Path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def quality_backend(name, device_index=None, config=None):
    if name == "cpu":
        return None
    # The CUDA bridge checks the selected compiled runtime/device. No fallback.
    from src.vision.cuda_quality import CudaQualityMeasurements
    classifier = config["classifier"] if config is not None else {}
    selected_index = device_index if device_index is not None else classifier.get("device_index", 0)
    return CudaQualityMeasurements(library_path=classifier.get("library_path"), device_index=selected_index)


def evaluate(manifest: Path, config, backend=None, synthetic_smoke=False):
    samples = read_manifest(manifest)
    test_samples = [sample for sample in samples if sample.split == "test"]
    if not test_samples:
        raise ValueError("Manifest has no test split; no evaluation was performed")
    rows = []
    for sample in test_samples:
        frame = read_frame(sample.image)
        mask = read_foreground_mask(sample.mask, frame.shape[:2]) if sample.mask else None
        detection = SimpleNamespace(bbox=sample.bbox, confidence=sample.confidence,
                                    class_name="apple", mask=mask)
        started = perf_counter()
        if backend is None:
            result = analyze(frame, detection, config)
        else:
            result = analyze(frame, detection, config, measurement_backend=backend)
        elapsed_ms = 1000 * (perf_counter() - started)
        if sample.mask:
            # This is an externally supplied foreground annotation, not a model prediction.
            result["validity"]["mask_source"] = "external_foreground_annotation"
        rows.append({"image": str(sample.image), "image_sha256": sample.sha256,
                     "fruit_id": sample.fruit_id, "truth": sample.label,
                     "recorded_detection_confidence": sample.confidence,
                     "result": result, "quality_wall_ms": elapsed_ms})
    metrics = classification_metrics([row["truth"] for row in rows],
                                     [row["result"]["status"] for row in rows])
    return {"application": "FruitGuard Vision", "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_kind": "synthetic_behavior_check" if synthetic_smoke else "human_labeled_test_frames",
            "claim": "Synthetic behavior only; no real-apple accuracy claim" if synthetic_smoke else
                     "Agreement with supplied visible-quality labels; no food-safety or freshness claim",
            "manifest": str(Path(manifest).resolve()),
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "split": "test", "audited_rows_all_splits": len(samples),
            "detection_source": "recorded boxes/confidence; detector is not evaluated",
            "config_snapshot": config, "metrics": metrics, "samples": rows,
            "timing_boundary": "quality analysis including CPU foreground estimate, measurements, decision; excludes image decoding and YOLO"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, help="Override classifier.device_index only with --backend cuda")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Mark all outputs as synthetic behavior checks")
    parser.add_argument("--output", type=Path, help="New JSON file; existing output is refused")
    args = parser.parse_args(argv)
    try:
        output = project_path(args.output) if args.output else new_report_path("visible_quality")
        if output.exists():
            raise ValueError("Output exists; refusing overwrite: {}".format(output))
        config = load_config(args.config)
        if args.cuda_device is not None and args.cuda_device < 0:
            raise ValueError("CUDA device index must not be negative")
        backend = quality_backend(args.backend, args.cuda_device, config)
        try:
            report = evaluate(project_path(args.manifest), config, backend, args.synthetic_smoke)
            if backend is not None:
                report["cuda_library"] = str(backend.library_path)
                report["cuda_library_sha256"] = hashlib.sha256(backend.library_path.read_bytes()).hexdigest()
        finally:
            if backend is not None:
                backend.close()
        report["measurement_backend"] = args.backend
        report["cuda_device"] = (args.cuda_device if args.cuda_device is not None else config["classifier"]["device_index"]) if args.backend == "cuda" else None
        save_report(report, output)
        print("Saved {} labeled test-frame results: {}".format(report["metrics"]["sample_count"], output.resolve()))
        print(report["claim"])
    except (ValueError, OSError, RuntimeError) as exc:
        print("Evaluation failed: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
