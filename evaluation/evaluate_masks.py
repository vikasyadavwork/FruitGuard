"""IoU/Dice of actual supplied ground-truth and predicted masks only."""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from evaluation.evaluate import new_report_path, project_path, save_report
from evaluation.metrics import segmentation_metrics


def load_mask(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path), allow_pickle=False)
    else:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        if mask is None or mask.ndim != 2 or not np.all(np.isin(mask, [0, 1, 255])):
            raise ValueError("Image masks must be single-channel binary 0/1 or 0/255")
        mask = (mask > 0).astype(np.uint8)
    return mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV columns ground_truth,predicted")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        output = project_path(args.output) if args.output else new_report_path("supplied_mask_metrics")
        if output.exists():
            raise ValueError("Output exists; refusing overwrite")
        rows = []
        manifest = project_path(args.manifest)
        with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["ground_truth", "predicted"]:
                raise ValueError("Mask manifest requires exactly ground_truth,predicted columns")
            for row in reader:
                if None in row or any(not value for value in row.values()):
                    raise ValueError("Malformed mask manifest row")
                paths = [(manifest.parent / row[key]).resolve()
                         for key in ("ground_truth", "predicted")]
                metrics = segmentation_metrics(load_mask(paths[0]), load_mask(paths[1]), args.threshold)
                rows.append({"ground_truth": str(paths[0]), "predicted": str(paths[1]), **metrics})
        if not rows:
            raise ValueError("No supplied mask pairs; no segmentation metrics calculated")
        report = {"application": "FruitGuard Vision", "sample_count": len(rows),
                  "threshold": args.threshold, "samples": rows,
                  "mean_iou": sum(row["iou"] for row in rows) / len(rows),
                  "mean_dice": sum(row["dice"] for row in rows) / len(rows),
                  "empty_mask_policy": "Both empty: 1; only one empty: 0",
                  "claim": "Agreement of supplied masks only. Current detector/classifier is not a defect segmentation model."}
        save_report(report, output)
        print("Saved supplied-mask evaluation: {}".format(output.resolve()))
    except (ValueError, OSError, RuntimeError) as exc:
        print("Mask evaluation failed: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
