#!/usr/bin/env python3
"""Average per-image count predictions from multiple eval.py runs.

This is for fixed, pre-declared checkpoint/seed ensembling. It intentionally
does not choose per-image models, tune thresholds, or use GT-dependent routing.
Each input must be a per-image results JSON produced by:

    python eval.py ... --per_image_results_file path.json
"""

import argparse
import json
import math
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser("Average per-image count predictions")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="per-image result JSON files from eval.py",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="optional nonnegative weights, one per input; default is uniform",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output JSON summary with per-image ensemble rows",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    keyed = {}
    for row in rows:
        key = (str(row.get("image_id", "")), str(row.get("image_path", "")))
        if key in keyed:
            raise ValueError(f"{path} contains duplicate image key {key}")
        keyed[key] = row
    return keyed


def main():
    args = get_args()
    input_paths = [Path(item) for item in args.inputs]
    if args.weights is None or len(args.weights) == 0:
        weights = [1.0] * len(input_paths)
    else:
        weights = list(args.weights)
    if len(weights) != len(input_paths):
        raise ValueError("--weights must have the same length as --inputs")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("--weights must be nonnegative and sum to a positive value")
    weight_sum = float(sum(weights))
    weights = [float(weight) / weight_sum for weight in weights]

    keyed_runs = [load_rows(path) for path in input_paths]
    keys = set(keyed_runs[0].keys())
    for path, keyed in zip(input_paths[1:], keyed_runs[1:]):
        missing = keys.difference(keyed.keys())
        extra = set(keyed.keys()).difference(keys)
        if missing or extra:
            raise ValueError(
                f"{path} image keys do not match first input: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    per_image = []
    abs_sum = 0.0
    sq_sum = 0.0
    pred_sum = 0.0
    gt_sum = 0.0
    for key in sorted(keys):
        rows = [run[key] for run in keyed_runs]
        gt_values = {float(row["gt_cnt"]) for row in rows}
        if len(gt_values) != 1:
            raise ValueError(f"GT mismatch for image key {key}: {sorted(gt_values)}")
        gt = gt_values.pop()
        pred = sum(weight * float(row["pred_cnt"]) for weight, row in zip(weights, rows))
        error = pred - gt
        abs_error = abs(error)
        sq_error = error * error
        abs_sum += abs_error
        sq_sum += sq_error
        pred_sum += pred
        gt_sum += gt
        per_image.append({
            "image_id": key[0],
            "image_path": key[1],
            "gt_cnt": gt,
            "pred_cnt": pred,
            "abs_error": abs_error,
            "sq_error": sq_error,
            "member_pred_cnt": [float(row["pred_cnt"]) for row in rows],
        })

    count = max(len(per_image), 1)
    summary = {
        "inputs": [str(path) for path in input_paths],
        "weights": weights,
        "num_images": len(per_image),
        "mae": abs_sum / count,
        "mse": math.sqrt(sq_sum / count),
        "pred_cnt": pred_sum / count,
        "gt_cnt": gt_sum / count,
        "per_image": sorted(per_image, key=lambda row: row["sq_error"], reverse=True),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Ensemble: mae={summary['mae']:.4f} mse={summary['mse']:.4f} "
        f"pred={summary['pred_cnt']:.4f} gt={summary['gt_cnt']:.4f}"
    )
    print(f"saved to: {output}")


if __name__ == "__main__":
    main()
