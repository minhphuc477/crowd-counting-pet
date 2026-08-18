#!/usr/bin/env python3
"""Sweep inference settings that actually affect a density-measure counter.

PET score and routing thresholds do not change ``eval_count_source=measure``.
This tool instead caches per-image measure counts for resolution/TTA views and
then searches pairwise view ensembles and optional scalar count calibration.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TEST_DATASETS = {
    "SHA", "SHB", "QNRF", "UCFCC50", "UCF_CC_50", "UCF-CC-50",
}


def parse_scale_set(value: str) -> tuple[float, ...]:
    scales = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError(f"invalid positive TTA scale set: {value!r}")
    return scales


def inclusive_grid(start: float, stop: float, step: float) -> list[float]:
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("grid bounds and step must be finite")
    if step <= 0 or stop < start:
        raise ValueError("grid requires step > 0 and stop >= start")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < stop - step * 1e-6:
        values.append(stop)
    return sorted({round(float(value), 10) for value in values})


def count_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("predictions and targets must be aligned one-dimensional arrays")
    if predictions.size == 0 or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("predictions and targets must be non-empty and finite")
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "pred_mean": float(np.mean(predictions)),
        "gt_mean": float(np.mean(targets)),
    }


def _candidate_record(
    name: str,
    kind: str,
    raw_predictions: np.ndarray,
    targets: np.ndarray,
    scale: float,
    bias: float,
) -> tuple[dict, np.ndarray]:
    calibrated = np.maximum(float(scale) * raw_predictions + float(bias), 0.0)
    record = {
        "name": name,
        "kind": kind,
        "calibration_scale": float(scale),
        "calibration_bias": float(bias),
    }
    record.update(count_metrics(calibrated, targets))
    return record, calibrated


def search_candidates(
    view_predictions: dict[str, np.ndarray],
    targets: np.ndarray,
    ensemble_alphas: list[float],
    calibration_scales: list[float],
    calibration_biases: list[float],
) -> dict:
    if not view_predictions:
        raise ValueError("at least one measure view is required")
    targets = np.asarray(targets, dtype=np.float64)
    base_candidates: list[tuple[str, str, np.ndarray]] = []
    for name, predictions in sorted(view_predictions.items()):
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape != targets.shape:
            raise ValueError(f"view {name!r} does not align with targets")
        base_candidates.append((name, "view", predictions))

    for (left_name, _, left), (right_name, _, right) in itertools.combinations(base_candidates.copy(), 2):
        for alpha in sorted({float(value) for value in ensemble_alphas if 0.0 < float(value) < 1.0}):
            name = f"blend({left_name},{right_name},right={alpha:.4g})"
            predictions = (1.0 - alpha) * left + alpha * right
            base_candidates.append((name, "view_blend", predictions))

    scale_grid = sorted({1.0, *(float(value) for value in calibration_scales)})
    bias_grid = sorted({0.0, *(float(value) for value in calibration_biases)})
    records: list[dict] = []
    best_predictions: dict[str, np.ndarray] = {}

    for name, kind, raw_predictions in base_candidates:
        record, predictions = _candidate_record(name, kind, raw_predictions, targets, 1.0, 0.0)
        record["calibration"] = "none"
        records.append(record)

        for scale in scale_grid:
            if abs(scale - 1.0) < 1e-12:
                continue
            record, predictions = _candidate_record(name, kind, raw_predictions, targets, scale, 0.0)
            record["calibration"] = "scale"
            records.append(record)

        for scale in scale_grid:
            for bias in bias_grid:
                if abs(bias) < 1e-12:
                    continue
                record, predictions = _candidate_record(name, kind, raw_predictions, targets, scale, bias)
                record["calibration"] = "affine"
                records.append(record)

    records.sort(key=lambda item: (item["mae"], item["mse"], abs(item["bias"])))
    winners = {}
    for label, accepted in (
        ("uncalibrated", {"none"}),
        ("scale_calibrated", {"none", "scale"}),
        ("affine_calibrated", {"none", "scale", "affine"}),
    ):
        winner = min(
            (record for record in records if record["calibration"] in accepted),
            key=lambda item: (item["mae"], item["mse"], abs(item["bias"])),
        )
        winners[label] = winner

    # Recreate only the three winning prediction vectors for export.
    base_by_name = {name: predictions for name, _, predictions in base_candidates}
    for label, winner in winners.items():
        raw = base_by_name[winner["name"]]
        best_predictions[label] = np.maximum(
            winner["calibration_scale"] * raw + winner["calibration_bias"],
            0.0,
        )
    return {
        "records": records,
        "winners": winners,
        "predictions": best_predictions,
    }


def _row_key(row: dict) -> str:
    return str(row.get("image_path") or row.get("image_id") or "")


def load_view_rows(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"per-image results must be a non-empty list: {path}")
    keyed = {}
    for row in rows:
        key = _row_key(row)
        if not key or key in keyed:
            raise ValueError(f"missing or duplicate image key in {path}: {key!r}")
        keyed[key] = row
    return keyed


def _tag(max_size: int, scales: tuple[float, ...], flip: bool) -> str:
    scale_text = "-".join(f"{scale:g}".replace(".", "p") for scale in scales)
    return f"max{max_size}_scales{scale_text}_{'flip' if flip else 'plain'}"


def run_measure_view(args, max_size: int, scales: tuple[float, ...], flip: bool) -> tuple[str, Path]:
    output_dir = Path(args.output_dir)
    tag = _tag(max_size, scales, flip)
    results_path = output_dir / "views" / f"{tag}.json"
    rows_path = output_dir / "views" / f"{tag}_per_image.json"
    log_path = output_dir / "views" / f"{tag}.log"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_path.is_file() and results_path.is_file() and not args.force:
        print(f"reuse {tag}: {rows_path}")
        return tag, rows_path

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval.py"),
        "--resume", args.resume,
        "--checkpoint_model_key", args.checkpoint_model_key,
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--backbone", args.backbone,
        "--eval_image_set", args.eval_image_set,
        "--eval_max_size", str(max_size),
        "--eval_count_source", "measure",
        "--tta_scales", ",".join(f"{scale:g}" for scale in scales),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--no_localization_metrics",
        "--results_file", str(results_path),
        "--per_image_results_file", str(rows_path),
    ]
    cmd.append("--tta_flip" if flip else "--no_tta_flip")
    print(f"run {tag}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=None if args.timeout <= 0 else args.timeout,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"measure view {tag} failed; inspect {log_path}")
    return tag, rows_path


def write_outputs(output_dir: Path, keys: list[str], targets: np.ndarray, search: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name", "kind", "calibration", "calibration_scale", "calibration_bias",
            "mae", "mse", "bias", "pred_mean", "gt_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in search["records"])

    report = {
        "images": len(keys),
        "winners": search["winners"],
        "note": (
            "uncalibrated uses only measure TTA/resolution views; scale and affine "
            "winners additionally fit scalar count calibration on this evaluation split"
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for label, predictions in search["predictions"].items():
        rows = [
            {
                "image_key": key,
                "gt_cnt": float(target),
                "pred_cnt": float(prediction),
                "abs_error": float(abs(prediction - target)),
            }
            for key, target, prediction in zip(keys, targets, predictions)
        ]
        with (output_dir / f"best_{label}_predictions.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sweep density-measure inference")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--dataset_file", default="QNRF")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--backbone", default="vgg16_bn")
    parser.add_argument("--checkpoint_model_key", default="model", choices=("auto", "model", "model_ema", "model_raw"))
    parser.add_argument("--eval_image_set", default="val", choices=("val", "train_eval", "train_holdout"))
    parser.add_argument("--allow_benchmark_test_sweep", action="store_true")
    parser.add_argument("--eval_max_sizes", nargs="+", type=int, default=[1536])
    parser.add_argument(
        "--tta_scale_sets", nargs="+", default=["1.0", "0.95,1.0,1.05", "0.9,1.0,1.1"],
        help="space-separated comma-delimited scale sets",
    )
    parser.add_argument("--flip_modes", nargs="+", choices=("plain", "flip"), default=["plain", "flip"])
    parser.add_argument("--ensemble_alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--calibration_scale_min", type=float, default=0.94)
    parser.add_argument("--calibration_scale_max", type=float, default=1.06)
    parser.add_argument("--calibration_scale_step", type=float, default=0.0025)
    parser.add_argument("--calibration_bias_min", type=float, default=-30.0)
    parser.add_argument("--calibration_bias_max", type=float, default=30.0)
    parser.add_argument("--calibration_bias_step", type=float, default=2.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    checkpoint = Path(args.resume)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if (
        args.dataset_file in BENCHMARK_TEST_DATASETS
        and args.eval_image_set == "val"
        and not args.allow_benchmark_test_sweep
    ):
        raise ValueError("benchmark-test measure sweep requires --allow_benchmark_test_sweep")

    scale_sets = list(dict.fromkeys(parse_scale_set(value) for value in args.tta_scale_sets))
    flip_modes = list(dict.fromkeys(args.flip_modes))
    view_paths = {}
    for max_size, scales, flip_mode in itertools.product(args.eval_max_sizes, scale_sets, flip_modes):
        name, path = run_measure_view(args, max_size, scales, flip_mode == "flip")
        view_paths[name] = path

    keyed_views = {name: load_view_rows(path) for name, path in view_paths.items()}
    first_name = next(iter(keyed_views))
    keys = sorted(keyed_views[first_name])
    if any(sorted(rows) != keys for rows in keyed_views.values()):
        raise ValueError("measure views contain different image keys")

    targets = np.asarray([keyed_views[first_name][key]["gt_cnt"] for key in keys], dtype=np.float64)
    predictions = {}
    for name, rows in keyed_views.items():
        view_targets = np.asarray([rows[key]["gt_cnt"] for key in keys], dtype=np.float64)
        if not np.array_equal(view_targets, targets):
            raise ValueError(f"ground-truth counts differ in view {name}")
        predictions[name] = np.asarray([rows[key]["pred_cnt"] for key in keys], dtype=np.float64)

    search = search_candidates(
        predictions,
        targets,
        args.ensemble_alphas,
        inclusive_grid(args.calibration_scale_min, args.calibration_scale_max, args.calibration_scale_step),
        inclusive_grid(args.calibration_bias_min, args.calibration_bias_max, args.calibration_bias_step),
    )
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, keys, targets, search)
    for label, winner in search["winners"].items():
        print(
            f"{label}: MAE={winner['mae']:.4f} MSE={winner['mse']:.4f} "
            f"bias={winner['bias']:.4f} source={winner['name']} "
            f"scale={winner['calibration_scale']:.5f} offset={winner['calibration_bias']:.2f}"
        )
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
