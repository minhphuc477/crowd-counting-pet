#!/usr/bin/env python3
"""Sweep PET inference thresholds for an existing checkpoint.

This is intentionally an evaluation-only tool. Including the current paper
thresholds in the grid gives a validation-set result that cannot be worse than
the baseline on the same checkpoint and split.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_THRESHOLDS = (0.35, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.65)
DEFAULT_SPLIT_THRESHOLDS = (0.5,)


def _unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(v), 6) for v in values})


def _load_checkpoint_args(checkpoint_path: Path):
    try:
        import torch
    except Exception:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")
    if args is None:
        return None
    return args


def _checkpoint_arg(checkpoint_args, key: str):
    if checkpoint_args is None:
        return None
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key)
    return getattr(checkpoint_args, key, None)


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_args = _load_checkpoint_args(checkpoint_path)
    if not args.dataset_file:
        args.dataset_file = _checkpoint_arg(checkpoint_args, "dataset_file") or "SHA"
    if not args.data_path:
        args.data_path = _checkpoint_arg(checkpoint_args, "data_path") or ""
    if not args.backbone:
        args.backbone = _checkpoint_arg(checkpoint_args, "backbone") or "vgg16_bn"
    if not args.output_dir:
        args.output_dir = str(checkpoint_path.resolve().parent / "threshold_sweep")
    return args


def run_eval(
    args: argparse.Namespace,
    score_threshold: float,
    split_threshold: float,
    run_dir: Path,
) -> dict:
    tag = f"score_{score_threshold:.6g}_split_{split_threshold:.6g}".replace(".", "p")
    results_file = run_dir / f"{tag}.json"
    log_file = run_dir / f"{tag}.log"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval.py"),
        "--backbone",
        args.backbone,
        "--dataset_file",
        args.dataset_file,
        "--resume",
        args.resume,
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--results_file",
        str(results_file),
        "--override_score_threshold",
        str(score_threshold),
        "--override_split_threshold",
        str(split_threshold),
    ]
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout if args.timeout > 0 else None,
    )
    log_file.write_text(
        "command: " + " ".join(cmd) + "\n\n" + proc.stdout,
        encoding="utf-8",
    )

    record = {
        "score_threshold": float(score_threshold),
        "split_threshold": float(split_threshold),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - started),
        "results_file": str(results_file),
        "log_file": str(log_file),
    }
    if proc.returncode == 0 and results_file.is_file():
        metrics = json.loads(results_file.read_text(encoding="utf-8"))
        record.update(metrics)
        record["ok"] = True
    else:
        record["ok"] = False
        record["error"] = f"eval.py failed with return code {proc.returncode}"
    return record


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_results.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "ok",
        "score_threshold",
        "split_threshold",
        "eval_mae",
        "eval_mse",
        "pred_cnt",
        "gt_cnt",
        "epoch",
        "returncode",
        "elapsed_sec",
        "results_file",
        "log_file",
    ]
    with (output_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    ok_records = [record for record in records if record.get("ok") and "eval_mae" in record]
    if ok_records:
        best = min(ok_records, key=lambda item: item["eval_mae"])
        (output_dir / "best_thresholds.json").write_text(
            json.dumps(best, indent=2) + "\n",
            encoding="utf-8",
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sweep PET eval thresholds")
    parser.add_argument("--resume", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--dataset_file", default="", help="Dataset name; default reads checkpoint args")
    parser.add_argument("--data_path", default="", help="Dataset root; default reads checkpoint args")
    parser.add_argument("--backbone", default="", help="Backbone; default reads checkpoint args")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--timeout", default=1800, type=int, help="Per-eval timeout in seconds; 0 disables")
    parser.add_argument("--output_dir", default="", help="Where to save sweep logs/results")
    parser.add_argument(
        "--score_thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCORE_THRESHOLDS),
    )
    parser.add_argument(
        "--split_thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_SPLIT_THRESHOLDS),
    )
    return parser.parse_args()


def main() -> int:
    args = resolve_runtime_args(get_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = _unique_sorted(args.score_thresholds)
    splits = _unique_sorted(args.split_thresholds)
    records = []
    total = len(scores) * len(splits)
    index = 0
    for split_threshold in splits:
        for score_threshold in scores:
            index += 1
            print(
                f"[{index}/{total}] score_threshold={score_threshold} "
                f"split_threshold={split_threshold}"
            )
            record = run_eval(args, score_threshold, split_threshold, output_dir)
            records.append(record)
            if record.get("ok"):
                print(f"  mae={record['eval_mae']:.4f} mse={record['eval_mse']:.4f}")
            else:
                print(f"  failed; see {record['log_file']}")
            write_outputs(records, output_dir)

    ok_records = [record for record in records if record.get("ok") and "eval_mae" in record]
    if not ok_records:
        print(f"No successful evaluations. Logs are in {output_dir}")
        return 1

    best = min(ok_records, key=lambda item: item["eval_mae"])
    print(
        "Best: "
        f"mae={best['eval_mae']:.4f} mse={best['eval_mse']:.4f} "
        f"score_threshold={best['score_threshold']} "
        f"split_threshold={best['split_threshold']}"
    )
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
