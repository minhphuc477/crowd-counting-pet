#!/usr/bin/env python3
"""Compare PET run directories, checkpoints, and sweep outputs.

This script is intentionally read-only. It is meant for cases where two runs
look equivalent from memory but produce different MAE. It prints the stored
checkpoint args, best/latest eval summaries, and threshold-sweep best records
side by side so config drift is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARG_KEYS = (
    "model_recipe",
    "backbone",
    "timm_adapter",
    "no_pretrained_backbone",
    "resume",
    "resume_model_only",
    "resume_allow_arch_change",
    "no_auto_freeze_bn_on_count_head_resume",
    "pet_loss_variant",
    "split_loss_variant",
    "apg_loss_coef",
    "apg_start_epoch",
    "apg_warmup_epochs",
    "apg_end_epoch",
    "apg_contrastive_coef",
    "apg_neg_k",
    "apg_margin",
    "count_head_loss_coef",
    "count_head_loss_type",
    "count_head_start_epoch",
    "count_head_end_epoch",
    "count_head_feature_grad_scale",
    "count_head_warmup_epochs",
    "density_map_loss_coef",
    "lr",
    "lr_backbone",
    "lr_scheduler",
    "lr_drop",
    "lr_gamma",
    "batch_size",
    "epochs",
    "warmup_epochs",
    "patch_size",
    "crop_attempts",
    "min_crop_points",
    "freeze_bn",
    "ema_decay",
    "eval_model",
    "score_threshold",
    "split_threshold",
    "eval_nms_radius",
    "eval_branch_gate",
    "eval_soft_split_gate",
    "eval_count_mode",
    "eval_score_calibration",
    "eval_foreground_gate",
    "eval_foreground_gate_mode",
    "eval_foreground_gate_strength",
    "seed",
)

METRIC_KEYS = (
    "epoch",
    "test_mae",
    "eval_mae",
    "mae",
    "test_mse",
    "eval_mse",
    "mse",
    "pred_cnt",
    "gt_cnt",
    "score_threshold",
    "split_threshold",
    "eval_nms_radius",
    "eval_branch_gate",
    "eval_soft_split_gate",
    "eval_count_mode",
    "eval_score_calibration",
    "loc_f1_large",
    "loc_prec_large",
    "loc_rec_large",
    "loc_f1_small",
    "loc_prec_small",
    "loc_rec_small",
)


def arg_value(args: Any, key: str) -> Any:
    if args is None:
        return None
    if isinstance(args, dict):
        return args.get(key)
    return getattr(args, key, None)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def metric_mae(record: dict[str, Any]) -> float:
    for key in ("test_mae", "eval_mae", "mae"):
        if key in record and record[key] is not None:
            return float(record[key])
    return float("inf")


def short_record(record: dict[str, Any] | None) -> str:
    if not record:
        return "missing"
    parts = []
    for key in METRIC_KEYS:
        if key not in record or record[key] is None:
            continue
        value = record[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "no standard metric keys"


def resolve_checkpoint(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".pth":
        return path
    if path.is_dir():
        for name in ("best_checkpoint.pth", "checkpoint.pth", "final_checkpoint.pth"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    return None


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    checkpoint_path = resolve_checkpoint(path)
    if checkpoint_path is None:
        return None
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "path": checkpoint_path,
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_mae": checkpoint.get("best_mae"),
        "best_mse": checkpoint.get("best_mse"),
        "has_model_ema": "model_ema" in checkpoint,
        "has_model_raw": "model_raw" in checkpoint,
        "args": checkpoint.get("args"),
    }


def load_best_sweep(path: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if path.is_file() and path.suffix.lower() == ".json":
        candidates.append(path)
    if path.is_dir():
        candidates.extend([
            path / "best_thresholds.json",
            path / "sweep_results.json",
        ])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = load_json(candidate)
        if isinstance(data, dict):
            if "eval_mae" in data or "test_mae" in data or "mae" in data:
                data = dict(data)
                data["_source"] = str(candidate)
                return data
        if isinstance(data, list):
            ok = [
                row for row in data
                if isinstance(row, dict) and row.get("ok", True) and metric_mae(row) < float("inf")
            ]
            if ok:
                best = dict(min(ok, key=metric_mae))
                best["_source"] = str(candidate)
                return best
    if path.is_dir():
        csv_path = path / "sweep_results.csv"
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = []
                for row in csv.DictReader(handle):
                    if row.get("ok", "True") not in ("True", "true", "1", "yes"):
                        continue
                    if not (row.get("eval_mae") or row.get("test_mae") or row.get("mae")):
                        continue
                    rows.append(row)
                if rows:
                    best = dict(min(rows, key=metric_mae))
                    best["_source"] = str(csv_path)
                    return best
    return None


def top_eval_history(path: Path, limit: int) -> list[dict[str, Any]]:
    run_dir = path.parent if path.is_file() else path
    rows = load_jsonl(run_dir / "eval_history.jsonl")
    rows = [row for row in rows if metric_mae(row) < float("inf")]
    return sorted(rows, key=metric_mae)[:limit]


def print_run(path: Path, limit: int) -> None:
    print(f"\n== {path}")
    checkpoint = load_checkpoint(path)
    if checkpoint:
        print(f"checkpoint: {checkpoint['path']}")
        print(
            "ckpt: "
            f"epoch={checkpoint['epoch']} best_epoch={checkpoint['best_epoch']} "
            f"best_mae={checkpoint['best_mae']} best_mse={checkpoint['best_mse']} "
            f"model_ema={checkpoint['has_model_ema']} model_raw={checkpoint['has_model_raw']}"
        )
        print("args:")
        for key in ARG_KEYS:
            print(f"  {key}: {arg_value(checkpoint['args'], key)}")

        run_dir = checkpoint["path"].parent
        print(f"best_eval_results: {short_record(load_json(run_dir / 'best_eval_results.json'))}")
        print(f"latest_eval_results: {short_record(load_json(run_dir / 'latest_eval_results.json'))}")
    else:
        print("checkpoint: not found")

    sweep = load_best_sweep(path)
    if sweep:
        print(f"sweep_best: {short_record(sweep)}")
        print(f"sweep_source: {sweep.get('_source')}")
    history = top_eval_history(path, limit)
    if history:
        print("eval_history_top:")
        for row in history:
            print(f"  {short_record(row)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PET run/checkpoint/sweep metadata")
    parser.add_argument("paths", nargs="+", help="Run dirs, checkpoint files, or sweep output dirs")
    parser.add_argument("--top", type=int, default=5, help="Number of best eval-history rows to print")
    args = parser.parse_args()

    for raw_path in args.paths:
        print_run(Path(raw_path), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
