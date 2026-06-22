#!/usr/bin/env python3
"""Ablate the APG+LC -> scalar count-head recovery stage.

All variants start from the same APG+LC stage-1 best checkpoint. This isolates
the mechanism that produced the archived 48-MAE run: scalar count-head
supervision, count-head-only trainability, BatchNorm adaptation, and optional
feature gradients.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    # Matches the archived recovery mechanism: only count-head parameters are
    # trainable, but BatchNorm stays in train mode so PET calibration can move.
    "bn_adapt_ch_only_coef1": [
        "--train_count_head_only",
        "--no_auto_freeze_bn_on_count_head_resume",
        "--count_head_loss_coef", "1.0",
        "--count_head_feature_grad_scale", "1.0",
    ],
    # If this fails to improve over stage 1, the 48 path depends on BN-buffer
    # adaptation rather than count-head weights alone.
    "freeze_bn_ch_only_coef1": [
        "--train_count_head_only",
        "--freeze_bn",
        "--count_head_loss_coef", "1.0",
        "--count_head_feature_grad_scale", "1.0",
    ],
    # Loss-strength ablations for the BN-adaptation path.
    "bn_adapt_ch_only_coef05": [
        "--train_count_head_only",
        "--no_auto_freeze_bn_on_count_head_resume",
        "--count_head_loss_coef", "0.5",
        "--count_head_feature_grad_scale", "1.0",
    ],
    "bn_adapt_ch_only_coef025": [
        "--train_count_head_only",
        "--no_auto_freeze_bn_on_count_head_resume",
        "--count_head_loss_coef", "0.25",
        "--count_head_feature_grad_scale", "1.0",
    ],
    # Full-model recovery with small count gradients. This tests whether a
    # real end-to-end fine-tune can replace the BN-only calibration effect.
    "full_model_freeze_bn_coef01_grad005": [
        "--freeze_bn",
        "--count_head_loss_coef", "0.10",
        "--count_head_feature_grad_scale", "0.05",
    ],
    "full_model_bn_adapt_coef01_grad005": [
        "--no_auto_freeze_bn_on_count_head_resume",
        "--count_head_loss_coef", "0.10",
        "--count_head_feature_grad_scale", "0.05",
    ],
}


def run(cmd: list[str], log_file: Path | None = None) -> None:
    print("\n" + " ".join(cmd) + "\n", flush=True)
    if log_file is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _metric_mae(record: dict) -> float:
    for key in ("eval_mae", "test_mae", "mae"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return float("inf")


def _load_best_sweep(eval_dir: Path) -> dict | None:
    candidates = [eval_dir / "best_thresholds.json", eval_dir / "sweep_results.json"]
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict) and row.get("ok", True)]
            if rows:
                return min(rows, key=_metric_mae)
    csv_path = eval_dir / "sweep_results.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if row.get("ok", "True") in ("True", "true", "1", "yes")
            ]
        if rows:
            return min(rows, key=_metric_mae)
    return None


def _compact_record(name: str, eval_dir: Path) -> dict:
    best = _load_best_sweep(eval_dir)
    if not best:
        return {"variant": name, "eval_dir": str(eval_dir), "ok": False}
    row = {
        "variant": name,
        "eval_dir": str(eval_dir),
        "ok": True,
        "mae": _metric_mae(best),
        "mse": float(best.get("eval_mse", best.get("test_mse", best.get("mse", 0.0)))),
        "pred_cnt": float(best.get("pred_cnt", 0.0)),
        "gt_cnt": float(best.get("gt_cnt", 0.0)),
        "score_threshold": best.get("score_threshold"),
        "split_threshold": best.get("split_threshold"),
        "query_prune_threshold": best.get("query_prune_threshold"),
        "eval_nms_radius": best.get("eval_nms_radius"),
        "eval_branch_gate": best.get("eval_branch_gate"),
        "eval_soft_split_gate": best.get("eval_soft_split_gate"),
        "loc_sigma_l_f1": float(best.get("loc_f1_large", best.get("loc_f1_sigma_l", 0.0))),
        "loc_sigma_l_prec": float(best.get("loc_prec_large", best.get("loc_prec_sigma_l", 0.0))),
        "loc_sigma_l_rec": float(best.get("loc_rec_large", best.get("loc_rec_sigma_l", 0.0))),
        "loc_sigma_s_f1": float(best.get("loc_f1_small", best.get("loc_f1_sigma_s", 0.0))),
        "loc_sigma_s_prec": float(best.get("loc_prec_small", best.get("loc_prec_sigma_s", 0.0))),
        "loc_sigma_s_rec": float(best.get("loc_rec_small", best.get("loc_rec_sigma_s", 0.0))),
    }
    gt_cnt = row["gt_cnt"]
    row["pred_gt_ratio"] = row["pred_cnt"] / gt_cnt if gt_cnt else 0.0
    return row


def write_ablation_summary(base_eval: Path, variants: list[str]) -> None:
    rows = [_compact_record(variant, base_eval / variant) for variant in variants]
    rows = sorted(rows, key=lambda row: (not row.get("ok", False), row.get("mae", float("inf"))))
    base_eval.mkdir(parents=True, exist_ok=True)
    json_path = base_eval / "ablation_summary.json"
    csv_path = base_eval / "ablation_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else ["variant"]
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAblation summary saved to: {csv_path}")
    for row in rows:
        if not row.get("ok"):
            print(f"  {row['variant']}: missing/failed")
            continue
        print(
            f"  {row['variant']}: mae={row['mae']:.4f} mse={row['mse']:.4f} "
            f"pred/gt={row['pred_gt_ratio']:.3f} "
            f"sigma_l(F1/P/R)={row['loc_sigma_l_f1']:.4f}/"
            f"{row['loc_sigma_l_prec']:.4f}/{row['loc_sigma_l_rec']:.4f} "
            f"sigma_s(F1/P/R)={row['loc_sigma_s_f1']:.4f}/"
            f"{row['loc_sigma_s_prec']:.4f}/{row['loc_sigma_s_rec']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate count-head recovery from a fixed APG+LC checkpoint.")
    parser.add_argument("--stage1_checkpoint", default="outputs/SHA/vgg16_bn_apglc_stage1_seed42/best_checkpoint.pth")
    parser.add_argument("--data_path", default="./data/ShanghaiTech/part_A")
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--eval_freq", default=2, type=int)
    parser.add_argument("--base_output_dir", default="outputs/SHA/counthead_recovery_ablation_seed42")
    parser.add_argument("--base_eval_dir", default="eval_results/SHA/counthead_recovery_ablation_seed42")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_sweep", action="store_true")
    parser.add_argument("--log_commands", action="store_true")
    return parser.parse_args()


def train_variant(args: argparse.Namespace, variant: str, output_dir: Path) -> None:
    common = [
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output_dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--eval_freq", str(args.eval_freq),
        "--eval_start_epoch", "0",
    ]
    cmd = [
        sys.executable, "main.py",
        "--model_recipe", "vgg_apglc",
        "--resume", args.stage1_checkpoint,
        "--resume_model_only",
        "--lr", "0.0001",
        "--lr_backbone", "0.0",
        "--lr_scheduler", "step",
        "--lr_drop", "50",
        "--lr_gamma", "0.1",
        "--apg_loss_coef", "0.0",
        "--count_head_loss_type", "log_l1",
        "--count_head_start_epoch", "0",
        "--count_head_end_epoch", "-1",
        "--count_head_warmup_epochs", "0",
        "--count_head_feature_grad_start_epoch", "0",
        "--count_head_feature_grad_warmup_epochs", "0",
        "--score_threshold", "0.59",
        "--split_threshold", "0.45",
        "--split_threshold_quantile", "0.55",
        "--eval_count_mode", "threshold",
        "--eval_score_calibration", "none",
        "--eval_nms_radius", "0",
        "--eval_branch_gate", "none",
        "--eval_soft_split_gate", "none",
        "--eval_count_head_min_score", "0.0",
        *VARIANTS[variant],
        *common,
    ]
    run(cmd, output_dir / "train.log" if args.log_commands else None)


def sweep_variant(args: argparse.Namespace, variant: str, output_dir: Path, eval_dir: Path) -> None:
    checkpoint = output_dir / "best_checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint for {variant}: {checkpoint}")
    cmd = [
        sys.executable, "scripts/sweep_eval_thresholds.py",
        "--resume", str(checkpoint),
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--output_dir", str(eval_dir),
        "--score_thresholds", "0.565", "0.575", "0.58", "0.585", "0.59", "0.60", "0.61",
        "--split_thresholds", "0.45", "0.47", "0.50",
        "--query_prune_thresholds", "0.5",
        "--eval_nms_radii", "0",
        "--eval_branch_gates", "none",
        "--eval_soft_split_gates", "none",
        "--eval_count_modes", "threshold",
        "--eval_count_sources", "pet",
    ]
    run(cmd, eval_dir / "sweep.log" if args.log_commands else None)


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    stage1_checkpoint = Path(args.stage1_checkpoint)
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(f"stage1 checkpoint not found: {stage1_checkpoint}")

    base_output = Path(args.base_output_dir)
    base_eval = Path(args.base_eval_dir)
    manifest = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "variants": args.variants,
        "variant_flags": {name: VARIANTS[name] for name in args.variants},
    }
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for variant in args.variants:
        output_dir = base_output / variant
        eval_dir = base_eval / variant
        if not args.skip_train:
            train_variant(args, variant, output_dir)
        if not args.skip_sweep:
            sweep_variant(args, variant, output_dir, eval_dir)
    write_ablation_summary(base_eval, args.variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
