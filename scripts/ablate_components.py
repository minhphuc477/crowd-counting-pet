#!/usr/bin/env python3
"""Component-removal ablations for PET/APG/LC/count-head recovery.

This runner trains stage-1 variants from scratch and optionally runs the
verified scalar count-head recovery stage on each best checkpoint. It is meant
for a clean ablation table:

    PET
    PET + APG
    PET + APG + LC
    each of the above + scalar count-head recovery

LC is implemented as APG's contrastive term (`apg_contrastive_coef`), so an
"LC only" row is not included because the current code has no standalone LC
loss independent of APG positive/negative assignment.
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


STAGE1_VARIANTS = {
    "pet": {
        "description": "PET baseline: no APG, no LC, no count-head auxiliary.",
        "flags": [
            "--model_recipe", "vgg_apglc",
            "--apg_loss_coef", "0.0",
            "--apg_contrastive_coef", "0.0",
            "--count_head_loss_coef", "0.0",
        ],
    },
    "apg_no_lc": {
        "description": "PET + APG only: APG point/class auxiliary, contrastive LC removed.",
        "flags": [
            "--model_recipe", "vgg_apglc",
            "--apg_loss_coef", "0.02",
            "--apg_contrastive_coef", "0.0",
        ],
    },
    "apg_lc": {
        "description": "PET + APG + LC: verified APG+LC stage-1 recipe.",
        "flags": [
            "--model_recipe", "vgg_apglc",
        ],
    },
}


COUNTHEAD_RECOVERY_FLAGS = [
    "--model_recipe", "vgg_apglc",
    "--resume_model_only",
    "--lr", "0.0001",
    "--lr_backbone", "0.0",
    "--lr_scheduler", "step",
    "--lr_drop", "50",
    "--lr_gamma", "0.1",
    "--apg_loss_coef", "0.0",
    "--count_head_loss_coef", "1.0",
    "--count_head_loss_type", "log_l1",
    "--count_head_start_epoch", "0",
    "--count_head_end_epoch", "-1",
    "--count_head_warmup_epochs", "0",
    "--count_head_feature_grad_scale", "1.0",
    "--count_head_feature_grad_start_epoch", "0",
    "--count_head_feature_grad_warmup_epochs", "0",
    "--train_count_head_only",
    "--no_auto_freeze_bn_on_count_head_resume",
    "--score_threshold", "0.59",
    "--split_threshold", "0.45",
    "--split_threshold_quantile", "0.55",
    "--eval_count_mode", "threshold",
    "--eval_score_calibration", "none",
    "--eval_nms_radius", "0",
    "--eval_branch_gate", "none",
    "--eval_soft_split_gate", "none",
    "--eval_count_head_min_score", "0.0",
]


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


def write_ablation_summary(base_eval: Path, rows: list[dict]) -> None:
    if not rows:
        return
    base_eval.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (not row.get("ok", False), row.get("mae", float("inf"))))
    json_path = base_eval / "ablation_summary.json"
    csv_path = base_eval / "ablation_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys())
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
    parser = argparse.ArgumentParser(description="Train component-removal ablations for PET/APG/LC/count-head.")
    parser.add_argument("--data_path", default="./data/ShanghaiTech/part_A")
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--stage1_epochs", default=1500, type=int)
    parser.add_argument("--counthead_epochs", default=80, type=int)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--base_output_dir", default="outputs/SHA/component_ablation_seed42")
    parser.add_argument("--base_eval_dir", default="eval_results/SHA/component_ablation_seed42")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--variants", nargs="+", choices=sorted(STAGE1_VARIANTS), default=list(STAGE1_VARIANTS))
    parser.add_argument("--with_counthead_stage", action="store_true",
                        help="after each stage-1 variant, run scalar count-head recovery from its best checkpoint")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_counthead", action="store_true")
    parser.add_argument("--skip_sweep", action="store_true")
    parser.add_argument("--log_commands", action="store_true")
    return parser.parse_args()


def common_train_args(args: argparse.Namespace, output_dir: Path, epochs: int, eval_freq: int) -> list[str]:
    return [
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
        "--eval_freq", str(eval_freq),
        "--eval_start_epoch", "0",
    ]


def train_stage1(args: argparse.Namespace, variant: str, output_dir: Path) -> None:
    cmd = [
        sys.executable, "main.py",
        *STAGE1_VARIANTS[variant]["flags"],
        *common_train_args(args, output_dir, args.stage1_epochs, args.eval_freq),
    ]
    run(cmd, output_dir / "train.log" if args.log_commands else None)


def train_counthead(args: argparse.Namespace, stage1_checkpoint: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable, "main.py",
        *COUNTHEAD_RECOVERY_FLAGS,
        "--resume", str(stage1_checkpoint),
        *common_train_args(args, output_dir, args.counthead_epochs, 2),
    ]
    run(cmd, output_dir / "train.log" if args.log_commands else None)


def sweep_checkpoint(args: argparse.Namespace, checkpoint: Path, eval_dir: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    cmd = [
        sys.executable, "scripts/sweep_eval_thresholds.py",
        "--resume", str(checkpoint),
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--output_dir", str(eval_dir),
        "--score_thresholds", "0.50", "0.52", "0.54", "0.56", "0.575", "0.58", "0.59", "0.60", "0.61",
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

    base_output = Path(args.base_output_dir)
    base_eval = Path(args.base_eval_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage1_variants": STAGE1_VARIANTS,
        "selected_variants": args.variants,
        "with_counthead_stage": bool(args.with_counthead_stage),
        "counthead_recovery_flags": COUNTHEAD_RECOVERY_FLAGS,
    }
    (base_output / "component_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for variant in args.variants:
        stage1_dir = base_output / variant
        stage1_ckpt = stage1_dir / "best_checkpoint.pth"
        if not args.skip_stage1:
            train_stage1(args, variant, stage1_dir)
        if not args.skip_sweep:
            sweep_checkpoint(args, stage1_ckpt, base_eval / variant)

        if args.with_counthead_stage and not args.skip_counthead:
            counthead_dir = base_output / f"{variant}_counthead"
            counthead_ckpt = counthead_dir / "best_checkpoint.pth"
            train_counthead(args, stage1_ckpt, counthead_dir)
            if not args.skip_sweep:
                sweep_checkpoint(args, counthead_ckpt, base_eval / f"{variant}_counthead")
    summary_rows = []
    for variant in args.variants:
        summary_rows.append(_compact_record(variant, base_eval / variant))
        if args.with_counthead_stage:
            summary_rows.append(_compact_record(f"{variant}_counthead", base_eval / f"{variant}_counthead"))
    write_ablation_summary(base_eval, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
