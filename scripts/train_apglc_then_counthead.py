#!/usr/bin/env python3
"""Run APG+LC/IFI training followed by count-head adaptation.

The detector architecture is identical in both stages. Stage 2 initializes the
additional scalar count head and applies its low-gradient auxiliary objective;
inference continues to use normal PET point thresholding.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified APG+LC/IFI from scratch, then adapt it with a scalar count head."
    )
    parser.add_argument("--data_path", default="./data/ShanghaiTech/part_A")
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--stage1_output", default=None)
    parser.add_argument("--stage1_checkpoint", default=None,
                        help="explicit stage-1 checkpoint for stage 2; overrides the complete MAE-best checkpoint")
    parser.add_argument("--stage2_output", default=None)
    parser.add_argument("--stage1_recipe", default=None)
    parser.add_argument("--stage2_recipe", default=None)
    parser.add_argument("--ifi_variant", default="branch", choices=("branch", "unified"),
                        help="branch uses branch-local IFI; unified uses one shared IFI for both branches")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--stage1_epochs", default=1500, type=int)
    parser.add_argument("--stage2_epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--patch_size_choices", default="")
    parser.add_argument("--crop_attempts", default=1, type=int)
    parser.add_argument("--min_crop_points", default=0, type=int)
    parser.add_argument("--eval_max_size", default=1536, type=int)
    parser.add_argument("--eval_tile_size", default=0, type=int)
    parser.add_argument("--eval_tile_overlap", default=0, type=int)
    parser.add_argument("--eval_tile_nms_radius", default=0.0, type=float)
    parser.add_argument("--eval_tile_min_gt", default=0, type=int)
    parser.add_argument("--eval_tile_max_tiles", default=0, type=int)
    parser.add_argument("--eval_tile_trigger_count", default=0.0, type=float)
    parser.add_argument("--eval_tile_trigger_area", default=0, type=int)
    parser.add_argument("--nwpu_eval_split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--jhu_eval_split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--nwpu_sigma_mode", default="area", choices=("area", "diag", "min_diag", "official"))
    parser.add_argument("--validation_protocol", default="auto",
                        choices=("auto", "benchmark_test", "train_holdout"))
    parser.add_argument("--train_holdout_fraction", default=0.1, type=float)
    parser.add_argument("--train_holdout_seed", default=None, type=int)
    parser.add_argument("--nwpu_dense_crop_prob", default=0.0, type=float)
    parser.add_argument("--nwpu_dense_crop_attempts", default=16, type=int)
    parser.add_argument("--train_count_weight_power", default=0.0, type=float)
    parser.add_argument("--train_count_weight_max", default=8.0, type=float)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--skip_stage1", action="store_true",
                        help="reuse stage1_output/best_checkpoint.pth and only run the count-head stage")
    parser.add_argument("--stage1_extra_args", default="",
                        help="extra args appended to the APG+LC stage, shell-style quoted")
    parser.add_argument("--stage2_extra_args", default="",
                        help="extra args appended to the count-head stage, shell-style quoted")
    args = parser.parse_args()
    args._explicit_args = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    if args.stage1_recipe is None:
        if args.ifi_variant == "unified":
            args.stage1_recipe = (
                "vgg_apglc_unified_ifi_nwpu"
                if args.dataset_file == "NWPU"
                else "vgg_apglc_unified_ifi"
            )
        else:
            args.stage1_recipe = (
                "vgg_apglc_branch_ifi_nwpu"
                if args.dataset_file == "NWPU"
                else "vgg_apglc_branch_ifi"
            )
    if args.stage2_recipe is None:
        if args.ifi_variant == "unified":
            args.stage2_recipe = (
                "vgg_apglc_unified_ifi_counthead_stage2_nwpu"
                if args.dataset_file == "NWPU"
                else "vgg_apglc_unified_ifi_counthead_stage2"
            )
        else:
            args.stage2_recipe = (
                "vgg_apglc_branch_ifi_counthead_stage2_nwpu"
                if args.dataset_file == "NWPU"
                else "vgg_apglc_branch_ifi_counthead_stage2"
            )
    dataset_dir = args.dataset_file
    if args.stage1_output is None:
        run_name = f"vgg16_bn_apglc_{args.ifi_variant}_ifi_stage1_seed{args.seed}"
        args.stage1_output = str(Path("outputs") / dataset_dir / run_name)
    if args.stage2_output is None:
        run_name = f"vgg16_bn_apglc_{args.ifi_variant}_ifi_counthead_stage2_seed{args.seed}"
        args.stage2_output = str(Path("outputs") / dataset_dir / run_name)
    return args


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    python = sys.executable
    common = [
        "--dataset_file", args.dataset_file,
        "--data_path", args.data_path,
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--patch_size", str(args.patch_size),
        "--nwpu_eval_split", args.nwpu_eval_split,
        "--jhu_eval_split", args.jhu_eval_split,
        "--validation_protocol", args.validation_protocol,
        "--train_holdout_fraction", str(args.train_holdout_fraction),
        "--train_holdout_seed", str(args.seed if args.train_holdout_seed is None else args.train_holdout_seed),
        "--seed", str(args.seed),
    ]
    recipe_owned = {
        "nwpu_sigma_mode": ("--nwpu_sigma_mode", args.nwpu_sigma_mode),
        "eval_max_size": ("--eval_max_size", str(args.eval_max_size)),
        "patch_size_choices": ("--patch_size_choices", args.patch_size_choices),
        "crop_attempts": ("--crop_attempts", str(args.crop_attempts)),
        "min_crop_points": ("--min_crop_points", str(args.min_crop_points)),
        "eval_tile_size": ("--eval_tile_size", str(args.eval_tile_size)),
        "eval_tile_overlap": ("--eval_tile_overlap", str(args.eval_tile_overlap)),
        "eval_tile_nms_radius": ("--eval_tile_nms_radius", str(args.eval_tile_nms_radius)),
        "eval_tile_min_gt": ("--eval_tile_min_gt", str(args.eval_tile_min_gt)),
        "eval_tile_max_tiles": ("--eval_tile_max_tiles", str(args.eval_tile_max_tiles)),
        "eval_tile_trigger_count": ("--eval_tile_trigger_count", str(args.eval_tile_trigger_count)),
        "eval_tile_trigger_area": ("--eval_tile_trigger_area", str(args.eval_tile_trigger_area)),
    }
    for key, (flag, value) in recipe_owned.items():
        if key in args._explicit_args and value != "":
            common.extend([flag, value])
    if args.dataset_file == "NWPU":
        for key, flag, value in (
            ("nwpu_dense_crop_prob", "--nwpu_dense_crop_prob", str(args.nwpu_dense_crop_prob)),
            ("nwpu_dense_crop_attempts", "--nwpu_dense_crop_attempts", str(args.nwpu_dense_crop_attempts)),
            ("train_count_weight_power", "--train_count_weight_power", str(args.train_count_weight_power)),
            ("train_count_weight_max", "--train_count_weight_max", str(args.train_count_weight_max)),
        ):
            if key in args._explicit_args:
                common.extend([flag, value])

    stage1_output = Path(args.stage1_output)
    if args.stage1_checkpoint:
        stage1_best = Path(args.stage1_checkpoint)
    else:
        complete_best = stage1_output / "best_complete_checkpoint.pth"
        stage1_best = complete_best if complete_best.is_file() else stage1_output / "best_checkpoint.pth"
    if not args.skip_stage1:
        if args.stage1_checkpoint:
            raise ValueError("--stage1_checkpoint is only valid with --skip_stage1")
        run([
            python, "main.py",
            "--model_recipe", args.stage1_recipe,
            "--output_dir", args.stage1_output,
            "--epochs", str(args.stage1_epochs),
            "--eval_freq", str(args.eval_freq),
            "--eval_start_epoch", "0",
            *common,
            *shlex.split(args.stage1_extra_args),
        ])

    if not args.stage1_checkpoint:
        complete_best = stage1_output / "best_complete_checkpoint.pth"
        if complete_best.is_file():
            stage1_best = complete_best
    if not stage1_best.is_file():
        raise FileNotFoundError(f"stage-1 best checkpoint not found: {stage1_best}")

    run([
        python, "main.py",
        "--model_recipe", args.stage2_recipe,
        "--resume", str(stage1_best),
        "--resume_model_only",
        "--output_dir", args.stage2_output,
        "--epochs", str(args.stage2_epochs),
        "--eval_freq", "2",
        "--eval_start_epoch", "0",
        *common,
        *shlex.split(args.stage2_extra_args),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
