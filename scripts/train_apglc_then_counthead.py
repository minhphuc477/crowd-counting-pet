#!/usr/bin/env python3
"""Run the verified APG+LC -> scalar count-head recovery pipeline.

This is intentionally a two-stage training protocol, not a hidden single-stage
architecture switch. The archived 48-MAE run was produced by first training the
PET/APG+LC point model, then loading its best checkpoint and training the
scalar density-sum count head while inference still used normal PET threshold
counting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train APG+LC from scratch, then run the verified count-head recovery stage."
    )
    parser.add_argument("--data_path", default="./data/ShanghaiTech/part_A")
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--stage1_output", default="outputs/SHA/vgg16_bn_apglc_stage1_seed42")
    parser.add_argument("--stage1_checkpoint", default=None,
                        help="explicit APG+LC checkpoint for stage 2; overrides stage1_output/best_checkpoint.pth")
    parser.add_argument("--stage2_output", default="outputs/SHA/vgg16_bn_apglc_counthead_stage2_seed42")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--stage1_epochs", default=1500, type=int)
    parser.add_argument("--stage2_epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--skip_stage1", action="store_true",
                        help="reuse stage1_output/best_checkpoint.pth and only run the count-head stage")
    return parser.parse_args()


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
        "--seed", str(args.seed),
    ]

    stage1_output = Path(args.stage1_output)
    stage1_best = Path(args.stage1_checkpoint) if args.stage1_checkpoint else stage1_output / "best_checkpoint.pth"
    if not args.skip_stage1:
        if args.stage1_checkpoint:
            raise ValueError("--stage1_checkpoint is only valid with --skip_stage1")
        run([
            python, "main.py",
            "--model_recipe", "vgg_apglc",
            "--output_dir", args.stage1_output,
            "--epochs", str(args.stage1_epochs),
            "--eval_freq", str(args.eval_freq),
            "--eval_start_epoch", "0",
            *common,
        ])

    if not stage1_best.is_file():
        raise FileNotFoundError(f"stage-1 best checkpoint not found: {stage1_best}")

    run([
        python, "main.py",
        "--model_recipe", "vgg_apglc_density_counthead_ft_legacy",
        "--resume", str(stage1_best),
        "--resume_model_only",
        "--output_dir", args.stage2_output,
        "--epochs", str(args.stage2_epochs),
        "--eval_freq", "2",
        "--eval_start_epoch", "0",
        *common,
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
