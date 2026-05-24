#!/usr/bin/env python3
"""
Multi-seed training orchestrator for PET backbone experiments.

Usage:
    python scripts/run_backbone_seeds.py \
        --backbone convnextv2_base \
        --seeds 42 7 13 99 1234 \
        --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine"

    python scripts/run_backbone_seeds.py \
        --preset latency \
        --seeds 42 7 \
        --extra_args "--epochs 300 --patch_size 256"

This script will:
1. Run main.py training for each seed
2. Save results and checkpoints per seed in results/{backbone}/seed_{seed}/
3. Aggregate MAE metrics across seeds
"""

import argparse
import json
import subprocess
import sys
import shlex
import re
from pathlib import Path
from datetime import datetime
import numpy as np


ABLATION_PRESETS = {
    "crowd_dense": [
        "convnextv2_base",
        "convnext_base",
        "swinv2_base",
        "maxvit_small",
        "pvtv2_b1",
    ],
    "latency": [
        "convnextv2_tiny",
        "fastvit_tiny",
        "efficientvit_tiny",
        "mobilenetv4_small",
        "repvit_tiny",
        "edgenext_tiny",
    ],
    "full": [
        "convnext_tiny",
        "convnext_base",
        "convnextv2_tiny",
        "convnextv2_base",
        "swinv2_tiny",
        "swinv2_base",
        "maxvit_tiny",
        "maxvit_small",
        "fastvit_tiny",
        "fastvit_small",
        "efficientvit_tiny",
        "efficientvit_small",
        "efficientnetv2_tiny",
        "efficientnetv2_small",
        "mobilenetv4_small",
        "mobilenetv4_hybrid",
        "hgnetv2_tiny",
        "pvtv2_b0",
        "pvtv2_b1",
        "edgenext_tiny",
        "repvit_tiny",
        "repvit_small",
    ],
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed training orchestrator for PET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        default="convnextv2_base",
        type=str,
        help="Single backbone architecture to train (default: convnextv2_base)",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Train multiple backbone architectures in one ablation run",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(ABLATION_PRESETS.keys()),
        default=None,
        help="Named backbone ablation preset: crowd_dense, latency, or full",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 13, 99, 1234],
        help="List of random seeds to use (default: 42 7 13 99 1234)",
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        default="--epochs 1500 --patch_size 256 --pet_loss_variant paper",
        help="Additional arguments to pass to main.py (as a quoted string)",
    )
    parser.add_argument(
        "--dataset_file",
        default="SHA",
        help="Dataset name used under outputs/<dataset_file>/ (default: SHA)",
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="Optional dataset path passed to main.py/eval.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--continue_from_seed",
        type=int,
        default=None,
        help="Continue from a specific seed (skip earlier seeds)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run training even when best_checkpoint.pth already exists",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from checkpoint.pth when it exists",
    )
    parser.add_argument(
        "--eval_after_training",
        action="store_true",
        help="Run eval.py on best_checkpoint.pth after each completed run",
    )
    parser.add_argument(
        "--check_contract",
        action="store_true",
        help="Run scripts/check_backbone_contract.py for each backbone before training",
    )
    return parser.parse_args()


def resolve_backbones(args):
    if args.preset:
        return ABLATION_PRESETS[args.preset]
    if args.backbones:
        return args.backbones
    return [args.backbone]


def run_training(
    backbone, seed, args
):
    """Run training for a single seed."""
    
    # Create output directory for this seed
    seed_output_dir = Path(args.output_dir) / backbone / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    actual_output_dir = Path("outputs") / args.dataset_file / seed_output_dir
    best_checkpoint = actual_output_dir / "best_checkpoint.pth"
    latest_checkpoint = actual_output_dir / "checkpoint.pth"

    if best_checkpoint.exists() and not args.force:
        print(f"\nSkipping {backbone} seed {seed}: {best_checkpoint} already exists")
        return True, seed_output_dir
    
    # Build the command
    cmd = [
        sys.executable,
        "main.py",
        "--backbone", backbone,
        "--dataset_file", args.dataset_file,
        "--seed", str(seed),
        "--output_dir", str(seed_output_dir),
    ]
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if latest_checkpoint.exists() and not args.no_resume:
        cmd.extend(["--resume", str(latest_checkpoint)])
    
    # Parse and add extra arguments
    if args.extra_args:
        extra_args_list = shlex.split(args.extra_args)
        cmd.extend(extra_args_list)
    
    print(f"\n{'='*80}")
    print(f"Training {backbone} with seed {seed}")
    print(f"Output directory: {seed_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        return True, None
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        return result.returncode == 0, seed_output_dir
    except OSError as e:
        print(f"Error running training: {e}")
        return False, seed_output_dir


def run_eval(backbone, seed, args):
    seed_output_dir = Path(args.output_dir) / backbone / f"seed_{seed}"
    actual_output_dir = Path("outputs") / args.dataset_file / seed_output_dir
    checkpoint = actual_output_dir / "best_checkpoint.pth"
    if not checkpoint.exists():
        checkpoint = actual_output_dir / "checkpoint.pth"
    if not checkpoint.exists():
        print(f"  Eval skipped for {backbone} seed {seed}: no checkpoint found")
        return None

    eval_log = actual_output_dir / "eval_log.txt"
    cmd = [
        sys.executable,
        "eval.py",
        "--backbone", backbone,
        "--dataset_file", args.dataset_file,
        "--resume", str(checkpoint),
    ]
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])

    print(f"  Eval command: {' '.join(cmd)}")
    if args.dry_run:
        return None

    with eval_log.open("w", encoding="utf-8", errors="replace") as log:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        print(f"  Eval failed for {backbone} seed {seed}; see {eval_log}")
        return None

    text = eval_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"epoch:\s*(\d+).*?mae:\s*([0-9.]+).*?mse:\s*([0-9.]+)", text, re.S)
    if not match:
        print(f"  Eval finished but metrics were not parsed from {eval_log}")
        return None
    eval_result = {
        "epoch": int(match.group(1)),
        "eval_mae": float(match.group(2)),
        "eval_mse": float(match.group(3)),
        "checkpoint": str(checkpoint),
        "eval_log": str(eval_log),
    }
    eval_result_path = actual_output_dir / "eval_results.json"
    eval_result_path.write_text(json.dumps(eval_result, indent=2) + "\n", encoding="utf-8")
    print(f"  Eval MAE = {eval_result['eval_mae']:.4f}, MSE = {eval_result['eval_mse']:.4f}")
    return eval_result


def collect_results(backbone, seeds, output_dir, dataset_file):
    """Collect MAE metrics from all seed runs."""
    
    results = {}
    mae_values = []
    
    for seed in seeds:
        seed_output_dir = Path(output_dir) / backbone / f"seed_{seed}"
        actual_output_dir = Path("outputs") / dataset_file / seed_output_dir
        
        # Look for log file with metrics
        stats_file = actual_output_dir / "run_log.txt"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line in reversed(lines):
                    if line.startswith('{'):
                        stats = json.loads(line)
                        if 'test_mae' in stats:
                            mae = stats['test_mae']
                            results[seed] = mae
                            mae_values.append(mae)
                            print(f"  Seed {seed}: MAE = {mae:.2f}")
                            break
            except (OSError, json.JSONDecodeError) as e:
                print(f"  Seed {seed}: Could not read stats ({e})")
    
    if mae_values:
        print(f"\nSummary for {backbone}:")
        print(f"  Mean MAE:   {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f}")
        print(f"  Min MAE:    {np.min(mae_values):.2f}")
        print(f"  Max MAE:    {np.max(mae_values):.2f}")
        print(f"  Seeds with results: {list(results.keys())}")
        return results
    else:
        print(f"No MAE results found for {backbone}")
        return {}


def save_experiment_log(backbone, seeds, output_dir, results):
    """Save experiment log with all seed results."""
    
    exp_dir = Path(output_dir) / backbone
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = exp_dir / "experiment_log.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "backbone": backbone,
        "seeds": seeds,
        "results": results,
        "num_seeds": len(seeds),
        "num_completed": len(results),
    }
    
    if results:
        mae_values = list(results.values())
        log_data["metrics"] = {
            "mean_mae": float(np.mean(mae_values)),
            "std_mae": float(np.std(mae_values)),
            "min_mae": float(np.min(mae_values)),
            "max_mae": float(np.max(mae_values)),
        }
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nExperiment log saved to: {log_file}")


def main():
    args = get_args()
    backbones = resolve_backbones(args)
    
    output_dir = Path(args.output_dir)
    exp_name = f"backbone_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*80}")
    print(f"Multi-seed Training Experiment: {exp_name}")
    print(f"Backbones: {backbones}")
    print(f"Seeds: {args.seeds}")
    print(f"Extra arguments: {args.extra_args}")
    print(f"{'='*80}\n")
    
    all_results = {}
    successful_runs = 0
    failed_runs = []

    for backbone in backbones:
        if args.check_contract:
            contract_cmd = [
                sys.executable,
                "scripts/check_backbone_contract.py",
                "--backbone",
                backbone,
                "--device",
                "cpu",
            ]
            print(f"Checking PET contract for {backbone}: {' '.join(contract_cmd)}")
            if not args.dry_run:
                contract = subprocess.run(contract_cmd, cwd=Path(__file__).parent.parent, check=False)
                if contract.returncode != 0:
                    print(f"Skipping {backbone}: PET contract check failed")
                    continue

        successful_seeds = []
        failed_seeds = []

        for seed in args.seeds:
            if args.continue_from_seed is not None and seed < args.continue_from_seed:
                print(f"Skipping seed {seed} (--continue_from_seed {args.continue_from_seed})")
                continue

            success, _ = run_training(
                backbone,
                seed,
                args,
            )

            if success:
                successful_seeds.append(seed)
                successful_runs += 1
                if args.eval_after_training:
                    run_eval(backbone, seed, args)
            else:
                failed_seeds.append(seed)
                failed_runs.append((backbone, seed))

        print(f"\n{'='*80}")
        print(f"Training Complete for {backbone} - Collecting Results")
        print(f"{'='*80}\n")

        results = collect_results(backbone, args.seeds, str(output_dir), args.dataset_file)
        all_results[backbone] = results

        if not args.dry_run:
            save_experiment_log(backbone, args.seeds, str(output_dir), results)

    print(f"\n{'='*80}")
    print("Final Summary")
    print(f"{'='*80}")
    print(f"Successful runs: {successful_runs}/{len(backbones) * len(args.seeds)}")
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
