#!/usr/bin/env python3
"""
Multi-seed training orchestrator for PET backbone experiments.

Usage:
    python scripts/run_backbone_seeds.py \
        --backbone convnextv2_base \
        --seeds 42 7 13 99 1234 \
        --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine"

This script will:
1. Run main.py training for each seed
2. Save results and checkpoints per seed in results/{backbone}/seed_{seed}/
3. Aggregate MAE metrics across seeds
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed training orchestrator for PET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        default="convnextv2_base",
        type=str,
        help="Backbone architecture to train (default: convnextv2_base)",
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
        default="--epochs 1500 --patch_size 256",
        help="Additional arguments to pass to main.py (as a quoted string)",
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
    return parser.parse_args()


def run_training(
    backbone, seed, extra_args, output_dir, exp_name, dry_run=False
):
    """Run training for a single seed."""
    
    # Create output directory for this seed
    seed_output_dir = Path(output_dir) / backbone / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable,
        "main.py",
        "--backbone", backbone,
        "--seed", str(seed),
        "--output_dir", str(seed_output_dir),
        "--exp_name", f"{exp_name}_seed{seed}",
    ]
    
    # Parse and add extra arguments
    if extra_args:
        extra_args_list = extra_args.split()
        cmd.extend(extra_args_list)
    
    print(f"\n{'='*80}")
    print(f"Training {backbone} with seed {seed}")
    print(f"Output directory: {seed_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        return True, None
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        return result.returncode == 0, seed_output_dir
    except Exception as e:
        print(f"Error running training: {e}")
        return False, seed_output_dir


def collect_results(backbone, seeds, output_dir):
    """Collect MAE metrics from all seed runs."""
    
    results = {}
    mae_values = []
    
    for seed in seeds:
        seed_output_dir = Path(output_dir) / backbone / f"seed_{seed}"
        
        # Look for log file with metrics
        stats_file = seed_output_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                    if "mae_val" in stats:
                        mae = stats["mae_val"][-1]  # Last epoch
                        results[seed] = mae
                        mae_values.append(mae)
                        print(f"  Seed {seed}: MAE = {mae:.2f}")
            except Exception as e:
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
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nExperiment log saved to: {log_file}")


def main():
    args = get_args()
    
    output_dir = Path(args.output_dir)
    exp_name = f"{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*80}")
    print(f"Multi-seed Training Experiment: {exp_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Seeds: {args.seeds}")
    print(f"Extra arguments: {args.extra_args}")
    print(f"{'='*80}\n")
    
    # Run training for each seed
    successful_seeds = []
    failed_seeds = []
    
    for seed in args.seeds:
        # Skip earlier seeds if continue_from_seed is specified
        if args.continue_from_seed is not None and seed < args.continue_from_seed:
            print(f"Skipping seed {seed} (--continue_from_seed {args.continue_from_seed})")
            continue
        
        success, seed_dir = run_training(
            args.backbone,
            seed,
            args.extra_args,
            str(output_dir),
            exp_name,
            dry_run=args.dry_run,
        )
        
        if success:
            successful_seeds.append(seed)
        else:
            failed_seeds.append(seed)
    
    # Collect and display results
    print(f"\n{'='*80}")
    print("Training Complete - Collecting Results")
    print(f"{'='*80}\n")
    
    results = collect_results(args.backbone, args.seeds, str(output_dir))
    
    # Save experiment log
    if not args.dry_run:
        save_experiment_log(args.backbone, args.seeds, str(output_dir), results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("Final Summary")
    print(f"{'='*80}")
    print(f"Successful runs: {len(successful_seeds)}/{len(args.seeds)}")
    if failed_seeds:
        print(f"Failed runs: {failed_seeds}")
    print(f"Output directory: {output_dir / args.backbone}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
