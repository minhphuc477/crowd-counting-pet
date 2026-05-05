#!/usr/bin/env python3
"""
Simple grid search over a few hyperparameters for PET training.

This script runs `main.py` for each configuration in the grid and records
the final evaluation MAE from the run log. It is deliberately lightweight
so it can be used without heavy dependencies.

Usage example:
  python scripts/hyperparam_search.py --backbone convnextv2_base \
      --lrs 2.5e-5 5e-5 --batch_sizes 2 4 --warmups 5 10 --seeds 42 \
      --extra_args "--epochs 150 --patch_size 256"

Notes:
- Uses same output layout as `run_backbone_seeds.py` (results/{backbone}/config_{id})
- Reads `outputs/<dataset>/<output_dir>/run_log.txt` to extract `test_mae`
"""

import argparse
import itertools
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search for PET")
    parser.add_argument("--backbone", default="convnextv2_base", type=str)
    parser.add_argument("--dataset_file", default="SHA", type=str)
    parser.add_argument("--lrs", nargs="+", type=float, default=[2.5e-5, 5e-5])
    parser.add_argument("--lr_backbones", nargs="+", type=float, default=[2.5e-6])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--warmups", nargs="+", type=int, default=[5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--extra_args", type=str, default="--epochs 150 --patch_size 256")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_config(cmd, dry_run=False):
    print("Running:", " ".join(cmd))
    if dry_run:
        return True
    try:
        r = subprocess.run(cmd, check=False)
        return r.returncode == 0
    except OSError as e:
        print("Execution error:", e)
        return False


def read_mae_from_log(dataset, seed_output_dir):
    # main.py writes logs into outputs/<dataset>/<output_dir>/run_log.txt
    out_dir = Path("outputs") / dataset / seed_output_dir
    log_path = out_dir / "run_log.txt"
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in reversed(lines):
        if line.startswith('{'):
            try:
                stats = json.loads(line)
                if 'test_mae' in stats:
                    return float(stats['test_mae'])
            except (OSError, json.JSONDecodeError):
                continue
    return None


def main():
    args = parse_args()

    grid = list(itertools.product(args.lrs, args.lr_backbones, args.batch_sizes, args.warmups))
    results = []
    base_extra = args.extra_args

    for i, (lr, lr_backbone, batch, warmup) in enumerate(grid):
        config_name = f"config_{i:03d}_lr{lr:.0e}_lrb{lr_backbone:.0e}_b{batch}_w{warmup}"
        out_dir = Path(args.output_dir) / args.backbone / config_name
        out_dir.mkdir(parents=True, exist_ok=True)

        extra = f"{base_extra} --lr {lr} --lr_backbone {lr_backbone} --batch_size {batch} --warmup_epochs {warmup}"
        extra_list = shlex.split(extra)

        best_maes = []
        for seed in args.seeds:
            seed_out_dir = out_dir / f"seed_{seed}"
            seed_out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "main.py", "--backbone", args.backbone, "--seed", str(seed), "--output_dir", str(seed_out_dir)] + extra_list
            success = run_config(cmd, dry_run=args.dry_run)
            if not success:
                print("Config failed:", config_name, "seed", seed)
                best_maes.append(None)
                continue
            mae = read_mae_from_log(args.dataset_file, seed_out_dir)
            best_maes.append(mae)

        # compute aggregate ignoring None
        maes = [m for m in best_maes if m is not None]
        mean_mae = float(np.mean(maes)) if maes else None
        results.append({
            'config': config_name,
            'lr': lr,
            'lr_backbone': lr_backbone,
            'batch_size': batch,
            'warmup': warmup,
            'per_seed_mae': best_maes,
            'mean_mae': mean_mae,
            'out_dir': str(out_dir),
        })
        # save interim results
        with open(Path(args.output_dir) / args.backbone / "hyper_search_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # final ranking
    ranked = [r for r in results if r['mean_mae'] is not None]
    ranked.sort(key=lambda x: x['mean_mae'])
    print("\nTop configs:")
    for r in ranked[:10]:
        print(r['config'], "mean_mae=", r['mean_mae'])


if __name__ == '__main__':
    main()
