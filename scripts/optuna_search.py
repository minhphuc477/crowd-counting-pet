#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning script for PET.

This script uses Optuna to optimize hyperparameters by launching `main.py`
for sampled configurations and reading the reported `test_mae` from run logs.

Note: Optuna should be installed (requirements.txt updated). This script
is a light wrapper and does not parallelize trials; for distributed tuning
consider using Optuna RDB storage and parallel workers.
"""

import argparse
import json
import shlex
import subprocess
import sys
from statistics import mean
from pathlib import Path
import optuna
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='convnextv2_base')
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 13])
    parser.add_argument('--seed_aggregate', choices=('mean', 'median'), default='mean')
    parser.add_argument('--extra_args', type=str, default='--epochs 150 --patch_size 256 --lr_scheduler warmup_hold_cosine')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--best_params_file', type=str, default=None)
    return parser.parse_args()


def read_mae(dataset, out_dir):
    log = Path('outputs') / dataset / out_dir / 'run_log.txt'
    if not log.exists():
        return None
    with open(log, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in reversed(lines):
        if line.startswith('{'):
            try:
                stats = json.loads(line)
                if 'test_mae' in stats:
                    return float(stats['test_mae'])
            except Exception:
                continue
    return None


def objective(trial, args):
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 1e-4, log=True)
    batch = trial.suggest_categorical('batch_size', [2, 4, 8])
    warmup = trial.suggest_int('warmup', 0, 10)
    score_threshold = trial.suggest_float('score_threshold', 0.2, 0.6)

    config_name = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_trial{trial.number}"
    out_dir = Path(args.output_dir) / args.backbone / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = f"{args.extra_args} --lr {lr} --lr_backbone {lr_backbone} --batch_size {batch} --warmup_epochs {warmup} --score_threshold {score_threshold}"
    maes = []
    for seed in args.seeds:
        seed_out_dir = out_dir / f"seed_{seed}"
        cmd = [
            sys.executable,
            'main.py',
            '--backbone',
            args.backbone,
            '--seed',
            str(seed),
            '--output_dir',
            str(seed_out_dir),
        ] + shlex.split(extra)
        print('Running trial', trial.number, f'seed {seed}', 'cmd:', ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise optuna.exceptions.TrialPruned()
        mae = read_mae(args.dataset_file, seed_out_dir)
        if mae is None:
            raise optuna.exceptions.TrialPruned()
        maes.append(mae)

    if args.seed_aggregate == 'median':
        maes_sorted = sorted(maes)
        mid = len(maes_sorted) // 2
        if len(maes_sorted) % 2 == 1:
            score = maes_sorted[mid]
        else:
            score = 0.5 * (maes_sorted[mid - 1] + maes_sorted[mid])
    else:
        score = mean(maes)

    trial.set_user_attr('per_seed_mae', maes)
    trial.set_user_attr('seed_aggregate', args.seed_aggregate)
    return score


def main():
    args = parse_args()

    # Use SQLite storage for persistence and resume capability
    storage_path = Path(args.output_dir) / args.backbone / 'optuna_study.db'
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f'sqlite:///{storage_path}'

    study = optuna.create_study(
        study_name=f'{args.backbone}_{args.dataset_file}',
        direction='minimize',
        storage=storage_url,
        load_if_exists=True,
    )

    # Count already completed trials for resume
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, args.trials - completed_trials)

    if remaining_trials < args.trials:
        print(f'Resuming study from {storage_url}: {completed_trials} trials already completed, running {remaining_trials} more.')

    func = lambda t: objective(t, args)
    study.optimize(func, n_trials=remaining_trials)

    best_params_path = Path(args.best_params_file) if args.best_params_file else Path(args.output_dir) / args.backbone / 'optuna_best.json'
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    # Guard against "No trials are completed yet" error
    completed_trials_final = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials_final:
        best_trial = study.best_trial
        best_value = float(study.best_value)
        best_params = best_trial.params
        best_trial_number = best_trial.number
        per_seed_mae = best_trial.user_attrs.get('per_seed_mae')
        print('Best trial:', best_params, 'MAE:', best_value)
    else:
        best_value = None
        best_params = None
        best_trial_number = None
        per_seed_mae = None
        print('Warning: No trials were completed successfully.')

    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'backbone': args.backbone,
            'dataset_file': args.dataset_file,
            'trials': args.trials,
            'trials_completed': len(completed_trials_final),
            'seed_aggregate': args.seed_aggregate,
            'extra_args': args.extra_args,
            'best_value': best_value,
            'best_params': best_params,
            'best_trial_number': best_trial_number,
            'per_seed_mae': per_seed_mae,
        }, f, indent=2)
    print('Saved best params to', best_params_path)


if __name__ == '__main__':
    main()
