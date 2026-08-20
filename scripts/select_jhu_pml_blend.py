#!/usr/bin/env python3
"""Select a PET/Direct-PML blend using only the official JHU validation set."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def blend_metrics(rows, alpha):
    gt = np.asarray([float(row['gt_cnt']) for row in rows], dtype=np.float64)
    pet = np.asarray([float(row['pet_pred_cnt']) for row in rows], dtype=np.float64)
    measure = np.asarray([float(row['measure_pred_cnt']) for row in rows], dtype=np.float64)
    prediction = (1.0 - float(alpha)) * pet + float(alpha) * measure
    error = prediction - gt
    return {
        'alpha': float(alpha),
        'mae': float(np.abs(error).mean()),
        'mse': float(np.sqrt(np.square(error).mean())),
        'bias': float(error.mean()),
        'pred_cnt': float(prediction.mean()),
        'gt_cnt': float(gt.mean()),
    }


def alpha_grid(step):
    step = float(step)
    if not math.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError('--alpha_step must be finite and in (0, 1]')
    values = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    return sorted({float(np.clip(value, 0.0, 1.0)) for value in values} | {0.0, 1.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_image_results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--alpha_step', type=float, default=0.05)
    parser.add_argument(
        '--selection_protocol',
        default='official_val',
        choices=('official_val',),
        help='the selector intentionally refuses benchmark-test selection',
    )
    args = parser.parse_args()

    input_path = Path(args.per_image_results)
    with input_path.open('r', encoding='utf-8') as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError('per-image results must be a non-empty JSON list')
    required = {'gt_cnt', 'pet_pred_cnt', 'measure_pred_cnt'}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise KeyError(f'row {index} is missing fields: {sorted(missing)}')

    candidates = [blend_metrics(rows, alpha) for alpha in alpha_grid(args.alpha_step)]
    candidates.sort(key=lambda row: (row['mae'], row['mse'], abs(row['bias']), row['alpha']))
    report = {
        'selection_protocol': args.selection_protocol,
        'source': str(input_path),
        'images': len(rows),
        'best': candidates[0],
        'pet_control': next(row for row in candidates if row['alpha'] == 0.0),
        'measure_control': next(row for row in candidates if row['alpha'] == 1.0),
        'candidates': candidates,
        'test_command_override': [
            '--eval_count_source',
            'measure_pet_blend',
            '--eval_count_blend_alpha',
            f"{candidates[0]['alpha']:.10g}",
        ],
        'warning': 'Freeze this alpha before evaluating the JHU test split.',
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report['best'], indent=2))
    print(f'wrote: {output_path}')


if __name__ == '__main__':
    main()
