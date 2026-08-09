#!/usr/bin/env python3
"""Select a low-count PET/count-head gate from per-image validation results."""

import argparse
import json
import math
from pathlib import Path


def parse_values(text):
    return [float(value) for value in str(text).split(',') if value.strip()]


def metrics(rows, threshold, alpha):
    errors = []
    used = 0
    for row in rows:
        pet = float(row['pet_pred_cnt'])
        head = row['count_head_pred_cnt']
        if pet <= threshold:
            if head is None:
                return None
            head = float(head)
            pred = alpha * head + (1.0 - alpha) * pet
            used += 1
        else:
            pred = pet
        errors.append(pred - float(row['gt_cnt']))
    return {
        'threshold': float(threshold),
        'alpha': float(alpha),
        'mae': sum(abs(error) for error in errors) / len(errors),
        'mse': math.sqrt(sum(error * error for error in errors) / len(errors)),
        'head_images': used,
        'images': len(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Offline official-val sweep for count_head_low_blend',
    )
    parser.add_argument('--input', required=True, help='eval.py --per_image_results_file JSON')
    parser.add_argument('--output', required=True)
    parser.add_argument('--thresholds', default='50,100,200,300,500,800')
    parser.add_argument('--alphas', default='0.25,0.5,0.75,1.0')
    parser.add_argument(
        '--input_threshold', default=500.0, type=float,
        help='gate used to generate legacy input rows lacking count_head_pred_cnt',
    )
    parser.add_argument(
        '--input_alpha', default=0.5, type=float,
        help='blend alpha used to generate legacy input rows lacking count_head_pred_cnt',
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('input must contain a non-empty JSON list')
    normalized = []
    for row in rows:
        if 'pet_pred_cnt' not in row:
            raise KeyError('per-image row is missing pet_pred_cnt; regenerate it with the updated eval.py')
        head = row.get('count_head_pred_cnt')
        if head is None and float(row.get('dbg_tile_used', 0.0)) > 0.0:
            head = row.get('dbg_tile_scalar_raw')
        elif head is None:
            head = row.get('dbg_count_pred')
        pet = float(row['pet_pred_cnt'])
        if head is None:
            # Rows produced before count_head_pred_cnt was exported can still
            # recover the head below the gate used for that evaluation:
            # pred = alpha * head + (1-alpha) * PET.
            if pet <= args.input_threshold and args.input_alpha > 0.0:
                head = (
                    float(row['pred_cnt']) - (1.0 - args.input_alpha) * pet
                ) / args.input_alpha
        normalized.append({
            'image_id': row.get('image_id', ''),
            'gt_cnt': float(row['gt_cnt']),
            'pet_pred_cnt': pet,
            'count_head_pred_cnt': None if head is None else float(head),
        })

    candidates = [candidate for candidate in (
        metrics(normalized, threshold, alpha)
        for threshold in parse_values(args.thresholds)
        for alpha in parse_values(args.alphas)
    ) if candidate is not None]
    if not candidates:
        raise ValueError(
            'no candidate can be evaluated from this file; regenerate per-image results '
            'or supply the --input_threshold/--input_alpha used for the run'
        )
    candidates.sort(key=lambda row: (row['mae'], row['mse']))
    pet_errors = [row['pet_pred_cnt'] - row['gt_cnt'] for row in normalized]
    payload = {
        'selection_split': 'official_val',
        'selection_rule': 'count head below PET-count threshold, PET otherwise',
        'legacy_reconstruction': {
            'input_threshold': args.input_threshold,
            'input_alpha': args.input_alpha,
        },
        'pet_baseline': {
            'mae': sum(abs(error) for error in pet_errors) / len(pet_errors),
            'mse': math.sqrt(sum(error * error for error in pet_errors) / len(pet_errors)),
        },
        'best': candidates[0],
        'candidates': candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload['pet_baseline'], indent=2))
    print('best:', json.dumps(payload['best'], indent=2))
    print(f'results saved to: {output}')


if __name__ == '__main__':
    main()
