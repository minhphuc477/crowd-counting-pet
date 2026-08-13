#!/usr/bin/env python3
"""Select a validation-only adaptive score-threshold policy for NWPU counts.

The policy uses a conservative prediction for low-count images, the reference
prediction in the middle, and a high-recall prediction for dense images. Gate
values are selected exclusively on official validation predictions and then
applied unchanged to the three matching test submissions.
"""

import argparse
import json
import math
from pathlib import Path


def parse_values(text):
    return [float(value) for value in str(text).split(',') if value.strip()]


def load_validation(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} must contain a non-empty JSON list')
    keyed = {}
    for row in rows:
        key = str(row.get('image_id') or row.get('image_path') or '')
        if not key or key in keyed:
            raise ValueError(f'{path} has a missing or duplicate image key: {key!r}')
        keyed[key] = {
            'pred': float(row['pred_cnt']),
            'gt': float(row['gt_cnt']),
        }
    return keyed


def load_submission(path):
    keyed = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding='utf-8', errors='strict').splitlines(),
        start=1,
    ):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{path}:{line_number} must contain image_id and count')
        image_id, count = fields
        if image_id in keyed:
            raise ValueError(f'{path} contains duplicate image ID {image_id}')
        value = float(count)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{path}:{line_number} has invalid count {count}')
        keyed[image_id] = value
    if len(keyed) != 1500:
        raise ValueError(f'{path} must contain 1,500 predictions, found {len(keyed)}')
    return keyed


def validate_keys(runs, label):
    keys = set(runs[0])
    for run in runs[1:]:
        if set(run) != keys:
            raise ValueError(f'{label} inputs do not contain identical image keys')
    return keys


def route(reference, conservative, high_recall, low_gate, high_gate):
    if reference <= low_gate:
        return conservative, 'conservative'
    if reference >= high_gate:
        return high_recall, 'high_recall'
    return reference, 'reference'


def score_policy(reference, conservative, high_recall, keys, low_gate, high_gate):
    errors = []
    usage = {'conservative': 0, 'reference': 0, 'high_recall': 0}
    for key in keys:
        gt_values = {
            reference[key]['gt'],
            conservative[key]['gt'],
            high_recall[key]['gt'],
        }
        if len(gt_values) != 1:
            raise ValueError(f'validation GT mismatch for {key}')
        pred, source = route(
            reference[key]['pred'],
            conservative[key]['pred'],
            high_recall[key]['pred'],
            low_gate,
            high_gate,
        )
        usage[source] += 1
        errors.append(pred - gt_values.pop())
    return {
        'low_gate': float(low_gate),
        'high_gate': float(high_gate),
        'mae': sum(abs(error) for error in errors) / len(errors),
        'mse': math.sqrt(sum(error * error for error in errors) / len(errors)),
        'usage': usage,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Select an adaptive NWPU count policy on official val and export test',
    )
    parser.add_argument('--val_reference', required=True)
    parser.add_argument('--val_conservative', required=True)
    parser.add_argument('--val_high_recall', required=True)
    parser.add_argument('--test_reference', required=True)
    parser.add_argument('--test_conservative', required=True)
    parser.add_argument('--test_high_recall', required=True)
    parser.add_argument('--low_gates', default='25,50,75,100')
    parser.add_argument('--high_gates', default='500,800,1200,1500')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    val_reference = load_validation(args.val_reference)
    val_conservative = load_validation(args.val_conservative)
    val_high_recall = load_validation(args.val_high_recall)
    val_keys = validate_keys(
        [val_reference, val_conservative, val_high_recall],
        'validation',
    )

    candidates = []
    for low_gate in parse_values(args.low_gates):
        for high_gate in parse_values(args.high_gates):
            if low_gate >= high_gate:
                continue
            candidates.append(score_policy(
                val_reference,
                val_conservative,
                val_high_recall,
                val_keys,
                low_gate,
                high_gate,
            ))
    if not candidates:
        raise ValueError('no valid low/high gate combinations')
    candidates.sort(key=lambda row: (row['mae'], row['mse'], row['low_gate'], row['high_gate']))
    best = candidates[0]

    test_reference = load_submission(args.test_reference)
    test_conservative = load_submission(args.test_conservative)
    test_high_recall = load_submission(args.test_high_recall)
    test_keys = validate_keys(
        [test_reference, test_conservative, test_high_recall],
        'test',
    )
    usage = {'conservative': 0, 'reference': 0, 'high_recall': 0}
    output_lines = []
    for image_id in sorted(test_keys, key=lambda value: int(value)):
        pred, source = route(
            test_reference[image_id],
            test_conservative[image_id],
            test_high_recall[image_id],
            best['low_gate'],
            best['high_gate'],
        )
        usage[source] += 1
        output_lines.append(f'{image_id} {pred:.6f}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(output_lines) + '\n', encoding='utf-8')
    report = {
        'selection_split': 'NWPU official validation',
        'policy': (
            'conservative if reference<=low_gate; high_recall if '
            'reference>=high_gate; reference otherwise'
        ),
        'validation_inputs': {
            'reference': args.val_reference,
            'conservative': args.val_conservative,
            'high_recall': args.val_high_recall,
        },
        'test_inputs': {
            'reference': args.test_reference,
            'conservative': args.test_conservative,
            'high_recall': args.test_high_recall,
        },
        'best': best,
        'test_usage': usage,
        'candidates': candidates,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('best validation policy:', json.dumps(best, indent=2))
    print('test usage:', json.dumps(usage, indent=2))
    print(f'wrote: {output}')
    print(f'report: {report_path}')


if __name__ == '__main__':
    main()
