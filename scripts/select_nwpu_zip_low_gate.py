#!/usr/bin/env python3
"""Select a validation-only ZIP correction for low-count NWPU images."""

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


def validate_keys(pet, zip_counts, label):
    if set(pet) != set(zip_counts):
        raise ValueError(f'{label} PET and ZIP inputs do not contain identical image keys')
    return set(pet)


def corrected_count(pet_count, zip_count, gate, alpha):
    if pet_count <= gate:
        return pet_count + alpha * (zip_count - pet_count), True
    return pet_count, False


def score_policy(pet, zip_counts, keys, gate, alpha):
    errors = []
    selected = 0
    for key in keys:
        if pet[key]['gt'] != zip_counts[key]['gt']:
            raise ValueError(f'validation GT mismatch for {key}')
        pred, used = corrected_count(
            pet[key]['pred'],
            zip_counts[key]['pred'],
            gate,
            alpha,
        )
        selected += int(used)
        errors.append(pred - pet[key]['gt'])
    return {
        'gate': float(gate),
        'alpha': float(alpha),
        'mae': sum(abs(error) for error in errors) / len(errors),
        'mse': math.sqrt(sum(error * error for error in errors) / len(errors)),
        'zip_selected_images': selected,
        'images': len(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Select PET<=gate ZIP blending on NWPU official val, then apply '
            'the frozen policy to matching test submissions'
        ),
    )
    parser.add_argument('--val_pet', required=True)
    parser.add_argument('--val_zip', required=True)
    parser.add_argument('--test_pet', required=True)
    parser.add_argument('--test_zip', required=True)
    parser.add_argument('--gates', default='-1,100,200,300,500,800')
    parser.add_argument('--alphas', default='0.25,0.5,0.75,1.0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    val_pet = load_validation(args.val_pet)
    val_zip = load_validation(args.val_zip)
    val_keys = validate_keys(val_pet, val_zip, 'validation')

    candidates = [score_policy(val_pet, val_zip, val_keys, -1.0, 0.0)]
    for gate in parse_values(args.gates):
        if gate < 0:
            continue
        for alpha in parse_values(args.alphas):
            if not 0 < alpha <= 1:
                raise ValueError('--alphas values must be in (0, 1]')
            candidates.append(score_policy(val_pet, val_zip, val_keys, gate, alpha))
    candidates.sort(key=lambda row: (row['mae'], row['mse'], row['alpha'], row['gate']))
    best = candidates[0]

    test_pet = load_submission(args.test_pet)
    test_zip = load_submission(args.test_zip)
    test_keys = validate_keys(test_pet, test_zip, 'test')
    selected = 0
    output_lines = []
    for image_id in sorted(test_keys, key=lambda value: int(value)):
        pred, used = corrected_count(
            test_pet[image_id],
            test_zip[image_id],
            best['gate'],
            best['alpha'],
        )
        selected += int(used)
        output_lines.append(f'{image_id} {pred:.6f}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(output_lines) + '\n', encoding='utf-8')
    report = {
        'selection_split': 'NWPU official validation',
        'policy': 'blend PET toward ZIP only when PET count <= gate',
        'best': best,
        'test_zip_selected_images': selected,
        'validation_inputs': {'pet': args.val_pet, 'zip': args.val_zip},
        'test_inputs': {'pet': args.test_pet, 'zip': args.test_zip},
        'candidates': candidates,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('best validation policy:', json.dumps(best, indent=2))
    print(f'test ZIP-selected images: {selected}/{len(test_keys)}')
    print(f'wrote: {output}')
    print(f'report: {report_path}')


if __name__ == '__main__':
    main()
