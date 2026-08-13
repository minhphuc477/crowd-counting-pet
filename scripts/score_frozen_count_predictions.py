#!/usr/bin/env python3
"""Score already-frozen count predictions against a reference per-image JSON."""

import argparse
import json
import math
from pathlib import Path


def key(row):
    value = str(row.get('image_id') or row.get('image_path') or '')
    if not value:
        raise ValueError('row has neither image_id nor image_path')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow_benchmark_test_scoring', action='store_true')
    args = parser.parse_args()
    if not args.allow_benchmark_test_scoring:
        raise ValueError(
            'Scoring may reveal benchmark-test metrics. Freeze the router '
            'first, then pass --allow_benchmark_test_scoring exactly once.'
        )

    prediction_rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    reference_rows = json.loads(Path(args.reference).read_text(encoding='utf-8'))
    predictions = {key(row): float(row['pred_cnt']) for row in prediction_rows}
    targets = {key(row): float(row['gt_cnt']) for row in reference_rows}
    if set(predictions) != set(targets):
        raise ValueError('prediction and reference keys differ')
    errors = [predictions[item] - targets[item] for item in targets]
    result = {
        'images': len(errors),
        'mae': sum(abs(error) for error in errors) / len(errors),
        'mse': math.sqrt(sum(error * error for error in errors) / len(errors)),
        'bias': sum(errors) / len(errors),
        'predictions': args.predictions,
        'reference': args.reference,
        'warning': 'benchmark test metrics; do not tune the router from this output',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
