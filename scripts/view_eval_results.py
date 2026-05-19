#!/usr/bin/env python3
"""Print a summary of existing eval_results.json files.

Scans outputs/<dataset_file>/results/*/eval_results.json and prints a table.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='View saved PET eval results')
    parser.add_argument('--dataset_file', default='SHA', help='Dataset folder under outputs/')
    parser.add_argument('--output_dir', default='outputs', help='Root outputs directory')
    parser.add_argument('--sort_by', choices=('eval_mae', 'train_best_mae', 'backbone'), default='eval_mae')
    return parser.parse_args()


def format_float(value):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.2f}'
    except (TypeError, ValueError):
        return 'N/A'


def main():
    args = parse_args()
    results_dir = Path(args.output_dir) / args.dataset_file / 'results'
    if not results_dir.exists():
        print(f'No results directory found: {results_dir}')
        return

    rows = []
    for eval_file in sorted(results_dir.glob('*/eval_results.json')):
        try:
            data = json.loads(eval_file.read_text(encoding='utf-8'))
            data['_path'] = str(eval_file)
            rows.append(data)
        except Exception as exc:
            print(f'Warning: could not read {eval_file}: {exc}')

    if not rows:
        print(f'No eval_results.json files found under {results_dir}')
        return

    if args.sort_by == 'backbone':
        rows.sort(key=lambda r: str(r.get('backbone', '')))
    else:
        rows.sort(key=lambda r: r.get(args.sort_by) if r.get(args.sort_by) is not None else 1e9)

    header = f"{'Backbone':<22} {'Eval MAE':>9} {'Eval MSE':>9} {'Train MAE':>10} {'Best Ep':>8} {'Optuna':>9}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{row.get('backbone', 'N/A'):<22} "
            f"{format_float(row.get('eval_mae')):>9} "
            f"{format_float(row.get('eval_mse')):>9} "
            f"{format_float(row.get('train_best_mae')):>10} "
            f"{str(row.get('best_epoch', 'N/A')):>8} "
            f"{format_float(row.get('optuna_search_mae')):>9}"
        )


if __name__ == '__main__':
    main()