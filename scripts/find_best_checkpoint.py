#!/usr/bin/env python3
"""Find PET checkpoints by stored training/eval metadata.

This does not run evaluation. It scans checkpoint files and prints metadata such
as best_mae, best_mse, best_epoch, and checkpoint epoch so you can locate the
old checkpoint that recorded a specific best result.
"""

import argparse
import json
import re
from pathlib import Path


DEFAULT_NAMES = (
    'best_checkpoint.pth',
    'checkpoint.pth',
    'final_checkpoint.pth',
)

MAE_KEYS = (
    'best_mae',
    'best_test_mae',
    'test_mae',
    'eval_mae',
    'mean_mae',
    'optuna_search_mae',
)

TEXT_MAE_RE = re.compile(
    r'(?P<label>best\s+mae|best_mae|best_test_mae|test_mae|eval_mae|mean_mae|mae)\s*[:=,]\s*(?P<value>[-+]?\d*\.?\d+)',
    re.IGNORECASE,
)


def load_metadata(path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    metadata = {}
    for key in ('best_mae', 'best_mse', 'best_epoch', 'epoch'):
        value = checkpoint.get(key)
        if hasattr(value, 'item'):
            value = value.item()
        metadata[key] = value
    args = checkpoint.get('args')
    if isinstance(args, dict):
        metadata['backbone'] = args.get('backbone')
        metadata['data_path'] = args.get('data_path')
        metadata['dataset_file'] = args.get('dataset_file')
    elif args is not None:
        metadata['backbone'] = getattr(args, 'backbone', None)
        metadata['data_path'] = getattr(args, 'data_path', None)
        metadata['dataset_file'] = getattr(args, 'dataset_file', None)
    else:
        metadata['backbone'] = None
        metadata['data_path'] = None
        metadata['dataset_file'] = None
    return metadata


def format_float(value):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return str(value)


def is_target_match(value, target, tolerance):
    if target is None:
        return True
    return abs(float(value) - target) <= tolerance


def walk_dict_values(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f'{prefix}.{key}' if prefix else str(key)
            yield from walk_dict_values(value, next_prefix)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = f'{prefix}[{index}]'
            yield from walk_dict_values(value, next_prefix)
    else:
        yield prefix, obj


def collect_log_matches(root, target_mae, tolerance, backbone_filter):
    rows = []
    extensions = {'.json', '.txt', '.log'}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if backbone_filter and backbone_filter.lower() not in str(path).lower():
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    data = None
                if data is not None:
                    for key_path, value in walk_dict_values(data):
                        key = key_path.split('.')[-1]
                        if key not in MAE_KEYS:
                            continue
                        try:
                            value_float = float(value)
                        except (TypeError, ValueError):
                            continue
                        if is_target_match(value_float, target_mae, tolerance):
                            rows.append((value_float, path, line_number, key_path, stripped[:220]))

            for match in TEXT_MAE_RE.finditer(stripped):
                try:
                    value_float = float(match.group('value'))
                except ValueError:
                    continue
                if is_target_match(value_float, target_mae, tolerance):
                    rows.append((value_float, path, line_number, match.group('label'), stripped[:220]))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Find PET checkpoints by stored best_mae')
    parser.add_argument('--root', default='outputs', help='Directory to scan recursively')
    parser.add_argument('--names', nargs='+', default=list(DEFAULT_NAMES), help='Checkpoint filenames to scan')
    parser.add_argument('--target_mae', type=float, default=None, help='Optional target best_mae, e.g. 49.6')
    parser.add_argument('--tolerance', type=float, default=1.0, help='Target match tolerance')
    parser.add_argument('--top_k', type=int, default=30, help='Print this many lowest best_mae checkpoints')
    parser.add_argument('--backbone_filter', default='', help='Only show paths/backbones containing this text')
    parser.add_argument('--search_logs', action='store_true', help='Also search JSON/txt/log files for MAE values')
    parser.add_argument('--logs_only', action='store_true', help='Only search JSON/txt/log files, not checkpoints')
    parser.add_argument('--include_errors', action='store_true', help='Print files that could not be loaded')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f'Root not found: {root}')
        return 1

    rows = []
    errors = []
    if not args.logs_only:
        for name in args.names:
            for path in sorted(root.rglob(name)):
                if not path.is_file():
                    continue
                if args.backbone_filter:
                    text = str(path).lower()
                    if args.backbone_filter.lower() not in text:
                        continue
                try:
                    metadata = load_metadata(path)
                except Exception as exc:
                    errors.append((path, exc))
                    continue
                best_mae = metadata.get('best_mae')
                if best_mae is None:
                    continue
                try:
                    best_mae_float = float(best_mae)
                except (TypeError, ValueError):
                    continue
                rows.append((best_mae_float, path, metadata))

    if args.target_mae is not None:
        rows = [
            row for row in rows
            if abs(row[0] - args.target_mae) <= args.tolerance
        ]
        rows.sort(key=lambda row: abs(row[0] - args.target_mae))
    else:
        rows.sort(key=lambda row: row[0])

    if args.logs_only:
        rows = []

    if not args.logs_only and not rows:
        print('No matching checkpoints found.')
    elif not args.logs_only:
        header = f"{'best_mae':>9} {'best_mse':>9} {'best_ep':>7} {'epoch':>7} {'backbone':<22} path"
        print(header)
        print('-' * len(header))
        for best_mae, path, metadata in rows[:args.top_k]:
            print(
                f"{best_mae:9.4f} "
                f"{format_float(metadata.get('best_mse')):>9} "
                f"{str(metadata.get('best_epoch', 'N/A')):>7} "
                f"{str(metadata.get('epoch', 'N/A')):>7} "
                f"{str(metadata.get('backbone') or 'N/A'):<22} "
                f"{path}"
            )

    if args.search_logs or args.logs_only:
        log_rows = collect_log_matches(root, args.target_mae, args.tolerance, args.backbone_filter)
        if args.target_mae is not None:
            log_rows.sort(key=lambda row: abs(row[0] - args.target_mae))
        else:
            log_rows.sort(key=lambda row: row[0])

        print('\nLog/JSON matches:')
        if not log_rows:
            print('No matching MAE values found in JSON/txt/log files.')
        else:
            header = f"{'mae':>9} {'line':>6} {'label':<20} path"
            print(header)
            print('-' * len(header))
            for value, path, line_number, label, line in log_rows[:args.top_k]:
                print(f"{value:9.4f} {line_number:6} {label:<20} {path}")
                print(f"  {line}")

    if errors and args.include_errors:
        print('\nErrors:')
        for path, exc in errors:
            print(f'  {path}: {exc}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
