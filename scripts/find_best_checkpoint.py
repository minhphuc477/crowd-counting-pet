#!/usr/bin/env python3
"""Find PET checkpoints by stored training/eval metadata.

This does not run evaluation. It scans checkpoint files and prints metadata such
as best_mae, best_mse, best_epoch, and checkpoint epoch so you can locate the
old checkpoint that recorded a specific best result.
"""

import argparse
from pathlib import Path


DEFAULT_NAMES = (
    'best_checkpoint.pth',
    'checkpoint.pth',
    'final_checkpoint.pth',
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


def parse_args():
    parser = argparse.ArgumentParser(description='Find PET checkpoints by stored best_mae')
    parser.add_argument('--root', default='outputs', help='Directory to scan recursively')
    parser.add_argument('--names', nargs='+', default=list(DEFAULT_NAMES), help='Checkpoint filenames to scan')
    parser.add_argument('--target_mae', type=float, default=None, help='Optional target best_mae, e.g. 49.6')
    parser.add_argument('--tolerance', type=float, default=1.0, help='Target match tolerance')
    parser.add_argument('--top_k', type=int, default=30, help='Print this many lowest best_mae checkpoints')
    parser.add_argument('--backbone_filter', default='', help='Only show paths/backbones containing this text')
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

    if not rows:
        print('No matching checkpoints found.')
    else:
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

    if errors and args.include_errors:
        print('\nErrors:')
        for path, exc in errors:
            print(f'  {path}: {exc}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
