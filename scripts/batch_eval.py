#!/usr/bin/env python3
"""
Batch evaluation script for PET checkpoints.
Scans outputs/SHA directory and evaluates all best_checkpoint.pth files.

Usage:
  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def get_backbone_from_dirname(dirname):
    """Infer backbone name from directory name."""
    if 'convnextv2' in dirname:
        return 'convnextv2_base'
    if 'maxvit' in dirname:
        # Try to extract the full model name
        for part in dirname.split('_'):
            if 'maxvit' in part:
                return part
        return 'maxvit_small_tf_224'
    if 'swinv2' in dirname:
        for part in dirname.split('_'):
            if 'swinv2' in part:
                return part
        return 'swinv2_base_window8_256'
    if 'vgg' in dirname:
        return 'vgg16_bn'
    return None


def run_eval(backbone, checkpoint_path, dataset_file, data_path):
    """Run eval.py for a single checkpoint."""
    cmd = [
        sys.executable,
        'eval.py',
        '--backbone', backbone,
        '--dataset_file', dataset_file,
        '--data_path', data_path,
        '--resume', str(checkpoint_path),
        '--device', 'cuda',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Extract MAE from output
        for line in result.stdout.split('\n'):
            if 'mae:' in line.lower():
                parts = line.split('mae:')
                if len(parts) > 1:
                    mae_str = parts[1].split(',')[0].strip()
                    try:
                        return float(mae_str)
                    except ValueError:
                        pass
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout evaluating {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error running eval for {checkpoint_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser('Batch evaluation for PET checkpoints')
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if summary already exists')
    parser.add_argument('--backbone_filter', default='', help='Only eval dirs containing this backbone name')
    args = parser.parse_args()

    outputs_dir = Path(args.output_dir) / args.dataset_file
    if not outputs_dir.exists():
        print(f"Output directory not found: {outputs_dir}")
        return

    # Collect all checkpoints
    checkpoints_to_eval = []
    for subdir in sorted(outputs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if args.backbone_filter and args.backbone_filter not in subdir.name:
            continue

        # Check for best_checkpoint.pth
        checkpoint_path = subdir / 'best_checkpoint.pth'
        if checkpoint_path.exists():
            backbone = get_backbone_from_dirname(subdir.name)
            if backbone:
                checkpoints_to_eval.append({
                    'dir': subdir.name,
                    'checkpoint': checkpoint_path,
                    'backbone': backbone,
                })
            else:
                print(f"Warning: Could not infer backbone for {subdir.name}")

    if not checkpoints_to_eval:
        print("No checkpoints found to evaluate")
        return

    # Run evaluations
    results = defaultdict(list)
    summary_path = outputs_dir / 'EVAL_SUMMARY.json'

    print(f"Found {len(checkpoints_to_eval)} checkpoints to evaluate")
    for i, item in enumerate(checkpoints_to_eval):
        print(f"\n[{i+1}/{len(checkpoints_to_eval)}] Evaluating {item['dir']} ({item['backbone']})...")
        mae = run_eval(item['backbone'], item['checkpoint'], args.dataset_file, args.data_path)
        if mae is not None:
            results[item['backbone']].append({
                'dir': item['dir'],
                'mae': float(mae),
                'checkpoint': str(item['checkpoint']),
            })
            print(f"  MAE: {mae:.4f}")
        else:
            print(f"  Failed to get MAE")

    # Aggregate by backbone
    summary = {}
    for backbone in sorted(results.keys()):
        maes = [r['mae'] for r in results[backbone]]
        summary[backbone] = {
            'count': len(maes),
            'mean_mae': float(np.mean(maes)),
            'std_mae': float(np.std(maes)),
            'min_mae': float(np.min(maes)),
            'max_mae': float(np.max(maes)),
            'results': results[backbone],
        }

    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSummary saved to {summary_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for backbone in sorted(summary.keys()):
        s = summary[backbone]
        print(f"\n{backbone}:")
        print(f"  Count: {s['count']}")
        print(f"  Mean MAE: {s['mean_mae']:.4f} ± {s['std_mae']:.4f}")
        print(f"  Range: [{s['min_mae']:.4f}, {s['max_mae']:.4f}]")
        for r in s['results']:
            print(f"    - {r['dir']}: {r['mae']:.4f}")


if __name__ == '__main__':
    main()
