#!/usr/bin/env python3
"""
Batch evaluation script for PET checkpoints.
Scans outputs/SHA directory and evaluates all best_checkpoint.pth files.

Usage:
  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A
  
  With verbose error output:
  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A --verbose
  
  Debug specific backbone:
  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A --verbose_for maxvit

Helper scripts:
  - scripts/list_checkpoints.py: List all checkpoints and inferred backbones
  - scripts/diagnose_checkpoint.py: Test if a checkpoint can load
  - scripts/test_eval.py: Manually run eval.py for one checkpoint
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
    dirname_lower = dirname.lower()
    
    if 'convnextv2' in dirname_lower:
        return 'convnextv2_base'
    
    if 'swinv2' in dirname_lower:
        # SwinV2 models from timm
        if 'swinv2_base_window8_256' in dirname_lower:
            return 'swinv2_base_window8_256'
        if 'swinv2_small' in dirname_lower:
            return 'swinv2_small_window8_256'
        # Default SwinV2
        return 'swinv2_base_window8_256'
    
    if 'maxvit' in dirname_lower:
        # MaxViT models from timm - note: no "poly" suffix in timm
        # Available: maxvit_tiny_tf_224, maxvit_small_tf_224, maxvit_base_tf_224
        #           maxvit_rmlp_tiny_rw_256, maxvit_rmlp_small_rw_256, maxvit_rmlp_base_rw_256
        
        # Try to match checkpoint names to actual timm model names
        if 'poly' in dirname_lower:
            # "poly" suffix doesn't exist in timm, use rmlp_tiny_rw_256 as substitute
            return 'maxvit_rmlp_tiny_rw_256'
        elif 'maxvit_rmlp_tiny_rw_256' in dirname_lower:
            return 'maxvit_rmlp_tiny_rw_256'
        elif 'maxvit_rmlp_small' in dirname_lower:
            return 'maxvit_rmlp_small_rw_256'
        elif 'maxvit_rmlp_base' in dirname_lower:
            return 'maxvit_rmlp_base_rw_256'
        elif 'maxvit_base' in dirname_lower:
            return 'maxvit_base_tf_224'
        elif 'maxvit_small' in dirname_lower:
            return 'maxvit_small_tf_224'
        else:
            # Default MaxViT
            return 'maxvit_small_tf_224'
    
    if 'vgg' in dirname_lower:
        return 'vgg16_bn'
    
    return None


def run_eval(backbone, checkpoint_path, dataset_file, data_path, verbose=False):
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
        
        output_text = result.stdout + '\n' + result.stderr
        
        # Extract MAE from output - check multiple patterns
        for line in output_text.split('\n'):
            line_lower = line.lower()
            if 'mae:' in line_lower or 'mae =' in line_lower:
                # Try to extract the number
                for sep in ['mae:', 'mae =', 'mae=']:
                    if sep in line_lower:
                        idx = line_lower.find(sep)
                        remainder = line[idx + len(sep):].strip()
                        # Extract first number
                        num_str = ''
                        for char in remainder:
                            if char.isdigit() or char == '.':
                                num_str += char
                            elif num_str:
                                break
                        if num_str:
                            try:
                                return float(num_str)
                            except ValueError:
                                pass
        
        # If no MAE found and verbose, print debug info
        if verbose:
            print(f"    No MAE found in output")
            if result.returncode != 0:
                print(f"    Return code: {result.returncode}")
            if result.stderr:
                print(f"    STDERR: {result.stderr[:500]}")
            last_lines = output_text.strip().split('\n')[-5:]
            print(f"    Last output lines:")
            for line in last_lines:
                if line.strip():
                    print(f"      {line}")
        
        return None
    except subprocess.TimeoutExpired:
        print(f"    Timeout evaluating {checkpoint_path}")
        return None
    except Exception as e:
        print(f"    Error running eval: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser('Batch evaluation for PET checkpoints')
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if summary already exists')
    parser.add_argument('--backbone_filter', default='', help='Only eval dirs containing this backbone name')
    parser.add_argument('--verbose', action='store_true', help='Show error details for failed evaluations')
    parser.add_argument('--verbose_for', default='', help='Show verbose output only for dirs containing this string')
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
        be_verbose = args.verbose or (args.verbose_for and args.verbose_for in item['dir'])
        mae = run_eval(item['backbone'], item['checkpoint'], args.dataset_file, args.data_path, verbose=be_verbose)
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
