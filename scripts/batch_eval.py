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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from models.backbones import get_supported_timm_backbones, resolve_timm_backbone_name
except ImportError:
    get_supported_timm_backbones = None
    resolve_timm_backbone_name = lambda name: name


def get_backbone_from_dirname(dirname):
    """Infer backbone name from directory name."""
    dirname_lower = dirname.lower()
    normalized_dirname = dirname_lower.replace('-', '_')

    if get_supported_timm_backbones is not None:
        known_backbones = sorted(
            set(get_supported_timm_backbones()),
            key=len,
            reverse=True,
        )
        for backbone in known_backbones:
            candidates = {
                backbone.lower(),
                resolve_timm_backbone_name(backbone).lower(),
            }
            if any(candidate in normalized_dirname for candidate in candidates):
                return backbone
    
    if 'convnextv2' in dirname_lower:
        return 'convnextv2_base'
    if 'convnext' in dirname_lower:
        return 'convnext_base'
    if 'fastvit' in dirname_lower:
        return 'fastvit_tiny'
    if 'efficientvit' in dirname_lower:
        return 'efficientvit_tiny'
    if 'efficientnetv2' in dirname_lower:
        return 'efficientnetv2_tiny'
    if 'mobilenetv4' in dirname_lower:
        return 'mobilenetv4_small'
    if 'hgnetv2' in dirname_lower or 'hgnet' in dirname_lower:
        return 'hgnetv2_tiny'
    if 'pvt_v2' in dirname_lower or 'pvtv2' in dirname_lower:
        return 'pvtv2_b0'
    if 'edgenext' in dirname_lower:
        return 'edgenext_tiny'
    if 'repvit' in dirname_lower:
        return 'repvit_tiny'
    
    if 'swinv2' in dirname_lower:
        # SwinV2 models from timm
        if 'swinv2_base_window8_256' in dirname_lower:
            return 'swinv2_base_window8_256'
        if 'swinv2_small' in dirname_lower:
            return 'swinv2_small_window8_256'
        # Default SwinV2
        return 'swinv2_base_window8_256'
    
    if 'maxvit' in dirname_lower:
        # Use 256-compatible MaxViT variants for PET's default crop/padding.
        
        # Try to match checkpoint names to actual timm model names
        if 'poly' in dirname_lower:
            # "poly" suffix doesn't exist in timm, use rmlp_tiny_rw_256 as substitute
            return 'maxvit_rmlp_tiny_rw_256'
        elif 'maxvit_rmlp_tiny_rw_256' in dirname_lower:
            return 'maxvit_rmlp_tiny_rw_256'
        elif 'maxvit_rmlp_small' in dirname_lower:
            return 'maxvit_rmlp_small_rw_256'
        elif 'maxvit_small' in dirname_lower:
            return 'maxvit_small'
        elif 'maxvit_tiny' in dirname_lower:
            return 'maxvit_tiny'
        else:
            # Default MaxViT
            return 'maxvit_tiny'
    
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
        
        # Extract MAE from output - try multiple patterns
        mae_value = None
        
        # Pattern 1: "mae: X" or "mae: X," or "mae = X"
        for line in output_text.split('\n'):
            line_lower = line.lower()
            if 'mae' in line_lower and any(sep in line_lower for sep in [':', '=']):
                # Find "mae:" or "mae="
                for sep in ['mae:', 'mae =', 'mae=']:
                    if sep in line_lower:
                        idx = line_lower.find(sep)
                        remainder = line[idx + len(sep):].strip()
                        # Extract first number (handle cases like "123.45," or "123.45 " etc)
                        num_str = ''
                        for char in remainder:
                            if char.isdigit() or char == '.':
                                num_str += char
                            elif num_str:
                                break
                        if num_str:
                            try:
                                mae_value = float(num_str)
                                break
                            except ValueError:
                                pass
                if mae_value is not None:
                    break
        
        # If still no MAE and verbose, print debug info
        if mae_value is None and verbose:
            print(f"    DEBUG: MAE extraction failed")
            print(f"    Return code: {result.returncode}")
            if result.stderr:
                err_lines = result.stderr.strip().split('\n')[-3:]
                print(f"    STDERR (last 3 lines):")
                for line in err_lines:
                    if line.strip():
                        print(f"      {line}")
            if result.stdout:
                out_lines = result.stdout.strip().split('\n')[-5:]
                print(f"    STDOUT (last 5 lines):")
                for line in out_lines:
                    if line.strip():
                        print(f"      {line}")
        
        return mae_value
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
