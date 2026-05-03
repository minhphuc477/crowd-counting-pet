#!/usr/bin/env python3
"""
List all checkpoints and their inferred backbones.
Helps diagnose why batch_eval is failing.

Usage:
  python scripts/list_checkpoints.py --output_dir outputs --dataset_file SHA
"""

import argparse
from pathlib import Path


def get_backbone_from_dirname(dirname):
    """Infer backbone name from directory name."""
    dirname_lower = dirname.lower()
    
    if 'convnextv2' in dirname_lower:
        return 'convnextv2_base'
    
    if 'swinv2' in dirname_lower:
        if 'swinv2_base_window8_256' in dirname_lower:
            return 'swinv2_base_window8_256'
        parts = dirname.split('_')
        for i, part in enumerate(parts):
            if 'swinv2' in part:
                extracted = [part]
                for j in range(i+1, min(i+4, len(parts))):
                    extracted.append(parts[j])
                    if parts[j].startswith('base') or parts[j].startswith('small'):
                        break
                backbone = '_'.join(extracted)
                if not any(x in backbone for x in ['window', 'base', 'small']):
                    return 'swinv2_base_window8_256'
                return backbone
        return 'swinv2_base_window8_256'
    
    if 'maxvit' in dirname_lower:
        if 'maxvit_rmlp_tiny_poly' in dirname_lower:
            return 'maxvit_rmlp_tiny_poly'
        if 'maxvit_rmlp_tiny_rw_256' in dirname_lower:
            return 'maxvit_rmlp_tiny_rw_256'
        parts = dirname.split('_')
        for i, part in enumerate(parts):
            if 'maxvit' in part:
                extracted = [part]
                for j in range(i+1, len(parts)):
                    extracted.append(parts[j])
                    if any(marker in parts[j] for marker in ['base', 'rw_256', 'poly', 'small']):
                        break
                backbone = '_'.join(extracted)
                return backbone if backbone.startswith('maxvit') else 'maxvit_small_tf_224'
        return 'maxvit_small_tf_224'
    
    if 'vgg' in dirname_lower:
        return 'vgg16_bn'
    
    return None


def main():
    parser = argparse.ArgumentParser('List all checkpoints')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--dataset_file', default='SHA')
    args = parser.parse_args()
    
    outputs_dir = Path(args.output_dir) / args.dataset_file
    if not outputs_dir.exists():
        print(f"Not found: {outputs_dir}")
        return
    
    print(f"\nCheckpoints in {outputs_dir}:\n")
    print(f"{'Directory':<50} {'Backbone':<30} {'Exists':<10}")
    print("-" * 90)
    
    for subdir in sorted(outputs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        checkpoint_path = subdir / 'best_checkpoint.pth'
        backbone = get_backbone_from_dirname(subdir.name)
        exists = "Yes" if checkpoint_path.exists() else "No"
        
        print(f"{subdir.name:<50} {str(backbone):<30} {exists:<10}")


if __name__ == '__main__':
    main()
