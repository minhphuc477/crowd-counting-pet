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
