#!/usr/bin/env python3
"""
Diagnose checkpoint loading issues.
Tests if a checkpoint can be loaded and what the error is.

Usage:
  python scripts/diagnose_checkpoint.py --checkpoint outputs/SHA/maxvit_rmlp_tiny_poly/best_checkpoint.pth --backbone maxvit_rmlp_tiny_poly
"""

import argparse
import sys
import torch
from pathlib import Path
from types import SimpleNamespace


def diagnose(checkpoint_path, backbone):
    print(f"\nDiagnose: {checkpoint_path}")
    print(f"Backbone: {backbone}\n")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
    
    # Load checkpoint
    print("\n1. Loading checkpoint...")
    try:
        ckpt = torch.load(str(checkpoint_path), weights_only=False, map_location='cpu')
        print("   OK - Checkpoint loaded")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Inspect checkpoint structure
    print(f"\n2. Checkpoint keys: {list(ckpt.keys())}")
    
    if 'model' in ckpt:
        model_keys = list(ckpt['model'].keys())
        print(f"   Model has {len(model_keys)} parameters")
        print(f"   First 5 keys: {model_keys[:5]}")
    
    # Try building model
    print("\n3. Building model...")
    try:
        from models import build_model

        args = SimpleNamespace(
            backbone=backbone,
            position_embedding='sine',
            dec_layers=2,
            dim_feedforward=512,
            hidden_dim=256,
            dropout=0.0,
            nheads=8,
            set_cost_class=1.0,
            set_cost_point=0.05,
            ce_loss_coef=1.0,
            point_loss_coef=5.0,
            eos_coef=0.5,
            aux_loss=False,
            device='cpu',
            num_classes=1,
            sparse_stride=8,
            dense_stride=4,
            negative_loss_coef=0.1,
            non_div_loss_coef=0.25,
            quadtree_loss_coef=0.1,
            quadtree_prior_coef=0.025,
            split_count_threshold=2,
            split_pos_weight=1.0,
            split_threshold=0.5,
            split_threshold_quantile=0.55,
            score_threshold=0.5,
            no_pretrained_backbone=True,
        )

        model, criterion = build_model(args)
        print("   OK - Model built")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load state dict
    print("\n4. Loading model state dict...")
    try:
        model.load_state_dict(ckpt['model'], strict=False)
        print("   OK - State dict loaded")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Diagnose checkpoint issues')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--backbone', required=True, help='Backbone name')
    args = parser.parse_args()
    
    success = diagnose(args.checkpoint, args.backbone)
    sys.exit(0 if success else 1)
