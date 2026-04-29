"""Ensemble evaluation: load multiple checkpoints, average predictions, report MAE.

Usage:
  python scripts/ensemble_evaluate.py \\
    --backbone swinv2_base_window8_256 \\
    --checkpoints outputs/SHA/swinv2_base_window8_256_seed_*/best_checkpoint.pth \\
    --data_path ./data/ShanghaiTech/PartA \\
    --patch_size 256

This script:
  1. Loads multiple model checkpoints (one per seed)
  2. Runs each on validation set, collects outputs_scores
  3. Averages scores across models
  4. Sweeps threshold and reports best MAE/MSE for ensemble
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import build_dataset
from models import build_model
import util.misc as utils
from engine import evaluate


def load_checkpoint_args(checkpoint_path):
    """Load args from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('args', None)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model and return both model and args."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint.get('args', None)
    
    if args is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'args'")
    
    model, criterion = build_model(args)
    model.to(device)
    
    # Load state dict
    utils.load_model_state(model, checkpoint['model'])
    model.eval()
    
    return model, args


@torch.no_grad()
def collect_predictions(model, data_loader, device, threshold=0.5):
    """Run model on validation set, return list of score tensors (one per sample)."""
    model.eval()
    all_scores = []
    all_gt_counts = []
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        
        # Forward pass
        outputs = model(samples, test=True, targets=targets)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0].float()
        
        all_scores.append(outputs_scores.detach().cpu())
        all_gt_counts.append(targets[0]['points'].shape[0])
    
    return all_scores, all_gt_counts


def compute_mae_mse_with_threshold(all_predictions, all_gt_counts, threshold):
    """Compute MAE/MSE for ensemble predictions at given threshold."""
    mae_sum, mse_sum = 0.0, 0.0
    
    for pred_scores, gt_count in zip(all_predictions, all_gt_counts):
        pred_count = (pred_scores > threshold).sum().item()
        error = abs(pred_count - gt_count)
        mae_sum += error
        mse_sum += error ** 2
    
    n_samples = len(all_predictions)
    mae = mae_sum / n_samples
    mse = math.sqrt(mse_sum / n_samples)
    
    return mae, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str)
    parser.add_argument('--checkpoints', required=True, nargs='+', type=str,
                        help='Paths to checkpoint files (supports glob patterns)')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/PartA', type=str)
    parser.add_argument('--dataset_file', default='SHA', type=str)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--threshold_min', default=0.30, type=float)
    parser.add_argument('--threshold_max', default=0.95, type=float)
    parser.add_argument('--threshold_step', default=0.025, type=float)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Expand glob patterns
    checkpoint_paths = []
    for pattern in args.checkpoints:
        checkpoint_paths.extend(glob.glob(pattern))
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found matching patterns: {args.checkpoints}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for cp in checkpoint_paths:
        print(f"  - {cp}")
    
    # Load first checkpoint to get args
    first_args = load_checkpoint_args(checkpoint_paths[0])
    if first_args is None:
        raise ValueError("Could not load args from checkpoint")
    
    # Override with CLI args
    first_args.backbone = args.backbone
    first_args.data_path = args.data_path
    first_args.dataset_file = args.dataset_file
    first_args.patch_size = args.patch_size
    first_args.device = args.device
    first_args.num_workers = args.num_workers
    
    # Build validation dataset (once)
    print("Building validation dataset...")
    dataset_val = build_dataset(image_set='val', args=first_args)
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, sampler=torch.utils.data.SequentialSampler(dataset_val),
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    
    # Load models and collect predictions
    print("Loading models and collecting predictions...")
    all_model_predictions = []
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n[{i+1}/{len(checkpoint_paths)}] Loading {checkpoint_path}...")
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        
        pred_scores, gt_counts = collect_predictions(model, data_loader_val, device)
        all_model_predictions.append(pred_scores)
        
        print(f"  Collected {len(pred_scores)} predictions")
    
    # Average predictions across models
    print("\nAveraging predictions across models...")
    ensemble_predictions = []
    for sample_idx in range(len(all_model_predictions[0])):
        scores_list = [model_preds[sample_idx] for model_preds in all_model_predictions]
        avg_scores = torch.stack(scores_list).mean(dim=0)
        ensemble_predictions.append(avg_scores)
    
    # Sweep thresholds and find best
    print("\nSweeping thresholds...")
    thresholds = []
    maes = []
    mses = []
    
    steps = int(round((args.threshold_max - args.threshold_min) / args.threshold_step))
    for idx in range(steps + 1):
        threshold = args.threshold_min + idx * args.threshold_step
        threshold = min(max(threshold, 0.0), 1.0)
        thresholds.append(threshold)
        
        mae, mse = compute_mae_mse_with_threshold(ensemble_predictions, gt_counts, threshold)
        maes.append(mae)
        mses.append(mse)
        
        if idx % 5 == 0:
            print(f"  threshold={threshold:.3f}: MAE={mae:.2f}, MSE={mse:.2f}")
    
    # Find best threshold
    best_idx = min(range(len(maes)), key=lambda i: (maes[i], mses[i]))
    best_threshold = thresholds[best_idx]
    best_mae = maes[best_idx]
    best_mse = mses[best_idx]
    
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION RESULT")
    print("="*60)
    print(f"Backbone: {args.backbone}")
    print(f"Number of models: {len(checkpoint_paths)}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Best MSE: {best_mse:.2f}")
    print("="*60)
    
    # Save results
    output_file = Path(f"ensemble_results_{args.backbone}.json")
    results = {
        'backbone': args.backbone,
        'num_models': len(checkpoint_paths),
        'checkpoints': checkpoint_paths,
        'best_threshold': float(best_threshold),
        'best_mae': float(best_mae),
        'best_mse': float(best_mse),
        'threshold_sweep': {
            'thresholds': [float(t) for t in thresholds],
            'maes': [float(m) for m in maes],
            'mses': [float(m) for m in mses],
        }
    }
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
