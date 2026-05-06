#!/usr/bin/env python3
"""
Ensemble evaluation script for multi-seed trained PET models.

Usage:
    python scripts/ensemble_evaluate.py \
        --backbone convnextv2_base \
        --seeds 42 7 13 99 1234 \
        --checkpoint_dir results/convnextv2_base

This script will:
1. Load checkpoint for each seed
2. Run evaluation on validation/test set
3. Aggregate predictions (averaging, voting, etc.)
4. Report ensemble MAE and individual seed MAEs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import PET modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import build_dataset
from models import build_model
from engine import evaluate
import util.misc as utils


def get_args():
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation for multi-seed PET models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        default="convnextv2_base",
        type=str,
        help="Backbone architecture (default: convnextv2_base)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 13, 99, 1234],
        help="List of random seeds to evaluate (default: 42 7 13 99 1234)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="results",
        help="Base checkpoint directory (default: results)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets",
        help="Path to datasets",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="SHA",
        choices=("SHA", "SHB", "QNRF", "UCF"),
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="average",
        choices=("average", "voting", "max"),
        help="Ensemble method for combining predictions",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for inference",
    )
    parser.add_argument(
        "--override_score_threshold",
        type=float,
        default=None,
        help="Override the checkpoint score threshold during evaluation",
    )
    parser.add_argument(
        "--override_split_threshold",
        type=float,
        default=None,
        help="Override the checkpoint split threshold during evaluation",
    )
    parser.add_argument(
        "--override_split_threshold_quantile",
        type=float,
        default=None,
        help="Override the checkpoint split-threshold quantile during evaluation",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model."""
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def merge_checkpoint_args(args, checkpoint):
    checkpoint_args = checkpoint.get('args')
    if checkpoint_args is None:
        return args
    if isinstance(checkpoint_args, dict):
        checkpoint_args = argparse.Namespace(**checkpoint_args)

    merged = argparse.Namespace(**vars(checkpoint_args))
    runtime_keys = {
        'device', 'checkpoint_dir', 'seeds', 'batch_size', 'num_workers', 'data_path', 'dataset_file',
        'ensemble_method', 'override_score_threshold', 'override_split_threshold',
        'override_split_threshold_quantile',
    }
    for key in runtime_keys:
        setattr(merged, key, getattr(args, key))
    return merged


def apply_eval_overrides(args):
    override_score_threshold = getattr(args, 'override_score_threshold', None)
    override_split_threshold = getattr(args, 'override_split_threshold', None)
    override_split_threshold_quantile = getattr(args, 'override_split_threshold_quantile', None)
    if override_score_threshold is not None:
        args.score_threshold = float(override_score_threshold)
    if override_split_threshold is not None:
        args.split_threshold = float(override_split_threshold)
    if override_split_threshold_quantile is not None:
        args.split_threshold_quantile = float(override_split_threshold_quantile)
    return args


def resolve_checkpoint_path(args, seed):
    rel_dir = Path(args.checkpoint_dir) / args.backbone / f"seed_{seed}"
    candidates = [
        Path("outputs") / args.dataset_file / rel_dir / "best_checkpoint.pth",
        Path("outputs") / args.dataset_file / rel_dir / "checkpoint.pth",
        rel_dir / "best_checkpoint.pth",
        rel_dir / "checkpoint.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def evaluate_single_seed(model, data_loader, device):
    """Evaluate model on data_loader using engine.evaluate()."""
    if getattr(data_loader, 'batch_size', 1) not in (None, 1):
        raise ValueError('ensemble_evaluate only supports batch_size=1 because evaluation scores one image at a time.')
    results = evaluate(model, data_loader, device)
    return results.get('mae', None)


def main():
    args = get_args()
    device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print("Ensemble Evaluation for Multi-Seed PET Models")
    print(f"{'='*80}")
    print(f"Backbone: {args.backbone}")
    print(f"Seeds: {args.seeds}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Ensemble method: {args.ensemble_method}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Build dataset
    dataset_val = build_dataset(
        image_set="val",
        args=argparse.Namespace(
            data_path=args.data_path,
            dataset_file=args.dataset_file,
            patch_size=args.patch_size,
        ),
    )
    eval_batch_size = 1
    if args.batch_size != 1:
        print(f"Warning: batch_size={args.batch_size} is not supported by the current evaluator; using 1.")
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    
    results = {}
    
    # Evaluate each seed
    for seed in args.seeds:
        print(f"\nEvaluating seed {seed}...")

        checkpoint_path = resolve_checkpoint_path(args, seed)
        if not checkpoint_path.exists():
            print(f"  Skipping seed {seed} (checkpoint not found)")
            continue
        
        # Build model
        model_args = argparse.Namespace(
            backbone=args.backbone,
            no_pretrained_backbone=False,
            device=str(device),  # Required by build_pet
            position_embedding="sine",
            hidden_dim=256,
            nheads=8,
            dim_feedforward=512,
            dropout=0.0,
            # Loss coefficients (required by SetCriterion)
            ce_loss_coef=1.0,
            point_loss_coef=5.0,
            eos_coef=0.5,
            negative_loss_coef=0.1,
            non_div_loss_coef=0.25,
            quadtree_loss_coef=0.1,
            quadtree_prior_coef=0.025,
            split_count_threshold=2,
            split_pos_weight=1.0,
            split_threshold=0.5,
            split_threshold_quantile=0.55,
            score_threshold=0.5,
            # Matcher parameters
            set_cost_class=1.0,
            set_cost_point=0.05,
            # Decoder parameters (will be set by PET)
            dec_layers=2,
            warmup_epochs=5,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_args = merge_checkpoint_args(model_args, checkpoint)
        model_args = apply_eval_overrides(model_args)
        
        model, _ = build_model(model_args)
        model = model.to(device)
        model.load_state_dict(checkpoint["model"])
        
        # Evaluate
        mae = evaluate_single_seed(model, data_loader_val, device)
        
        if mae is not None:
            results[seed] = mae
            print(f"  Seed {seed}: MAE = {mae:.2f}")
            print(
                f"    checkpoint={checkpoint_path} "
                f"score_threshold={model_args.score_threshold} "
                f"split_threshold={model_args.split_threshold} "
                f"split_q={model_args.split_threshold_quantile}"
            )
        else:
            print(f"  Seed {seed}: Evaluation failed")
    
    # Print summary
    if results:
        mae_values = list(results.values())
        print(f"\n{'='*80}")
        print("Ensemble Evaluation Summary")
        print(f"{'='*80}")
        print("Individual seed MAEs:")
        for seed, mae in sorted(results.items()):
            print(f"  Seed {seed}: {mae:.2f}")
        print("\nAggregate Statistics:")
        print(f"  Mean MAE:   {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f}")
        print(f"  Min MAE:    {np.min(mae_values):.2f}")
        print(f"  Max MAE:    {np.max(mae_values):.2f}")
        print(f"  Median MAE: {np.median(mae_values):.2f}")
        print(f"{'='*80}\n")
        
        # Save results
        results_file = (
            Path(args.checkpoint_dir)
            / args.backbone
            / "ensemble_results.json"
        )
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "individual": results,
                "mean": float(np.mean(mae_values)),
                "std": float(np.std(mae_values)),
                "min": float(np.min(mae_values)),
                "max": float(np.max(mae_values)),
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}\n")
    else:
        print("\nNo results collected (no valid checkpoints found)")


if __name__ == "__main__":
    main()
