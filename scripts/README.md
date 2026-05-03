# Multi-Seed Training Scripts for PET

This directory contains utilities for running PET training with multiple random seeds and evaluating ensemble results.

## Quick Start

### Single Multi-Seed Training Run

Train `convnextv2_base` backbone with 5 different seeds:

```bash
python scripts/run_backbone_seeds.py \
    --backbone convnextv2_base \
    --seeds 42 7 13 99 1234 \
    --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine"
```

For ConvNeXtV2, the current training code now applies a stronger default recipe than the original VGG path: `lr=5e-5`, `lr_backbone=5e-6`, `batch_size=4`, and `warmup_epochs=10` when those values are left at their defaults.

The same idea is now implemented as an explicit backbone recipe table in `main.py`, so future backbones like MaxViT can get their own named defaults instead of piggybacking on a single hard-coded branch.

### Key Arguments

- `--backbone`: Model backbone to train (default: `convnextv2_base`)
- `--seeds`: List of random seeds (default: `42 7 13 99 1234`)
- `--extra_args`: Additional arguments passed to `main.py` as a single quoted string
- `--output_dir`: Base output directory (default: `results`)
- `--dry_run`: Print commands without executing them
- `--continue_from_seed N`: Resume from seed N (skip earlier seeds)

### Output Structure

Results are organized as:
```
results/
  {backbone}/
    seed_{seed}/
      checkpoints/
      stats.json
      logs.txt
    experiment_log.json
```

## Scripts

### 1. `run_backbone_seeds.py`

Orchestrates multi-seed training experiments.

**Features:**
- Runs training sequentially for each seed
- Saves results per seed in organized directories
- Collects and aggregates MAE metrics across seeds
- Generates experiment log with summary statistics
- Supports resuming from a specific seed

**Usage Examples:**

```bash
# Basic: Train with default 5 seeds
python scripts/run_backbone_seeds.py --backbone convnextv2_base

# Advanced: Custom seeds and settings
python scripts/run_backbone_seeds.py \
    --backbone convnextv2_base \
    --seeds 42 7 13 99 1234 \
    --extra_args "--epochs 1500 --patch_size 256" \
    --output_dir /path/to/results

# Dry run: Preview commands without executing
python scripts/run_backbone_seeds.py \
    --backbone convnextv2_base \
    --dry_run

# Resume: Continue from seed 13
python scripts/run_backbone_seeds.py \
    --backbone convnextv2_base \
    --continue_from_seed 13
```

**Output:**
- Individual seed checkpoints and logs
- `experiment_log.json` with aggregate MAE statistics (mean, std, min, max)

### 2. `ensemble_evaluate.py`

Loads trained models from multiple seeds and evaluates them on a validation set.

**Features:**
- Loads checkpoints from all seeds
- Evaluates each model independently
- Computes aggregate statistics
- Saves results to `ensemble_results.json`

**Usage:**

```bash
# Evaluate with default seeds
python scripts/ensemble_evaluate.py \
    --backbone convnextv2_base

# Custom dataset and settings
python scripts/ensemble_evaluate.py \
    --backbone convnextv2_base \
    --seeds 42 7 13 99 1234 \
    --dataset_file SHA \
    --batch_size 8 \
    --patch_size 256
```

**Output:**
- Per-seed MAE values
- Aggregate statistics (mean ± std, min, max, median)
- Results saved to `results/{backbone}/ensemble_results.json`

## Typical Workflow

1. **Run multi-seed training:**
   ```bash
   python scripts/run_backbone_seeds.py \
       --backbone convnextv2_base \
       --seeds 42 7 13 99 1234 \
       --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine"
   ```

2. **Check intermediate results:**
   ```bash
   cat results/convnextv2_base/experiment_log.json | jq .metrics
   ```

3. **Evaluate ensemble on validation set:**
   ```bash
   python scripts/ensemble_evaluate.py --backbone convnextv2_base
   ```

4. **View results:**
   ```bash
   cat results/convnextv2_base/ensemble_results.json | jq .
   ```

## Expected Output

After running a multi-seed training experiment, you should see:

```
================================================================================
Multi-seed Training Experiment: convnextv2_base_20260430_143022
Backbone: convnextv2_base
Seeds: [42, 7, 13, 99, 1234]
Extra arguments: --epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine
================================================================================

================================================================================
Training convnextv2_base with seed 42
Output directory: results/convnextv2_base/seed_42
Command: python main.py --backbone convnextv2_base --seed 42 ...
================================================================================

[Training proceeds...]

================================================================================
Final Summary
================================================================================
Successful runs: 5/5
Output directory: results/convnextv2_base
================================================================================

Summary for convnextv2_base:
  Mean MAE:   42.15 ± 1.32
  Min MAE:    40.89
  Max MAE:    43.52
  Seeds with results: [42, 7, 13, 99, 1234]
```

## Notes

- The `--seed` argument is already defined in `main.py`, so multi-seed scripts will automatically use different seeds
- Output directories are created automatically per seed
- Training logs and checkpoints are saved in each seed directory
- Experiment logs are aggregated in the backbone directory for easy comparison across seeds

## Troubleshooting

**Issue: "Checkpoint not found"**
- Ensure training completed successfully for that seed
- Check `results/{backbone}/seed_{seed}/checkpoint.pth` exists

**Issue: CUDA/device errors**
- Verify `--device cuda` is set and GPU is available
- Check memory requirements for batch size
- Reduce batch size if out of memory

**Issue: Training diverges for some seeds**
- Review learning rate settings in `--extra_args`
- Check if warmup_epochs is appropriate for learning rate scheduler
- Verify batch size and loss scaling
