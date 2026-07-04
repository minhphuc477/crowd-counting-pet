# PET Experiment Scripts

This folder contains the maintained scripts for training, evaluating, and checking PET backbones.

## Maintained Scripts

- `check_backbone_contract.py`: verifies that a backbone produces PET-compatible 4x/8x feature maps and can run one synthetic forward pass.
- `run_backbone_seeds.py`: trains one or more backbones across seeds, with resume/skip support and optional eval after each run.
- `batch_eval.py`: evaluates existing `best_checkpoint.pth` files and writes per-run logs plus an aggregate summary.
- `find_best_checkpoint.py`: scans checkpoints and logs for stored MAE/MSE metadata.
- `optuna_search.py`: runs Optuna for one backbone and writes `optuna_best.json`.
- `ensemble_evaluate.py`: evaluates a set of seed checkpoints for one backbone.
- `run_ucfcc50_folds.py`: runs leakage-safe UCF-CC-50 five-fold training and
  aggregates the five held-out folds.
- `complete_partial_annotations.py`: creates auditable pseudo-completed
  ShanghaiTech training annotations from a partial-label stage-one model.
- `refine_point_annotations.py`: refines existing annotations with exactly one
  custom PET query per source point.
- `export_nwpu_test.py`: exports 1,500 ordered count predictions in the
  NWPU-Crowd hidden-test submission format.

The full protocols and their limitations are documented in
`docs/full_scope_protocols.md`.

## Paper Baseline

The default code path is paper-compatible PET:

```bash
python main.py \
  --backbone vgg16_bn \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --output_dir paper_vgg16_bn \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler step \
  --pet_loss_variant paper
```

## Backbone Contract Check

Run this before training a new timm backbone:

```bash
python scripts/check_backbone_contract.py --backbone convnextv2_base --device cpu
```

Check every supported backbone:

```bash
python scripts/check_backbone_contract.py --all --device cpu
```

## Batch Training

Train several backbones with one seed:

```bash
python scripts/run_backbone_seeds.py \
  --backbones convnextv2_base convnext_base fastvit_tiny efficientvit_tiny \
  --seeds 7 \
  --data_path ./data/ShanghaiTech/part_A \
  --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine --pet_loss_variant paper" \
  --check_contract \
  --eval_after_training
```

Use a preset:

```bash
python scripts/run_backbone_seeds.py \
  --preset crowd_dense \
  --seeds 7 \
  --data_path ./data/ShanghaiTech/part_A \
  --extra_args "--epochs 1500 --patch_size 256 --lr_scheduler warmup_hold_cosine --pet_loss_variant paper" \
  --check_contract \
  --eval_after_training
```

Useful flags:

- `--force`: retrain even if `best_checkpoint.pth` already exists.
- `--no_resume`: ignore an existing `checkpoint.pth`.
- `--dry_run`: print commands without running them.
- `--eval_after_training`: run `eval.py` after each seed.
- `--check_contract`: validate each backbone before training.

## Batch Evaluation

Evaluate all completed runs under `outputs/SHA`:

```bash
python scripts/batch_eval.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --verbose
```

Point to another checkpoint root:

```bash
python scripts/batch_eval.py \
  --checkpoint_root /path/to/outputs/SHA \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --verbose
```

Preview only:

```bash
python scripts/batch_eval.py --dataset_file SHA --dry_run
```

## Find Stored Best Checkpoints

```bash
python scripts/find_best_checkpoint.py \
  --root outputs \
  --backbone_filter convnextv2_base \
  --top_k 20
```

Search logs and JSON for a specific value:

```bash
python scripts/find_best_checkpoint.py \
  --root . \
  --target_mae 49.6 \
  --tolerance 1.0 \
  --search_logs
```
