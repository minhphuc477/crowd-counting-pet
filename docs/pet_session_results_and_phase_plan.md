# PET Improvement Session Log and Phase Plan

This note records the experiments/results reported in the session, the codebase
findings, and the implementation direction for `PET_improvement_plan.md` and
`PET_improve_phase2.md`.

## Current Best Known Result

Best checkpoint reported so far:

```text
outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_checkpoint.pth
```

Best validation sweep:

```text
MAE=50.4341
MSE=79.2122
score_threshold=0.59
split_threshold=0.45
eval_nms_radius=0.0
eval_branch_gate=none
```

This is the strongest known VGG16-BN PET variant from the session. It combines
the useful PET-compatible APG supervision with the existing lite FPN path.

## Results Received During The Session

### Baseline and threshold sweeps

- `outputs/SHA/outputs/SHA/vgg16_bn_step_drop700/best_checkpoint.pth`
  produced about `MAE=51.7363`, `MSE=84.1348` at epoch 370 with the paper-like
  step schedule and lite FPN.
- Threshold sweep around the baseline found best:
  `MAE=51.5769`, `MSE=83.3801`, `score_threshold=0.46`,
  `split_threshold=0.45`.
- APG run `vgg16_bn_drop700_apg_seed42` improved threshold sweep to:
  `MAE=50.6923`, `MSE=80.7227`, `score_threshold=0.55`.

### APG plus LC

- `outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_eval_results.json`
  reported:

```json
{
  "epoch": 585,
  "test_mae": 50.934065934065934,
  "test_mse": 79.4391051596846,
  "pred_cnt": 434.2637362637363,
  "gt_cnt": 433.3076923076923
}
```

- Threshold sweep of that checkpoint improved the measured result to:
  `MAE=50.4341`, `MSE=79.2122`, `score_threshold=0.59`,
  `split_threshold=0.45`.
- Predicted mean count was close to the GT mean, so this checkpoint is not
  suffering from a simple global overcount/undercount problem.

### Variants that did not help

- TTA horizontal flip worsened the APG+LC checkpoint:
  best about `MAE=50.5769`, `MSE=80.1469`.
- Eval NMS did not improve the APG+LC checkpoint:
  best stayed `MAE=50.4341`.
- `crowd_no_overlap` protocol returned the same result as PET full-image eval
  for this checkpoint.
- Local region count loss produced about `MAE=53`, worse than APG+LC.
- Count loss / delayed count loss produced about `MAE=64-68`, worse and
  unstable.
- QD-APG-style runs produced about `MAE=60-70+`, worse than plain APG.
- APG consistency run reported:
  `MAE=53.5495`, `MSE=85.7583`.
- Soft APG plus freeze BN threshold sweep reported:
  `MAE=52.8462`, `MSE=84.8151`.
- Swin shifted encoder produced about `MAE=106`, severe undercount.
- Global context / residual global context attempts produced catastrophic
  undercount around `MAE=332-337`, so decoder global context should remain off.
- Multi-crop 256-only fine-tuning from APG+LC did not improve; one run reported
  best `MAE=50.8901` at epoch 2 and later degraded to `MAE=55.2308`.

## Codebase Findings

### Resume bug

`--resume_model_only` originally did not preserve several runtime training
arguments from the command line. In particular, `--batch_size 1` could be
overwritten by the checkpoint's `batch_size=8`. That explains logs with
`0/37` iterations per epoch even when the command requested batch size 1.

The training resume path was updated to preserve:

- `batch_size`
- `accum_iter`
- `freeze_backbone_epochs`
- `clip_max_norm`
- `amp`

### OOM root cause

The collate path pads training images to a 256-pixel block:

```text
256 -> [3, 256, 256]
320 -> [3, 512, 512]
384 -> [3, 512, 512]
512 -> [3, 512, 512]
```

Therefore, `patch_size_choices 192,256,320` is not a clean three-scale crop
setup in this repo:

- `192` becomes a 256-padded sample.
- `320` becomes a 512-padded sample.
- `384` also becomes a 512-padded sample.

The safe choices without redesigning PET window sizes are `256` and `512`.

### Split supervision mismatch

The plan says `--quadtree_loss_coef 0.5` enables GT per-cell split supervision.
In the current code, that is only true in the non-paper split branch. With the
default `pet_loss_variant='paper'`, the coefficient still weights the original
PET min/max split prior.

Implementation direction: add an explicit split supervision selector so command
lines can request GT split supervision without also changing the point-query
classification/regression loss variant.

### Scale augmentation

The scale augmentation should skip a random downscale if the scaled image cannot
still supply the requested crop size. This avoids creating blurred upsampled
training crops from small images.

## Research Notes

Primary sources checked:

- PET: https://arxiv.org/abs/2308.13814
- APGCC: https://arxiv.org/abs/2405.10589
- ConvNeXt V2: https://arxiv.org/abs/2301.00808
- Bayesian Loss: https://arxiv.org/abs/1908.03684
- DM-Count / OT loss: https://arxiv.org/abs/2009.13077
- STEERER: https://openaccess.thecvf.com/content/ICCV2023/html/Han_STEERER_Resolving_Scale_Variations_for_Counting_and_Localization_via_Selective_ICCV_2023_paper.html
- PyTorch AMP: https://docs.pytorch.org/docs/stable/amp.html

The sources support the broad direction:

- PET's quadtree split is the central routing mechanism.
- APG-style auxiliary proposal guidance is relevant because PET's Hungarian
  matching supervises only a small subset of fixed point queries.
- ConvNeXt V2 is a stronger pure ConvNet feature backbone and is a better fit
  than adding another attention-window hierarchy on top of PET.
- Bayesian/OT-style losses are plausible smoother alternatives to hard matching,
  but should be optional and staged because several auxiliary count losses have
  already failed in this repo.

## Plan Corrections Based On Actual Results

- Do not treat TTA as guaranteed free MAE. Horizontal flip TTA already worsened
  the best APG+LC checkpoint.
- Do not use `320` or `384` as "medium crop" sizes unless the padding/window
  contract is redesigned.
- Do not enable global decoder context, shifted encoder windows, QD-APG, region
  count, or count loss as part of the next main run.
- Do not claim VGG cannot reproduce 49 as a proven fact. The session evidence
  shows VGG APG+LC reaches about 50.43 on this setup; lower than that likely
  needs either a better backbone or a structurally better loss.

## Implementation Tasks

Completed before this note:

- AMP training support.
- Gradient accumulation.
- Runtime resume fixes for batch size and related args.
- `eval_before_train`.
- `freeze_bn`.
- Multi-size crop parsing.
- APG, soft APG, IFI, QD-APG, region count, NMS, branch gate experiments/tools.

Implemented from the two phase plans:

- Safe random scale augmentation for SHA and QNRF.
  - Implemented as `safe_random_scale()` in `datasets/SHA.py`.
  - QNRF reuses the same helper.
  - Unsafe downscales that cannot still supply the requested crop are skipped.
- Explicit GT split supervision selector.
  - New flag: `--split_loss_variant {auto,paper,gt,paper_gt}`.
  - `paper` keeps PET min/max split prior.
  - `gt` uses per-cell GT BCE supervision.
  - `paper_gt` combines GT BCE with the old PET prior via
    `--quadtree_prior_coef`.
- Optional Bayesian point-count auxiliary loss.
  - New flags: `--bayesian_loss_coef`, `--bayesian_sigma`,
    `--bayesian_bg_coef`, `--bayesian_loss_gate`,
    `--bayesian_start_epoch`, `--bayesian_end_epoch`.
  - It is off by default.
  - It uses PET predicted points and gated person probabilities, not an
    external density map.
- Optional scale TTA support in eval tooling.
  - New `eval.py` / sweep flag: `--tta_scales`.
  - Scaled images are resized first and then padded to PET-compatible 256
    multiples.
- Optional STEERER-inspired soft split score gating at inference.
  - New flag: `--eval_soft_split_gate {none,query,pred}`.
  - This multiplies person scores by sparse/dense split responsibility before
    thresholding.
  - It is off by default because hard branch gates and NMS did not help the
    known APG+LC checkpoint.
- Inspector updates.
  - `scripts/inspect_pet_run.py` now prints AMP, accumulation, freeze, split
    loss, Bayesian, and soft split eval flags.

Validated smoke checks:

- `python -m py_compile main.py eval.py engine.py models/pet.py datasets/SHA.py datasets/QNRF.py scripts/sweep_eval_thresholds.py scripts/inspect_pet_run.py`
- Resume merge keeps `batch_size`, `accum_iter`, `split_loss_variant`, and
  `bayesian_loss_coef` under `--resume_model_only`.
- `--auto_backbone_recipe` no longer overrides an explicitly supplied
  `--batch_size`.
- Training startup prints `batch config: batch_size=... accum_iter=... effective_batch_size=... train_batches=...`.
- Dataset/TTA helper smoke check passed.
- Bayesian loss finite/backward smoke check passed.
- Soft split score-gate smoke check passed.

## Recommended Next Experiment Direction

The next main run should not be another VGG auxiliary-loss stack. Use the known
good ideas and move the primary feature extractor:

1. Start ConvNeXt V2 with ImageNet/timm pretrained weights.
2. Keep PET defaults that worked: lite FPN, no branch gate, no eval NMS.
3. Use explicit GT split supervision.
4. Delay APG instead of turning on every auxiliary from epoch 0.
5. Use AMP and gradient accumulation to fit the GPU.

If ConvNeXt V2 base is too large for 15 GB, use `convnextv2_tiny` or
`convnextv2_base` with `batch_size=1`, `accum_iter=4`, and AMP.

## Candidate Commands

### Main ConvNeXt V2 run

This is the command aligned with the two phase plans, adjusted for the actual
repo behavior and previous failed trials:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=./data/ShanghaiTech/part_A
export BATCH=15
export ACCUM=4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone convnextv2_base \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/convnextv2_base_gt_split_apg_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 200 \
  --min_lr 1e-7 \
  --lr 0.00005 \
  --lr_backbone 0.000005 \
  --lr_backbone_adapter 0.0001 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --ema_decay 0.9999 \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.5 \
  --split_count_threshold 2 \
  --split_pos_weight 2.0 \
  --apg_loss_coef 1.0 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --apg_start_epoch 100 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --eval_soft_split_gate none \
  --seed 42
```

Use this first without Bayesian loss. Add Bayesian only after the ConvNeXt + GT
split + APG run is healthy by epoch 300-600.

For a 15 GB GPU, start with `BATCH=1 ACCUM=4`. If memory allows, try
`BATCH=2 ACCUM=4` or `BATCH=4 ACCUM=2`. The first log lines should print the
resolved batch config; for SHA, `BATCH=1` should be about 300 train batches,
not 37. Do not include `320` unless you are ready for 512-padded crops.

### Bayesian ablation

This keeps the old high-weight Bayesian idea as an option. The command below is
the conservative combined-loss version; to run the aggressive original variant,
change `--bayesian_loss_coef 0.05` to `--bayesian_loss_coef 1.0` and set a
different `--output_dir`.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=./data/ShanghaiTech/part_A
export BATCH=1
export ACCUM=4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone convnextv2_base \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/convnextv2_base_gt_split_apg_bayes_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 200 \
  --min_lr 1e-7 \
  --lr 0.00005 \
  --lr_backbone 0.000005 \
  --lr_backbone_adapter 0.0001 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --ema_decay 0.9999 \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.5 \
  --split_count_threshold 2 \
  --split_pos_weight 2.0 \
  --apg_loss_coef 1.0 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --apg_start_epoch 100 \
  --bayesian_loss_coef 0.05 \
  --bayesian_sigma 8.0 \
  --bayesian_bg_coef 0.02 \
  --bayesian_loss_gate detach \
  --bayesian_start_epoch 150 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --eval_soft_split_gate none \
  --seed 42
```

### Eval TTA ablation

Run only after a checkpoint is trained. Because horizontal flip TTA worsened the
current best VGG checkpoint, treat this as a measured ablation.

```bash
python eval.py \
  --resume outputs/SHA/convnextv2_base_gt_split_apg_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path "$DATA" \
  --device cuda \
  --num_workers 2 \
  --tta_flip \
  --tta_scales 0.9,1.0,1.1 \
  --override_score_threshold 0.5 \
  --override_split_threshold 0.5 \
  --results_file eval_results/SHA/convnextv2_base_gt_split_apg_tta.json
```

### Soft Split Eval Sweep

Run after a checkpoint is trained to test the Phase 7 idea without changing the
training recipe:

```bash
python scripts/sweep_eval_thresholds.py \
  --resume outputs/SHA/convnextv2_base_gt_split_apg_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path "$DATA" \
  --device cuda \
  --num_workers 2 \
  --output_dir eval_results/SHA/convnextv2_base_gt_split_apg_soft_split_sweep \
  --score_thresholds 0.42 0.45 0.48 0.50 0.52 0.54 0.56 0.58 0.60 \
  --split_thresholds 0.45 0.50 \
  --eval_nms_radii 0 \
  --eval_branch_gates none \
  --eval_soft_split_gates none query pred
```
