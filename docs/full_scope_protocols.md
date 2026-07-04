# Full-Scope PET/APG Protocols

This document defines the maintained experiment paths. It separates published
behavior from new PET integrations so results are not mislabeled.

## Architecture Contracts

### PET Paper Baseline

`vgg_pet_paper` preserves the PET point-query quadtree, sparse/dense branches,
Hungarian matching, and paper loss. Use it as the required baseline for every
dataset.

### Legacy APG+LC

`vgg_apglc` is the repository's historical APG+LC recipe. Its
`apg_loss_coef` objective supervises the nearest existing PET grid query. This
is useful as an ablation, but it is not APGCC's independent auxiliary-point
sampling.

### Full Auxiliary-Point APG Integration

`vgg_apgcc_paper_ifi` integrates the APGCC mechanisms into PET:

- every ground-truth point independently samples two positive points within
  two pixels;
- every ground-truth point independently samples two negative points whose
  coordinate offsets have magnitude from two to eight pixels;
- arbitrary positions use four-neighbor implicit feature interpolation;
- positive queries learn confidence and displacement to the ground truth;
- negative queries learn background confidence and zero displacement;
- PET Hungarian matching remains active for the normal point queries.

This is an experimental PET/APG integration, not the published APGCC network.
It must beat `vgg_pet_paper` and `vgg_apglc` under the same protocol before it
can be claimed as an improvement.

## Localization Metric Correction

PET predicts coordinates relative to the padded tensor. Evaluation now converts
those coordinates with the padded tensor dimensions, then validates them
against each image's unpadded size. Older localization F1/precision/recall
numbers produced by this repository are invalid when an image dimension was
not a multiple of the padding divisor. Counting MAE/MSE is unaffected.

## Standard Training

Full APG integration on ShanghaiTech Part A:

```bash
python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --model_recipe vgg_apgcc_paper_ifi \
  --allow_experimental_model_recipe \
  --output_dir outputs/SHA/vgg_apgcc_paper_ifi_seed42 \
  --batch_size 8 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

Run the same command with `--model_recipe vgg_pet_paper` and
`--model_recipe vgg_apglc` as mandatory controls. Do not compare runs with
different image-size caps, validation splits, thresholds, or pretrained
backbone settings.

## UCF-CC-50

UCF-CC-50 has only 50 images and must be evaluated with five-fold
cross-validation. The runner persists the exact fold manifest, selects a
checkpoint only from a holdout inside the 40-image training partition, and
evaluates each untouched 10-image fold once.

```bash
python scripts/run_ucfcc50_folds.py \
  --data_path data/UCF_CC_50 \
  --model_recipe vgg_apgcc_paper_ifi \
  --output_root outputs/UCFCC50 \
  --results_dir eval_results/UCFCC50/vgg_apgcc_paper_ifi_seed42 \
  --fold_seed 42 \
  --device cuda \
  --num_workers 2 \
  --batch_size 8 \
  --epochs 1500
```

The generated deterministic folds are a reproducible protocol, not an
official fixed split. Publish the generated `fold_manifest.json`.

## Partial Annotation Learning

Stage one uses only the deterministic observed rectangle:

```bash
python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --model_recipe vgg_apgcc_paper_ifi \
  --allow_experimental_model_recipe \
  --partial_annotation_ratio 0.1 \
  --partial_annotation_seed 42 \
  --validation_protocol train_holdout \
  --output_dir outputs/SHA/partial_r010_stage1_seed42 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

Create auditable completed labels:

```bash
python scripts/complete_partial_annotations.py \
  --resume outputs/SHA/partial_r010_stage1_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --output_dir data/ShanghaiTech/part_A/completed_r010_seed42 \
  --partial_annotation_ratio 0.1 \
  --partial_annotation_seed 42 \
  --inference_band_pixels 256 \
  --score_threshold 0.5 \
  --device cuda
```

Train stage two from a pretrained backbone, not from the stage-one weights:

```bash
python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --annotation_override_dir data/ShanghaiTech/part_A/completed_r010_seed42 \
  --model_recipe vgg_apgcc_paper_ifi \
  --allow_experimental_model_recipe \
  --output_dir outputs/SHA/partial_r010_stage2_scratch_seed42 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

The completion manifest records observed bounds, pseudo labels, confidence,
and filtering. Results must be reported separately for each annotation ratio.

## Point Annotation Refinement

The refinement path creates exactly one PET decoder query for every source
annotation. It does not threshold or NMS the point set.

```bash
python scripts/refine_point_annotations.py \
  --resume outputs/SHA/vgg_apgcc_paper_ifi_seed42/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --output_dir data/ShanghaiTech/part_A/refined_seed42 \
  --min_person_score 0.5 \
  --device cuda
```

Retrain from scratch with `--annotation_override_dir` pointing to the refined
directory. Refining and evaluating the same benchmark test annotations is
prohibited.

## NWPU Hidden Test Export

```bash
python scripts/export_nwpu_test.py \
  --resume outputs/NWPU/run/best_checkpoint.pth \
  --data_path data/NWPU-Crowd \
  --output eval_results/NWPU/submission.txt \
  --device cuda
```

The exporter requires the official 1,500-entry `test.txt`, preserves its order,
and refuses malformed or duplicate IDs.

## Required Ablations

For every dataset, retain identical data and evaluation settings and compare:

1. `vgg_pet_paper`
2. `vgg_apglc`
3. `vgg_pet_branch_ifi`
4. `vgg_apgcc_paper_ifi`

Then isolate APG/IFI parameters:

- nearest query features versus implicit features;
- independent branch IFI versus shared multi-scale IFI;
- APG auxiliary loss off versus on;
- encoder and decoder window changes, one axis at a time;
- count head off versus on, with stage-two results reported separately.

Report MAE, RMSE, and localization F1/precision/recall for both large and small
thresholds. Store the exact checkpoint, threshold sweep, fold/split manifest,
seed, and per-image predictions for every reported result.
