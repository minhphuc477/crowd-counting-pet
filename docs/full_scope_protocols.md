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
- positive and negative groups are averaged separately and then summed;
- auxiliary offsets use MSE, matching the released APGCC criterion;
- PET Hungarian matching remains active for the normal point queries.

This is an experimental PET/APG integration, not the published APGCC network.
It must beat `vgg_pet_paper` and `vgg_apglc` under the same protocol before it
can be claimed as an improvement.

### Robust Residual IFI Ablation

`vgg_pet_apg_rifi` keeps PET's
native sparse/dense query feature as an identity path and adds shared 4x/8x IFI
through a zero-initialized learned residual. APG still trains from epoch zero
through the full run with the corrected per-point positive/negative loss and
PET's stable `0.02` auxiliary scale. This avoids replacing PET's usable
representation with a random interpolator at initialization or importing an
incompatible loss normalization from another network. The stage-one learning
rate drops by `0.1` at epoch 700; keeping `1e-4` through all 1,500 epochs caused
late counting drift even when localization remained strong.

Cross-dataset runs did not establish a consistent improvement over PET.
It is an experimental ablation, not the final or maintained default.

### Experimental Consolidated Candidate

`vgg_apglc_density_routed_ifi` is a maintained experimental ablation. It
preserves native sparse APG+LC and routes residual multi-scale IFI plus
arbitrary-point APG only through PET's dense branch. Both auxiliary families
end at epoch 350, followed by PET-only consolidation and an epoch-700 learning
rate drop.

The count head is excluded because it does not provide spatial supervision and
threshold inference does not consume it. This recipe is a falsifiable candidate,
not the repository default or a claim of universal improvement. It must recover
the established SHA result, remain strong on SHB, and avoid regressions on
QNRF/NWPU/JHU/UCF-CC-50 before promotion.

## Localization Metric Correction

PET predicts coordinates relative to the padded tensor. Evaluation now converts
those coordinates with the padded tensor dimensions, then validates them
against each image's unpadded size. Older localization F1/precision/recall
numbers produced by this repository are invalid when an image dimension was
not a multiple of the padding divisor. MAE/MSE can also change when invalid
point filtering removed coordinates produced with the old conversion.

NWPU uses the official `[x1, y1, x2, y2]` head boxes. For scale-aware training
and diagnostics, the `official` fallback derives
`sigma_s = ceil(min(width, height) / 2)` and
`sigma_l = ceil(sqrt(width^2 + height^2) / 2)`. This approximation is not
accepted for benchmark reporting: validation requires the released
`val_gt_loc.txt`, which always takes priority. Its stored point order is checked
against the JSON annotations before evaluation.

JHU-Crowd++ provides head boxes, but APGCC does not define a JHU localization
table/protocol comparable to NWPU's released evaluator. JHU `target_sigma`
values in this repository are therefore labeled approximate. Report JHU
counting MAE/RMSE as the benchmark result and identify any localization
protocol explicitly.

## Standard Development Training

Consolidated candidate on ShanghaiTech Part A:

```bash
python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol train_holdout \
  --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/SHA/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

The PET and APG+LC controls are existing evidence and need not be repeated
unless their data or evaluation protocol changes. Do not compare runs with
different image-size caps, validation splits, thresholds, or pretrained
backbone settings.

## Complete Cross-Dataset Commands

Use the same architecture without checkpoint initialization. SHA, SHB, and
QNRF use a deterministic training holdout:

```bash
python main.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol train_holdout --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/SHA/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 --epochs 1500 --num_workers 2 --device cuda --seed 42

python main.py --dataset_file SHB --data_path ./data/ShanghaiTech/part_B \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol train_holdout --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/SHB/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 --epochs 1500 --num_workers 2 --device cuda --seed 42

python main.py --dataset_file QNRF --data_path ./data/UCF-QNRF_ECCV18 \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol train_holdout --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/QNRF/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 --epochs 1500 --num_workers 2 --device cuda --seed 42
```

NWPU uses its dense-crop and tiled-evaluation recipe:

```bash
python main.py --dataset_file NWPU --data_path ./data/NWPU-Crowd \
  --model_recipe vgg_apglc_density_routed_ifi_nwpu \
  --allow_experimental_model_recipe \
  --validation_protocol official_val --nwpu_eval_split val \
  --output_dir outputs/NWPU/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 --epochs 1500 --num_workers 2 --device cuda --seed 42
```

JHU uses the generic architecture and official validation:

```bash
python main.py --dataset_file JHU --data_path ./data/jhu_crowd_v2.0 \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol official_val --jhu_eval_split val \
  --output_dir outputs/JHU/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 --epochs 1500 --num_workers 2 --device cuda --seed 42
```

Resume any interrupted run with the identical command plus
`--resume <output_dir>/checkpoint.pth`.

## UCF-CC-50

UCF-CC-50 has only 50 images and must be evaluated with five-fold
cross-validation. The runner persists the exact fold manifest, selects a
checkpoint only from a holdout inside the 40-image training partition, and
evaluates each untouched 10-image fold once.

```bash
python scripts/run_ucfcc50_folds.py \
  --data_path data/UCF_CC_50 \
  --model_recipe vgg_apglc_density_routed_ifi \
  --output_root outputs/UCFCC50 \
  --results_dir eval_results/UCFCC50/vgg_apglc_density_routed_ifi_seed42 \
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
  --model_recipe vgg_apglc_density_routed_ifi \
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
  --model_recipe vgg_apglc_density_routed_ifi \
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
  --resume outputs/SHA/vgg_apglc_density_routed_ifi_seed42/best_checkpoint.pth \
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
5. `vgg_pet_apg_rifi`
6. `vgg_apglc_density_routed_ifi`

Then isolate APG/IFI parameters:

- nearest query features versus implicit features;
- independent branch IFI versus shared multi-scale IFI;
- APG auxiliary loss off versus on;
- encoder and decoder window changes, one axis at a time;
- count head off versus on, with stage-two results reported separately.

Report MAE, RMSE, and localization F1/precision/recall for both large and small
thresholds. Store the exact checkpoint, threshold sweep, fold/split manifest,
seed, and per-image predictions for every reported result.
