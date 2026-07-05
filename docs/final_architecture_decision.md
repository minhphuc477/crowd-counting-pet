# Final Architecture And Evaluation Decision

## Decision

The repository does not currently contain a new architecture that has honestly
beaten PET across SHA, SHB, UCF-QNRF, JHU-Crowd++, NWPU-Crowd, and UCF-CC-50.
The reportable reference is therefore:

```text
model_recipe = vgg_pet_paper
backbone = ImageNet-pretrained vgg16_bn
count source = PET thresholded points
count head = disabled
IFI/APG grafts = disabled
```

`vgg_pet_apg_rifi`, `vgg_pet_rmi`, branch IFI, ZIP, and scalar count-head
variants remain ablations. None is the default final architecture.

This is a scientific decision, not a claim that PET cannot be improved. It
means the current evidence does not justify replacing the verified model.

## Why The Previous Success Does Not Establish A New Architecture

The historical SHA result near MAE 48.8 came from a model-only adaptation of an
already selected APG+LC checkpoint. Only the scalar count head was trainable,
but threshold inference did not consume that head. BatchNorm running buffers
were still mutable. The observed change can therefore include BatchNorm
recalibration and repeated benchmark-test checkpoint/threshold selection; it
does not prove that the scalar count head improved the point model.

Current two-stage experiments also regress on several datasets:

- SHA residual/branch IFI runs did not reproduce the historical 48.8 result.
- SHB stage two reduced holdout MAE in one run but worsened the untouched-test
  result and localization in others.
- UCF-QNRF remained substantially above PET's published result.
- NWPU benefited from IFI/APG in some runs, but this did not transfer
  consistently.

Consequently, count-head fine-tuning is an ablation only. It must not choose the
reported model unless its inference path uses the head and an untouched
validation protocol proves the gain.

## What The Papers Support

PET's progressive rectangle encoder/decoder and point-query quadtree are a
strong cross-density baseline. APGCC demonstrates that its complete proposal
architecture, implicit feature interpolation, matcher, and auxiliary point
guidance work together. Its ablation does not show that transplanting IFI or
APG into PET is sufficient.

APGCC also does not improve every PET metric: its published UCF-QNRF MAE is
slightly worse than PET while its RMSE is better. A universal improvement
cannot be assumed from either paper.

Primary sources:

- [PET paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html)
- [Official PET repository](https://github.com/cxliu0/PET)
- [APGCC paper](https://arxiv.org/abs/2405.10589)
- [Official APGCC repository](https://github.com/AaronCIH/APGCC)

## Final Research Path

1. Re-establish `vgg_pet_paper` on every dataset under the corrected protocol.
2. Treat `vgg_apglc` as a historical control, not as APGCC reproduction.
3. Implement a separate, paper-shaped APGCC proposal model if APGCC is pursued:
   four proposals per cell, multi-level IFI, matcher, positive/negative APG,
   and its crop/augmentation schedule. Do not keep adding auxiliary heads to
   PET and call the result APGCC.
4. Compare PET and the separate APGCC model with identical data accounting,
   pretrained backbone policy, seeds, and evaluation code.
5. Accept a replacement only after three seeds improve the validation mean and
   confidence interval on each target dataset. Dataset-specific winners are
   acceptable; claiming one universal winner without that evidence is not.

## Non-Negotiable Evaluation Contract

- SHA, SHB, and UCF-QNRF: select architecture, epoch, and thresholds only on a
  deterministic training holdout.
- Final SHA/SHB/QNRF report: retrain the frozen recipe on all training images
  and use `validation_protocol=final_test_once`.
- NWPU and JHU: select on their official validation splits.
- UCF-CC-50: five-fold cross-validation; each test fold is evaluated once.
- Never choose tiling from ground-truth count.
- Never sweep thresholds on a benchmark test split.
- Report MAE, RMSE, localization F1/precision/recall, all seeds, and the exact
  checkpoint. Do not compare adaptive and fixed localization protocols as if
  they were the same metric.

The code now rejects GT-controlled tiling and benchmark-test selection by
default. `scripts/audit_scientific_protocol.py` audits actual checkpoints and
result JSON independently of the unit tests.

## Commands

Development run:

```bash
python main.py \
  --dataset_file QNRF \
  --data_path ./data/UCF-QNRF_ECCV18 \
  --model_recipe vgg_pet_paper \
  --validation_protocol train_holdout \
  --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/QNRF/pet_paper_dev_seed42 \
  --batch_size 8 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

Sweep the same holdout. The sweep reads the split fraction and seed from the
checkpoint:

```bash
python scripts/sweep_eval_thresholds.py \
  --resume outputs/QNRF/pet_paper_dev_seed42/best_checkpoint.pth \
  --eval_image_set train_holdout \
  --eval_nms_radii 0 \
  --eval_branch_gates none \
  --eval_soft_split_gates none \
  --eval_count_sources pet \
  --score_thresholds 0.48 0.50 0.52 \
  --split_thresholds 0.45 0.50
```

After freezing the epoch and thresholds, run the full-data final refit. Replace
`1500` with the predeclared selected epoch count; do not inspect intermediate
test results:

```bash
python main.py \
  --dataset_file QNRF \
  --data_path ./data/UCF-QNRF_ECCV18 \
  --model_recipe vgg_pet_paper \
  --validation_protocol final_test_once \
  --output_dir outputs/QNRF/pet_paper_final_seed42 \
  --batch_size 8 \
  --epochs 1500 \
  --device cuda \
  --seed 42
```

Audit reportable artifacts:

```bash
python scripts/audit_scientific_protocol.py \
  outputs/QNRF/pet_paper_final_seed42 \
  --output eval_results/QNRF/pet_paper_final_seed42/protocol_audit.json
```
