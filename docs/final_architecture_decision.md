# Architecture Promotion Decision

## Current Status: Not Promoted

`vgg_apglc_density_routed_ifi` remains an experimental ablation. The latest
cross-dataset run reached MAE 96.87 and RMSE 183.39 with a well-calibrated mean
count (`pred/gt = 0.984`), but it did not improve localization and does not
support replacing the previously successful paths.

Do not call any current hybrid the final architecture. Circle back to a final
decision only after one unchanged architecture satisfies all promotion gates
below.

## Scope

The PET and APG+LC controls have already been run. Do not repeat them. The
density-routed model remains available as a falsification target:

```text
model_recipe = vgg_apglc_density_routed_ifi
backbone = ImageNet-pretrained vgg16_bn
sparse branch = PET nearest-query features + historical APG+LC
dense branch = residual shared 4x/8x IFI + routed auxiliary-point APG
count head = disabled
count source = thresholded PET points
```

It is not the repository default and is not a claim of a new state of the art.

## Why This Candidate

The repository evidence rejects three previous directions:

1. Shared IFI on both branches frequently improved localization while
   degrading counting, especially on SHA and QNRF.
2. Running IFI/APG auxiliary losses for all 1,500 epochs produced late count
   drift even when training loss remained low.
3. The scalar count head did not spatially supervise PET predictions, and PET
   threshold inference did not consume it. Its historical SHA result therefore
   does not establish a complete architecture gain.

PET deliberately separates sparse and dense query regimes. Its supplementary
analysis also reports matching ambiguity when query density is unnecessarily
high. The candidate therefore leaves the verified sparse representation alone
and confines the new representation and arbitrary-point guidance to queries
routed through PET's dense branch.

## Active Components

### Sparse path

- PET 8-pixel query stride and sparse rectangle-window decoder.
- Native nearest CNN feature lookup.
- Historical nearest-query APG+LC.
- APG active during epochs 0-350.

### Dense path

- PET 4-pixel dense queries and dense rectangle-window decoder.
- One shared residual IFI module using 4x and 8x feature maps.
- Residual scale initialized to `1e-3`: the initial perturbation is small, but
  the IFI MLP receives gradients on the first update.
- Routed positive/negative auxiliary points use PET's dense prediction heads.
- Exact per-GT negative sample count; dense crops no longer silently discard
  most negative supervision.
- IFI/APG active during epochs 0-350.

### Consolidation phase

- Epochs 351-699 optimize the normal PET matching and split objectives only.
- The learning rate drops by 0.1 at epoch 700.
- No count head, density map head, score calibration, NMS, foreground gate, or
  ground-truth-controlled inference is active.

## Correctness Fixes

- A zero APG branch coefficient now prevents that branch's APG computation.
- `ifi_branch_scope` requires `ifi_head_source=routed`; incompatible
  configurations fail instead of silently producing zero supervision.
- IFI's optimizer weight is zero after `ifi_end_epoch`.
- The final recipe routes normal IFI queries and auxiliary IFI points only to
  the dense branch.
- `ifi_negative_policy=paper` preserves the configured number of APG samples
  in dense regions. The older `filter` mode remains available as an ablation.

## Evidence And Limits

PET supports interpolated CNN point-query features, sparse-to-dense quadtree
routing, and branch-specific decoder windows. APGCC supports arbitrary-point
guidance and implicit feature interpolation as a coupled mechanism. Neither
paper establishes that a PET/APGCC hybrid will improve every dataset.

Recent alternatives such as segmentation prompting and consistency-based
pseudo-points target annotation noise or semi-supervised learning. They remain
separate ablations because adding them now would confound the dense-routing
hypothesis.

Primary sources:

- [PET paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html)
- [Official PET repository](https://github.com/cxliu0/PET)
- [APGCC paper](https://arxiv.org/abs/2405.10589)
- [Official APGCC repository](https://github.com/AaronCIH/APGCC)
- [Point-to-Region Loss](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Point-to-Region_Loss_for_Semi-Supervised_Point-Based_Crowd_Counting_CVPR_2025_paper.html)

## Promotion Gates

One unchanged architecture is promoted only if:

1. SHA reproduces the established approximately 48 MAE result under the frozen
   final protocol.
2. SHB remains competitive with the strongest properly matched SHB result.
3. QNRF, NWPU, JHU, and UCF-CC-50 each improve or remain within the
   predeclared non-inferiority margin of their matched reference.
4. Neither large- nor small-threshold localization F1 suffers a material
   regression.
5. The conclusion holds over three seeds, not one selected checkpoint.

Results from benchmark test, training holdout, and official validation are not
interchangeable. The SHA 48.8 adaptation and SHB 5.775 holdout result are
important historical evidence, but they cannot be combined into a universal
architecture claim until evaluated through the same declared model-selection
contract.

Report MAE, RMSE, localization F1/precision/recall, per-image predictions,
seed, exact checkpoint, split manifest, and inference settings. Select epochs
and thresholds on a training holdout for SHA/SHB/QNRF, official validation for
NWPU/JHU, and nested five-fold validation for UCF-CC-50.

## First Runs

ShanghaiTech Part A:

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
  --num_workers 2 \
  --device cuda \
  --seed 42
```

ShanghaiTech Part B:

```bash
python main.py \
  --dataset_file SHB \
  --data_path ./data/ShanghaiTech/part_B \
  --model_recipe vgg_apglc_density_routed_ifi \
  --allow_experimental_model_recipe \
  --validation_protocol train_holdout \
  --train_holdout_fraction 0.1 \
  --train_holdout_seed 42 \
  --output_dir outputs/SHB/vgg_apglc_density_routed_ifi_seed42 \
  --batch_size 8 \
  --epochs 1500 \
  --num_workers 2 \
  --device cuda \
  --seed 42
```

Resume an interrupted run with the same command plus:

```bash
--resume outputs/<DATASET>/vgg_apglc_density_routed_ifi_seed42/checkpoint.pth
```

Do not initialize either run from an existing PET, APG+LC, IFI, or count-head
checkpoint.
