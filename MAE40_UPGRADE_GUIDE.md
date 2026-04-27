# MAE40 Upgrade Guide

Important status update:

- the architectural upgrades in this note are still experimental
- after regression evidence from training runs, the repo default was restored to the known-good `ConvNeXtV2` auto-tuned path
- `train.sh` now points back to the conservative reference route
- Swin, shifted-window encoder mixing, and enhanced point-query fusion are opt-in experiments only

This note documents the upgrade path implemented in this repo for a training machine with:

- `RTX 5070 Ti`
- `16 GB VRAM`
- `32 GB RAM`

It does not promise a `40.x MAE` result by itself. That still depends on dataset integrity, run stability, and enough training time. What this upgrade does is remove the most obvious architecture and recipe limitations that were holding the repo below that target.

## Research conclusions

### 1. Threshold is not the main bottleneck

The counting threshold only calibrates how many predicted points are kept at inference. It does not fix weak features, limited cross-window context, or thin point-query tokens.

That is why the repo treats threshold as calibration, not architecture:

- `engine.py` can sweep validation thresholds
- threshold sweep is available when enabled explicitly
- the best threshold is stored in checkpoints

If MAE is still high after sweep, the bottleneck is upstream of thresholding.

### 2. The scheduler should stay warmup + cosine

For this PET fork, `warmup_cosine` is the right default:

- the original PET `StepLR(..., args.epochs)` path was effectively close to a constant LR schedule
- stronger modern backbones benefit from a short warmup and smooth decay
- the repo now exposes `--lr_scheduler`, `--warmup_epochs`, and `--min_lr`, but `warmup_cosine` remains the recommended setting

### 3. Best practical backbone upgrade for this repo

There is an important difference between "best on paper" and "best usable here".

- `Hiera` is a strong research candidate, but the current timm feature-extractor path is not robust for PET-style full-image dynamic inference
- `ConvNeXt V2` remains the safest strong backbone family
- `SwinV2` is the best new practical upgrade implemented in this repo because it supports dynamic full-image feature extraction with `strict_img_size=False` and gives PET a stronger hierarchical transformer backbone

For this `5070 Ti 16 GB` machine, the best stable default path in this repo is still:

- `auto` -> resolves to `convnextv2_base`

Best current experiment:

- `auto_swin` -> resolves to `swinv2_small_window8_256`

### 4. Encoder attention needed cross-window mixing

The old PET encoder used local windows only, but those windows did not communicate across boundaries. That is a real limitation in crowd scenes where density patterns and occlusion cues span multiple windows.

Implemented experimental change:

- alternating shifted windows in the PET context encoder via `--use_shifted_windows`

This is the same basic idea that makes Swin-style local attention much less myopic than plain non-overlapping windows.

Important implementation note:

- a shifted layer cannot stop at a cyclic roll
- it also needs the Swin-style per-window attention mask so wrapped tokens do not attend across artificial image boundaries
- this repo now applies that mask in the encoder path
- `--use_shifted_windows` still stays off by default until it proves better than the ConvNeXt baseline in full training

### 5. Decoder attention needed better query priors more than more depth

The repo already had decoder self-attention and cross-attention. The larger weakness was the input query token itself:

- sampled feature at a grid cell
- sampled dense positional embedding
- no explicit coordinate prior
- no local-context fusion around the query

Implemented experimental change:

- `--enhanced_point_query`

This now fuses:

- sampled branch feature
- local context around that feature
- explicit sine-coordinate prior derived from the point-query location
- branch-specific learned bias terms

This directly improves the content and positional signal that decoder attention sees.

## Implemented code changes

### Backbone and search space

- added dynamic timm backbone support for:
  - `ConvNeXtV2`
  - `Swin`
  - `SwinV2`
- added `auto_swin` backbone selector for 16 GB class GPUs
- expanded hyperparameter search neighbors so the search can move between strong ConvNeXtV2 and SwinV2 candidates

### Optimization

- kept `warmup_cosine` as default
- added configurable `--lr_scheduler`, `--warmup_epochs`, `--min_lr`

### PET architecture

- added shifted-window context encoding
- added enhanced point-query construction with local context and coordinate priors
- added checkpoint compatibility for old runs that do not contain the new query-fusion weights

### Training entrypoint

`train.sh` is now the restored reference recipe:

```bash
bash train.sh
```

That path uses:

- `--backbone auto`
- `--search_trials 6`
- `--search_epochs 8`
- `--search_eval_freq 1`
- `--target_mae 50`
- `--eval_freq 1`

The explicit copy of that same path also stays available in:

- [train_convnext_reference.sh](/f:/PET/train_convnext_reference.sh)

## Recommended usage on this machine

### Default stable run

```bash
bash train.sh
```

### Experimental Swin run

```bash
bash train.sh --backbone auto_swin --output_dir pet_swin_experiment
```

### Experimental query/encoder upgrade ablation

```bash
bash train.sh --use_shifted_windows --enhanced_point_query --output_dir pet_query_upgrade_experiment
```

### Query-only ablation

```bash
bash train.sh --enhanced_point_query --output_dir pet_query_only_experiment
```

### Shifted-window ablation

```bash
bash train.sh --use_shifted_windows --output_dir pet_shifted_window_experiment
```

### Direct ConvNeXt base run

```bash
bash train.sh --backbone convnextv2_base --output_dir pet_convnext_base_strong
```

## Research references

These informed the upgrade decisions:

- PET: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html>
- ConvNeXt V2: <https://arxiv.org/abs/2301.00808>
- Swin Transformer: <https://arxiv.org/abs/2103.14030>
- Swin Transformer V2: <https://arxiv.org/abs/2111.09883>
- DAB-DETR: <https://arxiv.org/abs/2201.12329>

## Bottom line

For this repo and this `5070 Ti 16 GB` machine:

- scheduler was not the primary problem, but it needed to stay on warmup + cosine
- threshold was not the primary bottleneck
- the stable path is still ConvNeXtV2
- the wider Swin and query/attention upgrades should be treated as experiments until they beat the 54 MAE baseline consistently

Those are the areas upgraded here, but they are no longer the default route.
