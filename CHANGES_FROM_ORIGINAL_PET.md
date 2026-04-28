# PET Fork Notes

This is the single fork-specific note for this repository. `README.md` is kept close to the original PET repo again.

Reference paper:

- PET: https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html

## What was removed

The shifted-window encoder experiment was removed from the code.

Reason:

- user ablations isolated it as a hard regression
- `ablate_base` stayed competitive
- `ablate_query` stayed competitive
- `ablate_shift` failed badly
- `ablate_both` failed badly

Conclusion:

- cross-window shifting inside PET's own encoder is not a good change for this fork
- the problem is architectural, not just implementation

## What remains from the upgrade work

### 1. Query upgrade

The query upgrade is still available through:

- `--enhanced_point_query`

What it changes:

- adds local context around each point query with a depthwise + pointwise context block
- adds an explicit sine-coordinate prior derived from the point-query location
- fuses query content with sampled backbone feature + local context + coordinate prior
- fuses query position embedding with the coordinate prior
- adds branch-specific learned biases for sparse and dense branches
- normalizes the fused query tensors before decoding

Why this is still worth testing:

- your ablations show the query-only path is not the source of the catastrophic regression
- it directly improves the representation seen by decoder self-attention and cross-attention

### 2. Modern backbone support

The fork keeps a generic timm backbone adapter with PET feature fusion. Supported practical options now include:

- `convnextv2_*`
- `swin*`
- `swinv2_*`
- `maxvit_*`

New auto selector:

- `auto_maxvit`

## Backbone research

The question is not "best classifier backbone on paper". The question is "best PET backbone that works with dynamic image sizes, hierarchical features, and this repo's FPN-style adapter."

### Local compatibility checks

Verified locally with `timm 1.0.26`:

- `maxvit_tiny_pm_256`: works with `features_only=True` and dynamic non-square inputs
- `swinv2_base_window8_256`: works with `features_only=True` and dynamic non-square inputs
- `hiera_small_abswin_256`: works at fixed `256x256`, but failed on dynamic non-square input in this PET-style feature extraction path

That matters. PET does not live on fixed-size square classification crops at inference time.

### Best next backbone to test

Best practical upgrade candidate for this repo:

- `MaxViT`

Recommended order:

1. `maxvit_tiny_pm_256`
2. `maxvit_small_tf_224`
3. `swinv2_base_window8_256`

Why MaxViT is the strongest next bet:

- official design mixes blocked local attention and dilated global attention inside the backbone itself
- hierarchical multi-scale features match PET better than plain ViT-style backbones
- it gives global context without modifying PET's own encoder
- it worked in local dynamic-size feature extraction checks

Why not Hiera right now:

- strong paper, but the current timm features-only path was not robust for PET-style dynamic non-square inference here

Why not keep forcing ConvNeXtV2 only:

- `convnextv2_base` is still the best known baseline in this repo
- but it is a baseline, not proof that no better backbone exists
- MaxViT is the cleanest next architecture to test without repeating the shifted-window mistake

Primary sources:

- ConvNeXt V2: https://arxiv.org/abs/2301.00808
- Swin Transformer V2: https://arxiv.org/abs/2111.09883
- MaxViT: https://arxiv.org/abs/2204.01697
- Hiera: https://arxiv.org/abs/2306.00989
- MaxViT official repo: https://github.com/google-research/maxvit
- Hiera official repo: https://github.com/facebookresearch/hiera

## Scheduler research

Current baseline scheduler:

- `warmup_cosine`

Why it stays:

- it is still the safest baseline for pretrained hierarchical backbones with AdamW
- ConvNeXt V2 and Swin-family training recipes are both built around warmup + cosine-style decay

New scheduler added for PET experiments:

- `warmup_poly`

Why add it:

- PET is a dense prediction problem, not plain classification
- polynomial decay is a common fit for dense prediction workloads because it decays more gradually late in training than a hard step drop and can stay more useful than cosine when long runs push the LR too close to zero

Recommended scheduler order:

1. baseline: `warmup_cosine`
2. next experiment: `warmup_poly`
3. avoid using `step` as the main path unless you have a strong reason

## Suggested next runs

Known baseline:

```bash
bash train.sh --backbone convnextv2_base --output_dir convnext_base_ref
```

Query-only run:

```bash
bash train.sh --backbone convnextv2_base --enhanced_point_query --output_dir convnext_query_only
```

Best next backbone experiment:

```bash
bash train.sh --backbone maxvit_tiny_pm_256 --lr_scheduler warmup_poly --output_dir maxvit_tiny_poly
```

Larger MaxViT experiment:

```bash
bash train.sh --backbone maxvit_small_tf_224 --lr_scheduler warmup_poly --output_dir maxvit_small_poly
```

Secondary transformer-backbone experiment:

```bash
bash train.sh --backbone swinv2_base_window8_256 --lr_scheduler warmup_cosine --output_dir swinv2_base_cosine
```

## Bottom line

- removed: shifted-window PET encoder
- kept: query upgrade
- restored: README surface
- added: MaxViT backbone path
- added: `warmup_poly` scheduler option

Current working position:

- best known baseline: `convnextv2_base`
- best next upgrade candidate: `maxvit_tiny_pm_256` first, then `maxvit_small_tf_224`
- best scheduler experiment to pair with new backbones: `warmup_poly`
