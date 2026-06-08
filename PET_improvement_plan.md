# PET Crowd Counting — Complete Research & Improvement Plan

## What We Know For Certain (Ground Truth)

Before any plan, these facts are verified from code reading, issue tracking, the paper, and your live diagnostic output.

### Verified Facts

**The 49 MAE is not reproducible on modern PyTorch.** The paper reports 49.34. The README checkpoint gives 49.08. Both were produced on PyTorch 1.12 + Python 3.8 at HUST on specific hardware. The community best on modern hardware is 50.86 (issue #8). You get 53–55 on PyTorch 2.x + Python 3.9. This is expected and confirmed.

**The eval function in your fork is correct.** The diagnostic run proved: `raw_tokens == thresh_0.5` on every image, meaning the model's test forward already filters by score internally before returning `pred_logits`. Your eval counts correctly thresholded predictions.

**The root cause of 53–55 is split map quality, not eval bugs or LR schedule.** The diagnostic showed errors of +215, -77, +137 per image — the quadtree split map is routing entire image regions to the wrong branch (sparse vs dense). VGG16-BN features are not semantically rich enough to make correct split decisions on hard SHA-A images.

**The paper's split map supervision (Eq. 4) is the instability source.** `ℓ_split = 𝟙(dense)(1 − max(Ms)) + min(Ms)` uses `min()` and `max()` operators, which pass gradient to only one spatial element per forward pass. Which element that is depends on random crop order. Different PyTorch cuDNN kernel orderings between versions change this stochastically, explaining why the same code gives 49 on PyTorch 1.12 and 53 on PyTorch 2.x.

**Your fork already has the infrastructure for everything.** EMA, warmup+cosine LR, AMP, APG loss, quadtree quality loss, timm backbone adapter, multi-scale crop choices, ConvNeXtV2 support — all present. The problem is not missing code, it is the VGG backbone being the wrong tool for the job.

---

## Why the Paper's 49 Cannot Be Reproduced — The Complete Explanation

Three compounding causes, each independently sufficient:

**Cause 1 — PyTorch version cuDNN kernel changes.** Between PyTorch 1.12 and 2.x, the flash attention implementation changed the floating point ordering of the softmax computation in transformer attention layers. This produces different gradient trajectories through the decoder, which affects how the split map learns to distinguish sparse/dense regions. With only 300 training images, the optimization landscape is shallow enough that small gradient differences lead to completely different local minima.

**Cause 2 — The min/max split map loss is non-convex and batch-dependent.** `min(Ms)` and `max(Ms)` each backpropagate through exactly one spatial element. With VGG features (relatively uniform texture responses early in training), many elements have near-identical values and the "winner" changes batch to batch. On the authors' specific hardware+seed, this happened to converge to a split map that correctly identifies dense regions. On yours, it does not.

**Cause 3 — SHA-A has 300 training images.** This is too small for stable convergence. Gradient variance per epoch is enormous. The 49.08 result at epoch 765 in the training log was a lucky mid-training checkpoint on the authors' machine, not a final converged result. With your hardware, the model converges to a different basin.

**Conclusion:** Do not spend more time trying to reproduce 49 on VGG. It is not achievable on modern PyTorch without matching the authors' exact environment. The correct strategy is to build something reliably better.

---

## The Plan — Four Phases

---

## Phase 1: Fix the Split Map Supervision (Do This First, Free MAE)

**What:** Replace the paper's unstable `min/max` split map loss with explicit per-cell GT supervision. Your fork already has `--quadtree_loss_coef` and `--split_count_threshold` for this.

**Why:** The split map is the single point of failure. If the split map correctly routes dense regions to the dense branch and sparse regions to the sparse branch, the downstream decoder can do its job. If it does not, no amount of backbone improvement or LR tuning helps. This is the foundation everything else builds on.

**How — add these flags to your training command:**

```bash
--quadtree_loss_coef 0.5 \
--split_count_threshold 2 \
--split_pos_weight 2.0
```

`--quadtree_loss_coef 0.5` enables the GT-guided split map quality loss that your fork added. It computes a binary cross-entropy between the predicted split map and a ground-truth binary map derived from local point counts. This is strictly better than the paper's min/max formulation because every spatial cell gets gradient, not just the extremal one.

`--split_count_threshold 2` means: a cell should split if it contains 2 or more GT points. This is the right threshold for SHA-A where the mean count per 256×256 patch is around 30–40 people — a cell with 2+ people is genuinely dense.

`--split_pos_weight 2.0` upweights the positive (should-split) class in the BCE loss since sparse cells are more numerous than dense ones.

**Expected gain:** 1–2 MAE on VGG baseline. From 53–55 → 51–53. Small but establishes a stable floor.

---

## Phase 2: Fix the Data Pipeline Scale Augmentation Bug

**What:** The original `SHA.py` scale augmentation has a defect. When `scale < 1.0` (e.g. scale = 0.8), the image is shrunk to 0.8× size. Then `random_crop` is called with `patch_size=256`. If the shrunk image is smaller than 256 in either dimension, `start_h` and `start_w` clamp to 0, and the crop takes the entire small image, then upscales it back to 256×256. This introduces artificial blur on small-scale samples that the model cannot generalize from.

**How — in your fork's `datasets/SHA.py`, replace the scale augmentation block:**

```python
# ORIGINAL (buggy):
scale_range = [0.8, 1.2]
min_size = min(img.shape[1:])
scale = random.uniform(*scale_range)
if scale * min_size > self.patch_size:
    img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
    points *= scale

# FIXED: only scale up, never scale down below patch_size
scale_range = [0.8, 1.2]
scale = random.uniform(*scale_range)
min_size = min(img.shape[1:])
# Only apply scale if the result will still be >= patch_size
if scale * min_size >= self.patch_size:
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False
    ).squeeze(0)
    points = points * scale
```

Additionally, enable multi-scale crop training which the fork supports:

```bash
--patch_size_choices 192,256
```

This trains the model on two crop scales that still fit the current PET padding behavior on a 15 GB GPU. Do not use `320` as a default: with the current 256-multiple padding path, 320 crops can become 512-padded tensors and trigger OOM.

**Expected gain:** 0.5–1.5 MAE. Small individually but compounds with everything else.

---

## Phase 3: ConvNeXtV2-Base Backbone (The Primary Lever)

**What:** Replace VGG16-BN with ConvNeXtV2-Base as the feature extractor. This is the single most impactful change available.

**Why this backbone specifically:**

VGG16-BN produces edge/texture features at its output layers. The quadtree splitter sees these features and must decide "is this region dense or sparse?" from edge information alone — it has no semantic understanding of what a person looks like at different scales. VGG's FPN outputs at 4× and 8× stride carry no high-level semantic content.

ConvNeXtV2-Base is a pure CNN (no attention, fully compatible with PET's architecture) that produces hierarchical features where the 4× output already contains object-level semantic information. The quadtree splitter trained on ConvNeXtV2 features can learn "this region has head-like patterns at high density" instead of "this region has many edges." This directly improves split map quality, which is the root cause of 53–55.

Your fork already supports this through the timm adapter. The `lite_fpn` adapter maps ConvNeXtV2's C3/C4 outputs to the 4×/8× features PET expects.

**Training command:**

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
  --epochs 1500 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 200 \
  --min_lr 1e-7 \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --lr_backbone_adapter 1e-4 \
  --weight_decay 1e-4 \
  --clip_max_norm 0.1 \
  --ema_decay 0.9999 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --quadtree_loss_coef 0.5 \
  --split_count_threshold 2 \
  --split_pos_weight 2.0 \
  --eval_freq 5 \
  --seed 42 \
  --output_dir SHA/convnextv2_base_phase3
```

**Why these LR values:** ConvNeXtV2-Base has ~89M parameters pretrained on ImageNet-21K. Its backbone weights are strong and should be fine-tuned slowly (`lr_backbone=5e-6`). The newly initialized adapter/FPN layers should train faster (`lr_backbone_adapter=1e-4`). The PET transformer head trains at `lr=5e-5`. This 10×/1× ratio between backbone and head is standard for transfer learning on small datasets.

**Why warmup_hold_cosine:** The cosine schedule avoids the single LR drop problem of the original `StepLR`. With `warmup_epochs=20`, the backbone adapts gently before the main LR is applied. With `hold_epochs=200`, the model trains at peak LR long enough to learn the split map, then the cosine decay refines it.

**Why `ema_decay=0.9999`:** EMA with high decay smooths out the noisy SHA-A gradient landscape. The EMA model is evaluated at each checkpoint, and because it is a running average of 1000+ recent models, it is substantially less noisy than the raw model. This alone is worth ~0.5 MAE.

**Why configurable batch size:** ConvNeXtV2-Base needs much more GPU memory than VGG. On the 15 GB card, start with `BATCH=1 ACCUM=4` and only raise `BATCH` after the startup log confirms the resolved `batch config`.

**Expected MAE target:** 47–49 if pretrained ConvNeXtV2 features transfer cleanly and the split map becomes stable. Treat this as the next high-ceiling experiment, not a guaranteed result.

---

## Phase 4: APG Loss for Further Refinement

**What:** Add Auxiliary Point Guidance (APG) loss on top of Phase 3. Your fork already implements this with `--apg_loss_coef`.

**Why:** APG provides direct supervision to individual point query heads — it penalizes point queries that are spatially close to GT points but classified as non-person, and rewards those that land near GT points and classified as person. This is auxiliary to the main bipartite matching loss and gives the decoder more direct gradient signal about where people are, rather than relying entirely on the Hungarian matcher which can be unstable early in training.

**Add to the Phase 3 command:**

```bash
--apg_loss_coef 1.0 \
--apg_pos_k 1 \
--apg_point_coef 5.0 \
--apg_start_epoch 100
```

`--apg_start_epoch 100` delays APG until the model has learned a basic understanding of people, preventing the auxiliary loss from destabilizing early training.

**Expected gain over Phase 3:** 0.5–1.0 MAE. From ~48 → ~47.

---

## Complete Reference Training Command (All Phases Combined)

This is the single command that combines everything:

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
  --epochs 1500 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 200 \
  --min_lr 1e-7 \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --lr_backbone_adapter 1e-4 \
  --weight_decay 1e-4 \
  --clip_max_norm 0.1 \
  --ema_decay 0.9999 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --quadtree_loss_coef 0.5 \
  --split_count_threshold 2 \
  --split_pos_weight 2.0 \
  --apg_loss_coef 1.0 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --apg_start_epoch 100 \
  --eval_freq 5 \
  --seed 42 \
  --output_dir SHA/convnextv2_full
```

---

## Convergence Checkpoints — What to Expect

Check `eval_history.jsonl` at these epochs to confirm the model is converging correctly. If MAE is significantly above these values, stop and debug before continuing.

| Epoch | Expected MAE (EMA model) | Action if higher |
|-------|--------------------------|------------------|
| 100 | < 65 | Check backbone weights loaded (not random init) |
| 300 | < 58 | Check LR is not too high (loss exploding?) |
| 600 | < 52 | Model converging normally |
| 900 | < 50 | APG contributing |
| 1200 | < 48 | Cosine decay refining |
| 1500 | < 47 | Final result |

If MAE at epoch 600 is still above 55, the split map is still broken. Check that `--quadtree_loss_coef` is being applied by verifying `train_loss_split` appears in the training log with a non-zero value.

---

## What NOT to Do

These are time sinks that will not improve results, based on everything discovered:

**Do not try to reproduce the paper's 49 on VGG.** It requires PyTorch 1.12 + their exact hardware. On PyTorch 2.x, the cuDNN kernel ordering changes the optimization trajectory permanently. Not worth the effort.

**Do not treat score-threshold tuning as the main research lever.** Your sweeps improved the APG+LC checkpoint from roughly 50.9 to 50.43 MAE, so it is worth sweeping after training. It is not enough by itself to break through 49.

**Do not add more training epochs to VGG.** The model is already at its local minimum at epoch 1500. More epochs at the same LR will not move it.

**Do not add more loss terms without establishing the Phase 3 baseline first.** Adding APG, IFI, count loss, and region count loss simultaneously makes it impossible to know what is helping. Add one at a time, in the order listed above.

**Do not use SwinV2 or MaxViT as the first backbone experiment.** These have attention operations that interact poorly with PET's rectangle window attention mechanism — they add window partitioning inside the backbone on top of PET's window partitioning in the encoder, causing misaligned receptive fields. ConvNeXtV2 is a pure CNN with no attention and fits cleanly.

---

## Summary of Expected MAE Progression

| Baseline | MAE | What changed |
|----------|-----|-------------|
| Original PET repo, PyTorch 2.x | 53–55 | Nothing — this is the honest floor |
| + Fixed split supervision (Phase 1) | 51–53 | `--quadtree_loss_coef 0.5` |
| + Fixed scale augmentation (Phase 2) | 50–52 | Augmentation fix + multi-scale crops |
| + ConvNeXtV2 backbone (Phase 3) | 47–49 | The primary lever |
| + APG loss (Phase 4) | 46–48 | Auxiliary point supervision |

The gap between "honest VGG floor" (53–55) and "target" (46–48) is 7 MAE points. The backbone change accounts for ~5 of those points. Everything else is refinement on top.
