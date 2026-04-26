# Changes From Original PET

Reference upstream repo: [`cxliu0/PET`](https://github.com/cxliu0/PET)  
Reference upstream commit audited here: `5b4dd7da8b11568a3305a88bb7c99a7fc831a998`  
Local repo HEAD after this repair: `2cfe33d9fe84f34116dd70222541d58c39deabc7`

## Scope

This document records the current functional differences between this repo and the original PET repository, with emphasis on the fixes applied in this repair pass.

The original PET paper is:

- [Point-Query Quadtree for Crowd Counting, Localization, and More (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html)

The paper defines PET around point-query loss and quadtree-splitter loss. The extra fork-only count regularizer was not part of the original formulation.

## Fork Additions Kept

- `models/backbones/backbone_convnextv2.py`
  Adds an optional ConvNeXt V2 backbone path for newer GPUs and larger pretrained models.
- `main.py`, `engine.py`
  Keeps mixed precision, gradient accumulation, optional threshold sweep, optional hyperparameter search, and more defensive training error handling.
- `test_single_image.py`
  Keeps the single-image inference utility.
- `models/matcher.py`, `models/pet.py`, `util/misc.py`
  Keeps numerical sanitization and empty-target/distributed robustness improvements that are useful beyond the original codebase.

## Fixes Applied In This Repair

### 1. Restored upstream-safe defaults

- `main.py`
  Default backbone changed back to `vgg16_bn`.
- `train.sh`
  Default training recipe changed back to the upstream-style VGG path instead of forcing the experimental auto/ConvNeXt configuration.
- `eval.py`, `test_single_image.py`, `eval.sh`
  Default evaluation backbone changed back to `vgg16_bn`.

Why:

- The fork had made the experimental ConvNeXt/auto path the default everywhere.
- That diverged from the published PET recipe and made the repo much easier to misconfigure.

### 2. Restored backbone dispatch instead of forcing ConvNeXt

- `models/backbones/__init__.py`

What changed:

- The repo now dispatches to VGG when `--backbone` starts with `vgg`.
- The ConvNeXt path remains available when `--backbone` is `auto` or starts with `convnextv2_`.

Why:

- The fork had hard-wired `build_backbone` to ConvNeXt V2, which silently broke the original PET backbone path.

### 3. Fixed dataset path override bug

- `datasets/__init__.py`

What changed:

- `--data_path` is now honored.
- Fallback resolution now supports both `./data/ShanghaiTech/PartA` and `./data/ShanghaiTech/part_A`.

Why:

- The fork always overwrote `args.data_path` internally, so user-provided dataset paths were ignored.
- On case-sensitive systems this could also break ShanghaiTech loading depending on folder naming.

### 4. Fixed checkpoint compatibility and resume/eval architecture mismatches

- `util/misc.py`
- `main.py`
- `eval.py`
- `test_single_image.py`

What changed:

- Added `utils.load_checkpoint()` to support both newer PyTorch and the original PET environment.
- Added `utils.restore_args_from_checkpoint()` so resume/eval rebuild the model with the checkpoint's actual architecture settings.

Why:

- The fork used `torch.load(..., weights_only=False)`, which is incompatible with the original PET environment (`PyTorch 1.12.1`).
- The fork also restored only a small subset of checkpoint args, which could rebuild the wrong architecture and make checkpoint loading fail.

### 5. Disabled the harmful count-loss path by default and corrected its semantics

- `main.py`
- `models/pet.py`

What changed:

- `--count_loss_coef` default changed from `0.01` to `0.0`.
- The optional count loss now works on the combined routed count instead of forcing the sparse branch and dense branch to each match the full-image count independently.

Why:

- The previous fork logic pushed both branches toward the entire ground-truth count, which fights the quadtree routing design and can flatten training progress.
- The original PET formulation does not include this extra loss term.

### 6. Restored less aggressive training defaults

- `main.py`
- `train.sh`

What changed:

- `--threshold_sweep` default changed from enabled to disabled.
- `--search_trials` default changed from `4` to `0`.
- `--split_warmup_epochs` default changed from `1` back to `5`.
- `--target_mae` default now disables early stopping unless explicitly set.
- The 5070 Ti auto profile no longer force-enables extra search/threshold/early-stop/count-loss behavior.

Why:

- The fork had turned several experimental behaviors on by default.
- That made the training path diverge sharply from the original PET recipe and obscured whether the model itself or the fork settings were causing regressions.

### 7. Fixed SHA dataset coordinate edge cases

- `datasets/SHA.py`

What changed:

- Fixed `index <= len(self)` to `index < len(self)`.
- Fixed random horizontal flip to mirror around `width - 1`.
- Fixed crop filtering to treat the right and bottom edges as exclusive.
- Made crop resize interpolation explicit with `mode='bilinear', align_corners=False`.

Why:

- These are small coordinate bugs, but they can corrupt training targets near crop/flip boundaries.

### 8. Simplified inference branch merge

- `models/pet.py`

What changed:

- The sparse/dense merge now concatenates the active branch outputs directly instead of applying a meaningless `score > 0` filter.

Why:

- Softmax probabilities are always positive, so that filter never removed anything and only added confusion.

### 9. Made shell entrypoints safer to reuse

- `train.sh`
- `eval.sh`

What changed:

- Both scripts now use `${CUDA_VISIBLE_DEVICES:-0}`.
- Both scripts accept additional CLI flags through `"$@"`.
- Training/eval defaults now match the restored upstream-safe path.

Why:

- The old scripts locked the repo into one experimental configuration and were awkward to override.

## Current Repo-Level Divergence From Original PET

These differences still exist on purpose after the repair:

- ConvNeXt V2 backbone support is still present and usable.
- Mixed precision, accumulation, and Optuna-based search are still present, but now opt-in instead of default.
- Threshold sweep support is still present, but now opt-in instead of default.
- Single-image inference is still present.
- Numeric sanitization and empty-target handling remain in matcher/loss/util code.

## Files Currently Different From Upstream

At the time of this audit, the tracked files that still differ from `origin/main` are:

- `datasets/SHA.py`
- `engine.py`
- `eval.py`
- `eval.sh`
- `main.py`
- `models/backbones/__init__.py`
- `models/backbones/backbone_convnextv2.py`
- `models/backbones/backbone_vgg.py`
- `models/matcher.py`
- `models/pet.py`
- `models/position_encoding.py`
- `models/transformer/__init__.py`
- `models/transformer/prog_win_transformer.py`
- `preprocess_dataset.py`
- `requirements.txt`
- `test_single_image.py`
- `train.sh`
- `util/__init__.py`
- `util/misc.py`

## Recommended Usage After This Repair

- For the closest behavior to the original PET paper/repo, use:
  - `bash train.sh`
  - `bash eval.sh --resume path_to_model`
- `train.sh` and `eval.sh` are shell scripts. Run them with `bash`/`sh`, not with `python`.
- Use the ConvNeXt path only intentionally, for example:
  - `bash train.sh --backbone convnextv2_nano`
  - `bash train.sh --backbone auto`
- If you want threshold sweep or hyperparameter search, enable them explicitly instead of relying on defaults.
