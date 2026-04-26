# Regression From `1baeb315d8bb3e64fb4bc27214e5d99e0340d994`

## Reference points

- Known-good checkpoint family from user report: `1baeb315d8bb3e64fb4bc27214e5d99e0340d994`
- First suspicious follow-up config commit: `d6b9e71` (`config for 5070Ti and lr change`)
- Later repo-wide behavior change: `cd0716f` (`major update`)

## What changed

### 1. The good recipe was a ConvNeXt auto-tuned path, not the current default script

At `1baeb31`, `train.sh` used:

- `--backbone auto`
- `--target_mae 50`
- `--search_trials 6`
- `--search_epochs 8`
- `--search_eval_freq 1`
- `--eval_freq 1`

That path used the generic ConvNeXt V2 auto-selection logic. On a 16 GB class GPU, that already resolved to `convnextv2_base` with the conservative `base` learning-rate and batch-size defaults.

### 2. `d6b9e71` changed the auto path for 5070 Ti class GPUs

`d6b9e71` did not change `train.sh`. The shell recipe before and after that commit stayed the same.

The regression came from `main.py`, where a dedicated `5070 Ti` branch was added on top of the existing `auto` path. That branch forced:

- `backbone = convnextv2_base`
- `hidden_dim >= 384` instead of `256`
- `dim_feedforward >= 768` instead of `512`
- `dec_layers >= 3` instead of `2`
- a different encoder window list with one extra stage
- larger decoder windows for both sparse and dense branches
- tighter early-stop target (`target_mae <= 40`)
- search pressure toward more trials

This is a materially different model and training objective from the `1baeb31` path. It is larger, more coupled, and harder to tune. The user-reported drop after this point is consistent with that change.

### 3. `cd0716f` changed the default entrypoints again

Later, `cd0716f` changed the default scripts and parser defaults away from the ConvNeXt reference path:

- `train.sh` defaulted back to `vgg16_bn`
- `eval.sh` defaulted back to `vgg16_bn`
- the repo stopped reproducing the `1baeb31` training path unless the user manually reintroduced the old flags

So the bad behavior after `1baeb31` is a two-step drift:

1. `d6b9e71` made the `auto` ConvNeXt recipe more aggressive for 16 GB / 5070 Ti class GPUs.
2. `cd0716f` later changed the default scripts so the known-good ConvNeXt recipe was no longer the out-of-box path at all.

## Root-cause conclusion

The strongest regression reason is not a single loss bug. It is recipe drift.

- `1baeb31` was effectively a conservative ConvNeXt V2 search recipe.
- `d6b9e71` replaced that with a larger special-case profile for the same hardware class.
- later scripts/defaults no longer matched the reference recipe anyway.

## Fixes applied in the current tree

- Removed the dedicated `5070 Ti` auto-profile from `main.py`
- Restored `--backbone auto` to the generic pre-`d6b9e71` memory-based ConvNeXt selection path
- Added [train_convnext_reference.sh](/f:/PET/train_convnext_reference.sh) to preserve the `1baeb31`-style training recipe explicitly

## How to reproduce the reference path now

Use:

```bash
bash train_convnext_reference.sh
```

On a 16 GB class GPU, this should resolve to the same generic `convnextv2_base` auto path that existed before `d6b9e71`.
