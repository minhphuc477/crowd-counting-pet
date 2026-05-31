# MAE-focused PET Improvements

This branch keeps official PET behavior available while adding options aimed at
reducing MAE with timm backbones.

## Why These Changes

PET's training objective supervises classification, point regression, and the
quadtree split prior. The final MAE, however, is determined by how many routed
queries pass the person-score threshold at inference. A stronger backbone alone
can improve features, but it does not directly calibrate the count.

The branch adds:

- `--transformer_activation gelu`: uses GELU in transformer FFN blocks.
- `--transformer_norm_style pre`: changes transformer blocks from official
  post-norm to pre-norm. This is mainly useful when increasing decoder depth.
- `--count_loss_coef`: an optional loss between soft predicted count and GT
  count. This directly trains count calibration.
- `--count_loss_type`: `log_l1` is recommended because raw count L1 can be
  very large early in training when thousands of query probabilities are
  uncalibrated.
- `--count_loss_start_epoch`: delays count calibration until classification has
  started to settle.
- `--count_loss_gate`: controls how sparse/dense routing gates are used by the
  count loss.
  - `detach`: calibrates query scores without backpropagating through the
    splitter. This is the recommended first run.
  - `soft`: lets count loss update both classifier scores and splitter.
  - `hard`: uses thresholded route masks.
- `--splitter_head conv`: adds a small local-context conv residual to the
  original `AvgPool2d -> Conv1x1` splitter. The residual is zero-initialized so
  training starts from the paper splitter instead of a new random router.
- `--ema_decay`: evaluates and saves an exponential moving average of model
  weights. This targets the common PET pattern where training loss continues to
  decrease after validation MAE has already peaked.
- `--lr_drop` / `--lr_gamma`: optional real StepLR decay. The original PET
  behavior is preserved by default because `--lr_drop` is negative, which drops
  only after `--epochs`.
- `--decoder_attention linear`: experimental kernelized linear attention inside
  decoder layers. `softmax` remains the default and matches official PET.
- `--decoder_memory_halo`: experimental overlap for decoder cross-attention
  memory windows. Query windows stay non-overlapped, so each point query remains
  unique and predictions are not duplicated.
- `--fusion_mhf_mode`: VGG FPN feature-fusion ablations inspired by VMambaCC's
  MHF/HS2FPN study. `cem` adds high-level channel guidance, `cem_msem` adds
  grouped spatial guidance, and `full` adds a second high-level channel filter.
  All gates are residual and zero-initialized so the model starts from the
  original FPN behavior instead of immediately suppressing features.
- `--fusion_mhf_impl`: chooses the implementation. `residual` is the PET-safe
  zero-init approximation. `vmambacc` follows the VMambaCC paper equations for
  CEM, MSEM, and HCEM.
- `--fusion_fpn_type hs2fpn`: marks the VGG feature fusion as the
  VMambaCC-style high-level semantic supervised path. Use it with
  `--fusion_mhf_impl vmambacc` for the paper-style ablation.
- `--quad_context_mixer lite`: retained only as a failed experimental switch.
  On SHA it severely undercounted, and a threshold sweep still stayed around
  MAE 333. Do not use it for the main experiment path.

Official PET reproduction remains:

```bash
--splitter_head pool --count_loss_coef 0.0 --transformer_activation relu --transformer_norm_style post --decoder_attention softmax --decoder_memory_halo 0 --fusion_mhf_mode none --fusion_mhf_impl residual --fusion_fpn_type fpn --quad_context_mixer none
```

For VGG paper-style runs, leave `--vgg_fpn_main_lr` off. That keeps the
original PET optimizer grouping where every VGG backbone/FPN parameter uses
`--lr_backbone`. Use `--vgg_fpn_main_lr` only as an ablation.

For a finished checkpoint, run an inference-threshold sweep before reporting
MAE. The default grid includes the paper threshold `0.5`, so the selected
validation result is non-worse than the baseline on that same validation split:

```bash
python scripts/sweep_eval_thresholds.py \
  --resume outputs/SHA/vgg16_bn_paper/best_checkpoint.pth \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --device cuda \
  --score_thresholds 0.35 0.4 0.45 0.48 0.5 0.52 0.55 0.6 0.65 \
  --split_thresholds 0.5
```

The script writes `best_thresholds.json`, `sweep_results.json`, and
`sweep_results.csv` under `<checkpoint_dir>/threshold_sweep` unless
`--output_dir` is provided.

## VMambaCC Paper-Style Optional Path

The paper's released text describes VMambaCC's MHF/HS2FPN blocks, but public
source code was not available when this branch was written. The optional
`vmambacc` path implements the equations from the paper:

- CEM: high-level max/avg pooled features share a conv stack, sum, sigmoid, and
  multiply the high-level feature.
- MSEM: channels are split into heads; each head builds a spatial gate from
  per-head channel max/avg maps.
- HCEM: the MSEM output is channel-enhanced again, projected with `Conv1x1`,
  upsampled, and multiplied into the lower-level feature before fusion.

This is intentionally off by default. To run the paper-style full MHF ablation
inside PET's VGG FPN:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_vmambacc_full_h4 \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode full \
  --fusion_mhf_heads 4 \
  --fusion_mhf_position before \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

`--fusion_mhf_output_activation none` follows the paper equation most closely.
If this suppresses or explodes counts in PET, use
`--fusion_mhf_output_activation sigmoid` only as a bounded ablation.

## Recommended VGG Improvement Run

For best MAE, keep the VGG backbone and paper loss, then add EMA:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_step_ema \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --ema_decay 0.999 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

For ShanghaiTech Part B, use the same loader through `--dataset_file SHB`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHB \
  --data_path ./data/ShanghaiTech/part_B \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_step_ema \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --ema_decay 0.999 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

For UCF-QNRF, use the larger official split through `--dataset_file QNRF`.
Training still uses 256x256 crops, while validation downsizes the long image
side to 1536 by default, matching the original PET preprocessing protocol:

Download and validate first:

```bash
sudo apt-get update
sudo apt-get install -y aria2 unzip
bash scripts/download_ucf_qnrf.sh
python scripts/check_qnrf_annotations.py --data_path ./data/UCF-QNRF_ECCV18
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file QNRF \
  --data_path ./data/UCF-QNRF_ECCV18 \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_paper \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --pet_loss_variant paper \
  --eval_max_size 1536 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

If MAE peaks early and then degrades while training loss keeps decreasing, test
a real LR drop instead of running at `1e-4` for all 1500 epochs:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_step_drop700 \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 700 \
  --lr_gamma 0.1 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

If a run already found a strong best checkpoint but later epochs degraded, start
a low-LR fine-tune from that best model without reusing the old optimizer and
scheduler:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --resume outputs/SHA/vgg16_bn_step/best_checkpoint.pth \
  --resume_model_only \
  --output_dir vgg16_bn_best_ft_ema \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-5 \
  --lr_backbone 1e-6 \
  --lr_scheduler step \
  --lr_drop 250 \
  --lr_gamma 0.1 \
  --ema_decay 0.999 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

## VMambaCC Paper-Style FPN Ablations

VMambaCC reports that feature fusion matters more than simply replacing the
backbone. Its ablation shows that CEM alone is harmful, CEM+MSEM with one head
helps, four heads can hurt without HCEM, and the full CEM+MSEM+HCEM block works
best in their architecture. In PET, test these one at a time with
`--fusion_mhf_impl vmambacc` and keep the paper splitter/loss unchanged.

Run short triage first. Stop a variant if best SHA MAE is still above 100 after
250-300 epochs or if `pred_cnt / gt_cnt` is below 0.5.

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_mhf_00_baseline \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode none \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

CEM only:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_mhf_01_cem \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode cem \
  --fusion_mhf_heads 1 \
  --fusion_mhf_position before \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

CEM + MSEM, one spatial head:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_mhf_02_cem_msem_h1 \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode cem_msem \
  --fusion_mhf_heads 1 \
  --fusion_mhf_position before \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

CEM + MSEM, four spatial heads:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_mhf_03_cem_msem_h4 \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode cem_msem \
  --fusion_mhf_heads 4 \
  --fusion_mhf_position before \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

Full CEM + MSEM + HCEM:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone vgg16_bn \
  --output_dir vgg16_bn_mhf_04_full_h4 \
  --epochs 500 \
  --eval_freq 5 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_scheduler step \
  --lr_drop 350 \
  --lr_gamma 0.1 \
  --fusion_fpn_type hs2fpn \
  --fusion_mhf_impl vmambacc \
  --fusion_mhf_mode full \
  --fusion_mhf_heads 4 \
  --fusion_mhf_position before \
  --fusion_mhf_output_activation none \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 42
```

## Failed Quad Context Ablation

The QuadMamba-inspired `--quad_context_mixer lite` run should be considered a
negative result for this PET codebase. On SHA, the best checkpoint stayed around
MAE 333 and a score-threshold sweep from 0.05 to 0.5 did not recover it. This
means the failure was not simple threshold calibration. Keep
`--quad_context_mixer none` unless deliberately reproducing that negative
ablation.

## Recommended Timm Run

The safest timm run is still the paper loss with FPN-style feature fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone convnextv2_base \
  --timm_adapter fpn \
  --output_dir convnextv2_base_fpn_paper \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 4 \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --lr_backbone_adapter 1e-4 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 10 \
  --freeze_backbone_epochs 5 \
  --splitter_head pool \
  --count_loss_coef 0.0 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 7
```

If training is stable but undercounts or overcounts persist, sweep:

```text
--transformer_activation gelu --transformer_norm_style pre --dim_feedforward 1024 --dec_layers 3
--decoder_memory_halo 1
--decoder_attention linear
--count_loss_coef 0.005
--count_loss_coef 0.02
--count_loss_gate soft
--count_loss_start_epoch 50
--splitter_head conv --splitter_hidden_dim 128 --splitter_activation gelu
```
