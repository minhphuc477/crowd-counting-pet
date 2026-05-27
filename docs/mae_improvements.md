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
- `--splitter_head conv`: replaces the original `AvgPool2d -> Conv1x1` splitter
  with a small local-context conv head before pooling.

Official PET reproduction remains:

```bash
--splitter_head pool --count_loss_coef 0.0 --transformer_activation relu --transformer_norm_style post
```

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
--count_loss_coef 0.005
--count_loss_coef 0.02
--count_loss_gate soft
--count_loss_start_epoch 50
--splitter_head conv --splitter_hidden_dim 128 --splitter_activation gelu
```
