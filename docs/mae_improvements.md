# MAE-focused PET Improvements

This branch keeps official PET behavior available while adding options aimed at
reducing MAE with timm backbones.

## Why These Changes

PET's training objective supervises classification, point regression, and the
quadtree split prior. The final MAE, however, is determined by how many routed
queries pass the person-score threshold at inference. A stronger backbone alone
can improve features, but it does not directly calibrate the count.

The branch adds:

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
--splitter_head pool --count_loss_coef 0.0
```

## Recommended Timm Run

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone convnextv2_base \
  --timm_adapter lite_fpn \
  --output_dir convnextv2_base_lite_fpn_count \
  --epochs 1500 \
  --eval_freq 5 \
  --batch_size 4 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --lr_backbone_adapter 1e-4 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 5 \
  --splitter_head conv \
  --splitter_hidden_dim 128 \
  --count_loss_coef 0.01 \
  --count_loss_gate detach \
  --count_loss_type log_l1 \
  --count_loss_start_epoch 20 \
  --pet_loss_variant paper \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --seed 7
```

If training is stable but undercounts or overcounts persist, sweep:

```text
--count_loss_coef 0.005
--count_loss_coef 0.02
--count_loss_gate soft
--count_loss_start_epoch 50
```
