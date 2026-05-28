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
- `--ema_decay`: evaluates and saves an exponential moving average of model
  weights. This targets the common PET pattern where training loss continues to
  decrease after validation MAE has already peaked.
- `--lr_drop` / `--lr_gamma`: optional real StepLR decay. The original PET
  behavior is preserved by default because `--lr_drop` is negative, which drops
  only after `--epochs`.

Official PET reproduction remains:

```bash
--splitter_head pool --count_loss_coef 0.0 --transformer_activation relu --transformer_norm_style post
```

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
  --output_dir outputs/SHA/vgg16_bn_best_ft_ema \
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
