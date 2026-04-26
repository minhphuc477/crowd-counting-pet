#!/usr/bin/env bash

# Reference recipe matching the pre-5070Ti auto-tuning path around
# commit 1baeb315d8bb3e64fb4bc27214e5d99e0340d994.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun \
    --standalone \
    --nproc_per_node=1 \
    main.py \
    --backbone="auto" \
    --target_mae=50 \
    --search_trials=6 \
    --search_epochs=8 \
    --search_eval_freq=1 \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=1 \
    --output_dir='pet_model_convnext_reference' \
    "$@"
