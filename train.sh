#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun \
    --standalone \
    --nproc_per_node=1 \
    main.py \
    --backbone="auto" \
    --search_trials=6 \
    --search_epochs=8 \
    --search_eval_freq=1 \
    --target_mae=50 \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=1 \
    --output_dir='pet_model' \
    "$@"
