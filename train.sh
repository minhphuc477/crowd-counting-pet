#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun \
    --standalone \
    --nproc_per_node=1 \
    main.py \
    --backbone="auto_swin" \
    --enhanced_point_query \
    --threshold_sweep \
    --search_trials=6 \
    --search_epochs=10 \
    --search_eval_freq=1 \
    --target_mae=45 \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=1 \
    --output_dir='pet_mae40' \
    "$@"
