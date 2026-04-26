#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun \
    --standalone \
    --nproc_per_node=1 \
    main.py \
    --backbone="vgg16_bn" \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=5 \
    --output_dir='pet_model' \
    "$@"
