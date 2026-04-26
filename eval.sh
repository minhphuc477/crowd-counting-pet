#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python eval.py \
    --backbone="vgg16_bn" \
    --dataset_file="SHA" \
    --resume="path_to_model" \
    --deterministic \
    --vis_dir="" \
    "$@"
