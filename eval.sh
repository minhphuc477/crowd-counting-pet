CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --backbone="auto" \
    --dataset_file="SHA" \
    --resume="path_to_model" \
    --deterministic \
    --vis_dir=""
