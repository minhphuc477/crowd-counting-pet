CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="SHA" \
    --backbone="convnextv2_base" \
    --resume="path_to_model" \
    --vis_dir=""
