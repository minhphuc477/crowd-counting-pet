CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="SHA" \
    --backbone="vgg16_bn" \
    --resume="path_to_model" \
    --vis_dir=""
