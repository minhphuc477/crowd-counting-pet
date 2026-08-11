import random

import numpy as np
import torch

import main
from datasets.QNRF import random_resized_square_crop
from models.pet import EBCZipCountHead, _point_block_indices


QNRF_CENTERS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.4309850967, 13.4325785098,
    15.436302706, 17.437926239, 19.4346368253, 21.8313277988,
    24.849660614, 27.8719289982, 31.243070164, 38.8572258533,
)
QNRF_RANGES = (
    '1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,'
    '11:12,13:14,15:16,17:18,19:20,21:23,24:26,27:29,30:33,34:inf'
)


def test_qnrf_zip_uses_exact_published_intervals():
    head = EBCZipCountHead(
        16,
        block_stride=8,
        count_bin_centers=QNRF_CENTERS,
        count_bin_ranges=QNRF_RANGES,
    )
    counts = torch.tensor([1.0, 10.0, 11.0, 12.0, 13.0, 33.0, 34.0, 200.0])
    assert head.assign_positive_bins(counts).tolist() == [0, 9, 10, 10, 11, 18, 19, 19]


def test_context_zip_excludes_padding_and_retains_boundary_blocks():
    head = EBCZipCountHead(
        16,
        block_stride=8,
        count_bin_centers=QNRF_CENTERS,
        count_bin_ranges=QNRF_RANGES,
        use_detail=True,
        variant='context',
    )
    context = torch.randn(1, 16, 96, 96)
    detail = torch.randn(1, 16, 192, 192)
    mask = torch.ones(1, 192, 192, dtype=torch.bool)
    mask[:, :165, :170] = False
    output = head(context, detail=detail, detail_mask=mask)
    valid = output['valid_fraction'] > 0
    assert output['expected_count'].shape == (1, 24, 24)
    assert int(valid.sum()) == 21 * 22
    assert torch.isfinite(output['expected_count']).all()
    assert torch.count_nonzero(output['expected_count'][~valid]) == 0


def test_zip_target_blocks_do_not_stretch_non_divisible_image_edges():
    points = torch.tensor([
        [31.9, 31.9],
        [32.0, 32.0],
        [639.0, 1023.0],
    ])
    y, x = _point_block_indices(
        points,
        block_h=20,
        block_w=32,
        image_h=641,
        image_w=1025,
        pixel_block_size=32,
    )
    assert y.tolist() == [0, 1, 19]
    assert x.tolist() == [0, 1, 31]


def test_qnrf_recount_recipe_is_isolated_and_paper_aligned():
    recipe = main.MODEL_RECIPES['vgg_apglc_qnrf_zip_recount_scale_rifi']
    assert recipe['backbone'] == 'vgg16_bn'
    assert recipe['train_zip_count_head_only'] is True
    assert recipe['zip_count_loss_variant'] == 'paper'
    assert recipe['zip_count_head_variant'] == 'context'
    assert recipe['zip_count_block_size'] == 32
    assert recipe['patch_size_choices'] == '672'
    assert recipe['qnrf_random_scale_min'] == 0.75
    assert recipe['qnrf_random_scale_max'] == 2.0
    assert recipe['qnrf_random_resized_crop'] is True
    assert recipe['qnrf_aug_brightness'] == 0.15
    assert recipe['qnrf_aug_contrast'] == 0.15
    assert recipe['qnrf_aug_saturation'] == 0.10
    assert recipe['qnrf_aug_saltiness'] == 0.001
    assert recipe['qnrf_aug_spiciness'] == 0.001
    assert recipe['eval_count_source'] == 'zip'


def test_qnrf_recount_recipe_requires_model_only_parent_resume():
    argv = [
        '--model_recipe', 'vgg_apglc_qnrf_zip_recount_scale_rifi',
        '--allow_experimental_model_recipe',
        '--dataset_file', 'QNRF',
        '--data_path', 'unused',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    try:
        main.sanitize_unstable_training_args(args)
    except ValueError as error:
        assert '--resume and --resume_model_only' in str(error)
    else:
        raise AssertionError('fresh QNRF recount training was not rejected')

    args.resume = 'parent.pth'
    try:
        main.sanitize_unstable_training_args(args)
    except ValueError as error:
        assert 'must use --resume_model_only' in str(error)
    else:
        raise AssertionError('full-state QNRF recount resume was not rejected')


def test_random_resized_square_crop_keeps_yx_points_aligned():
    random.seed(7)
    image = torch.zeros(3, 80, 120)
    points = np.array([[20.0, 30.0], [60.0, 90.0]], dtype=np.float32)
    cropped, transformed = random_resized_square_crop(
        image,
        points,
        output_size=64,
        scale_range=(1.0, 1.0),
    )
    assert cropped.shape == (3, 64, 64)
    assert transformed.ndim == 2 and transformed.shape[1] == 2
    assert np.all(transformed >= 0.0)
    assert np.all(transformed < 64.0)


def test_nwpu_zip_recipe_preserves_published_last_interval():
    recipe = main.MODEL_RECIPES['vgg_apglc_nwpu_zip_recount']
    assert recipe['zip_count_bin_ranges'].endswith('10:inf')
