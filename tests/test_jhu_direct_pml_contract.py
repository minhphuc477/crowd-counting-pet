import argparse

import pytest
import torch

import main
from models import build_model


RECIPE = 'vgg_apglc_jhu_direct_pml_scale_rifi'
PARENT_RECIPE = 'vgg_apglc_jhu_tail_scale_rifi'


def test_jhu_direct_pml_recipe_preserves_detector_and_adds_balanced_measure():
    recipe = main.MODEL_RECIPES[RECIPE]

    assert recipe['backbone'] == 'vgg16_bn'
    assert recipe['measure_head_variant'] == 'direct_fpn'
    assert recipe['measure_loss_mode'] == 'pml'
    assert recipe['measure_pml_normalization'] == 'points'
    assert recipe['measure_loss_feature_grad_scale'] == pytest.approx(0.05)
    assert recipe['measure_loss_feature_grad_start_epoch'] > 0
    assert recipe['measure_loss_zero_coef'] > 0
    assert recipe['eval_count_source'] == 'measure_pet_blend'
    assert recipe['validation_protocol'] == 'official_val'
    assert recipe['jhu_eval_split'] == 'val'
    assert recipe['min_crop_points'] == 0
    assert recipe['train_count_strata'].startswith('0,')
    assert recipe['apg_loss_coef'] == 0.0
    assert recipe['ifi_loss_coef'] == 0.0
    assert recipe['scale_point_loss_coef'] == 0.0


def _guard_args(resume='', model_only=False, dataset='JHU'):
    return argparse.Namespace(
        model_recipe=RECIPE,
        dataset_file=dataset,
        resume=resume,
        resume_model_only=model_only,
        density_map_loss_coef=0.0,
        count_head_loss_coef=0.0,
        _explicit_args=set(),
    )


def test_jhu_direct_pml_requires_model_only_for_initial_transfer():
    with pytest.raises(ValueError, match='requires a trained JHU Scale-RIFI checkpoint'):
        main.sanitize_unstable_training_args(_guard_args())

    checkpoint = {'args': argparse.Namespace(model_recipe=PARENT_RECIPE)}
    with pytest.raises(ValueError, match='must use --resume_model_only'):
        main.sanitize_unstable_training_args(
            _guard_args(resume='parent.pth'),
            checkpoint=checkpoint,
        )

    args = _guard_args(resume='parent.pth', model_only=True)
    assert main.sanitize_unstable_training_args(args, checkpoint=checkpoint) is args


def test_jhu_direct_pml_allows_full_same_recipe_resume():
    args = _guard_args(resume='checkpoint.pth')
    checkpoint = {'args': argparse.Namespace(model_recipe=RECIPE)}

    assert main.sanitize_unstable_training_args(args, checkpoint=checkpoint) is args


def test_jhu_direct_pml_rejects_other_datasets():
    args = _guard_args(resume='checkpoint.pth', dataset='NWPU')
    checkpoint = {'args': argparse.Namespace(model_recipe=RECIPE)}

    with pytest.raises(ValueError, match='only defined for JHU-Crowd'):
        main.sanitize_unstable_training_args(args, checkpoint=checkpoint)


def test_jhu_direct_pml_eval_emits_pet_measure_blend():
    argv = [
        '--dataset_file', 'JHU',
        '--data_path', 'unused',
        '--model_recipe', RECIPE,
        '--allow_experimental_model_recipe',
        '--resume', 'parent.pth',
        '--resume_model_only',
        '--no_pretrained_backbone',
        '--hidden_dim', '64',
        '--dim_feedforward', '128',
        '--nheads', '8',
        '--device', 'cpu',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    args = main.sanitize_unstable_training_args(
        args,
        checkpoint={'args': {'model_recipe': PARENT_RECIPE}},
    )
    model, _ = build_model(args)
    model.eval()

    with torch.no_grad():
        output = model([torch.rand(3, 128, 160)], epoch=30)

    pet_count = output['pred_points'].shape[1]
    measure_count = output['measure_count']
    expected = (
        (1.0 - args.eval_count_blend_alpha) * pet_count
        + args.eval_count_blend_alpha * measure_count
    )
    assert torch.allclose(output['count_for_mae'], expected)
    assert torch.equal(output['count_density'], output['measure_density'])
