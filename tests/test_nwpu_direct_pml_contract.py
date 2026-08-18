import argparse

import pytest
import torch

import main


def build_recipe_args(*extra):
    argv = [
        '--dataset_file', 'NWPU',
        '--data_path', 'unused',
        '--model_recipe', 'vgg_apglc_nwpu_direct_pml_tail_rifi',
        '--allow_experimental_model_recipe',
        '--resume', 'tail_rifi_checkpoint.pth',
        '--resume_model_only',
        '--device', 'cpu',
        *extra,
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    return main.sanitize_unstable_training_args(args)


def test_nwpu_direct_pml_recipe_is_an_isolated_spatial_recount_stage():
    args = build_recipe_args()

    assert args.backbone == 'vgg16_bn'
    assert args.train_measure_head_only is True
    assert args.measure_head_variant == 'direct_fpn'
    assert args.measure_loss_mode == 'pml'
    assert args.eval_count_source == 'measure'
    assert args.eval_tile_trigger_area == 0
    assert args.measure_loss_feature_grad_scale == 0.0
    assert args.measure_loss_image_count_coef > 0.0
    assert args.measure_loss_relative_count_coef > 0.0
    assert args.measure_loss_zero_coef > 0.0
    assert args.train_count_strata == '0,100,500,5000'
    assert args.train_count_strata_strength > 0.0
    assert args.validation_protocol == 'official_val'


def test_nwpu_direct_pml_recipe_rejects_scratch_and_optimizer_resume():
    base = argparse.Namespace(
        model_recipe='vgg_apglc_nwpu_direct_pml_tail_rifi',
        dataset_file='NWPU',
        resume='',
        resume_model_only=False,
        density_map_loss_coef=0.0,
        count_head_loss_coef=0.0,
        _explicit_args=set(),
    )
    with pytest.raises(ValueError, match='requires a trained NWPU Tail-RIFI checkpoint'):
        main.sanitize_unstable_training_args(base)

    base.resume = 'checkpoint.pth'
    with pytest.raises(ValueError, match='must use --resume_model_only'):
        main.sanitize_unstable_training_args(base)


def test_measure_head_only_trainability_freezes_every_inherited_parameter():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = torch.nn.Linear(4, 4)
            self.measure_head = torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 1),
            )

    model = ToyModel()
    trainable, frozen = main.set_measure_head_only_trainability(model)

    assert trainable == sum(parameter.numel() for parameter in model.measure_head.parameters())
    assert frozen == sum(parameter.numel() for parameter in model.detector.parameters())
    assert all(parameter.requires_grad for parameter in model.measure_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.detector.parameters())


def test_count_strata_equalize_mass_and_keep_zero_images_separate():
    counts = [0, 0, 50, 100, 101, 500, 501, 5000, 5001]
    weights = main.build_count_stratified_weights(
        counts,
        boundaries='0,100,500,5000',
        strength=1.0,
        max_weight=0.0,
    )
    groups = ([0, 1], [2, 3], [4, 5], [6, 7], [8])
    mass = torch.tensor([weights[list(indices)].sum() for indices in groups])

    assert torch.allclose(mass, torch.full_like(mass, mass[0]))
    assert weights[0] == weights[1]
    assert weights[8] > weights[0]


def test_count_strata_and_monotonic_count_weighting_are_mutually_exclusive():
    class CountDataset:
        def get_sample_counts(self):
            return [0, 10, 1000]

    args = argparse.Namespace(
        train_count_weight_power=0.5,
        train_count_weight_max=8.0,
        train_count_strata='0,100,500,5000',
        train_count_strata_strength=0.5,
        train_count_strata_max_weight=4.0,
    )
    with pytest.raises(ValueError, match='cannot be enabled together'):
        main.resolve_training_sampling_weights(CountDataset(), args)
