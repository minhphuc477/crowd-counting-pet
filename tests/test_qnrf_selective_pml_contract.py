import torch

import main
from models import build_model
from models.pet import SelectiveContextToDetailFusion, proximal_mapping_measure_loss


def test_selective_fusion_is_exact_identity_at_initialization():
    module = SelectiveContextToDetailFusion(32)
    detail = torch.randn(2, 32, 16, 20)
    context = torch.randn(2, 32, 8, 10)

    fused_detail, fused_context, selector_logits = module(detail, context)

    assert torch.equal(fused_detail, detail)
    assert torch.equal(fused_context, context)
    assert selector_logits.shape == (2, 1, 8, 10)


def test_proximal_mapping_has_zero_loss_for_unit_voronoi_mass():
    density = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
    targets = [{'points': torch.tensor([[0.5, 0.5], [0.5, 1.5]])}]
    valid = torch.ones_like(density, dtype=torch.bool)

    losses = proximal_mapping_measure_loss(
        density,
        targets,
        valid,
        image_size=(1, 2),
        radius=1.0,
        chunk_size=1,
    )

    assert torch.allclose(losses['count'], torch.tensor(0.0))
    assert torch.allclose(losses['spatial'], torch.tensor(0.0))
    (losses['count'] + losses['spatial']).backward()
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()


def test_proximal_mapping_penalizes_wrong_per_point_mass_and_empty_images():
    density = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    valid = torch.ones_like(density, dtype=torch.bool)
    targets = [{'points': torch.tensor([[0.5, 0.5], [0.5, 1.5]])}]
    losses = proximal_mapping_measure_loss(
        density,
        targets,
        valid,
        image_size=(1, 2),
        radius=1.0,
    )
    assert torch.allclose(losses['count'], torch.tensor(1.0))

    empty_losses = proximal_mapping_measure_loss(
        density,
        [{'points': torch.empty(0, 2)}],
        valid,
        image_size=(1, 2),
    )
    assert torch.allclose(empty_losses['count'], torch.tensor(2.0))
    assert torch.allclose(empty_losses['spatial'], torch.tensor(2.0))


def test_qnrf_selective_pml_recipe_keeps_pet_as_eval_source():
    recipe = main.MODEL_RECIPES['vgg_apglc_qnrf_selective_pml_scale_rifi']
    assert recipe['backbone'] == 'vgg16_bn'
    assert recipe['scale_fusion'] == 'selective_context_to_detail'
    assert recipe['measure_loss_mode'] == 'pml'
    assert recipe['measure_feature_source'] == 'detail4x'
    assert recipe['measure_loss_init_cells'] == 4096.0
    assert recipe['measure_loss_coef'] > 0
    assert recipe['scale_selection_loss_coef'] > 0
    assert recipe['qnrf_max_train_outside_fraction'] == 0.25
    assert recipe['eval_count_source'] == 'pet'
    assert recipe['eval_score_calibration'] == 'none'


def test_qnrf_selective_pml_finetune_does_not_restart_converged_auxiliaries():
    recipe = main.MODEL_RECIPES[
        'vgg_apglc_qnrf_selective_pml_scale_rifi_finetune'
    ]
    assert recipe['apg_loss_coef'] == 0.0
    assert recipe['ifi_loss_coef'] == 0.0
    assert recipe['scale_point_loss_coef'] == 0.0
    assert recipe['scale_selection_loss_coef'] > 0.0
    assert recipe['measure_loss_coef'] > 0.0
    assert recipe['freeze_bn'] is True


def test_qnrf_selective_pml_recipe_parses_without_hidden_overrides():
    argv = [
        '--dataset_file', 'QNRF',
        '--data_path', 'unused',
        '--model_recipe', 'vgg_apglc_qnrf_selective_pml_scale_rifi',
        '--allow_experimental_model_recipe',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    assert args.scale_fusion == 'selective_context_to_detail'
    assert args.measure_loss_mode == 'pml'
    assert args.measure_feature_source == 'detail4x'
    assert args.eval_count_source == 'pet'


def test_qnrf_selective_pml_real_model_forward_backward():
    torch.manual_seed(7)
    argv = [
        '--dataset_file', 'QNRF',
        '--data_path', 'unused',
        '--model_recipe', 'vgg_apglc_qnrf_selective_pml_scale_rifi',
        '--allow_experimental_model_recipe',
        '--device', 'cpu',
        '--no_pretrained_backbone',
        '--hidden_dim', '64',
        '--dim_feedforward', '128',
        '--nheads', '8',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    args = main.sanitize_unstable_training_args(args)
    model, criterion = build_model(args)
    model.train()

    image = torch.rand(3, 128, 128)
    points = torch.tensor([[28.0, 30.0], [32.0, 34.0], [90.0, 82.0]])
    target = {
        'points': points,
        'labels': torch.ones(points.shape[0], dtype=torch.long),
        'density': torch.tensor(float(points.shape[0])),
    }
    output = model(
        [image],
        train=True,
        criterion=criterion,
        targets=[target],
        epoch=30,
    )

    assert torch.isfinite(output['losses'])
    assert 'loss_scale_selection' in output['loss_dict']
    assert 'loss_measure_dist' in output['loss_dict']
    assert 'loss_measure_count' in output['loss_dict']
    output['losses'].backward()
    for module in (model.scale_fusion, model.measure_head):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
