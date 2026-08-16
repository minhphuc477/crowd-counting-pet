import torch

import eval as eval_script
import main
from models import build_model
from models.pet import DirectPMLDensityHead, proximal_mapping_measure_loss


def build_direct_pml_args(hidden_dim=64):
    argv = [
        '--dataset_file', 'QNRF',
        '--data_path', 'unused',
        '--model_recipe', 'vgg_apglc_qnrf_direct_pml_scale_rifi',
        '--allow_experimental_model_recipe',
        '--device', 'cpu',
        '--no_pretrained_backbone',
        '--hidden_dim', str(hidden_dim),
        '--dim_feedforward', str(hidden_dim * 2),
        '--nheads', '8',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    return main.sanitize_unstable_training_args(args)


def test_direct_pml_head_is_nonnegative_masked_and_differentiable():
    torch.manual_seed(3)
    head = DirectPMLDensityHead(24, 32, activation='relu')
    detail = torch.randn(2, 24, 8, 10, requires_grad=True)
    context = torch.randn(2, 24, 4, 5, requires_grad=True)
    mask = torch.zeros(2, 8, 10, dtype=torch.bool)
    mask[1, 6:, :] = True

    density, count = head(detail, context, mask)

    assert density.shape == (2, 16, 20)
    assert count.shape == (2,)
    assert (density >= 0).all()
    assert torch.count_nonzero(density[1, 12:, :]) == 0
    assert torch.allclose(count, density.flatten(1).sum(1))
    count.sum().backward()
    assert detail.grad is not None and torch.isfinite(detail.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_direct_pml_batch_reduction_matches_published_sum_over_points():
    density = torch.tensor([[[0.8, 0.2, 0.4, 0.6]]], requires_grad=True)
    points = torch.tensor([[0.5, 0.5], [0.5, 3.5]])
    targets = [{'points': points}]
    valid = torch.ones_like(density, dtype=torch.bool)

    normalized = proximal_mapping_measure_loss(
        density,
        targets,
        valid,
        image_size=(1, 4),
        radius=1.0,
        normalization='points',
    )
    published = proximal_mapping_measure_loss(
        density,
        targets,
        valid,
        image_size=(1, 4),
        radius=1.0,
        normalization='batch',
    )

    assert torch.allclose(published['count'], normalized['count'] * 2)
    assert torch.allclose(published['spatial'], normalized['spatial'] * 2)
    (published['count'] + published['spatial']).backward()
    assert density.grad is not None and torch.isfinite(density.grad).all()


def test_direct_pml_recipe_is_a_primary_counter_not_selective_auxiliary():
    recipe = main.MODEL_RECIPES['vgg_apglc_qnrf_direct_pml_scale_rifi']
    assert recipe['backbone'] == 'vgg16_bn'
    assert recipe['measure_head_variant'] == 'direct_fpn'
    assert recipe['measure_loss_mode'] == 'pml'
    assert recipe['measure_pml_normalization'] == 'batch'
    assert recipe['measure_loss_coef'] == 1.0
    assert recipe['eval_count_source'] == 'measure'
    assert recipe['lr_measure_head'] == 3e-5
    assert recipe['freeze_bn'] is True
    assert recipe.get('scale_fusion', 'none') != 'selective_context_to_detail'
    assert recipe['patch_size_choices'] == '512'
    assert recipe['qnrf_random_scale_min'] == 0.5
    assert recipe['qnrf_random_scale_max'] == 1.5
    assert recipe['train_sample_multiplier'] == 1.0
    assert recipe['qnrf_max_train_outside_fraction'] == 1.0


def test_direct_pml_optimizer_uses_a_dedicated_reference_lr():
    args = build_direct_pml_args()
    model, _ = build_model(args)
    groups, summary = main.build_optimizer_param_groups(model, args)
    summary_by_name = {name: (tensors, parameters, lr) for name, tensors, parameters, lr in summary}

    assert 'measure_head' in summary_by_name
    assert summary_by_name['measure_head'][2] == 3e-5
    assert summary_by_name['main'][2] == args.lr
    measure_parameter_ids = {id(parameter) for parameter in model.measure_head.parameters()}
    grouped_parameter_ids = {
        id(parameter)
        for group in groups
        for parameter in group['params']
    }
    assert measure_parameter_ids <= grouped_parameter_ids


def test_direct_pml_real_model_forward_backward_and_inference_count():
    torch.manual_seed(7)
    args = build_direct_pml_args()
    model, criterion = build_model(args)
    model.train()
    image = torch.rand(3, 128, 160)
    points = torch.tensor([[28.0, 30.0], [32.0, 34.0], [90.0, 120.0]])
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
    assert 'loss_measure_dist' in output['loss_dict']
    assert 'loss_measure_count' in output['loss_dict']
    output['losses'].backward()
    head_gradients = [
        parameter.grad
        for parameter in model.measure_head.parameters()
        if parameter.grad is not None
    ]
    assert head_gradients
    assert all(torch.isfinite(gradient).all() for gradient in head_gradients)

    model.eval()
    with torch.no_grad():
        prediction = model([image], epoch=30)
    assert prediction['measure_density'].ndim == 3
    assert prediction['measure_density'].shape[0] == 1
    assert (prediction['measure_density'] >= 0).all()
    assert torch.allclose(
        prediction['count_for_mae'],
        prediction['measure_density'].flatten(1).sum(1),
    )
    assert torch.equal(prediction['count_density'], prediction['measure_density'])


def test_direct_pml_checkpoint_args_reconstruct_the_eval_architecture():
    train_args = build_direct_pml_args()
    checkpoint = {'args': vars(train_args)}
    argv = [
        '--resume', 'direct_pml_checkpoint.pth',
        '--dataset_file', 'QNRF',
        '--data_path', 'unused',
        '--device', 'cpu',
    ]
    eval_args = eval_script.get_args_parser().parse_args(argv)
    eval_args._explicit_args = main.get_explicit_arg_names(argv)
    merged = eval_script.merge_checkpoint_args(eval_args, checkpoint)

    assert merged.measure_head_variant == 'direct_fpn'
    assert merged.measure_pml_normalization == 'batch'
    assert merged.eval_count_source == 'measure'
    rebuilt, _ = build_model(merged)
    assert isinstance(rebuilt.measure_head, DirectPMLDensityHead)
