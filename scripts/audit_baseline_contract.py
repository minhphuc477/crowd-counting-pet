"""Emit deterministic paper-PET tensors for cross-revision comparison.

Run this script through ``exec`` from the repository revision being audited so
that imports resolve against that revision. The output is intentionally a
plain torch checkpoint: a second process can compare two revisions without
loading both source trees into one Python interpreter.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

# The audit is intentionally executed from the root of each compared worktree.
sys.path.insert(0, str(Path.cwd()))

import main
from models import build_model


def _supported_args(parser, requested):
    supported = {option for action in parser._actions for option in action.option_strings}
    result = []
    index = 0
    while index < len(requested):
        option = requested[index]
        value = requested[index + 1] if index + 1 < len(requested) else None
        if option in supported:
            result.append(option)
            if value is not None and not value.startswith('--'):
                result.append(value)
        index += 2 if value is not None and not value.startswith('--') else 1
    return result


def _tensor_dict_stats(tensors):
    stats = {}
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            continue
        data = value.detach().cpu().double()
        stats[name] = {
            'shape': list(data.shape),
            'sum': data.sum().item(),
            'abs_sum': data.abs().sum().item(),
            'square_sum': data.square().sum().item(),
        }
    return stats


def main_audit():
    cli = argparse.ArgumentParser()
    cli.add_argument('--output', required=True)
    cli.add_argument('--seed', type=int, default=1234)
    cli.add_argument('--image-size', type=int, default=256)
    cli.add_argument('--pretrained', action='store_true')
    cli.add_argument('--skip-eval', action='store_true')
    cli.add_argument('--skip-train', action='store_true')
    audit_args = cli.parse_args()

    requested = [
        '--backbone', 'vgg16_bn',
        '--timm_adapter', 'lite_fpn',
        '--dataset_file', 'SHA',
        '--data_path', '.',
        '--device', 'cpu',
        '--batch_size', '1',
        '--pet_loss_variant', 'paper',
        '--split_loss_variant', 'paper',
        '--apg_loss_coef', '0.0',
        '--ifi_loss_coef', '0.0',
        '--count_head_loss_coef', '0.0',
        '--score_threshold', '0.5',
        '--split_threshold', '0.5',
        '--query_prune_threshold', '0.5',
        '--eval_nms_radius', '0.0',
        '--eval_branch_gate', 'none',
        '--eval_soft_split_gate', 'none',
    ]
    if not audit_args.pretrained:
        requested.append('--no_pretrained_backbone')
    parser = main.get_args_parser()
    args = parser.parse_args(_supported_args(parser, requested))

    torch.manual_seed(audit_args.seed)
    model, criterion = build_model(args)
    model.to('cpu')
    criterion.to('cpu')

    size = audit_args.image_size
    generator = torch.Generator().manual_seed(audit_args.seed + 1)
    samples = [torch.rand(3, size, size, generator=generator)]
    targets = [
        {
            'labels': torch.ones(5, dtype=torch.long),
            'points': torch.tensor(
                [[32.0, 48.0], [64.0, 192.0], [112.0, 96.0], [176.0, 208.0], [224.0, 40.0]],
                dtype=torch.float32,
            ),
            'density': torch.tensor(5.0),
        },
    ]

    initial_stats = _tensor_dict_stats(model.state_dict())
    eval_tensors = {}
    if not audit_args.skip_eval:
        model.eval()
        with torch.no_grad():
            eval_output = model(samples, test=True, targets=targets, epoch=6)
        eval_tensors = {
            key: value.detach().cpu()
            for key, value in eval_output.items()
            if torch.is_tensor(value) and key in {'pred_logits', 'pred_points', 'pred_offsets', 'points_queries'}
        }

    loss_values = {}
    gradient_stats = {}
    if not audit_args.skip_train:
        model.train()
        model.zero_grad(set_to_none=True)
        train_output = model(samples, epoch=6, train=True, criterion=criterion, targets=targets)
        total_loss = train_output['losses']
        total_loss.backward()
        loss_values = {
            key: float(value.detach().cpu())
            for key, value in train_output['loss_dict'].items()
            if torch.is_tensor(value)
        }
        loss_values['losses'] = float(total_loss.detach().cpu())
        gradient_stats = _tensor_dict_stats({
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        })

    payload = {
        'args': vars(args),
        'initial_stats': initial_stats,
        'eval_tensors': eval_tensors,
        'loss_values': loss_values,
        'gradient_stats': gradient_stats,
    }
    output = Path(audit_args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        'output': str(output),
        'loss_values': loss_values,
        'eval_shapes': {key: list(value.shape) for key, value in eval_tensors.items()},
        'parameter_tensors': len(initial_stats),
        'gradient_tensors': len(gradient_stats),
    }, indent=2))


if __name__ == '__main__':
    main_audit()
