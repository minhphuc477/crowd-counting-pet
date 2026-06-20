"""Compare model tensors in two trusted local PET checkpoints."""

import argparse
import json
from pathlib import Path

import torch


BN_BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--base-model-key', default='model')
    parser.add_argument('--candidate-model-key', default='model')
    parser.add_argument('--atol', default=0.0, type=float)
    parser.add_argument('--top', default=30, type=int)
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def category(name):
    if name.startswith('count_head.'):
        return 'count_head'
    if name.endswith(BN_BUFFER_SUFFIXES):
        return 'batchnorm_buffer'
    return 'shared_model'


def load_state(path, model_key):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if model_key not in checkpoint:
        raise KeyError(f'{path}: missing model key {model_key!r}; keys={list(checkpoint)}')
    return checkpoint, checkpoint[model_key]


def tensor_delta(name, base, candidate, atol):
    if base.shape != candidate.shape:
        return {
            'name': name,
            'category': category(name),
            'changed': True,
            'shape_mismatch': [list(base.shape), list(candidate.shape)],
        }
    if not (torch.is_tensor(base) and torch.is_tensor(candidate)):
        changed = base != candidate
        return {'name': name, 'category': category(name), 'changed': bool(changed)}
    base_f = base.detach().cpu().double()
    candidate_f = candidate.detach().cpu().double()
    delta = candidate_f - base_f
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    return {
        'name': name,
        'category': category(name),
        'changed': max_abs > atol,
        'shape': list(base.shape),
        'max_abs': max_abs,
        'mean_abs': float(delta.abs().mean().item()) if delta.numel() else 0.0,
        'l2': float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0,
    }


def main():
    args = get_args()
    base_checkpoint, base = load_state(args.base, args.base_model_key)
    candidate_checkpoint, candidate = load_state(args.candidate, args.candidate_model_key)

    base_keys = set(base)
    candidate_keys = set(candidate)
    common = sorted(base_keys & candidate_keys)
    deltas = [tensor_delta(name, base[name], candidate[name], args.atol) for name in common]
    changed = [item for item in deltas if item['changed']]

    categories = {}
    for name in ('shared_model', 'batchnorm_buffer', 'count_head'):
        members = [item for item in deltas if item['category'] == name]
        changed_members = [item for item in members if item['changed']]
        categories[name] = {
            'common_tensors': len(members),
            'changed_tensors': len(changed_members),
            'max_abs': max((item.get('max_abs', 0.0) for item in changed_members), default=0.0),
            'l2_sum': sum(item.get('l2', 0.0) for item in changed_members),
        }

    report = {
        'base': str(args.base),
        'candidate': str(args.candidate),
        'base_epoch': base_checkpoint.get('epoch'),
        'candidate_epoch': candidate_checkpoint.get('epoch'),
        'base_best_mae': base_checkpoint.get('best_mae'),
        'candidate_best_mae': candidate_checkpoint.get('best_mae'),
        'common_tensors': len(common),
        'base_only': sorted(base_keys - candidate_keys),
        'candidate_only': sorted(candidate_keys - base_keys),
        'categories': categories,
        'top_changed': sorted(
            changed,
            key=lambda item: item.get('l2', item.get('max_abs', 0.0)),
            reverse=True,
        )[:max(0, args.top)],
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
