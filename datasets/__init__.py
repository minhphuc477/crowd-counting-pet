from pathlib import Path

import torch.utils.data
import torchvision

from .SHA import build as build_sha


DEFAULT_DATA_PATHS = {
    'SHA': (
        './data/ShanghaiTech/PartA',
        './data/ShanghaiTech/part_A',
    ),
}


def resolve_data_path(dataset_file, requested_path=''):
    candidates = DEFAULT_DATA_PATHS.get(dataset_file)
    if candidates is None:
        raise ValueError(f'dataset {dataset_file} not supported')

    requested_path = str(requested_path or '').strip()
    if requested_path:
        requested = Path(requested_path)
        if requested.exists():
            return str(requested)

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return requested_path or candidates[0]


def build_dataset(image_set, args):
    args.data_path = resolve_data_path(args.dataset_file, getattr(args, 'data_path', ''))
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
