from .SHA import build as build_sha
from pathlib import Path

data_paths = {
    'SHA': ('./data/ShanghaiTech/part_A/', './data/ShanghaiTech/part_A_final/'),
    'SHB': ('./data/ShanghaiTech/part_B/', './data/ShanghaiTech/part_B_final/'),
}


def _split_images_dir(data_root, image_set):
    split = 'train_data' if image_set == 'train' else 'test_data'
    return Path(data_root) / split / 'images'


def _candidate_data_paths(dataset_file, requested_path):
    candidates = []
    if requested_path:
        requested = Path(requested_path)
        candidates.append(requested)
        if not requested.name.endswith('_final'):
            candidates.append(requested.with_name(f'{requested.name}_final'))
    candidates.extend(Path(path) for path in data_paths[dataset_file])

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def _resolve_data_path(dataset_file, requested_path, image_set):
    candidates = _candidate_data_paths(dataset_file, requested_path)
    for candidate in candidates:
        if _split_images_dir(candidate, image_set).is_dir():
            return str(candidate)

    tried = '\n'.join(f'  - {_split_images_dir(candidate, image_set)}' for candidate in candidates)
    raise FileNotFoundError(
        f'Could not find {dataset_file} {image_set} images directory. Tried:\n{tried}'
    )


def build_dataset(image_set, args):
    if args.dataset_file in ('SHA', 'SHB'):
        args.data_path = _resolve_data_path(args.dataset_file, getattr(args, 'data_path', None), image_set)
        return build_sha(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
