from pathlib import Path

from .NWPU import build as build_nwpu
from .SHA import build as build_sha


data_paths = {
    'SHA': ('./data/ShanghaiTech/part_A/', './data/ShanghaiTech/part_A_final/'),
    'SHB': ('./data/ShanghaiTech/part_B/', './data/ShanghaiTech/part_B_final/'),
    'NWPU': ('./data/NWPU-Crowd/', './data/NWPU/'),
    'NWPU_Crowd': ('./data/NWPU-Crowd/', './data/NWPU/'),
    'NWPU-Crowd': ('./data/NWPU-Crowd/', './data/NWPU/'),
}

dataset_dir_names = {
    'SHA': ('part_A', 'part_A_final'),
    'SHB': ('part_B', 'part_B_final'),
    'NWPU': ('NWPU-Crowd', 'NWPU'),
    'NWPU_Crowd': ('NWPU-Crowd', 'NWPU'),
    'NWPU-Crowd': ('NWPU-Crowd', 'NWPU'),
}


def _split_images_dir(data_root, image_set, dataset_file):
    source_set = 'train' if image_set == 'train_eval' else image_set
    if dataset_file in ('NWPU', 'NWPU_Crowd', 'NWPU-Crowd'):
        return Path(data_root) / 'images'
    split = 'train_data' if source_set == 'train' else 'test_data'
    return Path(data_root) / split / 'images'


def _add_unique(candidates, seen, path):
    key = str(path)
    if key not in seen:
        candidates.append(path)
        seen.add(key)


def _iter_named_descendants(root, names, max_depth=4):
    if not root.is_dir():
        return

    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if current.name in names:
            yield current
        if depth >= max_depth:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in reversed(children):
            if child.is_dir():
                stack.append((child, depth + 1))


def _candidate_data_paths(dataset_file, requested_path):
    candidates = []
    seen = set()
    search_roots = []

    if requested_path:
        requested = Path(requested_path)
        _add_unique(candidates, seen, requested)
        if not requested.name.endswith('_final'):
            _add_unique(candidates, seen, requested.with_name(f'{requested.name}_final'))
        search_roots.extend([requested.parent, requested.parent.parent])

    for path in data_paths[dataset_file]:
        default_path = Path(path)
        _add_unique(candidates, seen, default_path)
        search_roots.extend([default_path.parent, default_path.parent.parent])

    for root in search_roots:
        for found in _iter_named_descendants(root, dataset_dir_names[dataset_file]):
            _add_unique(candidates, seen, found)

    return candidates


def _resolve_data_path(dataset_file, requested_path, image_set):
    candidates = _candidate_data_paths(dataset_file, requested_path)
    for candidate in candidates:
        if _split_images_dir(candidate, image_set, dataset_file).is_dir():
            return str(candidate)

    tried = '\n'.join(
        f'  - {_split_images_dir(candidate, image_set, dataset_file)}'
        for candidate in candidates
    )
    raise FileNotFoundError(
        f'Could not find {dataset_file} {image_set} images directory. Tried:\n{tried}'
    )


def build_dataset(image_set, args):
    if args.dataset_file in ('SHA', 'SHB'):
        args.data_path = _resolve_data_path(args.dataset_file, getattr(args, 'data_path', None), image_set)
        return build_sha(image_set, args)
    if args.dataset_file in ('NWPU', 'NWPU_Crowd', 'NWPU-Crowd'):
        args.data_path = _resolve_data_path(args.dataset_file, getattr(args, 'data_path', None), image_set)
        return build_nwpu(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported by the best-model branch')
