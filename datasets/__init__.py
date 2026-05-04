from .SHA import build as build_sha

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
}

def build_dataset(image_set, args):
    if not getattr(args, 'data_path', None):
        args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
