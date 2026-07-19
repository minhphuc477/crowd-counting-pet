import eval as pet_eval


def test_checkpoint_merge_preserves_single_process_runtime_state():
    runtime_args = pet_eval.get_args_parser().parse_args([])
    runtime_args.distributed = False
    runtime_args.rank = 0
    runtime_args.gpu = 0
    runtime_args.dist_backend = 'nccl'

    checkpoint = {
        'args': {
            'distributed': True,
            'world_size': 8,
            'rank': 7,
            'gpu': 7,
            'dist_backend': 'nccl',
            'dist_url': 'env://',
        },
    }

    merged = pet_eval.merge_checkpoint_args(runtime_args, checkpoint)

    assert merged.distributed is False
    assert merged.world_size == 1
    assert merged.rank == 0
    assert merged.gpu == 0
    assert merged.dist_backend == 'nccl'
    assert merged.dist_url == runtime_args.dist_url
