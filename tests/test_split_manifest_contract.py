from util.splits import build_train_holdout_manifest


def test_train_holdout_manifest_is_deterministic_and_hash_stable():
    counts = list(range(20))
    first = build_train_holdout_manifest(
        len(counts),
        0.2,
        17,
        strategy='count_stratified',
        sample_counts=counts,
        dataset_file='SHA',
        data_path='data/ShanghaiTech/part_A',
    )
    second = build_train_holdout_manifest(
        len(counts),
        0.2,
        17,
        strategy='count_stratified',
        sample_counts=counts,
        dataset_file='SHA',
        data_path='data/ShanghaiTech/part_A',
    )

    train_indices, holdout_indices, manifest = first

    assert first == second
    assert manifest['split_kind'] == 'train_holdout'
    assert manifest['holdout_strategy'] == 'count_stratified'
    assert manifest['holdout_fraction'] == 0.2
    assert manifest['holdout_seed'] == 17
    assert manifest['dataset_file'] == 'SHA'
    assert len(manifest['train_indices']) == 16
    assert len(manifest['holdout_indices']) == 4
    assert set(train_indices).isdisjoint(holdout_indices)
    assert sorted(train_indices + holdout_indices) == list(range(20))
    assert len(manifest['manifest_hash']) == 64
